// Binary FBX for the viewer: the two slow parts of three's FBXLoader done in
// Rust, the rest left to FBXLoader itself.
//
// FBXLoader spends nearly all its time reading the binary file (inflating
// and boxing every array into JS numbers) and turning mesh geometry into
// vertex buffers. Here the file is read into the node list BinaryParser
// reads, and every mesh geometry is expanded exactly as GeometryParser
// expands it: same triangulation (earcut on the projected polygon), same
// layer lookups (getData() with its shared scratch array), same skin weight
// selection and material groups. The JS side (app/vite/fbx-native.js,
// appended to FBXLoader's module) rebuilds FBXLoader's tree from the node
// list with BinaryParser's own code and runs the unchanged FBXTreeParser on
// it, so materials, textures, skeletons and animations are FBXLoader's
// own. Only GeometryParser's buffers come from here, which is why the mesh
// arrays need not be sent at all.
//
// Layout: "3DVF", u32 JSON length, JSON (padded to 8 bytes), binary data.
// The JSON is { "nodes": [...], "geometries": [...] }: a node is
// [name, properties, children, singleProperty]; a property is a number,
// string or boolean, or a tagged array: ["n", "NaN"] for a non-finite
// number, ["r", offset, length] for raw bytes, ["f", offset, count] for an
// f64 array, ["b", offset, count] for a boolean array (bytes), ["x"] for an
// array left out because its geometry comes ready-made.
//
// ASCII files and FBX 6 are left to FBXLoader (LoadError::Unsupported).

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;

use crate::formats::{invalid, LoadError};

const MAGIC: &[u8] = b"Kaydara FBX Binary  \0";

pub fn load(path: &Path) -> Result<Vec<u8>, LoadError> {
    let data = std::fs::read(path)?;
    if !data.starts_with(MAGIC) {
        return Err(LoadError::Unsupported("ASCII FBX".into()));
    }
    let mut nodes = parse(&data)?;
    inflate_all(&mut nodes, &data)?;
    let geometries = build_geometries(&nodes)?;
    // The arrays the geometries were built from are not needed in JS.
    if let Some(objects) = last_top(&mut nodes, "Objects") {
        for g in objects.children.iter_mut().filter(|c| c.name == "Geometry") {
            if let Some(Prop::Num(id)) = g.props.first() {
                if geometries.consumed.contains(&key(*id)) {
                    strip_arrays(g);
                }
            }
        }
    }
    Ok(encode(&nodes, &geometries))
}

// --- reading: BinaryParser / BinaryReader ---------------------------------------

pub(crate) enum Prop {
    Bool(bool),
    Num(f64),
    Str(String),
    Raw(Vec<u8>),
    Nums(Vec<f64>),
    Bools(Vec<bool>),
    /// An array whose content is not sent (see strip_arrays).
    Stripped,
    /// A compressed array not inflated yet: type, length, range in the file.
    Deflated(u8, usize, std::ops::Range<usize>),
}

pub(crate) struct Node {
    name: String,
    props: Vec<Prop>,
    children: Vec<Node>,
    /// BinaryParser's singleProperty: one property and no nested records.
    single: bool,
}

impl Node {
    /// The id FBXLoader sees: the first property when it is a number.
    fn id(&self) -> Option<f64> {
        match self.props.first() {
            Some(Prop::Num(n)) => Some(*n),
            _ => None,
        }
    }

    /// attrType: the third property unless it is an empty string.
    fn attr_type(&self) -> Option<&str> {
        match self.props.get(2) {
            Some(Prop::Str(s)) if !s.is_empty() => Some(s),
            _ => None,
        }
    }
}

struct Reader<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], LoadError> {
        let end = self.offset.checked_add(n).filter(|&e| e <= self.data.len()).ok_or_else(|| invalid("FBX: read past the end"))?;
        let s = &self.data[self.offset..end];
        self.offset = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, LoadError> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, LoadError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, LoadError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    /// getString(): cut at the first NUL ("Name\0\x01Class" reads "Name"),
    /// decoded as TextDecoder does (BOM dropped, invalid bytes replaced).
    fn string(&mut self, n: usize) -> Result<String, LoadError> {
        let bytes = self.take(n)?;
        let bytes = &bytes[..bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len())];
        let bytes = bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(bytes);
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }
}

fn parse(data: &[u8]) -> Result<Vec<Node>, LoadError> {
    let mut r = Reader { data, offset: 23 };
    let version = r.u32()?;
    // FBX 6 identifies objects by name; FBXLoader's handling of it is left
    // to FBXLoader.
    if version < 7000 {
        return Err(LoadError::Unsupported(format!("FBX version {version}")));
    }
    let mut nodes = Vec::new();
    while !end_of_content(r.offset, data.len()) {
        if let Some(node) = parse_node(&mut r, version, 0)? {
            nodes.push(node);
        }
    }
    Ok(nodes)
}

/// The footer: 160 bytes plus padding to 16.
fn end_of_content(offset: usize, size: usize) -> bool {
    if size % 16 == 0 {
        ((offset + 160 + 16) & !0xf) >= size
    } else {
        offset + 160 + 16 >= size
    }
}

fn parse_node(r: &mut Reader, version: u32, depth: usize) -> Result<Option<Node>, LoadError> {
    if depth > 256 {
        return Err(invalid("FBX: nodes nested too deep"));
    }
    let (end, count) = if version >= 7500 {
        let end = r.u64()?;
        let count = r.u64()?;
        r.u64()?;
        (end, count)
    } else {
        let end = r.u32()? as u64;
        let count = r.u32()? as u64;
        r.u32()?;
        (end, count)
    };
    let name_len = r.u8()? as usize;
    let name = r.string(name_len)?;
    if end == 0 {
        return Ok(None);
    }
    let mut props = Vec::new();
    for _ in 0..count {
        props.push(parse_property(r)?);
    }
    let single = count == 1 && r.offset as u64 == end;
    let mut children = Vec::new();
    while end > r.offset as u64 {
        if let Some(child) = parse_node(r, version, depth + 1)? {
            children.push(child);
        }
    }
    Ok(Some(Node { name, props, children, single }))
}

fn parse_property(r: &mut Reader) -> Result<Prop, LoadError> {
    let kind = r.u8()?;
    Ok(match kind {
        b'C' => Prop::Bool(r.u8()? & 1 == 1),
        b'D' => Prop::Num(f64::from_le_bytes(r.take(8)?.try_into().unwrap())),
        b'F' => Prop::Num(f32::from_le_bytes(r.take(4)?.try_into().unwrap()) as f64),
        b'I' => Prop::Num(i32::from_le_bytes(r.take(4)?.try_into().unwrap()) as f64),
        // getInt64() rounds to the nearest double, as `as` does.
        b'L' => Prop::Num(i64::from_le_bytes(r.take(8)?.try_into().unwrap()) as f64),
        b'R' => {
            let n = r.u32()? as usize;
            Prop::Raw(r.take(n)?.to_vec())
        }
        b'S' => {
            let n = r.u32()? as usize;
            Prop::Str(r.string(n)?)
        }
        b'Y' => Prop::Num(i16::from_le_bytes(r.take(2)?.try_into().unwrap()) as f64),
        b'b' | b'c' | b'd' | b'f' | b'i' | b'l' => {
            let len = r.u32()? as usize;
            let encoding = r.u32()?;
            let compressed = r.u32()? as usize;
            let size = element_size(kind);
            let bytes = len.checked_mul(size).ok_or_else(|| invalid("FBX: array too large"))?;
            if encoding == 0 {
                array(kind, r.take(bytes)?)
            } else {
                // Inflated later, all arrays at once on several threads.
                let start = r.offset;
                r.take(compressed)?;
                Prop::Deflated(kind, len, start..r.offset)
            }
        }
        _ => return Err(invalid(format!("FBX: unknown property type {}", kind as char))),
    })
}

/// An array property from its little-endian bytes.
fn array(kind: u8, raw: &[u8]) -> Prop {
    match kind {
        b'b' | b'c' => Prop::Bools(raw.iter().map(|b| b & 1 == 1).collect()),
        b'd' => Prop::Nums(raw.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()),
        b'f' => Prop::Nums(raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64).collect()),
        b'i' => Prop::Nums(raw.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f64).collect()),
        _ => Prop::Nums(raw.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f64).collect()),
    }
}

fn element_size(kind: u8) -> usize {
    match kind {
        b'b' | b'c' => 1,
        b'f' | b'i' => 4,
        _ => 8,
    }
}

/// fflate's unzlibSync: the zlib header is checked, the checksum is not.
fn unzlib(data: &[u8]) -> Result<Vec<u8>, LoadError> {
    if data.len() < 2 || (data[0] & 15) != 8 || (data[0] >> 4) > 7 || ((data[0] as u32) << 8 | data[1] as u32) % 31 != 0 || data[1] & 32 != 0 {
        return Err(invalid("FBX: invalid zlib data"));
    }
    miniz_oxide::inflate::decompress_to_vec(&data[2..]).map_err(|e| invalid(format!("FBX: inflate failed ({:?})", e.status)))
}

/// Inflates every Deflated property, the big ones spread over threads.
fn inflate_all(nodes: &mut [Node], data: &[u8]) -> Result<(), LoadError> {
    fn collect<'n>(nodes: &'n mut [Node], out: &mut Vec<&'n mut Prop>) {
        for n in nodes {
            for p in n.props.iter_mut() {
                if matches!(p, Prop::Deflated(..)) {
                    out.push(p);
                }
            }
            collect(&mut n.children, out);
        }
    }
    let mut pending = Vec::new();
    collect(nodes, &mut pending);
    let inflate = |p: &mut Prop| -> Result<(), LoadError> {
        let Prop::Deflated(kind, len, range) = p else { return Ok(()) };
        let bytes = *len * element_size(*kind);
        let raw = unzlib(&data[range.clone()])?;
        if raw.len() < bytes {
            return Err(invalid("FBX: array shorter than declared"));
        }
        *p = array(*kind, &raw[..bytes]);
        Ok(())
    };
    // Largest first, dealt round-robin: even shares without a queue.
    pending.sort_by_key(|p| std::cmp::Reverse(if let Prop::Deflated(_, _, r) = p { r.len() } else { 0 }));
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(8);
    let total: usize = pending.iter().map(|p| if let Prop::Deflated(_, _, r) = p { r.len() } else { 0 }).sum();
    if threads < 2 || total < 1 << 20 {
        return pending.into_iter().try_for_each(inflate);
    }
    let mut shares: Vec<Vec<&mut Prop>> = (0..threads).map(|_| Vec::new()).collect();
    for (i, p) in pending.into_iter().enumerate() {
        shares[i % threads].push(p);
    }
    std::thread::scope(|scope| {
        let handles: Vec<_> = shares.into_iter().map(|share| scope.spawn(move || share.into_iter().try_for_each(inflate))).collect();
        handles.into_iter().try_for_each(|h| h.join().expect("FBX inflate thread"))
    })
}

fn last_top<'a>(nodes: &'a mut [Node], name: &str) -> Option<&'a mut Node> {
    nodes.iter_mut().rev().find(|n| n.name == name)
}

fn strip_arrays(node: &mut Node) {
    for p in node.props.iter_mut() {
        if matches!(p, Prop::Nums(_) | Prop::Bools(_)) {
            *p = Prop::Stripped;
        }
    }
    for c in node.children.iter_mut() {
        strip_arrays(c);
    }
}

// --- FBXLoader's view of the tree ------------------------------------------------
//
// BinaryParser.parseSubNode() turns children into object properties with
// rules (single-property children become values, repeated named children
// become id-keyed maps, Properties70 entries are copied up...). Geometry
// building reads a handful of such properties; `get` answers what
// `node[name]` would be.

enum Value<'a> {
    Undefined,
    /// A single-property child holding an array: the child, with .a set.
    Array(&'a [f64]),
    /// A single-property child holding anything else.
    Scalar(&'a Prop),
    /// Named children keyed by their numeric ids, first one per id.
    Map(Vec<(f64, &'a Node)>),
    /// Any other object (a child without numeric id, a P entry...).
    Object,
}

impl<'a> Value<'a> {
    /// `.a` of the value, when it is an array.
    fn array(&self) -> Option<&'a [f64]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }

    fn string(&self) -> Option<&'a str> {
        match self {
            Value::Scalar(Prop::Str(s)) => Some(s),
            _ => None,
        }
    }

    /// `value[index]` on an id-keyed map.
    fn entry(&self, index: f64) -> Option<&'a Node> {
        match self {
            Value::Map(m) => m.iter().find(|(id, _)| *id == index).map(|(_, n)| *n),
            _ => None,
        }
    }

    /// JS truthiness.
    fn truthy(&self) -> bool {
        match self {
            Value::Undefined => false,
            Value::Scalar(Prop::Bool(b)) => *b,
            Value::Scalar(Prop::Num(n)) => *n != 0.0 && !n.is_nan(),
            Value::Scalar(Prop::Str(s)) => !s.is_empty(),
            _ => true,
        }
    }
}

/// `node[name]` after BinaryParser has parsed `node`.
fn get<'a>(node: &'a Node, name: &str) -> Value<'a> {
    let mut v = Value::Undefined;
    let mut seen = std::collections::HashSet::new();
    for sub in &node.children {
        if sub.single {
            if sub.name == name {
                v = match &sub.props[0] {
                    Prop::Nums(a) => Value::Array(a),
                    Prop::Bools(_) | Prop::Stripped => Value::Object,
                    p => Value::Scalar(p),
                };
            }
        } else if node.name == "Connections" && sub.name == "C" {
        } else if sub.name == "Properties70" {
            // Its keys are copied into the node.
            if ["singleProperty", "propertyList", "name"].contains(&name) || sub.children.iter().any(|p| p.name == "P" && !p.single && p70_name(p).as_deref() == Some(name)) {
                v = Value::Object;
            }
        } else if node.name == "Properties70" && sub.name == "P" {
            if p70_name(sub).as_deref() == Some(name) {
                v = Value::Object;
            }
        } else if sub.name == name {
            v = match v {
                Value::Undefined => match sub.id() {
                    Some(id) => {
                        seen.insert(key(id));
                        Value::Map(vec![(id, sub)])
                    }
                    None => Value::Object,
                },
                Value::Map(mut m) => {
                    if let Some(id) = sub.id() {
                        if seen.insert(key(id)) {
                            m.push((id, sub));
                        }
                    }
                    Value::Map(m)
                }
                other => other,
            };
        }
    }
    v
}

fn p70_name(p: &Node) -> Option<String> {
    match p.props.first() {
        Some(Prop::Str(s)) => Some(if let Some(rest) = s.strip_prefix("Lcl ") { format!("Lcl_{rest}") } else { s.clone() }),
        _ => None,
    }
}

/// Map key for an id, as JS objects and Maps compare them.
fn key(id: f64) -> u64 {
    if id == 0.0 {
        0
    } else {
        id.to_bits()
    }
}

struct Relationships {
    parents: Vec<(f64, Option<String>)>,
    children: Vec<(f64, Option<String>)>,
}

/// FBXTreeParser.parseConnections().
fn connections(nodes: &[Node]) -> Result<HashMap<u64, Relationships>, LoadError> {
    let mut map: HashMap<u64, Relationships> = HashMap::new();
    let Some(conn) = nodes.iter().rev().find(|n| n.name == "Connections") else { return Ok(map) };
    for c in conn.children.iter().filter(|c| c.name == "C" && !c.single) {
        let (Some(Prop::Num(from)), Some(Prop::Num(to))) = (c.props.get(1), c.props.get(2)) else {
            return Err(LoadError::Unsupported("FBX connection without numeric ids".into()));
        };
        let relationship = match c.props.get(3) {
            None => None,
            Some(Prop::Str(s)) => Some(s.clone()),
            Some(_) => Some(String::new()),
        };
        map.entry(key(*from)).or_insert_with(|| Relationships { parents: vec![], children: vec![] }).parents.push((*to, relationship.clone()));
        map.entry(key(*to)).or_insert_with(|| Relationships { parents: vec![], children: vec![] }).children.push((*from, relationship));
    }
    Ok(map)
}

// --- geometry: GeometryParser ----------------------------------------------------

struct Geometry {
    id: f64,
    skin: Option<f64>,
    position: Vec<f32>,
    normal: Vec<f32>,
    color: Vec<f32>,
    uvs: Vec<Vec<f32>>,
    skin_index: Option<Vec<u16>>,
    skin_weight: Option<Vec<f32>>,
    groups: Option<Vec<(usize, usize, f64)>>,
    morphs: Vec<(f64, Vec<f32>)>,
}

struct Geometries {
    list: Vec<Geometry>,
    /// Geometry nodes whose arrays are no longer needed.
    consumed: std::collections::HashSet<u64>,
}

struct RawBone<'a> {
    indices: &'a [f64],
    weights: &'a [f64],
}

fn unsupported(what: &str) -> LoadError {
    LoadError::Unsupported(format!("FBX: {what}"))
}

fn build_geometries(nodes: &[Node]) -> Result<Geometries, LoadError> {
    let mut out = Geometries { list: Vec::new(), consumed: Default::default() };
    let Some(objects) = nodes.iter().rev().find(|n| n.name == "Objects") else { return Ok(out) };
    let geometry_nodes = match get(objects, "Geometry") {
        Value::Map(m) => m,
        Value::Undefined => return Ok(out),
        _ => return Err(unsupported("unusual Geometry list")),
    };
    let deformers = match get(objects, "Deformer") {
        Value::Map(m) => m,
        Value::Undefined => Vec::new(),
        _ => return Err(unsupported("unusual Deformer list")),
    };
    let deformer_map: HashMap<u64, &Node> = deformers.iter().map(|&(id, n)| (key(id), n)).collect();
    let geometry_map: HashMap<u64, &Node> = geometry_nodes.iter().map(|&(id, n)| (key(id), n)).collect();
    let deformer = |id: f64| deformer_map.get(&key(id)).copied();
    let geometry = |id: f64| geometry_map.get(&key(id)).copied();
    let conns = connections(nodes)?;

    for &(id, geo) in &geometry_nodes {
        if geo.attr_type() != Some("Mesh") {
            continue;
        }
        let rel = conns.get(&key(id)).ok_or_else(|| unsupported("geometry without connections"))?;
        if rel.parents.is_empty() {
            continue;
        }
        // parseMeshGeometry(): the last skin among the children, the blend
        // shapes in order.
        let mut skin = None;
        let mut morph_shapes: Vec<(f64, &Node)> = Vec::new();
        for &(child, _) in &rel.children {
            let Some(d) = deformer(child) else { continue };
            match d.attr_type() {
                Some("Skin") => skin = Some((child, d)),
                Some("BlendShape") => {
                    let channels = conns.get(&key(child)).ok_or_else(|| unsupported("blend shape without connections"))?;
                    for &(channel_id, _) in &channels.children {
                        let channel = deformer(channel_id).ok_or_else(|| unsupported("blend shape channel"))?;
                        if channel.attr_type() != Some("BlendShapeChannel") {
                            break;
                        }
                        let shape_id = conns
                            .get(&key(channel_id))
                            .and_then(|r| r.children.iter().find(|(_, rel)| rel.is_none()))
                            .map(|(id, _)| *id)
                            .ok_or_else(|| unsupported("blend shape channel without shape"))?;
                        if let Some(shape) = geometry(shape_id) {
                            morph_shapes.push((shape_id, shape));
                        }
                    }
                }
                _ => {}
            }
        }
        let bones = match skin {
            Some((skin_id, _)) => {
                let mut bones = Vec::new();
                let rel = conns.get(&key(skin_id)).ok_or_else(|| unsupported("skin without connections"))?;
                for &(child, _) in &rel.children {
                    let bone = deformer(child).ok_or_else(|| unsupported("skin child is not a deformer"))?;
                    if bone.attr_type() != Some("Cluster") {
                        continue;
                    }
                    if matches!(get(bone, "Indexes"), Value::Undefined) {
                        bones.push(RawBone { indices: &[], weights: &[] });
                    } else {
                        let indices = get(bone, "Indexes").array().ok_or_else(|| unsupported("cluster indexes"))?;
                        let weights = get(bone, "Weights").array().ok_or_else(|| unsupported("cluster weights"))?;
                        bones.push(RawBone { indices, weights });
                    }
                }
                Some(bones)
            }
            None => None,
        };

        let info = GeoInfo::new(geo, bones.as_deref())?;
        let buffers = info.buffers()?;
        let mut g = Geometry {
            id,
            skin: skin.map(|(id, _)| id),
            position: buffers.vertex,
            normal: buffers.normal,
            color: buffers.colors,
            uvs: buffers.uvs,
            skin_index: bones.as_ref().map(|_| buffers.weight_indices.iter().map(|&i| to_uint16(i)).collect()),
            skin_weight: bones.as_ref().map(|_| buffers.weights.iter().map(|&w| w as f32).collect()),
            groups: None,
            morphs: Vec::new(),
        };
        if info.material.as_ref().is_some_and(|m| m.mapping != Mapping::AllSame) {
            g.groups = Some(groups(&buffers.material_index)?);
        }
        // genMorphGeometry(): the shape's sparse positions over the base
        // polygons, triangulated as the base is.
        for (shape_id, shape) in morph_shapes {
            let length = g.position.len();
            let mut positions = vec![0f32; length];
            let sparse = array_or_empty(shape, "Vertices")?;
            let indices = array_or_empty(shape, "Indexes")?;
            for (i, &index) in indices.iter().enumerate() {
                let at = index * 3.0;
                for k in 0..3 {
                    let v = sparse.get(i * 3 + k).copied().unwrap_or(f64::NAN);
                    let target = at + k as f64;
                    if let Some(t) = as_index(target).filter(|&t| t < length) {
                        positions[t] = v as f32;
                    }
                }
            }
            let morph_info = GeoInfo {
                vertex_positions: Positions::F32(&positions),
                base_positions: Some(info.vertex_positions_f64()),
                vertex_indices: info.vertex_indices,
                color: None,
                normal: None,
                material: None,
                uvs: None,
                weight_table: None,
            };
            let morph = morph_info.buffers()?;
            out.consumed.insert(key(shape_id));
            g.morphs.push((shape_id, morph.vertex));
        }
        out.consumed.insert(key(id));
        out.list.push(g);
    }
    Ok(out)
}

/// `node.X !== undefined ? node.X.a : []`.
fn array_or_empty<'a>(node: &'a Node, name: &str) -> Result<&'a [f64], LoadError> {
    match get(node, name) {
        Value::Undefined => Ok(&[]),
        v => v.array().ok_or_else(|| unsupported(name)),
    }
}

/// genGeometry(): consecutive runs of one material become groups.
fn groups(material_index: &[f64]) -> Result<Vec<(usize, usize, f64)>, LoadError> {
    for &m in material_index {
        if m.is_nan() || m.trunc() != m {
            return Err(unsupported("material index"));
        }
    }
    let mut groups = Vec::new();
    let Some(&first) = material_index.first() else { return Ok(vec![(0, 0, 0.0)]) };
    let mut prev = first;
    let mut start = 0;
    for (i, &current) in material_index.iter().enumerate() {
        if current != prev {
            groups.push((start, i - start, prev));
            prev = current;
            start = i;
        }
    }
    if let Some(&(s, c, _)) = groups.last() {
        let last = s + c;
        if last != material_index.len() {
            groups.push((last, material_index.len() - last, prev));
        }
    }
    if groups.is_empty() {
        groups.push((0, material_index.len(), first));
    }
    Ok(groups)
}

#[derive(Clone, Copy, PartialEq)]
enum Mapping {
    ByPolygonVertex,
    ByPolygon,
    ByVertice,
    AllSame,
    /// Anything else: getData() finds no index.
    Unknown,
}

struct Layer<'a> {
    size: usize,
    buffer: std::borrow::Cow<'a, [f64]>,
    /// None: the indices are 0, 1, 2... (material layers).
    indices: Option<&'a [f64]>,
    identity_len: usize,
    mapping: Mapping,
    index_to_direct: bool,
}

impl<'a> Layer<'a> {
    /// `required`: the index array must be there (FBXLoader reads it
    /// unconditionally for colours and UVs).
    fn new(node: &'a Node, data: &str, index_names: &[&str], required: bool, size: usize) -> Result<Layer<'a>, LoadError> {
        let mapping = mapping(get(node, "MappingInformationType").string());
        let index_to_direct = get(node, "ReferenceInformationType").string() == Some("IndexToDirect");
        let buffer = get(node, data).array().ok_or_else(|| unsupported(data))?;
        let mut indices: &[f64] = &[];
        if index_to_direct {
            for name in index_names {
                let v = get(node, name);
                if !matches!(v, Value::Undefined) {
                    indices = v.array().ok_or_else(|| unsupported(name))?;
                    break;
                }
            }
            if required && matches!(get(node, index_names[0]), Value::Undefined) {
                return Err(unsupported(index_names[0]));
            }
        }
        Ok(Layer { size, buffer: buffer.into(), indices: Some(indices), identity_len: 0, mapping, index_to_direct })
    }

    fn index(&self, i: f64) -> f64 {
        let Some(u) = as_index(i) else { return f64::NAN };
        match self.indices {
            Some(ix) => ix.get(u).copied().unwrap_or(f64::NAN),
            None => {
                if u < self.identity_len {
                    i
                } else {
                    f64::NAN
                }
            }
        }
    }

    /// getData(): copies the item into the shared scratch array; an index
    /// that resolves to nothing leaves the scratch as it was, and the
    /// answer is false.
    fn get(&self, polygon_vertex: usize, polygon: usize, vertex: f64, scratch: &mut [f64; 4]) -> bool {
        let mut index = match self.mapping {
            Mapping::ByPolygonVertex => polygon_vertex as f64,
            Mapping::ByPolygon => polygon as f64,
            Mapping::ByVertice => vertex,
            Mapping::AllSame => self.index(0.0),
            Mapping::Unknown => f64::NAN,
        };
        if self.index_to_direct {
            index = self.index(index);
        }
        let from = index * self.size as f64;
        if from.is_nan() {
            return false;
        }
        for (j, slot) in scratch.iter_mut().enumerate().take(self.size) {
            let i = from + j as f64;
            *slot = as_index(i).and_then(|i| self.buffer.get(i).copied()).unwrap_or(f64::NAN);
        }
        true
    }
}

fn mapping(s: Option<&str>) -> Mapping {
    match s {
        Some("ByPolygonVertex") => Mapping::ByPolygonVertex,
        Some("ByPolygon") => Mapping::ByPolygon,
        Some("ByVertice") => Mapping::ByVertice,
        Some("AllSame") => Mapping::AllSame,
        _ => Mapping::Unknown,
    }
}

enum Positions<'a> {
    F64(&'a [f64]),
    F32(&'a [f32]),
}

impl Positions<'_> {
    fn at(&self, i: f64) -> f64 {
        let Some(i) = as_index(i) else { return f64::NAN };
        match self {
            Positions::F64(p) => p.get(i).copied().unwrap_or(f64::NAN),
            Positions::F32(p) => p.get(i).map(|&v| v as f64).unwrap_or(f64::NAN),
        }
    }
}

struct GeoInfo<'a> {
    vertex_positions: Positions<'a>,
    base_positions: Option<&'a [f64]>,
    vertex_indices: &'a [f64],
    color: Option<Layer<'a>>,
    normal: Option<Layer<'a>>,
    material: Option<Layer<'a>>,
    /// Some(empty) when the geometry has LayerElementUV nodes but no UVs.
    uvs: Option<Vec<Layer<'a>>>,
    /// Per control point: (bone, weight) in rawBones order.
    weight_table: Option<Vec<Vec<(f64, f64)>>>,
}

#[derive(Default)]
struct Buffers {
    vertex: Vec<f32>,
    normal: Vec<f32>,
    colors: Vec<f32>,
    uvs: Vec<Vec<f32>>,
    material_index: Vec<f64>,
    weights: Vec<f64>,
    weight_indices: Vec<f64>,
}

impl<'a> GeoInfo<'a> {
    /// parseGeoNode().
    fn new(geo: &'a Node, bones: Option<&[RawBone<'a>]>) -> Result<GeoInfo<'a>, LoadError> {
        let vertex_positions = array_or_empty(geo, "Vertices")?;
        let vertex_indices = array_or_empty(geo, "PolygonVertexIndex")?;
        let first_layer = |name: &str| -> Result<Option<&'a Node>, LoadError> {
            let v = get(geo, name);
            if !v.truthy() {
                return Ok(None);
            }
            v.entry(0.0).map(Some).ok_or_else(|| unsupported(name))
        };
        let mut color = None;
        if let Some(node) = first_layer("LayerElementColor")? {
            if get(node, "Colors").truthy() {
                let mut layer = Layer::new(node, "Colors", &["ColorIndex"], true, 4)?;
                if layer.buffer.len() % 4 != 0 {
                    return Err(unsupported("colour count"));
                }
                // parseVertexColors(): sRGB to linear, in place.
                let mut b = layer.buffer.into_owned();
                for c in b.chunks_exact_mut(4) {
                    for v in &mut c[..3] {
                        *v = srgb_to_linear(*v);
                    }
                }
                layer.buffer = b.into();
                color = Some(layer);
            }
        }
        let mut material = None;
        if let Some(node) = first_layer("LayerElementMaterial")? {
            let mapping = mapping(get(node, "MappingInformationType").string());
            let index_to_direct = get(node, "ReferenceInformationType").string() == Some("IndexToDirect");
            if get(node, "MappingInformationType").string() == Some("NoMappingInformation") {
                material = Some(Layer { size: 1, buffer: vec![0.0].into(), indices: Some(&[0.0]), identity_len: 0, mapping: Mapping::AllSame, index_to_direct });
            } else {
                let buffer = get(node, "Materials").array().ok_or_else(|| unsupported("material indices"))?;
                material = Some(Layer { size: 1, buffer: buffer.into(), indices: None, identity_len: buffer.len(), mapping, index_to_direct });
            }
        }
        let mut normal = None;
        if let Some(node) = first_layer("LayerElementNormal")? {
            normal = Some(Layer::new(node, "Normals", &["NormalIndex", "NormalsIndex"], false, 3)?);
        }
        let mut uvs = None;
        let uv_value = get(geo, "LayerElementUV");
        if uv_value.truthy() {
            let mut list = Vec::new();
            let mut i = 0.0;
            while let Some(node) = uv_value.entry(i) {
                if get(node, "UV").truthy() {
                    list.push(Layer::new(node, "UV", &["UVIndex"], true, 2)?);
                }
                i += 1.0;
            }
            uvs = Some(list);
        }
        let weight_table = bones.map(|bones| {
            // Only control points that faces use are looked up (and only
            // ones with a position are worth keeping).
            let max = vertex_indices.iter().map(|&v| control_point(v)).fold(-1.0f64, f64::max).min((vertex_positions.len() / 3) as f64);
            let mut table: Vec<Vec<(f64, f64)>> = vec![Vec::new(); (max + 1.0).max(0.0) as usize];
            for (i, bone) in bones.iter().enumerate() {
                for (j, &index) in bone.indices.iter().enumerate() {
                    if let Some(index) = as_index(index).filter(|&i| i < table.len()) {
                        table[index].push((i as f64, bone.weights.get(j).copied().unwrap_or(f64::NAN)));
                    }
                }
            }
            table
        });
        Ok(GeoInfo { vertex_positions: Positions::F64(vertex_positions), base_positions: None, vertex_indices, color, normal, material, uvs, weight_table })
    }

    fn vertex_positions_f64(&self) -> &'a [f64] {
        match self.vertex_positions {
            Positions::F64(p) => p,
            Positions::F32(_) => &[],
        }
    }

    /// genBuffers() + genFace(), on several threads for large meshes: the
    /// polygons are split into runs built separately and joined. That is
    /// the loader's result as long as no lookup falls back on the scratch
    /// array's previous contents (an invalid layer index); if one does,
    /// the mesh is built again in one go.
    fn buffers(&self) -> Result<Buffers, LoadError> {
        let indices = self.vertex_indices;
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(8);
        if threads < 2 || indices.len() < 100_000 {
            return Ok(self.buffers_range(0, indices.len(), 0).0);
        }
        // Runs end after a polygon's last vertex (a negative index).
        let mut cuts = vec![(0usize, 0usize)];
        let step = indices.len() / threads;
        let mut polygons = 0;
        let mut next = step;
        for (i, &v) in indices.iter().enumerate() {
            if v < 0.0 {
                polygons += 1;
                if i + 1 >= next && i + 1 < indices.len() {
                    cuts.push((i + 1, polygons));
                    next = i + 1 + step;
                }
            }
        }
        cuts.push((indices.len(), polygons));
        let parts: Vec<(Buffers, bool)> = std::thread::scope(|scope| {
            let handles: Vec<_> = cuts.windows(2).map(|w| scope.spawn(move || self.buffers_range(w[0].0, w[1].0, w[0].1))).collect();
            handles.into_iter().map(|h| h.join().expect("FBX geometry thread")).collect()
        });
        if parts.iter().any(|(_, stale)| *stale) {
            return Ok(self.buffers_range(0, indices.len(), 0).0);
        }
        let mut b = Buffers::default();
        let total = |f: fn(&Buffers) -> usize| parts.iter().map(|(p, _)| f(p)).sum::<usize>();
        b.vertex.reserve(total(|p| p.vertex.len()));
        b.normal.reserve(total(|p| p.normal.len()));
        b.colors.reserve(total(|p| p.colors.len()));
        b.material_index.reserve(total(|p| p.material_index.len()));
        b.weights.reserve(total(|p| p.weights.len()));
        b.weight_indices.reserve(total(|p| p.weight_indices.len()));
        for (p, _) in &parts {
            b.vertex.extend_from_slice(&p.vertex);
            b.normal.extend_from_slice(&p.normal);
            b.colors.extend_from_slice(&p.colors);
            b.material_index.extend_from_slice(&p.material_index);
            b.weights.extend_from_slice(&p.weights);
            b.weight_indices.extend_from_slice(&p.weight_indices);
            // A run without triangles has no UV arrays yet.
            if b.uvs.len() < p.uvs.len() {
                b.uvs.resize(p.uvs.len(), Vec::new());
            }
            for (j, uv) in p.uvs.iter().enumerate() {
                b.uvs[j].extend_from_slice(uv);
            }
        }
        Ok(b)
    }

    /// genBuffers() over the polygons in vertex_indices[start..end], the
    /// first of them polygon number `first_polygon`. Also answers whether
    /// a lookup used the scratch array's previous contents.
    fn buffers_range(&self, start: usize, end: usize, first_polygon: usize) -> (Buffers, bool) {
        let mut b = Buffers::default();
        let mut stale = false;
        // Quads, the usual case, expand four indices into six vertices.
        let estimate = (end - start) * 3 / 2;
        b.vertex.reserve(estimate * 3);
        if self.normal.is_some() {
            b.normal.reserve(estimate * 3);
        }
        let uv_count = self.uvs.as_ref().map_or(0, |u| u.len());
        let mut scratch = [f64::NAN; 4];
        let mut polygon = first_polygon;
        let mut face = Face::default();
        face.uvs.resize(uv_count, Vec::new());
        let material_mapped = self.material.as_ref().is_some_and(|m| m.mapping != Mapping::AllSame);
        let mut tri = Triangulator::default();

        for (polygon_vertex, &raw) in self.vertex_indices.iter().enumerate().take(end).skip(start) {
            let mut vertex = raw;
            let mut end_of_face = false;
            if vertex < 0.0 {
                vertex = (to_int32(vertex) ^ -1) as f64;
                end_of_face = true;
            }
            face.positions.extend_from_slice(&[vertex * 3.0, vertex * 3.0 + 1.0, vertex * 3.0 + 2.0]);
            if let Some(color) = &self.color {
                stale |= !color.get(polygon_vertex, polygon, vertex, &mut scratch);
                face.colors.extend_from_slice(&scratch[..3]);
            }
            if let Some(table) = &self.weight_table {
                let entries: &[(f64, f64)] = as_index(vertex).and_then(|v| table.get(v)).map_or(&[], |v| v);
                // Up to four as they come, zero-padded; more go through the
                // loader's insertion into a top four.
                let mut weights = [0.0f64; 4];
                let mut indices = [0.0f64; 4];
                if entries.len() <= 4 {
                    for (k, &(index, weight)) in entries.iter().enumerate() {
                        weights[k] = weight;
                        indices[k] = index;
                    }
                } else {
                    for &(index, weight) in entries {
                        let mut current_weight = weight;
                        let mut current_index = index;
                        for k in 0..4 {
                            if current_weight > weights[k] {
                                std::mem::swap(&mut weights[k], &mut current_weight);
                                std::mem::swap(&mut indices[k], &mut current_index);
                            }
                        }
                    }
                }
                face.weights.extend_from_slice(&weights);
                face.weight_indices.extend_from_slice(&indices);
            }
            if let Some(normal) = &self.normal {
                stale |= !normal.get(polygon_vertex, polygon, vertex, &mut scratch);
                face.normals.extend_from_slice(&scratch[..3]);
            }
            let mut material_index = f64::NAN;
            if material_mapped {
                stale |= !self.material.as_ref().unwrap().get(polygon_vertex, polygon, vertex, &mut scratch);
                material_index = scratch[0];
                if material_index < 0.0 {
                    material_index = 0.0;
                }
            }
            if let Some(uvs) = &self.uvs {
                for (i, uv) in uvs.iter().enumerate() {
                    stale |= !uv.get(polygon_vertex, polygon, vertex, &mut scratch);
                    face.uvs[i].extend_from_slice(&scratch[..2]);
                }
            }
            face.length += 1;
            if end_of_face {
                self.gen_face(&mut b, &face, material_index, material_mapped, uv_count, &mut tri);
                polygon += 1;
                face.clear();
            }
        }
        (b, stale)
    }

    fn gen_face(&self, b: &mut Buffers, face: &Face, material_index: f64, material_mapped: bool, uv_count: usize, tri: &mut Triangulator) {
        const SINGLE: [[usize; 3]; 1] = [[0, 1, 2]];
        let triangles: &[[usize; 3]] = if face.length > 3 {
            let positions = match self.base_positions {
                Some(p) => Positions::F64(p),
                None => match self.vertex_positions {
                    Positions::F64(p) => Positions::F64(p),
                    Positions::F32(p) => Positions::F32(p),
                },
            };
            tri.vertices.clear();
            tri.vertices.extend(face.positions.chunks_exact(3).map(|c| [positions.at(c[0]), positions.at(c[1]), positions.at(c[2])]));
            tri.triangulate_face();
            &tri.faces
        } else {
            &SINGLE
        };
        let at = |v: &[f64], i: usize| v.get(i).copied().unwrap_or(f64::NAN);
        for &t in triangles {
            for &i in &t {
                for k in 0..3 {
                    b.vertex.push(self.vertex_positions.at(at(&face.positions, i * 3 + k)) as f32);
                }
            }
            if self.weight_table.is_some() {
                for &i in &t {
                    for k in 0..4 {
                        b.weights.push(at(&face.weights, i * 4 + k));
                    }
                }
                for &i in &t {
                    for k in 0..4 {
                        b.weight_indices.push(at(&face.weight_indices, i * 4 + k));
                    }
                }
            }
            if self.color.is_some() {
                for &i in &t {
                    for k in 0..3 {
                        b.colors.push(at(&face.colors, i * 3 + k) as f32);
                    }
                }
            }
            if material_mapped {
                b.material_index.extend_from_slice(&[material_index; 3]);
            }
            if self.normal.is_some() {
                for &i in &t {
                    for k in 0..3 {
                        b.normal.push(at(&face.normals, i * 3 + k) as f32);
                    }
                }
            }
            if uv_count > 0 {
                b.uvs.resize(uv_count, Vec::new());
                for (j, uv) in face.uvs.iter().enumerate() {
                    for &i in &t {
                        b.uvs[j].push(at(uv, i * 2) as f32);
                        b.uvs[j].push(at(uv, i * 2 + 1) as f32);
                    }
                }
            }
        }
    }
}

#[derive(Default)]
struct Face {
    length: usize,
    positions: Vec<f64>,
    normals: Vec<f64>,
    colors: Vec<f64>,
    uvs: Vec<Vec<f64>>,
    weights: Vec<f64>,
    weight_indices: Vec<f64>,
}

impl Face {
    fn clear(&mut self) {
        self.length = 0;
        self.positions.clear();
        self.normals.clear();
        self.colors.clear();
        for uv in &mut self.uvs {
            uv.clear();
        }
        self.weights.clear();
        self.weight_indices.clear();
    }
}

/// A JS number used as an array index: a non-negative integer, else none.
#[inline]
fn as_index(i: f64) -> Option<usize> {
    // The casts are exact for every integer below 2^53 and reject the rest.
    if i >= 0.0 && i < 9.0e15 {
        let u = i as usize;
        if u as f64 == i {
            return Some(u);
        }
    }
    None
}

/// The control point a PolygonVertexIndex entry names.
fn control_point(v: f64) -> f64 {
    if v < 0.0 {
        (to_int32(v) ^ -1) as f64
    } else {
        v
    }
}

/// ECMAScript ToInt32.
fn to_int32(v: f64) -> i32 {
    // Integers in range, the usual case, convert directly.
    if v >= -2147483648.0 && v <= 2147483647.0 && v as i32 as f64 == v {
        return v as i32;
    }
    if !v.is_finite() {
        return 0;
    }
    (v.trunc().rem_euclid(4294967296.0) as u64 as u32) as i32
}

/// ECMAScript ToUint16 (Uint16Array element conversion).
fn to_uint16(v: f64) -> u16 {
    if (0.0..65536.0).contains(&v) && v as u16 as f64 == v {
        return v as u16;
    }
    if !v.is_finite() {
        return 0;
    }
    v.trunc().rem_euclid(65536.0) as u16
}

/// three's SRGBToLinear, in double precision.
fn srgb_to_linear(c: f64) -> f64 {
    if c < 0.04045 {
        c * 0.0773993808
    } else {
        (c * 0.9478672986 + 0.0521327014).powf(2.4)
    }
}

// --- triangulation: genFace() polygons, ShapeUtils.triangulateShape, earcut ------

/// Math.min / Math.max: NaN wins.
fn js_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a < b || (a == b && a.is_sign_negative()) {
        a
    } else {
        b
    }
}

fn js_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a > b || (a == b && b.is_sign_negative()) {
        a
    } else {
        b
    }
}

/// JS truthiness of a number.
fn truthy(v: f64) -> bool {
    v != 0.0 && !v.is_nan()
}

type V3 = [f64; 3];

fn length(v: V3) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Vector3.normalize(): divideScalar(length || 1).
fn normalize(v: V3) -> V3 {
    let l = length(v);
    let s = 1.0 / if truthy(l) { l } else { 1.0 };
    [v[0] * s, v[1] * s, v[2] * s]
}

fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[derive(Default)]
struct Triangulator {
    /// The polygon to triangulate.
    vertices: Vec<V3>,
    /// The result: triangles of polygon vertex indices.
    faces: Vec<[usize; 3]>,
    nodes: Vec<ENode>,
    data: Vec<f64>,
    triangles: Vec<usize>,
}

const NIL: usize = usize::MAX;

#[derive(Clone)]
struct ENode {
    i: usize,
    x: f64,
    y: f64,
    prev: usize,
    next: usize,
    z: i32,
    prev_z: usize,
    next_z: usize,
    steiner: bool,
}

impl Triangulator {
    /// genFace() for more than three vertices: projected onto the plane of
    /// the Newell normal, then ShapeUtils.triangulateShape(contour, []).
    fn triangulate_face(&mut self) {
        let vertices = &self.vertices;
        // getNormalNewell
        let mut n = [0.0f64; 3];
        for i in 0..vertices.len() {
            let c = vertices[i];
            let x = vertices[(i + 1) % vertices.len()];
            n[0] += (c[1] - x[1]) * (c[2] + x[2]);
            n[1] += (c[2] - x[2]) * (c[0] + x[0]);
            n[2] += (c[0] - x[0]) * (c[1] + x[1]);
        }
        let normal = normalize(n);
        let up = if normal[2].abs() > 0.5 { [0.0, 1.0, 0.0] } else { [0.0, 0.0, 1.0] };
        let tangent = normalize(cross(up, normal));
        let bitangent = normalize(cross(normal, tangent));
        self.data.clear();
        for &v in vertices {
            self.data.push(dot(v, tangent));
            self.data.push(dot(v, bitangent));
        }
        // removeDupEndPts
        let l = self.data.len();
        if l > 4 && self.data[l - 2] == self.data[0] && self.data[l - 1] == self.data[1] {
            self.data.truncate(l - 2);
        }
        self.earcut();
        self.faces.clear();
        self.faces.extend(self.triangles.chunks_exact(3).map(|c| [c[0], c[1], c[2]]));
    }

    // earcut 3.0.2, without holes, dim 2.
    fn earcut(&mut self) {
        self.nodes.clear();
        self.triangles.clear();
        let outer_len = self.data.len();
        let outer = self.linked_list(0, outer_len, true);
        if outer == NIL || self.nodes[outer].next == self.nodes[outer].prev {
            return;
        }
        let (mut min_x, mut min_y, mut inv_size) = (f64::NAN, f64::NAN, 0.0);
        let hashed = self.data.len() > 80 * 2;
        if hashed {
            min_x = self.data[0];
            min_y = self.data[1];
            let mut max_x = min_x;
            let mut max_y = min_y;
            let mut i = 2;
            while i < outer_len {
                let x = self.data[i];
                let y = self.data[i + 1];
                if x < min_x {
                    min_x = x;
                }
                if y < min_y {
                    min_y = y;
                }
                if x > max_x {
                    max_x = x;
                }
                if y > max_y {
                    max_y = y;
                }
                i += 2;
            }
            inv_size = js_max(max_x - min_x, max_y - min_y);
            inv_size = if inv_size != 0.0 { 32767.0 / inv_size } else { 0.0 };
        }
        self.earcut_linked(outer, min_x, min_y, inv_size, 0);
    }

    fn create_node(&mut self, i: usize, x: f64, y: f64) -> usize {
        self.nodes.push(ENode { i, x, y, prev: NIL, next: NIL, z: 0, prev_z: NIL, next_z: NIL, steiner: false });
        self.nodes.len() - 1
    }

    fn insert_node(&mut self, i: usize, x: f64, y: f64, last: usize) -> usize {
        let p = self.create_node(i, x, y);
        if last == NIL {
            self.nodes[p].prev = p;
            self.nodes[p].next = p;
        } else {
            let ln = self.nodes[last].next;
            self.nodes[p].next = ln;
            self.nodes[p].prev = last;
            self.nodes[ln].prev = p;
            self.nodes[last].next = p;
        }
        p
    }

    fn remove_node(&mut self, p: usize) {
        let (prev, next, pz, nz) = (self.nodes[p].prev, self.nodes[p].next, self.nodes[p].prev_z, self.nodes[p].next_z);
        self.nodes[next].prev = prev;
        self.nodes[prev].next = next;
        if pz != NIL {
            self.nodes[pz].next_z = nz;
        }
        if nz != NIL {
            self.nodes[nz].prev_z = pz;
        }
    }

    fn signed_area(&self, start: usize, end: usize) -> f64 {
        let mut sum = 0.0;
        let mut j = end.wrapping_sub(2);
        let mut i = start;
        while i < end {
            sum += (self.data[j] - self.data[i]) * (self.data[i + 1] + self.data[j + 1]);
            j = i;
            i += 2;
        }
        sum
    }

    fn linked_list(&mut self, start: usize, end: usize, clockwise: bool) -> usize {
        let mut last = NIL;
        if end < start + 2 {
            return NIL;
        }
        if clockwise == (self.signed_area(start, end) > 0.0) {
            let mut i = start;
            while i < end {
                last = self.insert_node(i / 2, self.data[i], self.data[i + 1], last);
                i += 2;
            }
        } else {
            let mut i = end as isize - 2;
            while i >= start as isize {
                let u = i as usize;
                last = self.insert_node(u / 2, self.data[u], self.data[u + 1], last);
                i -= 2;
            }
        }
        if last != NIL && self.equals(last, self.nodes[last].next) {
            self.remove_node(last);
            last = self.nodes[last].next;
        }
        last
    }

    fn equals(&self, a: usize, b: usize) -> bool {
        self.nodes[a].x == self.nodes[b].x && self.nodes[a].y == self.nodes[b].y
    }

    fn area(&self, p: usize, q: usize, r: usize) -> f64 {
        let (p, q, r) = (&self.nodes[p], &self.nodes[q], &self.nodes[r]);
        (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
    }

    fn filter_points(&mut self, start: usize, end: usize) -> usize {
        if start == NIL {
            return start;
        }
        let mut end = if end == NIL { start } else { end };
        let mut p = start;
        loop {
            let mut again = false;
            let next = self.nodes[p].next;
            let prev = self.nodes[p].prev;
            if !self.nodes[p].steiner && (self.equals(p, next) || self.area(prev, p, next) == 0.0) {
                self.remove_node(p);
                p = self.nodes[p].prev;
                end = p;
                if p == self.nodes[p].next {
                    break;
                }
                again = true;
            } else {
                p = next;
            }
            if !(again || p != end) {
                break;
            }
        }
        end
    }

    fn earcut_linked(&mut self, ear: usize, min_x: f64, min_y: f64, inv_size: f64, pass: u8) {
        if ear == NIL {
            return;
        }
        let mut ear = ear;
        if pass == 0 && truthy(inv_size) {
            self.index_curve(ear, min_x, min_y, inv_size);
        }
        let mut stop = ear;
        while self.nodes[ear].prev != self.nodes[ear].next {
            let prev = self.nodes[ear].prev;
            let next = self.nodes[ear].next;
            let is_ear = if truthy(inv_size) { self.is_ear_hashed(ear, min_x, min_y, inv_size) } else { self.is_ear(ear) };
            if is_ear {
                self.triangles.extend_from_slice(&[self.nodes[prev].i, self.nodes[ear].i, self.nodes[next].i]);
                self.remove_node(ear);
                ear = self.nodes[next].next;
                stop = self.nodes[next].next;
                continue;
            }
            ear = next;
            if ear == stop {
                if pass == 0 {
                    let f = self.filter_points(ear, NIL);
                    self.earcut_linked(f, min_x, min_y, inv_size, 1);
                } else if pass == 1 {
                    let f = self.filter_points(ear, NIL);
                    let e = self.cure_local_intersections(f);
                    self.earcut_linked(e, min_x, min_y, inv_size, 2);
                } else if pass == 2 {
                    self.split_earcut(ear, min_x, min_y, inv_size);
                }
                break;
            }
        }
    }

    fn bbox(&self, a: usize, b: usize, c: usize) -> (f64, f64, f64, f64) {
        let (a, b, c) = (&self.nodes[a], &self.nodes[b], &self.nodes[c]);
        (js_min(js_min(a.x, b.x), c.x), js_min(js_min(a.y, b.y), c.y), js_max(js_max(a.x, b.x), c.x), js_max(js_max(a.y, b.y), c.y))
    }

    fn blocks(&self, p: usize, a: usize, b: usize, c: usize, bb: (f64, f64, f64, f64), check_ends: bool) -> bool {
        let n = &self.nodes[p];
        let (x0, y0, x1, y1) = bb;
        n.x >= x0
            && n.x <= x1
            && n.y >= y0
            && n.y <= y1
            && (!check_ends || (p != a && p != c))
            && point_in_triangle_except_first(self.nodes[a].x, self.nodes[a].y, self.nodes[b].x, self.nodes[b].y, self.nodes[c].x, self.nodes[c].y, n.x, n.y)
            && self.area(n.prev, p, n.next) >= 0.0
    }

    fn is_ear(&self, ear: usize) -> bool {
        let a = self.nodes[ear].prev;
        let b = ear;
        let c = self.nodes[ear].next;
        if self.area(a, b, c) >= 0.0 {
            return false;
        }
        let bb = self.bbox(a, b, c);
        let mut p = self.nodes[c].next;
        while p != a {
            if self.blocks(p, a, b, c, bb, false) {
                return false;
            }
            p = self.nodes[p].next;
        }
        true
    }

    fn is_ear_hashed(&self, ear: usize, min_x: f64, min_y: f64, inv_size: f64) -> bool {
        let a = self.nodes[ear].prev;
        let b = ear;
        let c = self.nodes[ear].next;
        if self.area(a, b, c) >= 0.0 {
            return false;
        }
        let bb = self.bbox(a, b, c);
        let min_z = z_order(bb.0, bb.1, min_x, min_y, inv_size);
        let max_z = z_order(bb.2, bb.3, min_x, min_y, inv_size);
        let mut p = self.nodes[ear].prev_z;
        let mut n = self.nodes[ear].next_z;
        while p != NIL && self.nodes[p].z >= min_z && n != NIL && self.nodes[n].z <= max_z {
            if self.blocks(p, a, b, c, bb, true) {
                return false;
            }
            p = self.nodes[p].prev_z;
            if self.blocks(n, a, b, c, bb, true) {
                return false;
            }
            n = self.nodes[n].next_z;
        }
        while p != NIL && self.nodes[p].z >= min_z {
            if self.blocks(p, a, b, c, bb, true) {
                return false;
            }
            p = self.nodes[p].prev_z;
        }
        while n != NIL && self.nodes[n].z <= max_z {
            if self.blocks(n, a, b, c, bb, true) {
                return false;
            }
            n = self.nodes[n].next_z;
        }
        true
    }

    fn cure_local_intersections(&mut self, start: usize) -> usize {
        let mut start = start;
        let mut p = start;
        loop {
            let a = self.nodes[p].prev;
            let pn = self.nodes[p].next;
            let b = self.nodes[pn].next;
            if !self.equals(a, b) && self.intersects(a, p, pn, b) && self.locally_inside(a, b) && self.locally_inside(b, a) {
                self.triangles.extend_from_slice(&[self.nodes[a].i, self.nodes[p].i, self.nodes[b].i]);
                self.remove_node(p);
                self.remove_node(pn);
                p = b;
                start = b;
            }
            p = self.nodes[p].next;
            if p == start {
                break;
            }
        }
        self.filter_points(p, NIL)
    }

    fn split_earcut(&mut self, start: usize, min_x: f64, min_y: f64, inv_size: f64) {
        let mut a = start;
        loop {
            let mut b = self.nodes[self.nodes[a].next].next;
            while b != self.nodes[a].prev {
                if self.nodes[a].i != self.nodes[b].i && self.is_valid_diagonal(a, b) {
                    let c = self.split_polygon(a, b);
                    let an = self.nodes[a].next;
                    let a2 = self.filter_points(a, an);
                    let cn = self.nodes[c].next;
                    let c2 = self.filter_points(c, cn);
                    self.earcut_linked(a2, min_x, min_y, inv_size, 0);
                    self.earcut_linked(c2, min_x, min_y, inv_size, 0);
                    return;
                }
                b = self.nodes[b].next;
            }
            a = self.nodes[a].next;
            if a == start {
                break;
            }
        }
    }

    fn index_curve(&mut self, start: usize, min_x: f64, min_y: f64, inv_size: f64) {
        let mut p = start;
        loop {
            if self.nodes[p].z == 0 {
                self.nodes[p].z = z_order(self.nodes[p].x, self.nodes[p].y, min_x, min_y, inv_size);
            }
            self.nodes[p].prev_z = self.nodes[p].prev;
            self.nodes[p].next_z = self.nodes[p].next;
            p = self.nodes[p].next;
            if p == start {
                break;
            }
        }
        let pz = self.nodes[p].prev_z;
        self.nodes[pz].next_z = NIL;
        self.nodes[p].prev_z = NIL;
        self.sort_linked(p);
    }

    fn sort_linked(&mut self, list: usize) {
        let mut list = list;
        let mut in_size = 1;
        loop {
            let mut p = list;
            list = NIL;
            let mut tail = NIL;
            let mut num_merges = 0;
            while p != NIL {
                num_merges += 1;
                let mut q = p;
                let mut p_size = 0;
                for _ in 0..in_size {
                    p_size += 1;
                    q = self.nodes[q].next_z;
                    if q == NIL {
                        break;
                    }
                }
                let mut q_size = in_size;
                while p_size > 0 || (q_size > 0 && q != NIL) {
                    let e;
                    if p_size != 0 && (q_size == 0 || q == NIL || self.nodes[p].z <= self.nodes[q].z) {
                        e = p;
                        p = self.nodes[p].next_z;
                        p_size -= 1;
                    } else {
                        e = q;
                        q = self.nodes[q].next_z;
                        q_size -= 1;
                    }
                    if tail != NIL {
                        self.nodes[tail].next_z = e;
                    } else {
                        list = e;
                    }
                    self.nodes[e].prev_z = tail;
                    tail = e;
                }
                p = q;
            }
            self.nodes[tail].next_z = NIL;
            in_size *= 2;
            if num_merges <= 1 {
                break;
            }
        }
    }

    fn is_valid_diagonal(&self, a: usize, b: usize) -> bool {
        let (na, nb) = (&self.nodes[a], &self.nodes[b]);
        self.nodes[na.next].i != nb.i
            && self.nodes[na.prev].i != nb.i
            && !self.intersects_polygon(a, b)
            && ((self.locally_inside(a, b) && self.locally_inside(b, a) && self.middle_inside(a, b) && (truthy(self.area(na.prev, a, nb.prev)) || truthy(self.area(a, nb.prev, b))))
                || (self.equals(a, b) && self.area(na.prev, a, na.next) > 0.0 && self.area(nb.prev, b, nb.next) > 0.0))
    }

    fn intersects(&self, p1: usize, q1: usize, p2: usize, q2: usize) -> bool {
        let o1 = sign(self.area(p1, q1, p2));
        let o2 = sign(self.area(p1, q1, q2));
        let o3 = sign(self.area(p2, q2, p1));
        let o4 = sign(self.area(p2, q2, q1));
        if o1 != o2 && o3 != o4 {
            return true;
        }
        (o1 == 0 && self.on_segment(p1, p2, q1)) || (o2 == 0 && self.on_segment(p1, q2, q1)) || (o3 == 0 && self.on_segment(p2, p1, q2)) || (o4 == 0 && self.on_segment(p2, q1, q2))
    }

    fn on_segment(&self, p: usize, q: usize, r: usize) -> bool {
        let (p, q, r) = (&self.nodes[p], &self.nodes[q], &self.nodes[r]);
        q.x <= js_max(p.x, r.x) && q.x >= js_min(p.x, r.x) && q.y <= js_max(p.y, r.y) && q.y >= js_min(p.y, r.y)
    }

    fn intersects_polygon(&self, a: usize, b: usize) -> bool {
        let (ai, bi) = (self.nodes[a].i, self.nodes[b].i);
        let mut p = a;
        loop {
            let pn = self.nodes[p].next;
            if self.nodes[p].i != ai && self.nodes[pn].i != ai && self.nodes[p].i != bi && self.nodes[pn].i != bi && self.intersects(p, pn, a, b) {
                return true;
            }
            p = pn;
            if p == a {
                return false;
            }
        }
    }

    fn locally_inside(&self, a: usize, b: usize) -> bool {
        let (prev, next) = (self.nodes[a].prev, self.nodes[a].next);
        if self.area(prev, a, next) < 0.0 {
            self.area(a, b, next) >= 0.0 && self.area(a, prev, b) >= 0.0
        } else {
            self.area(a, b, prev) < 0.0 || self.area(a, next, b) < 0.0
        }
    }

    fn middle_inside(&self, a: usize, b: usize) -> bool {
        let mut p = a;
        let mut inside = false;
        let px = (self.nodes[a].x + self.nodes[b].x) / 2.0;
        let py = (self.nodes[a].y + self.nodes[b].y) / 2.0;
        loop {
            let n = &self.nodes[p];
            let nx = &self.nodes[n.next];
            if ((n.y > py) != (nx.y > py)) && nx.y != n.y && (px < (nx.x - n.x) * (py - n.y) / (nx.y - n.y) + n.x) {
                inside = !inside;
            }
            p = n.next;
            if p == a {
                return inside;
            }
        }
    }

    fn split_polygon(&mut self, a: usize, b: usize) -> usize {
        let a2 = self.create_node(self.nodes[a].i, self.nodes[a].x, self.nodes[a].y);
        let b2 = self.create_node(self.nodes[b].i, self.nodes[b].x, self.nodes[b].y);
        let an = self.nodes[a].next;
        let bp = self.nodes[b].prev;
        self.nodes[a].next = b;
        self.nodes[b].prev = a;
        self.nodes[a2].next = an;
        self.nodes[an].prev = a2;
        self.nodes[b2].next = a2;
        self.nodes[a2].prev = b2;
        self.nodes[bp].next = b2;
        self.nodes[b2].prev = bp;
        b2
    }
}

fn sign(v: f64) -> i32 {
    if v > 0.0 {
        1
    } else if v < 0.0 {
        -1
    } else {
        0
    }
}

fn point_in_triangle(ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64, px: f64, py: f64) -> bool {
    (cx - px) * (ay - py) >= (ax - px) * (cy - py) && (ax - px) * (by - py) >= (bx - px) * (ay - py) && (bx - px) * (cy - py) >= (cx - px) * (by - py)
}

fn point_in_triangle_except_first(ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64, px: f64, py: f64) -> bool {
    !(ax == px && ay == py) && point_in_triangle(ax, ay, bx, by, cx, cy, px, py)
}

fn z_order(x: f64, y: f64, min_x: f64, min_y: f64, inv_size: f64) -> i32 {
    let mut x = to_int32((x - min_x) * inv_size);
    let mut y = to_int32((y - min_y) * inv_size);
    x = (x | x.wrapping_shl(8)) & 0x00FF00FF;
    x = (x | x.wrapping_shl(4)) & 0x0F0F0F0F;
    x = (x | x.wrapping_shl(2)) & 0x33333333;
    x = (x | x.wrapping_shl(1)) & 0x55555555;
    y = (y | y.wrapping_shl(8)) & 0x00FF00FF;
    y = (y | y.wrapping_shl(4)) & 0x0F0F0F0F;
    y = (y | y.wrapping_shl(2)) & 0x33333333;
    y = (y | y.wrapping_shl(1)) & 0x55555555;
    x | y.wrapping_shl(1)
}

// --- output ------------------------------------------------------------------------

struct Blob {
    data: Vec<u8>,
}

impl Blob {
    fn push(&mut self, bytes: &[u8]) -> usize {
        while self.data.len() % 8 != 0 {
            self.data.push(0);
        }
        let offset = self.data.len();
        self.data.extend_from_slice(bytes);
        offset
    }
}

fn number(out: &mut String, v: f64) {
    if v.is_finite() {
        // Shortest decimal that reads back as the same double.
        let _ = write!(out, "{v}");
    } else {
        out.push_str(if v.is_nan() {
            r#"["n","NaN"]"#
        } else if v > 0.0 {
            r#"["n","Infinity"]"#
        } else {
            r#"["n","-Infinity"]"#
        });
    }
}

fn string(out: &mut String, s: &str) {
    out.push_str(&serde_json::to_string(s).unwrap_or_else(|_| "\"\"".into()));
}

fn node_json(out: &mut String, blob: &mut Blob, node: &Node) {
    out.push('[');
    string(out, &node.name);
    out.push_str(",[");
    for (i, p) in node.props.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        match p {
            Prop::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Prop::Num(n) => number(out, *n),
            Prop::Str(s) => string(out, s),
            Prop::Raw(r) => {
                let off = blob.push(r);
                let _ = write!(out, r#"["r",{off},{}]"#, r.len());
            }
            Prop::Nums(a) => {
                let off = blob.push(bytemuck::cast_slice(a));
                let _ = write!(out, r#"["f",{off},{}]"#, a.len());
            }
            Prop::Bools(a) => {
                let bytes: Vec<u8> = a.iter().map(|&b| b as u8).collect();
                let off = blob.push(&bytes);
                let _ = write!(out, r#"["b",{off},{}]"#, a.len());
            }
            Prop::Stripped => out.push_str(r#"["x"]"#),
            Prop::Deflated(..) => unreachable!("inflated before encoding"),
        }
    }
    out.push_str("],[");
    for (i, c) in node.children.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        node_json(out, blob, c);
    }
    out.push_str("],");
    out.push_str(if node.single { "true" } else { "false" });
    out.push(']');
}

fn floats(out: &mut String, blob: &mut Blob, v: &[f32]) {
    let off = blob.push(bytemuck::cast_slice(v));
    let _ = write!(out, "[{off},{}]", v.len());
}

fn encode(nodes: &[Node], geometries: &Geometries) -> Vec<u8> {
    let mut blob = Blob { data: Vec::new() };
    let mut json = String::from(r#"{"nodes":["#);
    for (i, n) in nodes.iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        node_json(&mut json, &mut blob, n);
    }
    json.push_str(r#"],"geometries":["#);
    for (i, g) in geometries.list.iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        json.push_str(r#"{"id":"#);
        number(&mut json, g.id);
        json.push_str(r#","skin":"#);
        match g.skin {
            Some(s) => number(&mut json, s),
            None => json.push_str("null"),
        }
        json.push_str(r#","position":"#);
        floats(&mut json, &mut blob, &g.position);
        if !g.color.is_empty() {
            json.push_str(r#","color":"#);
            floats(&mut json, &mut blob, &g.color);
        }
        if !g.normal.is_empty() {
            json.push_str(r#","normal":"#);
            floats(&mut json, &mut blob, &g.normal);
        }
        json.push_str(r#","uvs":["#);
        for (j, uv) in g.uvs.iter().enumerate() {
            if j > 0 {
                json.push(',');
            }
            floats(&mut json, &mut blob, uv);
        }
        json.push(']');
        if let (Some(index), Some(weight)) = (&g.skin_index, &g.skin_weight) {
            let off = blob.push(bytemuck::cast_slice(index));
            let _ = write!(json, r#","skinIndex":[{off},{}]"#, index.len());
            json.push_str(r#","skinWeight":"#);
            floats(&mut json, &mut blob, weight);
        }
        if let Some(groups) = &g.groups {
            json.push_str(r#","groups":["#);
            for (j, (s, c, m)) in groups.iter().enumerate() {
                if j > 0 {
                    json.push(',');
                }
                let _ = write!(json, "[{s},{c},");
                number(&mut json, *m);
                json.push(']');
            }
            json.push(']');
        }
        json.push_str(r#","morphs":["#);
        for (j, (shape, positions)) in g.morphs.iter().enumerate() {
            if j > 0 {
                json.push(',');
            }
            json.push('[');
            number(&mut json, *shape);
            json.push(',');
            floats(&mut json, &mut blob, positions);
            json.push(']');
        }
        json.push_str("]}");
    }
    json.push_str("]}");

    let mut out = Vec::with_capacity(16 + json.len() + blob.data.len());
    out.extend_from_slice(b"3DVF");
    out.extend_from_slice(&(json.len() as u32).to_le_bytes());
    out.extend_from_slice(json.as_bytes());
    while out.len() % 8 != 0 {
        out.push(b' ');
    }
    out.extend_from_slice(&blob.data);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn js_conversions() {
        assert_eq!(to_int32(-1.0), -1);
        assert_eq!(to_int32(4294967296.0 + 5.0), 5);
        assert_eq!(to_int32(f64::NAN), 0);
        assert_eq!((to_int32(-4.0) ^ -1), 3);
        assert_eq!(to_uint16(65537.0), 1);
        assert_eq!(to_uint16(-1.0), 65535);
    }

    #[test]
    fn quad_triangulation() {
        let mut t = Triangulator::default();
        t.vertices = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
        t.triangulate_face();
        assert_eq!(t.faces.len(), 2);
    }

    #[test]
    fn groups_like_the_loader() {
        assert_eq!(groups(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap(), vec![(0, 3, 0.0), (3, 3, 1.0)]);
        assert_eq!(groups(&[2.0, 2.0, 2.0]).unwrap(), vec![(0, 3, 2.0)]);
    }
}
