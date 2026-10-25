// PLY, ASCII and binary (both byte orders): port of three's PLYLoader plus
// loaders.ts's choice of a mesh (faces present) or a point cloud.
//
// Like the loader, only triangles and quads become faces, vertex and face
// colours are sRGB, and face colours or per-face UVs turn the geometry
// non-indexed - which loaders.ts then shows as a point cloud. That is how
// the viewer displays such files, so the thumbnail does the same.

use std::path::Path;

use super::{invalid, mesh_default_material, points_default_material, LoadError};
use crate::scene::{srgb_to_linear, Mesh, Placement, Scene, Topology};

#[derive(Clone, Copy, PartialEq)]
enum Ty {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl Ty {
    fn parse(s: &str) -> Option<Ty> {
        Some(match s {
            "char" | "int8" => Ty::I8,
            "uchar" | "uint8" => Ty::U8,
            "short" | "int16" => Ty::I16,
            "ushort" | "uint16" => Ty::U16,
            "int" | "int32" => Ty::I32,
            "uint" | "uint32" => Ty::U32,
            "float" | "float32" => Ty::F32,
            "double" | "float64" => Ty::F64,
            _ => return None,
        })
    }

    fn size(self) -> usize {
        match self {
            Ty::I8 | Ty::U8 => 1,
            Ty::I16 | Ty::U16 => 2,
            Ty::I32 | Ty::U32 | Ty::F32 => 4,
            Ty::F64 => 8,
        }
    }

    fn is_float(self) -> bool {
        matches!(self, Ty::F32 | Ty::F64)
    }

    /// PLYLoader getColorScale().
    fn color_scale(self) -> f64 {
        match self {
            Ty::U8 => 1.0 / 255.0,
            Ty::U16 => 1.0 / 65535.0,
            Ty::F32 | Ty::F64 => 1.0,
            _ => 1.0 / 255.0,
        }
    }
}

struct Property {
    name: String,
    ty: Ty,
    /// Some(count type) for list properties.
    list: Option<Ty>,
}

struct Element {
    name: String,
    count: usize,
    properties: Vec<Property>,
}

#[derive(Clone, Copy, PartialEq)]
enum Format {
    Ascii,
    BinaryLe,
    BinaryBe,
}

struct Header {
    format: Format,
    elements: Vec<Element>,
    body_offset: usize,
}

fn parse_header(data: &[u8]) -> Result<Header, LoadError> {
    if !data.starts_with(b"ply") {
        return Err(invalid("not a PLY file"));
    }
    let end = find(data, b"end_header").ok_or_else(|| invalid("PLY without end_header"))?;
    let mut body_offset = end + b"end_header".len();
    // The line break after end_header: \r\n, \r or \n.
    if data.get(body_offset) == Some(&b'\r') {
        body_offset += 1;
    }
    if data.get(body_offset) == Some(&b'\n') {
        body_offset += 1;
    }
    let text = String::from_utf8_lossy(&data[..end]);
    let mut format = None;
    let mut elements: Vec<Element> = Vec::new();
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        match parts.next() {
            Some("format") => {
                format = match parts.next() {
                    Some("ascii") => Some(Format::Ascii),
                    Some("binary_little_endian") => Some(Format::BinaryLe),
                    Some("binary_big_endian") => Some(Format::BinaryBe),
                    _ => None,
                }
            }
            Some("element") => {
                let name = parts.next().unwrap_or("").to_string();
                let count = parts.next().and_then(|c| c.parse().ok()).unwrap_or(0);
                elements.push(Element { name, count, properties: Vec::new() });
            }
            Some("property") => {
                let Some(element) = elements.last_mut() else { continue };
                let words: Vec<&str> = parts.collect();
                let property = if words.first() == Some(&"list") && words.len() >= 4 {
                    match (Ty::parse(words[1]), Ty::parse(words[2])) {
                        (Some(count), Some(item)) => Property { name: words[3].to_string(), ty: item, list: Some(count) },
                        _ => return Err(invalid("PLY list type")),
                    }
                } else if words.len() >= 2 {
                    match Ty::parse(words[0]) {
                        Some(ty) => Property { name: words[1].to_string(), ty, list: None },
                        None => return Err(invalid("PLY property type")),
                    }
                } else {
                    continue;
                };
                element.properties.push(property);
            }
            _ => {}
        }
    }
    let format = format.ok_or_else(|| invalid("PLY format"))?;
    Ok(Header { format, elements, body_offset })
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// One parsed element by property index: scalar values, and the values of
/// list properties. Reused from one element to the next.
struct Values {
    scalars: Vec<f64>,
    lists: Vec<Vec<f64>>,
}

impl Values {
    fn new(properties: &[Property]) -> Values {
        Values { scalars: vec![0.0; properties.len()], lists: vec![Vec::new(); properties.len()] }
    }
}

/// PLYLoader getAttributeDescriptor(): the property names it recognises.
struct Descriptor {
    position: Option<[usize; 3]>,
    normal: Option<[usize; 3]>,
    uv: Option<[usize; 2]>,
    color: Option<([usize; 3], Ty)>,
    /// vertex_indices, or vertex_index (three.js issue #9338).
    indices: Option<usize>,
    texcoord: Option<usize>,
}

fn descriptor(properties: &[Property]) -> Descriptor {
    let find = |names: &[&str]| -> Option<usize> { names.iter().find_map(|n| properties.iter().position(|p| p.name == *n)) };
    let three = |a: &[&str], b: &[&str], c: &[&str]| -> Option<[usize; 3]> { Some([find(a)?, find(b)?, find(c)?]) };
    let position = three(&["x", "px", "posx"], &["y", "py", "posy"], &["z", "pz", "posz"]);
    let normal = three(&["nx", "normalx"], &["ny", "normaly"], &["nz", "normalz"]);
    let uv = match (find(&["s", "u", "texture_u", "tx"]), find(&["t", "v", "texture_v", "ty"])) {
        (Some(s), Some(t)) => Some([s, t]),
        _ => None,
    };
    let color = match (
        find(&["red", "diffuse_red", "r", "diffuse_r"]),
        find(&["green", "diffuse_green", "g", "diffuse_g"]),
        find(&["blue", "diffuse_blue", "b", "diffuse_b"]),
    ) {
        (Some(r), Some(g), Some(b)) => Some(([r, g, b], properties[r].ty)),
        _ => None,
    };
    let indices = find(&["vertex_indices"]).or_else(|| find(&["vertex_index"]));
    let texcoord = find(&["texcoord"]);
    Descriptor { position, normal, uv, color, indices, texcoord }
}

#[derive(Default)]
struct Buffers {
    indices: Vec<u32>,
    vertices: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    colors: Vec<[f32; 4]>,
    face_uvs: Vec<[f32; 2]>,
    face_colors: Vec<[f32; 4]>,
}

/// The linear colour three ends up with: setRGB(..., SRGBColorSpace), then
/// stored back in the attribute's type (rounded for integer types) and
/// normalised again when drawn.
fn color(values: &Values, names: &[usize; 3], ty: Ty, round: bool) -> [f32; 4] {
    let scale = ty.color_scale();
    let mut out = [0.0f32, 0.0, 0.0, 1.0];
    for i in 0..3 {
        let raw = values.scalars[names[i]];
        let lin = srgb_to_linear((raw * scale) as f32) as f64;
        out[i] = if ty.is_float() || !round { lin as f32 } else { ((lin / scale).round() * scale) as f32 };
    }
    out
}

fn handle(buffers: &mut Buffers, element: &str, values: &Values, d: &Descriptor) {
    let get = |n: &usize| values.scalars[*n] as f32;
    if element == "vertex" {
        if let Some(p) = &d.position {
            buffers.vertices.push([get(&p[0]), get(&p[1]), get(&p[2])]);
        }
        if let Some(n) = &d.normal {
            buffers.normals.push([get(&n[0]), get(&n[1]), get(&n[2])]);
        }
        if let Some(uv) = &d.uv {
            buffers.uvs.push([get(&uv[0]), get(&uv[1])]);
        }
        if let Some((names, ty)) = &d.color {
            buffers.colors.push(color(values, names, *ty, true));
        }
    } else if element == "face" {
        let Some(idx) = d.indices.map(|i| &values.lists[i]) else { return };
        let texcoord = d.texcoord.map(|i| &values.lists[i]);
        if idx.len() == 3 {
            buffers.indices.extend([idx[0] as u32, idx[1] as u32, idx[2] as u32]);
            if let Some(t) = texcoord.filter(|t| t.len() == 6) {
                for k in 0..3 {
                    buffers.face_uvs.push([t[k * 2] as f32, t[k * 2 + 1] as f32]);
                }
            }
        } else if idx.len() == 4 {
            buffers.indices.extend([idx[0] as u32, idx[1] as u32, idx[3] as u32]);
            buffers.indices.extend([idx[1] as u32, idx[2] as u32, idx[3] as u32]);
        }
        if let Some((names, ty)) = &d.color {
            // Face colours are pushed unrounded, three per face.
            let c = color(values, names, *ty, false);
            buffers.face_colors.extend([c, c, c]);
        }
    }
}

pub fn load(path: &Path) -> Result<Scene, LoadError> {
    let data = std::fs::read(path)?;
    let header = parse_header(&data)?;
    let mut buffers = Buffers::default();
    let body = &data[header.body_offset.min(data.len())..];
    match header.format {
        Format::Ascii => parse_ascii(body, &header, &mut buffers),
        Format::BinaryLe => parse_binary(body, &header, &mut buffers, true),
        Format::BinaryBe => parse_binary(body, &header, &mut buffers, false),
    }
    build(buffers)
}

fn parse_ascii(body: &[u8], header: &Header, buffers: &mut Buffers) {
    let text = String::from_utf8_lossy(body);
    let mut tokens = text.split_ascii_whitespace();
    let number = |t: Option<&str>, ty: Ty| -> Option<f64> {
        let t = t?;
        if ty.is_float() {
            t.parse::<f64>().ok().or(Some(f64::NAN))
        } else {
            // parseInt: leading integer part.
            let end = t.find(|c: char| !(c.is_ascii_digit() || c == '-' || c == '+')).unwrap_or(t.len());
            t[..end].parse::<i64>().ok().map(|v| v as f64).or(Some(f64::NAN))
        }
    };
    'elements: for element in &header.elements {
        let d = descriptor(&element.properties);
        let mut values = Values::new(&element.properties);
        for _ in 0..element.count {
            for (pi, p) in element.properties.iter().enumerate() {
                match p.list {
                    Some(count_ty) => {
                        let Some(n) = number(tokens.next(), count_ty) else { break 'elements };
                        let n = if n.is_finite() && n >= 0.0 { n as usize } else { 0 };
                        let list = &mut values.lists[pi];
                        list.clear();
                        for _ in 0..n {
                            let Some(v) = number(tokens.next(), p.ty) else { break 'elements };
                            list.push(v);
                        }
                    }
                    None => {
                        let Some(v) = number(tokens.next(), p.ty) else { break 'elements };
                        values.scalars[pi] = v;
                    }
                }
            }
            handle(buffers, &element.name, &values, &d);
        }
    }
}

fn read(body: &[u8], at: usize, ty: Ty, le: bool) -> Option<f64> {
    let b = body.get(at..at + ty.size())?;
    macro_rules! num {
        ($t:ty, $n:expr) => {{
            let mut a = [0u8; $n];
            a.copy_from_slice(b);
            if le {
                <$t>::from_le_bytes(a)
            } else {
                <$t>::from_be_bytes(a)
            }
        }};
    }
    Some(match ty {
        Ty::I8 => b[0] as i8 as f64,
        Ty::U8 => b[0] as f64,
        Ty::I16 => num!(i16, 2) as f64,
        Ty::U16 => num!(u16, 2) as f64,
        Ty::I32 => num!(i32, 4) as f64,
        Ty::U32 => num!(u32, 4) as f64,
        Ty::F32 => num!(f32, 4) as f64,
        Ty::F64 => num!(f64, 8),
    })
}

fn parse_binary(body: &[u8], header: &Header, buffers: &mut Buffers, le: bool) {
    let mut at = 0usize;
    for element in &header.elements {
        let d = descriptor(&element.properties);
        let mut values = Values::new(&element.properties);
        for _ in 0..element.count {
            for (pi, p) in element.properties.iter().enumerate() {
                match p.list {
                    Some(count_ty) => {
                        let Some(n) = read(body, at, count_ty, le) else { return };
                        at += count_ty.size();
                        let n = if n.is_finite() && n >= 0.0 { n as usize } else { 0 };
                        if at + n * p.ty.size() > body.len() {
                            return;
                        }
                        let list = &mut values.lists[pi];
                        list.clear();
                        for _ in 0..n {
                            list.push(read(body, at, p.ty, le).unwrap_or(0.0));
                            at += p.ty.size();
                        }
                    }
                    None => {
                        let Some(v) = read(body, at, p.ty, le) else { return };
                        at += p.ty.size();
                        values.scalars[pi] = v;
                    }
                }
            }
            handle(buffers, &element.name, &values, &d);
        }
    }
}

/// PLYLoader postProcess() + loaders.ts asMesh / asPoints.
fn build(b: Buffers) -> Result<Scene, LoadError> {
    if b.vertices.is_empty() {
        return Err(invalid("PLY without vertices"));
    }
    let n = b.vertices.len();
    let mut mesh = Mesh::new(Topology::Triangles, b.vertices);
    if b.normals.len() == n {
        mesh.normals = Some(b.normals);
    }
    if b.uvs.len() == n {
        // TextureLoader flipY; no material reads them, but keep parity.
        mesh.uv0 = Some(b.uvs.into_iter().map(crate::scene::flip_v).collect());
    }
    if b.colors.len() == n {
        mesh.colors = Some(b.colors);
    }
    let mut indexed = !b.indices.is_empty();
    if indexed {
        let max = n as u32;
        let indices: Vec<u32> = b.indices;
        if indices.iter().any(|&i| i >= max) {
            return Err(invalid("PLY face index out of range"));
        }
        mesh = mesh.with_indices(indices);
    }
    if indexed && (!b.face_uvs.is_empty() || !b.face_colors.is_empty()) {
        // geometry.toNonIndexed(): the index goes away, which makes
        // loaders.ts draw the result as points.
        mesh = to_non_indexed(mesh);
        if b.face_uvs.len() == mesh.positions.len() {
            mesh.uv0 = Some(b.face_uvs.into_iter().map(crate::scene::flip_v).collect());
        }
        if b.face_colors.len() == mesh.positions.len() {
            mesh.colors = Some(b.face_colors);
        }
        indexed = false;
    }
    let vertex_colors = mesh.colors.is_some();
    let mut scene = Scene::default();
    if indexed {
        let m = scene.add_material(mesh_default_material(vertex_colors));
        scene.meshes.push(Placement::identity().apply(mesh.with_material(m)));
    } else {
        mesh.topology = Topology::Points;
        mesh.indices = None;
        let count = mesh.positions.len() as u32;
        mesh.groups = vec![crate::scene::Group { start: 0, count, material: 0 }];
        let m = scene.add_material(points_default_material(vertex_colors));
        scene.meshes.push(Placement::identity().apply(mesh.with_material(m)));
    }
    Ok(scene)
}

fn to_non_indexed(mesh: Mesh) -> Mesh {
    let Some(indices) = mesh.indices.clone() else { return mesh };
    let pick = |v: &Vec<[f32; 3]>| indices.iter().map(|&i| v[i as usize]).collect::<Vec<_>>();
    let positions = pick(&mesh.positions);
    let mut out = Mesh::new(Topology::Triangles, positions);
    out.normals = mesh.normals.as_ref().map(pick);
    out.uv0 = mesh.uv0.as_ref().map(|v| indices.iter().map(|&i| v[i as usize]).collect());
    out.colors = mesh.colors.as_ref().map(|v| indices.iter().map(|&i| v[i as usize]).collect());
    out
}
