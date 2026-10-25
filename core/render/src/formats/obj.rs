// Wavefront OBJ: port of three's OBJLoader and MTLLoader as loaders.ts uses
// them. Notable parity details:
//   - the material file is `<model>.mtl` next to the model, whatever the
//     mtllib lines say (loaders.ts probes exactly that name);
//   - faces are fan-triangulated and unshared; a face without normals gets
//     its flat face normal right away;
//   - with an MTL file every material is a MeshPhongMaterial built from it
//     (a name the file lacks still gets a plain white one) and vertex colours
//     are ignored; without one, materials are default Phong with vertexColors
//     when the file has them;
//   - 'l' becomes LineSegments, 'p' Points, a file of bare vertices a point
//     cloud.

use std::collections::HashMap;
use std::path::Path;

use glam::{Vec2, Vec3};

use super::LoadError;
use crate::scene::{srgb3, AlphaMode, Group, Material, Mesh, Placement, Scene, TextureRef, Topology, UvTransform, Wrap};
use crate::texture::Textures;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Kind {
    Mesh,
    Line,
    Points,
}

struct MaterialUse {
    name: String,
    group_start: usize,
    group_end: Option<usize>,
    inherited: bool,
}

impl MaterialUse {
    fn count(&self) -> isize {
        match self.group_end {
            Some(end) => end as isize - self.group_start as isize,
            None => -1,
        }
    }
}

struct Object {
    kind: Kind,
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    colors: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    has_uv_indices: bool,
    materials: Vec<MaterialUse>,
    from_declaration: bool,
}

impl Object {
    fn new(from_declaration: bool) -> Object {
        Object {
            kind: Kind::Mesh,
            positions: Vec::new(),
            normals: Vec::new(),
            colors: Vec::new(),
            uvs: Vec::new(),
            has_uv_indices: false,
            materials: Vec::new(),
            from_declaration,
        }
    }

    fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    /// ParserState object._finalize().
    fn finalize(&mut self, end: bool) -> Option<usize> {
        let count = self.vertex_count();
        let last = self.materials.len().checked_sub(1);
        if let Some(i) = last {
            let m = &mut self.materials[i];
            if m.group_end.is_none() {
                m.group_end = Some(count);
                m.inherited = false;
            }
        }
        if end && self.materials.len() > 1 {
            self.materials.retain(|m| m.count() > 0);
        }
        if end && self.materials.is_empty() {
            self.materials.push(MaterialUse { name: String::new(), group_start: 0, group_end: Some(count), inherited: false });
        }
        last
    }

    /// object.startMaterial().
    fn start_material(&mut self, name: &str) {
        let previous = self.finalize(false);
        let mut group_start = 0;
        if let Some(i) = previous {
            group_start = self.materials[i].group_end.unwrap_or(0);
            if self.materials[i].inherited || self.materials[i].count() <= 0 {
                self.materials.remove(i);
            }
        }
        self.materials.push(MaterialUse { name: name.to_string(), group_start, group_end: None, inherited: false });
    }
}

struct State {
    objects: Vec<Object>,
    vertices: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    colors: Vec<Option<[f32; 3]>>,
    uvs: Vec<[f32; 2]>,
}

impl State {
    fn object(&mut self) -> &mut Object {
        self.objects.last_mut().expect("there is always a current object")
    }

    /// ParserState.startObject().
    fn start_object(&mut self) {
        if let Some(o) = self.objects.last_mut() {
            if !o.from_declaration {
                o.from_declaration = true;
                return;
            }
        }
        let previous = self.objects.last().and_then(|o| o.materials.last()).map(|m| m.name.clone());
        if let Some(o) = self.objects.last_mut() {
            o.finalize(true);
        }
        let mut object = Object::new(true);
        if let Some(name) = previous.filter(|n| !n.is_empty()) {
            object.materials.push(MaterialUse { name, group_start: 0, group_end: None, inherited: true });
        }
        self.objects.push(object);
    }

    /// OBJ indices: 1-based, negative ones count from the end.
    fn resolve(index: &str, len: usize) -> Option<usize> {
        let i: i64 = parse_int(index)?;
        let r = if i >= 0 { i - 1 } else { i + len as i64 };
        (r >= 0 && (r as usize) < len).then_some(r as usize)
    }

    fn add_face(&mut self, v: [&[&str]; 3]) {
        let vlen = self.vertices.len();
        let Some(ia) = Self::resolve(v[0][0], vlen) else { return };
        let Some(ib) = Self::resolve(v[1][0], vlen) else { return };
        let Some(ic) = Self::resolve(v[2][0], vlen) else { return };
        let (pa, pb, pc) = (self.vertices[ia], self.vertices[ib], self.vertices[ic]);
        let colors = [self.colors[ia], self.colors[ib], self.colors[ic]];
        let normal_refs = [v[0].get(2), v[1].get(2), v[2].get(2)];
        let normals = if normal_refs.iter().all(|r| r.is_some_and(|s| !s.is_empty())) {
            let nlen = self.normals.len();
            let get = |r: &Option<&&str>| r.and_then(|s| Self::resolve(s, nlen)).map(|i| self.normals[i]).unwrap_or([0.0; 3]);
            [get(&normal_refs[0]), get(&normal_refs[1]), get(&normal_refs[2])]
        } else {
            // addFaceNormal: (c - b) x (a - b), normalised.
            let (a, b, c) = (Vec3::from_array(pa), Vec3::from_array(pb), Vec3::from_array(pc));
            let n = (c - b).cross(a - b).normalize_or_zero().to_array();
            [n, n, n]
        };
        let uv_refs = [v[0].get(1), v[1].get(1), v[2].get(1)];
        let uvs = if uv_refs.iter().all(|r| r.is_some_and(|s| !s.is_empty())) {
            let ulen = self.uvs.len();
            let get = |r: &Option<&&str>| r.and_then(|s| Self::resolve(s, ulen)).map(|i| self.uvs[i]).unwrap_or([0.0; 2]);
            Some([get(&uv_refs[0]), get(&uv_refs[1]), get(&uv_refs[2])])
        } else {
            None
        };
        let o = self.object();
        o.positions.extend([pa, pb, pc]);
        for c in colors.into_iter().flatten() {
            o.colors.push(c);
        }
        o.normals.extend(normals);
        match uvs {
            Some(uv) => {
                o.uvs.extend(uv);
                o.has_uv_indices = true;
            }
            None => o.uvs.extend([[0.0, 0.0]; 3]),
        }
    }
}

/// parseInt(): the leading integer of the token, if any.
fn parse_int(s: &str) -> Option<i64> {
    let s = s.trim();
    let end = s
        .char_indices()
        .find(|&(i, c)| !(c.is_ascii_digit() || (i == 0 && (c == '-' || c == '+'))))
        .map(|(i, _)| i)
        .unwrap_or(s.len());
    s[..end].parse().ok()
}

/// parseFloat(): NaN when there is no number.
fn parse_float(s: Option<&str>) -> f32 {
    s.and_then(|t| t.parse::<f32>().ok()).unwrap_or(f32::NAN)
}

pub fn load(path: &Path, textures: &mut Textures) -> Result<Scene, LoadError> {
    let bytes = std::fs::read(path)?;
    let text = String::from_utf8_lossy(&bytes);
    let mut scene = Scene::default();
    let mtl = load_mtl(path, textures, &mut scene);
    let state = parse(&text);
    build(state, mtl, scene)
}

fn parse(text: &str) -> State {
    let text = text.replace("\r\n", "\n").replace("\\\n", "");
    let mut state = State { objects: vec![Object::new(false)], vertices: Vec::new(), normals: Vec::new(), colors: Vec::new(), uvs: Vec::new() };
    for raw in text.split('\n') {
        let line = raw.trim_start();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let first = line.as_bytes()[0];
        if first == b'v' {
            let data: Vec<&str> = line.split_whitespace().collect();
            match data[0] {
                "v" => {
                    state.vertices.push([parse_float(data.get(1).copied()), parse_float(data.get(2).copied()), parse_float(data.get(3).copied())]);
                    state.colors.push(if data.len() >= 7 {
                        Some(srgb3([parse_float(Some(data[4])), parse_float(Some(data[5])), parse_float(Some(data[6]))]).to_array())
                    } else {
                        None
                    });
                }
                "vn" => state.normals.push([parse_float(data.get(1).copied()), parse_float(data.get(2).copied()), parse_float(data.get(3).copied())]),
                "vt" => state.uvs.push([parse_float(data.get(1).copied()), parse_float(data.get(2).copied())]),
                _ => {}
            }
        } else if first == b'f' {
            let verts: Vec<Vec<&str>> = line[1..].split_whitespace().map(|v| v.split('/').collect()).collect();
            if verts.len() < 3 {
                continue;
            }
            for j in 1..verts.len() - 1 {
                state.add_face([&verts[0], &verts[j], &verts[j + 1]]);
            }
        } else if first == b'l' {
            let parts: Vec<&str> = line[1..].trim().split(' ').collect();
            let vlen = state.vertices.len();
            let refs: Vec<&str> = if line.contains('/') { parts.iter().map(|p| p.split('/').next().unwrap_or("")).filter(|s| !s.is_empty()).collect() } else { parts };
            let points: Vec<[f32; 3]> = refs.iter().filter_map(|r| State::resolve(r, vlen)).map(|i| state.vertices[i]).collect();
            let o = state.object();
            o.kind = Kind::Line;
            o.positions.extend(points);
        } else if first == b'p' {
            let vlen = state.vertices.len();
            let idx: Vec<usize> = line[1..].trim().split(' ').filter_map(|r| State::resolve(r, vlen)).collect();
            let pts: Vec<([f32; 3], Option<[f32; 3]>)> = idx.iter().map(|&i| (state.vertices[i], state.colors[i])).collect();
            let o = state.object();
            o.kind = Kind::Points;
            for (p, c) in pts {
                o.positions.push(p);
                if let Some(c) = c {
                    o.colors.push(c);
                }
            }
        } else if first == b'o' || first == b'g' {
            // /^[og]\s*(.+)?/ matches any line starting with o or g.
            state.start_object();
        } else if let Some(name) = line.strip_prefix("usemtl ") {
            state.object().start_material(name.trim());
        }
        // mtllib, usemap, s: ignored, as in the loader (smoothing groups are
        // commented out there).
    }
    if let Some(o) = state.objects.last_mut() {
        o.finalize(true);
    }
    state
}

/// MTLLoader result: three materials by name, plus a plain white Phong for
/// names the file does not define (MaterialCreator.create on an unknown
/// name).
struct Mtl {
    materials: HashMap<String, Material>,
}

fn load_mtl(path: &Path, textures: &mut Textures, scene: &mut Scene) -> Option<Mtl> {
    let mtl_path = path.with_extension("mtl");
    // loaders.ts replaces a trailing ".obj" case-insensitively; the sibling
    // may also be ".MTL" on case-sensitive file systems.
    let mtl_path = if mtl_path.is_file() { mtl_path } else { path.with_extension("MTL") };
    let bytes = std::fs::read(&mtl_path).ok()?;
    let text = String::from_utf8_lossy(&bytes);
    let mut infos: Vec<(String, Vec<(String, String)>)> = Vec::new();
    for raw in text.split('\n') {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (key, value) = match line.find(' ') {
            Some(p) => (line[..p].to_ascii_lowercase(), line[p + 1..].trim().to_string()),
            None => (line.to_ascii_lowercase(), String::new()),
        };
        if key == "newmtl" {
            infos.push((value, Vec::new()));
        } else if let Some(info) = infos.last_mut() {
            // Later keys overwrite earlier ones of the same name.
            info.1.retain(|(k, _)| *k != key);
            info.1.push((key, value));
        }
    }
    let mut materials = HashMap::new();
    for (name, props) in infos {
        materials.insert(name, mtl_material(&props, textures, scene));
    }
    Some(Mtl { materials })
}

fn mtl_material(props: &[(String, String)], textures: &mut Textures, scene: &mut Scene) -> Material {
    let mut m = Material::phong();
    let rgb = |v: &str| -> [f32; 3] {
        let mut it = v.split_whitespace().map(|s| s.parse::<f32>().unwrap_or(f32::NAN));
        [it.next().unwrap_or(f32::NAN), it.next().unwrap_or(f32::NAN), it.next().unwrap_or(f32::NAN)]
    };
    for (key, value) in props {
        if value.is_empty() {
            continue;
        }
        match key.as_str() {
            "kd" => m.color = srgb3(rgb(value)),
            "ks" => m.specular = srgb3(rgb(value)),
            "ke" => m.emissive = srgb3(rgb(value)),
            "map_kd" => set_map(&mut m.map, value, textures, scene, None),
            "map_ks" => set_map(&mut m.specular_map, value, textures, scene, None),
            "map_ke" => set_map(&mut m.emissive_map, value, textures, scene, None),
            "norm" => set_map(&mut m.normal_map, value, textures, scene, None),
            "map_bump" | "bump" => {
                let mut scale = None;
                set_map(&mut m.bump_map, value, textures, scene, Some(&mut scale));
                if let Some(s) = scale {
                    m.bump_scale = s;
                }
            }
            "map_d" => {
                set_map(&mut m.alpha_map, value, textures, scene, None);
                m.alpha_mode = AlphaMode::Blend;
            }
            "ns" => m.shininess = value.parse().unwrap_or(f32::NAN),
            "d" => {
                let n: f32 = value.parse().unwrap_or(f32::NAN);
                if n < 1.0 {
                    m.set_opacity(n);
                }
            }
            "tr" => {
                let n: f32 = value.parse().unwrap_or(f32::NAN);
                if n > 0.0 {
                    m.set_opacity(1.0 - n);
                }
            }
            _ => {}
        }
    }
    m
}

/// setMapForType(): the first texture of a kind wins; "-s u v w" scales
/// and "-o u v w" offsets it, "-bm s" is the bump scale.
fn set_map(slot: &mut Option<TextureRef>, value: &str, textures: &mut Textures, scene: &mut Scene, bump_scale: Option<&mut Option<f32>>) {
    if slot.is_some() {
        return;
    }
    let mut items: Vec<&str> = value.split_whitespace().collect();
    let mut repeat = Vec2::ONE;
    let mut offset = Vec2::ZERO;
    if let Some(p) = items.iter().position(|s| *s == "-bm") {
        if let Some(b) = bump_scale {
            *b = items.get(p + 1).and_then(|v| v.parse().ok());
        }
        items.drain(p..(p + 2).min(items.len()));
    }
    if let Some(p) = items.iter().position(|s| *s == "-mm") {
        items.drain(p..(p + 3).min(items.len()));
    }
    if let Some(p) = items.iter().position(|s| *s == "-s") {
        repeat = Vec2::new(parse_float(items.get(p + 1).copied()), parse_float(items.get(p + 2).copied()));
        items.drain(p..(p + 4).min(items.len()));
    }
    if let Some(p) = items.iter().position(|s| *s == "-o") {
        offset = Vec2::new(parse_float(items.get(p + 1).copied()), parse_float(items.get(p + 2).copied()));
        items.drain(p..(p + 4).min(items.len()));
    }
    let url = items.join(" ");
    let url = url.trim();
    if let Some(image) = textures.load(scene, url, false) {
        let mut t = TextureRef::new(image);
        t.wrap_u = Wrap::Repeat;
        t.wrap_v = Wrap::Repeat;
        if repeat.is_finite() && offset.is_finite() {
            t.transform = UvTransform::three(offset, repeat, 0.0, Vec2::ZERO, true);
        }
        *slot = Some(t);
    }
}

fn build(state: State, mtl: Option<Mtl>, mut scene: Scene) -> Result<Scene, LoadError> {
    let has_primitives = !(state.objects.len() == 1 && state.objects[0].positions.is_empty());
    // Default materials are shared by name, smoothing and vertex colours,
    // like OBJLoader's state.materials hash.
    let mut defaults: HashMap<(String, bool, Kind), usize> = HashMap::new();
    let mut mtl_ids: HashMap<(String, Kind), usize> = HashMap::new();
    if has_primitives {
        for o in state.objects {
            if o.positions.is_empty() {
                continue;
            }
            let n = o.positions.len();
            let has_colors = !o.colors.is_empty();
            let mut materials = Vec::with_capacity(o.materials.len());
            for mu in &o.materials {
                let id = match &mtl {
                    Some(mtl) => *mtl_ids.entry((mu.name.clone(), o.kind)).or_insert_with(|| {
                        let base = mtl.materials.get(&mu.name).cloned().unwrap_or_else(Material::phong);
                        scene.add_material(adapt(base, o.kind))
                    }),
                    None => *defaults.entry((mu.name.clone(), has_colors, o.kind)).or_insert_with(|| {
                        let mut m = match o.kind {
                            Kind::Line => Material::basic(),
                            Kind::Points => Material::points(1.0, false),
                            Kind::Mesh => Material::phong(),
                        };
                        m.vertex_colors = has_colors;
                        scene.add_material(m)
                    }),
                };
                materials.push(id);
            }
            let topology = match o.kind {
                Kind::Mesh => Topology::Triangles,
                Kind::Line => Topology::Lines,
                Kind::Points => Topology::Points,
            };
            let mut mesh = Mesh::new(topology, o.positions);
            if !o.normals.is_empty() && o.normals.len() == n {
                mesh.normals = Some(o.normals);
            }
            if o.colors.len() == n {
                mesh.colors = Some(o.colors.into_iter().map(crate::scene::rgba).collect());
            }
            if o.has_uv_indices && o.uvs.len() == n {
                mesh.uv0 = Some(o.uvs.into_iter().map(crate::scene::flip_v).collect());
            }
            mesh.groups = if materials.len() > 1 {
                o.materials
                    .iter()
                    .zip(&materials)
                    .map(|(mu, &material)| Group { start: mu.group_start as u32, count: mu.count().max(0) as u32, material })
                    .collect()
            } else {
                vec![Group { start: 0, count: n as u32, material: materials.first().copied().unwrap_or(0) }]
            };
            scene.meshes.push(Placement::identity().apply(mesh));
        }
    } else if !state.vertices.is_empty() {
        // Only vertices: a point cloud.
        let mut m = Material::points(1.0, false);
        let mut mesh = Mesh::new(Topology::Points, state.vertices);
        if state.colors.first().is_some_and(|c| c.is_some()) {
            m.vertex_colors = true;
            mesh.colors = Some(state.colors.into_iter().map(|c| crate::scene::rgba(c.unwrap_or([0.0; 3]))).collect());
        }
        let id = scene.add_material(m);
        scene.meshes.push(Placement::identity().apply(mesh.with_material(id)));
    }
    Ok(scene)
}

/// OBJLoader converts MTL materials for lines and points: LineBasicMaterial
/// with the colour, PointsMaterial size 10 without attenuation.
fn adapt(m: Material, kind: Kind) -> Material {
    match kind {
        Kind::Mesh => m,
        Kind::Line => Material { shading: crate::scene::Shading::Basic, ..m },
        Kind::Points => Material { shading: crate::scene::Shading::Basic, point_size: 10.0, size_attenuation: false, ..m },
    }
}

