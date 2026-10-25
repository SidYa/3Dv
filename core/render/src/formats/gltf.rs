// glTF 2.0 (.gltf and .glb): port of what three's GLTFLoader builds for a
// static thumbnail, read straight from the JSON so that files every viewer
// opens - textures without a "source" (gltfpack), unknown required
// extensions - open here too. Parity details:
//   - materials: metallic-roughness → MeshStandardMaterial (default material
//     metalness 1, roughness 1), KHR_materials_unlit → MeshBasicMaterial,
//     BLEND without depth writes, MASK cutoff 0.5 by default;
//   - a primitive without normals is drawn flat-shaded; without tangents
//     the normal map's green axis is flipped (derivative tangent frames);
//   - skinned meshes are posed by their joints' rest transforms (three's
//     "attached" bind mode makes the mesh node's own transform cancel out),
//     morph targets get their default weights;
//   - bounds come from the POSITION accessor's min/max widened by morph
//     displacements, as GLTFLoader's computeBounds; skinned meshes use the
//     posed vertices like SkinnedMesh.computeBoundingBox;
//   - compression: KHR_draco_mesh_compression (draco-core),
//     EXT/KHR_meshopt_compression (meshoptimizer), KHR_mesh_quantization,
//     KHR_texture_basisu (KTX2 through the Basis Universal transcoder),
//     EXT_texture_webp; EXT_mesh_gpu_instancing draws every instance.

use std::collections::HashMap;
use std::path::Path;

use glam::{Mat3, Mat4, Quat, Vec2, Vec3};
use serde_json::Value;

use super::{invalid, LoadError};
use crate::scene::{Aabb, AlphaMode, Material, Mesh, Placement, Scene, Side, TextureRef, Topology, UvTransform, Wrap};
use crate::texture::{Missing, Textures};

const MAX_DEPTH: usize = 256;
/// Instanced geometry is expanded per instance; beyond this many vertices
/// in total the file is refused rather than exhausting memory.
const MAX_INSTANCED_VERTICES: usize = 40_000_000;

// --- JSON helpers -----------------------------------------------------------

fn arr<'a>(v: &'a Value, key: &str) -> &'a [Value] {
    v.get(key).and_then(Value::as_array).map(Vec::as_slice).unwrap_or(&[])
}

fn idx(v: &Value, key: &str) -> Option<usize> {
    v.get(key)?.as_u64().map(|i| i as usize)
}

fn num(v: &Value, key: &str) -> Option<f64> {
    v.get(key)?.as_f64()
}

fn f32s<const N: usize>(v: Option<&Value>) -> Option<[f32; N]> {
    let a = v?.as_array()?;
    if a.len() < N {
        return None;
    }
    let mut out = [0f32; N];
    for (o, x) in out.iter_mut().zip(a) {
        *o = x.as_f64()? as f32;
    }
    Some(out)
}

fn ext<'a>(v: &'a Value, name: &str) -> Option<&'a Value> {
    v.get("extensions")?.get(name)
}

// --- buffers and accessors ----------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Component {
    I8,
    U8,
    I16,
    U16,
    U32,
    F32,
}

impl Component {
    fn from_gl(t: u64) -> Option<Component> {
        Some(match t {
            5120 => Component::I8,
            5121 => Component::U8,
            5122 => Component::I16,
            5123 => Component::U16,
            5125 => Component::U32,
            5126 => Component::F32,
            _ => return None,
        })
    }

    fn size(self) -> usize {
        match self {
            Component::I8 | Component::U8 => 1,
            Component::I16 | Component::U16 => 2,
            Component::U32 | Component::F32 => 4,
        }
    }

    fn read(self, b: &[u8]) -> f32 {
        match self {
            Component::I8 => b[0] as i8 as f32,
            Component::U8 => b[0] as f32,
            Component::I16 => i16::from_le_bytes([b[0], b[1]]) as f32,
            Component::U16 => u16::from_le_bytes([b[0], b[1]]) as f32,
            Component::U32 => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32,
            Component::F32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        }
    }

    /// How WebGL turns a normalized integer into a float (signed values
    /// clamp at -1), three's getNormalizedComponentScale.
    fn normalize(self, v: f32) -> f32 {
        match self {
            Component::I8 => (v / 127.0).max(-1.0),
            Component::U8 => v / 255.0,
            Component::I16 => (v / 32767.0).max(-1.0),
            Component::U16 => v / 65535.0,
            Component::U32 | Component::F32 => v,
        }
    }
}

fn components_of(ty: &str) -> Option<usize> {
    Some(match ty {
        "SCALAR" => 1,
        "VEC2" => 2,
        "VEC3" => 3,
        "VEC4" | "MAT2" => 4,
        "MAT3" => 9,
        "MAT4" => 16,
        _ => return None,
    })
}

struct Doc {
    json: Value,
    buffers: Vec<Vec<u8>>,
    /// Buffer views decoded from meshopt streams.
    decoded: HashMap<usize, Vec<u8>>,
}

/// An accessor read into floats: `count` elements of `components` values.
struct Data {
    components: usize,
    values: Vec<f32>,
}

impl Data {
    fn get<const N: usize>(&self, i: usize) -> [f32; N] {
        let mut out = [0f32; N];
        let base = i * self.components;
        for (k, o) in out.iter_mut().enumerate().take(self.components.min(N)) {
            *o = self.values.get(base + k).copied().unwrap_or(0.0);
        }
        out
    }

    fn count(&self) -> usize {
        self.values.len() / self.components.max(1)
    }

    fn vec<const N: usize>(&self) -> Vec<[f32; N]> {
        (0..self.count()).map(|i| self.get::<N>(i)).collect()
    }
}

impl Doc {
    fn accessor(&self, i: usize) -> Option<&Value> {
        arr(&self.json, "accessors").get(i)
    }

    /// The bytes of a buffer view, meshopt-decoded where the view says so.
    fn view_bytes(&self, i: usize) -> Result<&[u8], LoadError> {
        if let Some(d) = self.decoded.get(&i) {
            return Ok(d);
        }
        let view = arr(&self.json, "bufferViews").get(i).ok_or_else(|| invalid("glTF bufferView index"))?;
        let buffer = self.buffers.get(idx(view, "buffer").unwrap_or(usize::MAX)).ok_or_else(|| invalid("glTF buffer index"))?;
        let offset = idx(view, "byteOffset").unwrap_or(0);
        let length = idx(view, "byteLength").unwrap_or(0);
        buffer.get(offset..offset.saturating_add(length)).ok_or_else(|| invalid("glTF bufferView out of range"))
    }

    /// EXT_meshopt_compression / KHR_meshopt_compression: decodes every
    /// compressed buffer view up front.
    fn decode_meshopt(&mut self) -> Result<(), LoadError> {
        let views = arr(&self.json, "bufferViews").to_vec();
        for (i, view) in views.iter().enumerate() {
            let Some(m) = ext(view, "EXT_meshopt_compression").or_else(|| ext(view, "KHR_meshopt_compression")) else { continue };
            let buffer = self.buffers.get(idx(m, "buffer").unwrap_or(usize::MAX)).ok_or_else(|| invalid("meshopt buffer"))?;
            let offset = idx(m, "byteOffset").unwrap_or(0);
            let length = idx(m, "byteLength").unwrap_or(0);
            let source = buffer.get(offset..offset.saturating_add(length)).ok_or_else(|| invalid("meshopt range"))?;
            let stride = idx(m, "byteStride").unwrap_or(0);
            let count = idx(m, "count").unwrap_or(0);
            if stride == 0 || stride > 256 || count.saturating_mul(stride) > 1 << 31 {
                return Err(invalid("meshopt stride/count"));
            }
            let mut out = vec![0u8; count * stride];
            let mode = m.get("mode").and_then(Value::as_str).unwrap_or("ATTRIBUTES");
            let filter = m.get("filter").and_then(Value::as_str).unwrap_or("NONE");
            // SAFETY: the decoders write at most count*stride bytes into
            // `out` and read at most `source.len()` bytes; both are valid.
            let rc = unsafe {
                match mode {
                    "ATTRIBUTES" => meshopt::ffi::meshopt_decodeVertexBuffer(out.as_mut_ptr().cast(), count, stride, source.as_ptr(), source.len()),
                    "TRIANGLES" => meshopt::ffi::meshopt_decodeIndexBuffer(out.as_mut_ptr().cast(), count, stride, source.as_ptr(), source.len()),
                    "INDICES" => meshopt::ffi::meshopt_decodeIndexSequence(out.as_mut_ptr().cast(), count, stride, source.as_ptr(), source.len()),
                    _ => return Err(invalid("meshopt mode")),
                }
            };
            if rc != 0 {
                return Err(invalid("meshopt stream"));
            }
            unsafe {
                match filter {
                    "NONE" => {}
                    "OCTAHEDRAL" => meshopt::ffi::meshopt_decodeFilterOct(out.as_mut_ptr().cast(), count, stride),
                    "QUATERNION" => meshopt::ffi::meshopt_decodeFilterQuat(out.as_mut_ptr().cast(), count, stride),
                    "EXPONENTIAL" => meshopt::ffi::meshopt_decodeFilterExp(out.as_mut_ptr().cast(), count, stride),
                    "COLOR" => meshopt::ffi::meshopt_decodeFilterColor(out.as_mut_ptr().cast(), count, stride),
                    _ => return Err(invalid("meshopt filter")),
                }
            }
            self.decoded.insert(i, out);
        }
        Ok(())
    }

    /// Reads an accessor into floats, normalizing integers when the accessor
    /// says so, sparse substitutions applied.
    fn read(&self, i: usize) -> Result<Data, LoadError> {
        let a = self.accessor(i).ok_or_else(|| invalid("glTF accessor index"))?;
        let component = a.get("componentType").and_then(Value::as_u64).and_then(Component::from_gl).ok_or_else(|| invalid("glTF componentType"))?;
        let components = a.get("type").and_then(Value::as_str).and_then(components_of).ok_or_else(|| invalid("glTF accessor type"))?;
        let count = idx(a, "count").unwrap_or(0);
        if count.saturating_mul(components) > 1 << 28 {
            return Err(invalid("glTF accessor too large"));
        }
        let normalized = a.get("normalized").and_then(Value::as_bool).unwrap_or(false);
        let element = component.size() * components;
        let mut values = vec![0f32; count * components];
        if let Some(view_index) = idx(a, "bufferView") {
            let bytes = self.view_bytes(view_index)?;
            let view = &arr(&self.json, "bufferViews")[view_index];
            let stride = idx(view, "byteStride").filter(|&s| s > 0).unwrap_or(element);
            let offset = idx(a, "byteOffset").unwrap_or(0);
            for e in 0..count {
                let start = offset + e * stride;
                let Some(src) = bytes.get(start..start + element) else { return Err(invalid("glTF accessor out of range")) };
                for c in 0..components {
                    values[e * components + c] = component.read(&src[c * component.size()..]);
                }
            }
        }
        if let Some(sparse) = a.get("sparse") {
            let n = idx(sparse, "count").unwrap_or(0);
            let (si, sv) = (sparse.get("indices"), sparse.get("values"));
            if let (Some(si), Some(sv)) = (si, sv) {
                let it = si.get("componentType").and_then(Value::as_u64).and_then(Component::from_gl).ok_or_else(|| invalid("sparse index type"))?;
                let ib = self.view_bytes(idx(si, "bufferView").unwrap_or(usize::MAX))?;
                let io = idx(si, "byteOffset").unwrap_or(0);
                let vb = self.view_bytes(idx(sv, "bufferView").unwrap_or(usize::MAX))?;
                let vo = idx(sv, "byteOffset").unwrap_or(0);
                for k in 0..n {
                    let Some(ibytes) = ib.get(io + k * it.size()..io + (k + 1) * it.size()) else { break };
                    let target = it.read(ibytes) as usize;
                    if target >= count {
                        continue;
                    }
                    let Some(src) = vb.get(vo + k * element..vo + (k + 1) * element) else { break };
                    for c in 0..components {
                        values[target * components + c] = component.read(&src[c * component.size()..]);
                    }
                }
            }
        }
        if normalized {
            for v in values.iter_mut() {
                *v = component.normalize(*v);
            }
        }
        Ok(Data { components, values })
    }

    fn read_indices(&self, i: usize) -> Result<Vec<u32>, LoadError> {
        let d = self.read(i)?;
        Ok(d.values.iter().map(|&v| v as u32).collect())
    }

    /// The accessor's min/max as GLTFLoader's computeBounds reads them.
    fn bounds(&self, i: usize) -> Option<Aabb> {
        let a = self.accessor(i)?;
        let mut min = Vec3::from_array(f32s::<3>(a.get("min"))?);
        let mut max = Vec3::from_array(f32s::<3>(a.get("max"))?);
        if a.get("normalized").and_then(Value::as_bool).unwrap_or(false) {
            let c = a.get("componentType").and_then(Value::as_u64).and_then(Component::from_gl)?;
            let s = c.normalize(1.0);
            min *= s;
            max *= s;
        }
        Some(Aabb { min, max })
    }

    /// computeBounds' per-target displacement: the larger absolute bound per
    /// axis.
    fn displacement(&self, i: usize) -> Option<Vec3> {
        let b = self.bounds(i)?;
        Some(b.min.abs().max(b.max.abs()))
    }
}

/// Attributes of one primitive, from accessors or a Draco stream.
#[derive(Default)]
struct Attributes {
    positions: Vec<[f32; 3]>,
    normals: Option<Vec<[f32; 3]>>,
    tangents: Option<Vec<[f32; 4]>>,
    uv0: Option<Vec<[f32; 2]>>,
    uv1: Option<Vec<[f32; 2]>>,
    colors: Option<Vec<[f32; 4]>>,
    joints: Option<Vec<[f32; 4]>>,
    weights: Option<Vec<[f32; 4]>>,
    indices: Option<Vec<u32>>,
}

const SEMANTICS: &[&str] = &["POSITION", "NORMAL", "TANGENT", "TEXCOORD_0", "TEXCOORD_1", "COLOR_0", "JOINTS_0", "WEIGHTS_0"];

impl Attributes {
    fn set(&mut self, semantic: &str, d: Data) {
        match semantic {
            "POSITION" => self.positions = d.vec::<3>(),
            "NORMAL" => self.normals = Some(d.vec::<3>()),
            "TANGENT" => self.tangents = Some(d.vec::<4>()),
            "TEXCOORD_0" => self.uv0 = Some(d.vec::<2>()),
            "TEXCOORD_1" => self.uv1 = Some(d.vec::<2>()),
            "COLOR_0" => {
                // vec3 colours are opaque.
                let rgb = d.components == 3;
                self.colors = Some((0..d.count()).map(|i| {
                    let c = d.get::<4>(i);
                    if rgb {
                        [c[0], c[1], c[2], 1.0]
                    } else {
                        c
                    }
                }).collect());
            }
            "JOINTS_0" => self.joints = Some(d.vec::<4>()),
            "WEIGHTS_0" => self.weights = Some(d.vec::<4>()),
            _ => {}
        }
    }
}

/// KHR_draco_mesh_compression: the primitive's attributes and indices come
/// from the Draco stream; the accessors only describe their types.
fn decode_draco(doc: &Doc, primitive: &Value, draco: &Value) -> Result<Attributes, LoadError> {
    let bytes = doc.view_bytes(idx(draco, "bufferView").ok_or_else(|| invalid("draco bufferView"))?)?;
    let mut buffer = draco_core::DecoderBuffer::new(bytes);
    let mut mesh = draco_core::Mesh::new();
    draco_core::MeshDecoder::new().decode(&mut buffer, &mut mesh).map_err(|e| invalid(format!("draco: {e:?}")))?;
    let points = mesh.num_points();
    let mut out = Attributes::default();
    let ids = draco.get("attributes").and_then(Value::as_object).ok_or_else(|| invalid("draco attributes"))?;
    let accessors = primitive.get("attributes").and_then(Value::as_object);
    for (semantic, id) in ids {
        if !SEMANTICS.contains(&semantic.as_str()) {
            continue;
        }
        let Some(id) = id.as_u64() else { continue };
        let Some(att) = mesh.attribute_by_unique_id(id as u32) else { continue };
        // Component count and normalization as the glTF accessor declares,
        // which is what DRACOLoader decodes into.
        let accessor = accessors.and_then(|a| a.get(semantic)).and_then(Value::as_u64).and_then(|i| doc.accessor(i as usize));
        let components = accessor.and_then(|a| a.get("type")).and_then(Value::as_str).and_then(components_of).unwrap_or(att.num_components() as usize);
        let mut values = att.read_f32s(points, components);
        let normalized = accessor.and_then(|a| a.get("normalized")).and_then(Value::as_bool).unwrap_or(false);
        if normalized {
            if let Some(c) = accessor.and_then(|a| a.get("componentType")).and_then(Value::as_u64).and_then(Component::from_gl) {
                for v in values.iter_mut() {
                    *v = c.normalize(*v);
                }
            }
        }
        out.set(semantic, Data { components, values });
    }
    let mut indices = Vec::with_capacity(mesh.num_faces() * 3);
    for f in mesh.faces() {
        indices.extend([f[0].0, f[1].0, f[2].0]);
    }
    out.indices = Some(indices);
    Ok(out)
}

// --- loading --------------------------------------------------------------------

fn parse_container(bytes: &[u8]) -> Result<(Value, Option<Vec<u8>>), LoadError> {
    if bytes.starts_with(b"glTF") {
        if bytes.len() < 20 {
            return Err(invalid("GLB header"));
        }
        let mut offset = 12;
        let mut json = None;
        let mut bin = None;
        while offset + 8 <= bytes.len() {
            let len = u32::from_le_bytes([bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]]) as usize;
            let kind = &bytes[offset + 4..offset + 8];
            let body = bytes.get(offset + 8..offset + 8 + len).ok_or_else(|| invalid("GLB chunk"))?;
            match kind {
                b"JSON" => json = Some(serde_json::from_slice::<Value>(body).map_err(|e| invalid(format!("glTF JSON: {e}")))?),
                b"BIN\0" => bin = Some(body.to_vec()),
                _ => {}
            }
            offset += 8 + len;
        }
        Ok((json.ok_or_else(|| invalid("GLB without JSON"))?, bin))
    } else {
        let json = serde_json::from_slice::<Value>(bytes).map_err(|e| invalid(format!("glTF JSON: {e}")))?;
        Ok((json, None))
    }
}

fn data_uri(uri: &str) -> Option<Vec<u8>> {
    let rest = uri.strip_prefix("data:")?;
    let (_, payload) = rest.split_once(";base64,")?;
    base64_decode(payload)
}

fn load_buffers(json: &Value, mut bin: Option<Vec<u8>>, textures: &Textures) -> Result<Vec<Vec<u8>>, LoadError> {
    let mut out = Vec::new();
    for (i, b) in arr(json, "buffers").iter().enumerate() {
        let length = idx(b, "byteLength").unwrap_or(0);
        let data = match b.get("uri").and_then(Value::as_str) {
            Some(uri) if uri.starts_with("data:") => data_uri(uri).ok_or_else(|| invalid("glTF data URI"))?,
            Some(uri) => {
                let path = textures.resolve(uri).ok_or_else(|| invalid(format!("glTF buffer {uri} missing")))?;
                std::fs::read(path)?
            }
            // The GLB body is buffer 0; a meshopt fallback buffer has no
            // data of its own, its views are decoded.
            None if i == 0 && bin.is_some() => bin.take().unwrap_or_default(),
            None if ext(b, "EXT_meshopt_compression").or_else(|| ext(b, "KHR_meshopt_compression")).is_some() => Vec::new(),
            None => return Err(invalid("glTF buffer without data")),
        };
        if data.len() < length && !data.is_empty() {
            return Err(invalid("glTF buffer shorter than byteLength"));
        }
        out.push(data);
    }
    Ok(out)
}

pub fn load(path: &Path, textures: &mut Textures) -> Result<Scene, LoadError> {
    let bytes = std::fs::read(path)?;
    let (json, bin) = parse_container(&bytes)?;
    drop(bytes);
    let buffers = load_buffers(&json, bin, textures)?;
    let mut doc = Doc { json, buffers, decoded: HashMap::new() };
    doc.decode_meshopt()?;

    let scenes = arr(&doc.json, "scenes");
    let scene_index = idx(&doc.json, "scene").unwrap_or(0);
    let gscene = scenes.get(scene_index).ok_or_else(|| invalid("glTF without a scene"))?;
    let roots: Vec<usize> = arr(gscene, "nodes").iter().filter_map(|v| v.as_u64().map(|i| i as usize)).collect();

    // World matrices of every node of the scene, for meshes and joints.
    let node_count = arr(&doc.json, "nodes").len();
    let mut world = vec![None; node_count];
    for &n in &roots {
        compute_world(&doc.json, n, Mat4::IDENTITY, &mut world, 0);
    }
    let mut loader = Loader { scene: Scene::default(), textures, doc: &doc, materials: HashMap::new(), images: HashMap::new(), instanced_vertices: 0 };
    for &n in &roots {
        loader.visit(n, &world, 0)?;
    }
    Ok(loader.scene)
}

fn local_matrix(node: &Value) -> Mat4 {
    if let Some(m) = f32s::<16>(node.get("matrix")) {
        return Mat4::from_cols_array(&m);
    }
    let t = f32s::<3>(node.get("translation")).unwrap_or([0.0; 3]);
    let r = f32s::<4>(node.get("rotation")).unwrap_or([0.0, 0.0, 0.0, 1.0]);
    let s = f32s::<3>(node.get("scale")).unwrap_or([1.0; 3]);
    Mat4::from_scale_rotation_translation(Vec3::from_array(s), Quat::from_array(r).normalize(), Vec3::from_array(t))
}

fn compute_world(json: &Value, n: usize, parent: Mat4, world: &mut Vec<Option<Mat4>>, depth: usize) {
    if depth > MAX_DEPTH {
        return;
    }
    let Some(node) = arr(json, "nodes").get(n) else { return };
    let m = parent * local_matrix(node);
    if let Some(slot) = world.get_mut(n) {
        *slot = Some(m);
    }
    for c in arr(node, "children").iter().filter_map(Value::as_u64) {
        compute_world(json, c as usize, m, world, depth + 1);
    }
}

struct Loader<'a> {
    scene: Scene,
    textures: &'a mut Textures,
    doc: &'a Doc,
    /// (material index or None for the default, flat, derivative tangents,
    /// vertex colours, topology) → scene material.
    materials: HashMap<(Option<usize>, bool, bool, bool, u8), usize>,
    images: HashMap<usize, Option<usize>>,
    instanced_vertices: usize,
}

impl<'a> Loader<'a> {
    fn visit(&mut self, n: usize, world: &[Option<Mat4>], depth: usize) -> Result<(), LoadError> {
        if depth > MAX_DEPTH {
            return Ok(());
        }
        let doc = self.doc;
        let Some(node) = arr(&doc.json, "nodes").get(n) else { return Ok(()) };
        let m = world.get(n).copied().flatten().unwrap_or(Mat4::IDENTITY);
        if let Some(mesh) = idx(node, "mesh").and_then(|i| arr(&doc.json, "meshes").get(i)) {
            let weights: Vec<f32> = node.get("weights").or_else(|| mesh.get("weights")).and_then(Value::as_array).map(|w| w.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect()).unwrap_or_default();
            let instances = self.instances(node)?;
            for primitive in arr(mesh, "primitives") {
                match &instances {
                    None => self.primitive(primitive, node, m, world, &weights)?,
                    Some(list) => {
                        for inst in list {
                            self.primitive(primitive, node, m * *inst, world, &weights)?;
                        }
                    }
                }
            }
        }
        for c in arr(node, "children").iter().filter_map(Value::as_u64) {
            self.visit(c as usize, world, depth + 1)?;
        }
        Ok(())
    }

    /// EXT_mesh_gpu_instancing: per-instance TRS on top of the node's matrix.
    fn instances(&self, node: &Value) -> Result<Option<Vec<Mat4>>, LoadError> {
        let Some(e) = ext(node, "EXT_mesh_gpu_instancing") else { return Ok(None) };
        let attrs = e.get("attributes");
        let read = |name: &str| -> Result<Option<Data>, LoadError> {
            match attrs.and_then(|a| a.get(name)).and_then(Value::as_u64) {
                Some(i) => self.doc.read(i as usize).map(Some),
                None => Ok(None),
            }
        };
        let (t, r, s) = (read("TRANSLATION")?, read("ROTATION")?, read("SCALE")?);
        let count = [&t, &r, &s].iter().filter_map(|d| d.as_ref().map(Data::count)).max().unwrap_or(0);
        Ok(Some(
            (0..count)
                .map(|i| {
                    let tr = t.as_ref().map_or([0.0; 3], |d| d.get::<3>(i));
                    let ro = r.as_ref().map_or([0.0, 0.0, 0.0, 1.0], |d| d.get::<4>(i));
                    let sc = s.as_ref().map_or([1.0; 3], |d| d.get::<3>(i));
                    Mat4::from_scale_rotation_translation(Vec3::from_array(sc), Quat::from_array(ro).normalize(), Vec3::from_array(tr))
                })
                .collect(),
        ))
    }

    fn primitive(&mut self, primitive: &Value, node: &Value, node_world: Mat4, world: &[Option<Mat4>], weights: &[f32]) -> Result<(), LoadError> {
        let doc = self.doc;
        let attr_index = |name: &str| primitive.get("attributes").and_then(|a| a.get(name)).and_then(Value::as_u64).map(|i| i as usize);
        let Some(position_accessor) = attr_index("POSITION") else { return Ok(()) };

        let mut a = match ext(primitive, "KHR_draco_mesh_compression") {
            Some(draco) => decode_draco(doc, primitive, draco)?,
            None => {
                let mut a = Attributes::default();
                for s in SEMANTICS {
                    if let Some(i) = attr_index(s) {
                        a.set(s, doc.read(i)?);
                    }
                }
                if let Some(i) = idx(primitive, "indices") {
                    a.indices = Some(doc.read_indices(i)?);
                }
                a
            }
        };
        let n = a.positions.len();
        if n == 0 {
            return Ok(());
        }
        if let Some(i) = &a.indices {
            if i.iter().any(|&v| v as usize >= n) {
                return Err(invalid("glTF index out of range"));
            }
        }
        self.instanced_vertices += n;
        if self.instanced_vertices > MAX_INSTANCED_VERTICES {
            return Err(invalid("glTF too many vertices"));
        }
        let has_normals = a.normals.is_some();
        let has_tangents = a.tangents.is_some();
        let has_colors = a.colors.is_some();

        // Accessor bounds (GLTFLoader computeBounds), before morphing.
        let mut local_box = doc.bounds(position_accessor).unwrap_or_else(|| Aabb::from_points(a.positions.iter()));

        // Morph targets at their default weights; the box grows by the
        // largest displacement any single target allows, whatever the
        // weights (computeBounds' "reasonable size" compromise).
        let mut max_displacement = Vec3::ZERO;
        for (t, target) in arr(primitive, "targets").iter().enumerate() {
            let w = weights.get(t).copied().unwrap_or(0.0);
            if let Some(pi) = idx(target, "POSITION") {
                let d = doc.read(pi)?;
                if w != 0.0 {
                    for (k, p) in a.positions.iter_mut().enumerate() {
                        let dv = Vec3::from_array(d.get::<3>(k));
                        *p = (Vec3::from_array(*p) + dv * w).to_array();
                    }
                }
                if let Some(extent) = doc.displacement(pi) {
                    max_displacement = max_displacement.max(extent);
                }
            }
            if let (Some(ni), Some(normals)) = (idx(target, "NORMAL"), a.normals.as_mut()) {
                if w != 0.0 {
                    let d = doc.read(ni)?;
                    for (k, nrm) in normals.iter_mut().enumerate() {
                        *nrm = (Vec3::from_array(*nrm) + Vec3::from_array(d.get::<3>(k)) * w).to_array();
                    }
                }
            }
        }
        local_box.min -= max_displacement;
        local_box.max += max_displacement;

        let mode = primitive.get("mode").and_then(Value::as_u64).unwrap_or(4);
        let topology = match mode {
            0 => Topology::Points,
            1..=3 => Topology::Lines,
            _ => Topology::Triangles,
        };
        let indices = expand_mode(mode, a.indices.take(), n);

        let mut mesh = Mesh::new(topology, std::mem::take(&mut a.positions));
        mesh.normals = a.normals.take().filter(|v| v.len() == n);
        mesh.tangents = a.tangents.take().filter(|v| v.len() == n);
        mesh.uv0 = a.uv0.take().filter(|v| v.len() == n);
        mesh.uv1 = a.uv1.take().filter(|v| v.len() == n);
        mesh.colors = a.colors.take().filter(|v| v.len() == n);
        mesh = match indices {
            Some(i) => mesh.with_indices(i),
            None => mesh,
        };
        let topo_key = match topology {
            Topology::Triangles => 0,
            Topology::Lines => 1,
            Topology::Points => 2,
        };
        let material = self.material(idx(primitive, "material"), !has_normals, !has_tangents, has_colors, topo_key);
        mesh = mesh.with_material(material);

        let skin = idx(node, "skin").and_then(|s| arr(&doc.json, "skins").get(s));
        let placed = match skin.and_then(|s| self.skin(s, &a, &mesh, node_world, world)) {
            Some(skinned) => skinned,
            None => {
                let mut placed = Placement::new(node_world).apply(mesh);
                placed.bounds = local_box.transformed(&node_world);
                placed
            }
        };
        self.scene.meshes.push(placed);
        Ok(())
    }

    /// Rest pose of a skinned primitive: Σ w·(joint world · inverse bind)·v,
    /// expressed back in the mesh node's space so the bounds come out as
    /// SkinnedMesh.computeBoundingBox's.
    fn skin(&self, skin: &Value, a: &Attributes, mesh: &Mesh, node_world: Mat4, world: &[Option<Mat4>]) -> Option<Mesh> {
        let joints = a.joints.as_ref()?;
        let weights = a.weights.as_ref()?;
        if joints.len() != mesh.positions.len() || weights.len() != mesh.positions.len() {
            return None;
        }
        let ibms: Vec<Mat4> = idx(skin, "inverseBindMatrices")
            .and_then(|i| self.doc.read(i).ok())
            .map(|d| (0..d.count()).map(|k| Mat4::from_cols_array(&d.get::<16>(k))).collect())
            .unwrap_or_default();
        let bones: Vec<Mat4> = arr(skin, "joints")
            .iter()
            .enumerate()
            .map(|(i, j)| {
                let jw = j.as_u64().and_then(|j| world.get(j as usize).copied().flatten()).unwrap_or(Mat4::IDENTITY);
                jw * ibms.get(i).copied().unwrap_or(Mat4::IDENTITY)
            })
            .collect();
        let to_local = if node_world.determinant().abs() > 1e-30 { node_world.inverse() } else { Mat4::IDENTITY };
        let mut out = mesh.clone();
        let matrix = |i: usize| to_local * skin_matrix(&bones, joints[i], weights[i]);
        for (i, p) in out.positions.iter_mut().enumerate() {
            *p = matrix(i).transform_point3(Vec3::from_array(*p)).to_array();
        }
        if let Some(normals) = out.normals.as_mut() {
            for (i, n) in normals.iter_mut().enumerate() {
                *n = (Mat3::from_mat4(matrix(i)) * Vec3::from_array(*n)).to_array();
            }
        }
        if let Some(tangents) = out.tangents.as_mut() {
            for (i, t) in tangents.iter_mut().enumerate() {
                let v = Mat3::from_mat4(matrix(i)) * Vec3::new(t[0], t[1], t[2]);
                *t = [v.x, v.y, v.z, t[3]];
            }
        }
        Some(Placement::new(node_world).apply(out))
    }

    fn material(&mut self, index: Option<usize>, flat: bool, derivative_tangents: bool, vertex_colors: bool, topology: u8) -> usize {
        let key = (index, flat, derivative_tangents, vertex_colors, topology);
        if let Some(&m) = self.materials.get(&key) {
            return m;
        }
        let doc = self.doc;
        let mut m = match index.and_then(|i| arr(&doc.json, "materials").get(i)) {
            Some(def) => self.build_material(def),
            // GLTFLoader's default material.
            None => Material { metalness: 1.0, roughness: 1.0, ..Material::standard() },
        };
        if flat {
            m.flat_shading = true;
        }
        if derivative_tangents {
            m.normal_scale.y = -m.normal_scale.y;
        }
        if vertex_colors {
            m.vertex_colors = true;
        }
        match topology {
            // PointsMaterial copy: colour, map, 1px dots.
            2 => {
                m = Material { color: m.color, opacity: m.opacity, map: m.map, vertex_colors: m.vertex_colors, alpha_mode: m.alpha_mode, depth_write: m.depth_write, ..Material::points(1.0, false) };
            }
            1 => {
                m = Material { color: m.color, opacity: m.opacity, map: m.map, vertex_colors: m.vertex_colors, alpha_mode: m.alpha_mode, depth_write: m.depth_write, side: m.side, ..Material::basic() };
            }
            _ => {}
        }
        let id = self.scene.add_material(m);
        self.materials.insert(key, id);
        id
    }

    fn build_material(&mut self, def: &Value) -> Material {
        let pbr = def.get("pbrMetallicRoughness").cloned().unwrap_or(Value::Null);
        let unlit = ext(def, "KHR_materials_unlit").is_some();
        let mut m = if unlit { Material::basic() } else { Material::standard() };
        let base = f32s::<4>(pbr.get("baseColorFactor")).unwrap_or([1.0; 4]);
        m.color = Vec3::new(base[0], base[1], base[2]);
        m.opacity = base[3];
        m.map = pbr.get("baseColorTexture").and_then(|t| self.texture_info(t));
        if !unlit {
            m.metalness = num(&pbr, "metallicFactor").unwrap_or(1.0) as f32;
            m.roughness = num(&pbr, "roughnessFactor").unwrap_or(1.0) as f32;
            m.metal_rough_map = pbr.get("metallicRoughnessTexture").and_then(|t| self.texture_info(t));
            if let Some(n) = def.get("normalTexture") {
                m.normal_map = self.texture_info(n);
                m.normal_scale = Vec2::splat(num(n, "scale").unwrap_or(1.0) as f32);
            }
            if let Some(o) = def.get("occlusionTexture") {
                m.ao_map = self.texture_info(o);
                m.ao_intensity = num(o, "strength").unwrap_or(1.0) as f32;
            }
            let e = f32s::<3>(def.get("emissiveFactor")).unwrap_or([0.0; 3]);
            let strength = ext(def, "KHR_materials_emissive_strength").and_then(|s| num(s, "emissiveStrength")).unwrap_or(1.0) as f32;
            m.emissive = Vec3::from_array(e) * strength;
            m.emissive_map = def.get("emissiveTexture").and_then(|t| self.texture_info(t));
            // MeshPhysicalMaterial: transmission, volume and ior.
            if let Some(tr) = ext(def, "KHR_materials_transmission") {
                m.transmission = num(tr, "transmissionFactor").unwrap_or(0.0) as f32;
                m.transmission_map = tr.get("transmissionTexture").and_then(|t| self.texture_info(t));
            }
            if let Some(v) = ext(def, "KHR_materials_volume") {
                m.thickness = num(v, "thicknessFactor").unwrap_or(0.0) as f32;
                m.thickness_map = v.get("thicknessTexture").and_then(|t| self.texture_info(t));
                m.attenuation_distance = num(v, "attenuationDistance").map_or(f32::INFINITY, |d| d as f32);
                m.attenuation_color = Vec3::from_array(f32s::<3>(v.get("attenuationColor")).unwrap_or([1.0; 3]));
            }
            if let Some(i) = ext(def, "KHR_materials_ior").and_then(|e| num(e, "ior")) {
                m.ior = i as f32;
            }
        }
        if def.get("doubleSided").and_then(Value::as_bool).unwrap_or(false) {
            m.side = Side::Double;
        }
        match def.get("alphaMode").and_then(Value::as_str).unwrap_or("OPAQUE") {
            "BLEND" => {
                m.alpha_mode = AlphaMode::Blend;
                m.depth_write = false;
            }
            "MASK" => m.alpha_mode = AlphaMode::Mask(num(def, "alphaCutoff").unwrap_or(0.5) as f32),
            _ => {}
        }
        m
    }

    /// A textureInfo: texture, texCoord and KHR_texture_transform.
    fn texture_info(&mut self, info: &Value) -> Option<TextureRef> {
        let doc = self.doc;
        let texture = arr(&doc.json, "textures").get(idx(info, "index")?)?;
        // GLTFLoader's texture plugins in their order: KTX2, WebP, AVIF
        // (not decoded here: the plain source when there is one), then the
        // core source.
        let source = ext(texture, "KHR_texture_basisu")
            .and_then(|e| idx(e, "source"))
            .or_else(|| ext(texture, "EXT_texture_webp").and_then(|e| idx(e, "source")))
            .or_else(|| idx(texture, "source"))?;
        let image = self.image(source)?;
        let mut t = TextureRef::new(image);
        t.uv_set = idx(info, "texCoord").unwrap_or(0).min(1) as u8;
        let sampler = idx(texture, "sampler").and_then(|s| arr(&doc.json, "samplers").get(s));
        t.wrap_u = wrap(sampler.and_then(|s| s.get("wrapS")).and_then(Value::as_u64));
        t.wrap_v = wrap(sampler.and_then(|s| s.get("wrapT")).and_then(Value::as_u64));
        if let Some(tr) = ext(info, "KHR_texture_transform") {
            if let Some(tc) = idx(tr, "texCoord") {
                t.uv_set = tc.min(1) as u8;
            }
            let offset = f32s::<2>(tr.get("offset")).unwrap_or([0.0; 2]);
            let scale = f32s::<2>(tr.get("scale")).unwrap_or([1.0; 2]);
            let rotation = num(tr, "rotation").unwrap_or(0.0) as f32;
            t.transform = UvTransform::three(Vec2::from_array(offset), Vec2::from_array(scale), rotation, Vec2::ZERO, false);
        }
        Some(t)
    }

    fn image(&mut self, index: usize) -> Option<usize> {
        if let Some(&cached) = self.images.get(&index) {
            return cached;
        }
        let doc = self.doc;
        let image = arr(&doc.json, "images").get(index)?;
        let key = format!("gltf-image-{index}");
        let result = if let Some(view) = idx(image, "bufferView") {
            let bytes = doc.view_bytes(view).ok()?;
            self.textures.load_bytes(&mut self.scene, &key, bytes, Missing::Skip)
        } else {
            let uri = image.get("uri").and_then(Value::as_str)?;
            if uri.starts_with("data:") {
                let bytes = data_uri(uri)?;
                self.textures.load_bytes(&mut self.scene, &key, &bytes, Missing::Skip)
            } else {
                self.textures.load_with(&mut self.scene, uri, false, Missing::Skip)
            }
        };
        self.images.insert(index, result);
        result
    }
}

fn skin_matrix(bones: &[Mat4], joints: [f32; 4], weights: [f32; 4]) -> Mat4 {
    // normalizeSkinWeights(): scale to a sum of 1.
    let sum: f32 = weights.iter().sum();
    if sum <= 0.0 {
        return Mat4::IDENTITY;
    }
    let mut acc = Mat4::ZERO;
    for k in 0..4 {
        let w = weights[k] / sum;
        if w != 0.0 {
            let b = bones.get(joints[k] as usize).copied().unwrap_or(Mat4::IDENTITY);
            acc += b * w;
        }
    }
    acc
}

fn wrap(mode: Option<u64>) -> Wrap {
    match mode {
        Some(33071) => Wrap::Clamp,
        Some(33648) => Wrap::Mirror,
        _ => Wrap::Repeat,
    }
}

/// Strips, fans and line strips/loops as plain lists, the way three's
/// toTrianglesDrawMode and LineSegments see them.
fn expand_mode(mode: u64, indices: Option<Vec<u32>>, n: usize) -> Option<Vec<u32>> {
    let seq = || -> Vec<u32> { (0..n as u32).collect() };
    match mode {
        5 => {
            let src = indices.unwrap_or_else(seq);
            let mut out = Vec::with_capacity(src.len().saturating_sub(2) * 3);
            for i in 0..src.len().saturating_sub(2) {
                if i % 2 == 0 {
                    out.extend([src[i], src[i + 1], src[i + 2]]);
                } else {
                    out.extend([src[i + 2], src[i + 1], src[i]]);
                }
            }
            Some(out)
        }
        6 => {
            let src = indices.unwrap_or_else(seq);
            let mut out = Vec::new();
            for i in 1..src.len().saturating_sub(1) {
                out.extend([src[0], src[i], src[i + 1]]);
            }
            Some(out)
        }
        2 | 3 => {
            let src = indices.unwrap_or_else(seq);
            let mut out = Vec::new();
            for w in src.windows(2) {
                out.extend([w[0], w[1]]);
            }
            if mode == 2 && src.len() > 2 {
                out.extend([src[src.len() - 1], src[0]]);
            }
            Some(out)
        }
        _ => indices,
    }
}

fn base64_decode(s: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let mut buf = 0u32;
    let mut bits = 0;
    for c in s.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' => break,
            b' ' | b'\n' | b'\r' | b'\t' => continue,
            _ => return None,
        } as u32;
        buf = (buf << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buf >> bits) as u8);
        }
    }
    Some(out)
}
