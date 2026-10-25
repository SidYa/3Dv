// Models for the viewer window, parsed in Rust and handed to three.js.
//
// For text formats the native loaders are several times faster than the
// three.js ones (a 50 MB OBJ: ~0.3 s instead of ~1.6 s) and run off the UI
// thread. The geometry goes over as one binary blob that JS wraps in typed
// arrays without copying; materials go as JSON. Textures are not decoded
// here: their references go along and three's TextureLoader loads them at
// full size, as for every other format, so missing or undecodable files
// behave as before.
//
// Layout: "3DVG", u32 JSON length, JSON (padded to 4 bytes), binary data.
// Offsets in the JSON are byte offsets into the binary data.

use std::path::Path;

use serde_json::{json, Value};

use crate::formats::{self, LoadError};
use crate::scene::{AlphaMode, Material, Scene, Shading, Side, TextureRef, Topology, Wrap};
use crate::texture::Textures;

/// Formats the viewer loads through Rust: the ones whose three.js parsers
/// are slow on large files. FBX keeps FBXLoader for everything but reading
/// the file and building geometry (see viewer_fbx.rs).
pub const VIEWER_EXTENSIONS: &[&str] = &["obj", "ply", "stl", "fbx"];

/// An f32 as the shortest decimal that reads back as it (0.65, not
/// 0.6499999761581421), the value three's JS constant would have been.
fn n(x: f32) -> f64 {
    format!("{x}").parse().unwrap_or(x as f64)
}

fn n3(v: glam::Vec3) -> [f64; 3] {
    [n(v.x), n(v.y), n(v.z)]
}

struct Blob {
    data: Vec<u8>,
}

impl Blob {
    fn push(&mut self, bytes: &[u8]) -> Value {
        let offset = self.data.len();
        self.data.extend_from_slice(bytes);
        json!([offset, bytes.len()])
    }

    fn floats<const N: usize>(&mut self, v: &[[f32; N]]) -> Value {
        // Little-endian targets: the in-memory floats are the wire format.
        self.push(bytemuck::cast_slice(v))
    }
}

/// Parses `path` for the viewer.
pub fn load(path: &Path) -> Result<Vec<u8>, LoadError> {
    let ext = path.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).unwrap_or_default();
    if !VIEWER_EXTENSIONS.contains(&ext.as_str()) {
        return Err(LoadError::Unsupported(format!("viewer .{ext}")));
    }
    if ext == "fbx" {
        return crate::viewer_fbx::load(path);
    }
    // three's STLLoader reads binary STL about as fast as memory copies it;
    // only ASCII STL, which it parses with regular expressions, is worth it.
    if ext == "stl" && formats::stl_is_binary(path)? {
        return Err(LoadError::Unsupported("binary STL".into()));
    }
    let mut scene = formats::load_with(path, Textures::references(path))?;
    // ensureNormals() as loaders.ts does it; framing is the viewer's job.
    let _ = scene.finish();
    Ok(encode(&scene))
}

fn encode(scene: &Scene) -> Vec<u8> {
    let mut blob = Blob { data: Vec::new() };
    let mut meshes = Vec::with_capacity(scene.meshes.len());
    for m in &scene.meshes {
        let mut mesh = json!({
            "topology": match m.topology {
                Topology::Triangles => "triangles",
                Topology::Lines => "lines",
                Topology::Points => "points",
            },
            "groups": m.groups.iter().map(|g| json!([g.start, g.count, g.material])).collect::<Vec<_>>(),
            "position": blob.floats(&m.positions),
        });
        if let Some(n) = &m.normals {
            mesh["normal"] = blob.floats(n);
        }
        // The loaders store image-space texture coordinates; three's run
        // bottom-up for these formats (TextureLoader flipY).
        if let Some(uv) = &m.uv0 {
            mesh["uv"] = blob.floats(&uv.iter().map(|&[u, v]| [u, 1.0 - v]).collect::<Vec<_>>());
        }
        if let Some(uv) = &m.uv1 {
            mesh["uv1"] = blob.floats(&uv.iter().map(|&[u, v]| [u, 1.0 - v]).collect::<Vec<_>>());
        }
        if let Some(c) = &m.colors {
            // OBJ, PLY and STL colours are RGB.
            mesh["color"] = blob.floats(&c.iter().map(|c| [c[0], c[1], c[2]]).collect::<Vec<_>>());
        }
        if let Some(i) = &m.indices {
            mesh["index"] = blob.push(bytemuck::cast_slice(i));
        }
        meshes.push(mesh);
    }
    let materials: Vec<Value> = scene.materials.iter().map(|m| material(m, scene)).collect();
    let header = json!({ "meshes": meshes, "materials": materials }).to_string();
    let mut out = Vec::with_capacity(8 + header.len() + 3 + blob.data.len());
    out.extend_from_slice(b"3DVG");
    out.extend_from_slice(&(header.len() as u32).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    while out.len() % 4 != 0 {
        out.push(b' ');
    }
    out.extend_from_slice(&blob.data);
    out
}

fn material(m: &Material, scene: &Scene) -> Value {
    let mut v = json!({
        "type": match m.shading {
            Shading::Standard => "standard",
            Shading::Phong => "phong",
            Shading::Lambert => "lambert",
            Shading::Basic => "basic",
        },
        "color": n3(m.color),
        "opacity": n(m.opacity),
        "transparent": m.alpha_mode == AlphaMode::Blend,
        "alphaTest": n(if let AlphaMode::Mask(c) = m.alpha_mode { c } else { 0.0 }),
        "depthWrite": m.depth_write,
        "side": match m.side {
            Side::Front => 0,
            Side::Back => 1,
            Side::Double => 2,
        },
        "flatShading": m.flat_shading,
        "vertexColors": m.vertex_colors,
        "emissive": n3(m.emissive),
        "specular": n3(m.specular),
        "shininess": n(m.shininess),
        "roughness": n(m.roughness),
        "metalness": n(m.metalness),
        "bumpScale": n(m.bump_scale),
        "normalScale": [n(m.normal_scale.x), n(m.normal_scale.y)],
        "aoMapIntensity": n(m.ao_intensity),
        "additive": m.additive,
        "size": n(m.point_size),
        "sizeAttenuation": m.size_attenuation,
    });
    let maps = [
        ("map", m.map),
        ("emissiveMap", m.emissive_map),
        ("normalMap", m.normal_map),
        ("bumpMap", m.bump_map),
        ("aoMap", m.ao_map),
        ("alphaMap", m.alpha_map),
        ("specularMap", m.specular_map),
    ];
    let mut out = serde_json::Map::new();
    for (name, t) in maps {
        if let Some(t) = t.and_then(|t| texture(&t, scene)) {
            out.insert(name.to_string(), t);
        }
    }
    v["maps"] = Value::Object(out);
    v
}

fn texture(t: &TextureRef, scene: &Scene) -> Option<Value> {
    let source = scene.image_references.get(&t.image)?;
    let wrap = |w: Wrap| match w {
        Wrap::Repeat => 1000,
        Wrap::Clamp => 1001,
        Wrap::Mirror => 1002,
    };
    Some(json!({
        "source": source,
        "channel": t.uv_set,
        "wrapS": wrap(t.wrap_u),
        "wrapT": wrap(t.wrap_v),
        "matrix": three_matrix(t.transform.0).map(n),
    }))
}

/// The texture matrix back from image space to three's v-up space: F·T·F
/// for F = [[1,0,0],[0,-1,1],[0,0,1]] and T = [[a,b,c],[d,e,f],[0,0,1]],
/// row-major as Matrix3.set takes it.
fn three_matrix([a, b, c, d, e, f]: [f32; 6]) -> [f32; 9] {
    [a, -b, b + c, -d, e, 1.0 - e - f, 0.0, 0.0, 1.0]
}
