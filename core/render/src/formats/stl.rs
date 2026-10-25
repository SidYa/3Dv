// STL, binary and ASCII: port of three's STLLoader plus the material
// loaders.ts gives it. Faces are unshared (non-indexed); the file's facet
// normal goes to all three vertices, and all-zero normals - common in STL -
// are recomputed later by Scene::finish, as in the viewer.

use std::path::Path;

use super::{invalid, mesh_default_material, LoadError};
use crate::scene::{srgb_to_linear, Mesh, Placement, Scene, Topology};

pub fn load(path: &Path) -> Result<Scene, LoadError> {
    let data = std::fs::read(path)?;
    let mut mesh = if is_binary(&data) { parse_binary(&data)? } else { parse_ascii(&data)? };
    let vertex_colors = mesh.colors.is_some();
    let mut scene = Scene::default();
    let material = scene.add_material(mesh_default_material(vertex_colors));
    mesh = mesh.with_material(material);
    scene.meshes.push(Placement::identity().apply(mesh));
    Ok(scene)
}

/// STLLoader.isBinary(): the size a binary file of that face count would
/// have decides; failing that, a file starting with "solid" is ASCII.
fn is_binary(data: &[u8]) -> bool {
    is_binary_header(data, data.len() as u64)
}

/// is_binary() from the first 84 bytes and the file size.
pub(crate) fn is_binary_header(data: &[u8], size: u64) -> bool {
    if data.len() >= 84 {
        let faces = u32::from_le_bytes([data[80], data[81], data[82], data[83]]) as u64;
        if 84 + faces * 50 == size {
            return true;
        }
    }
    // STLLoader looks for "solid" at any of the first five offsets.
    for off in 0..5 {
        if data.len() >= off + 5 && &data[off..off + 5] == b"solid" {
            return false;
        }
    }
    true
}

fn parse_binary(data: &[u8]) -> Result<Mesh, LoadError> {
    if data.len() < 84 {
        return Err(invalid("STL too short"));
    }
    let faces = u32::from_le_bytes([data[80], data[81], data[82], data[83]]) as usize;
    let faces = faces.min((data.len() - 84) / 50);
    let f32_at = |o: usize| f32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);

    // "COLOR=rgba" in the header: Magics colours. The last match wins.
    let mut default_color: Option<[f32; 3]> = None;
    for i in 0..(80 - 10) {
        if &data[i..i + 6] == b"COLOR=" {
            default_color = Some([data[i + 6] as f32 / 255.0, data[i + 7] as f32 / 255.0, data[i + 8] as f32 / 255.0]);
        }
    }

    let mut positions = Vec::with_capacity(faces * 3);
    let mut normals = Vec::with_capacity(faces * 3);
    let mut colors = default_color.map(|_| Vec::with_capacity(faces * 3));
    for face in 0..faces {
        let start = 84 + face * 50;
        let n = [f32_at(start), f32_at(start + 4), f32_at(start + 8)];
        let color = default_color.map(|default| {
            let packed = u16::from_le_bytes([data[start + 48], data[start + 49]]);
            let c = if packed & 0x8000 == 0 {
                [
                    (packed & 0x1F) as f32 / 31.0,
                    ((packed >> 5) & 0x1F) as f32 / 31.0,
                    ((packed >> 10) & 0x1F) as f32 / 31.0,
                ]
            } else {
                default
            };
            // setRGB(r, g, b, SRGBColorSpace)
            [srgb_to_linear(c[0]), srgb_to_linear(c[1]), srgb_to_linear(c[2]), 1.0]
        });
        for i in 1..=3 {
            let v = start + i * 12;
            positions.push([f32_at(v), f32_at(v + 4), f32_at(v + 8)]);
            normals.push(n);
            if let (Some(colors), Some(c)) = (colors.as_mut(), color) {
                colors.push(c);
            }
        }
    }
    let mut mesh = Mesh::new(Topology::Triangles, positions);
    mesh.normals = Some(normals);
    mesh.colors = colors;
    Ok(mesh)
}

/// STLLoader.parseASCII(): every "vertex x y z" inside a facet, each with the
/// facet's last "normal". Groups per solid are ignored: the viewer draws the
/// mesh with a single material.
fn parse_ascii(data: &[u8]) -> Result<Mesh, LoadError> {
    let text = String::from_utf8_lossy(data);
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut normal = [0.0f32; 3];
    let mut in_solid = false;
    let mut tokens = text.split_ascii_whitespace().peekable();
    while let Some(tok) = tokens.next() {
        match tok {
            "solid" => in_solid = true,
            "endsolid" => in_solid = false,
            // "facet normal nx ny nz"; also a stray "normal" keyword.
            "normal" if in_solid => {
                normal = read3(&mut tokens);
            }
            "vertex" if in_solid => {
                positions.push(read3(&mut tokens));
                normals.push(normal);
            }
            _ => {}
        }
    }
    if positions.is_empty() {
        return Err(invalid("STL without vertices"));
    }
    let mut mesh = Mesh::new(Topology::Triangles, positions);
    mesh.normals = Some(normals);
    Ok(mesh)
}

/// parseFloat on the next three tokens; like the loader's regex, a token
/// that is not a number reads as NaN and leaves the token in place.
fn read3<'a>(tokens: &mut std::iter::Peekable<impl Iterator<Item = &'a str>>) -> [f32; 3] {
    let mut out = [f32::NAN; 3];
    for v in out.iter_mut() {
        match tokens.peek().and_then(|t| t.parse::<f32>().ok()) {
            Some(x) => {
                *v = x;
                tokens.next();
            }
            None => break,
        }
    }
    out
}
