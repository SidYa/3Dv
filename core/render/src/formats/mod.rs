// Model loaders, one per format, each a port of the three.js loader the
// viewer uses (app/src/lib/loaders.ts), so a thumbnail shows the model the
// way the viewer does: same materials, same colours, same orientation.

use std::fmt;
use std::path::Path;

use crate::scene::Scene;
use crate::texture::Textures;

mod dae;
mod fbx;
mod gltf;
mod obj;
mod ply;
mod stl;
mod tds;

#[derive(Debug)]
pub enum LoadError {
    Io(std::io::Error),
    /// Valid file using something only the three.js loaders handle (Draco,
    /// meshopt, KTX2...). The caller falls back to the WebView renderer.
    Unsupported(String),
    Invalid(String),
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::Io(e) => write!(f, "io: {e}"),
            LoadError::Unsupported(s) => write!(f, "unsupported: {s}"),
            LoadError::Invalid(s) => write!(f, "invalid: {s}"),
        }
    }
}

impl From<std::io::Error> for LoadError {
    fn from(e: std::io::Error) -> Self {
        LoadError::Io(e)
    }
}

pub fn invalid(msg: impl Into<String>) -> LoadError {
    LoadError::Invalid(msg.into())
}

pub const EXTENSIONS: &[&str] = &["glb", "gltf", "obj", "fbx", "stl", "ply", "dae", "3ds"];

/// Loads any supported model. `max_texture` caps texture sizes.
pub fn load(path: &Path, max_texture: u32) -> Result<Scene, LoadError> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();
    let mut textures = Textures::new(path, max_texture);
    let mut scene = match ext.as_str() {
        "glb" | "gltf" => gltf::load(path, &mut textures),
        "obj" => obj::load(path, &mut textures),
        "fbx" => fbx::load(path, &mut textures),
        "stl" => stl::load(path),
        "ply" => ply::load(path),
        "dae" => dae::load(path, &mut textures),
        "3ds" => tds::load(path, &mut textures),
        _ => Err(LoadError::Unsupported(format!("extension .{ext}"))),
    }?;
    textures.finish(&mut scene);
    Ok(scene)
}

/// The material loaders.ts gives STL and PLY meshes (asMesh): grey, or white
/// with vertex colours, slightly glossy.
pub(crate) fn mesh_default_material(vertex_colors: bool) -> crate::scene::Material {
    let mut m = crate::scene::Material::standard();
    m.color = if vertex_colors { glam::Vec3::ONE } else { glam::Vec3::splat(crate::scene::srgb_to_linear(0x80 as f32 / 255.0)) };
    m.vertex_colors = vertex_colors;
    m.metalness = 0.05;
    m.roughness = 0.65;
    m
}

/// The material loaders.ts gives a PLY point cloud (asPoints): light grey or
/// white with vertex colours, attenuated dots.
pub(crate) fn points_default_material(vertex_colors: bool) -> crate::scene::Material {
    let mut m = crate::scene::Material::points(1.0, true);
    m.color = if vertex_colors { glam::Vec3::ONE } else { glam::Vec3::splat(crate::scene::srgb_to_linear(0xcc as f32 / 255.0)) };
    m.vertex_colors = vertex_colors;
    m
}
