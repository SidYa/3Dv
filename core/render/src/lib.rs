// Model loading and thumbnail rendering of 3Dv.
//
// Rust ports of the three.js loaders the viewer uses, an offscreen wgpu
// renderer reproducing the viewer's scene (app/src/lib/scene.ts) closely
// enough that a cached thumbnail looks like the opened model, and (viewer.rs)
// the fast path the viewer itself takes for large text formats.

mod env;
pub mod formats;
mod gpu;
pub mod scene;
pub mod texture;
pub mod viewer;
mod viewer_fbx;

use std::path::Path;
use std::time::{Duration, Instant};

pub use formats::{LoadError, EXTENSIONS};
pub use gpu::{RenderError, Renderer};

#[derive(Debug)]
pub enum ThumbnailError {
    Load(LoadError),
    /// The model has nothing to draw (scene.ts hasRenderableGeometry /
    /// frameObject returned nothing).
    NoGeometry,
    Render(RenderError),
}

impl std::fmt::Display for ThumbnailError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ThumbnailError::Load(e) => write!(f, "load: {e}"),
            ThumbnailError::NoGeometry => write!(f, "no geometry"),
            ThumbnailError::Render(e) => write!(f, "render: {e}"),
        }
    }
}

/// A rendered thumbnail: `size`×`size` RGBA8, top row first.
pub struct Thumbnail {
    pub size: u32,
    pub rgba: Vec<u8>,
    pub load_time: Duration,
    pub render_time: Duration,
}

/// A model loaded and prepared for rendering. Loading needs no GPU, so
/// several can be prepared in parallel while one renderer draws.
pub struct Prepared {
    scene: scene::Scene,
    framing: scene::Framing,
    load_time: Duration,
}

/// Loads `path` for a thumbnail of `size` pixels. Textures are decoded at
/// most at twice that size, which is all a minified sample can use.
pub fn prepare(path: &Path, size: u32) -> Result<Prepared, ThumbnailError> {
    let started = Instant::now();
    let max_texture = (size * 2).clamp(256, 2048);
    let mut scene = formats::load(path, max_texture).map_err(ThumbnailError::Load)?;
    if !scene.has_renderable_geometry() {
        return Err(ThumbnailError::NoGeometry);
    }
    let framing = scene.finish().ok_or(ThumbnailError::NoGeometry)?;
    Ok(Prepared { scene, framing, load_time: started.elapsed() })
}

/// Renders a prepared model at `size`×`size` pixels.
pub fn render(renderer: &mut Renderer, prepared: &Prepared, size: u32) -> Result<Thumbnail, ThumbnailError> {
    let started = Instant::now();
    let rgba = renderer.render(&prepared.scene, &prepared.framing, size).map_err(ThumbnailError::Render)?;
    Ok(Thumbnail { size, rgba, load_time: prepared.load_time, render_time: started.elapsed() })
}

/// prepare() and render() in one go.
pub fn thumbnail(renderer: &mut Renderer, path: &Path, size: u32) -> Result<Thumbnail, ThumbnailError> {
    let prepared = prepare(path, size)?;
    render(renderer, &prepared, size)
}

/// Encodes RGBA8 pixels as a PNG.
pub fn encode_png(rgba: &[u8], width: u32, height: u32) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        // Thumbnails are written often and read rarely: fast compression.
        encoder.set_compression(png::Compression::Fast);
        let mut writer = encoder.write_header().ok()?;
        writer.write_image_data(rgba).ok()?;
    }
    Some(out)
}

pub fn is_supported(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
}
