// The thumbnail renderer (core/render: Rust loaders + wgpu), shared by the
// 3dv-thumbnail CLI, the Linux thumbnail cache and the Windows thumbnail
// server.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use dv3_render::{RenderError, Renderer, ThumbnailError};

/// Size of the thumbnails the CLI and the freedesktop cache store.
pub const THUMB_SIZE: u32 = 256;

/// Why there is no thumbnail.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Failure {
    /// The model itself gives no picture: unreadable, nothing to draw, or
    /// nothing in view. Trying again will not change that.
    NoPicture,
    /// The renderer failed (no device, a device error, a crash); another
    /// attempt, possibly on the software rasterizer, may succeed.
    Renderer,
}

/// One device per process, created on first use and kept warm.
fn renderer_slot() -> &'static Mutex<Option<Renderer>> {
    static SLOT: OnceLock<Mutex<Option<Renderer>>> = OnceLock::new();
    SLOT.get_or_init(|| Mutex::new(None))
}

/// Runs `f` with the process renderer, creating it if needed. A panic
/// inside (a lost device, a driver error surfacing as a wgpu panic) drops
/// the renderer, so the next request starts on a fresh one.
fn with_renderer<T>(software: bool, f: impl FnOnce(&mut Renderer) -> Result<T, Failure>) -> Result<T, Failure> {
    let mut slot = renderer_slot().lock().unwrap_or_else(|e| e.into_inner());
    if slot.is_none() {
        let created = catch_unwind(|| Renderer::new(software).or_else(|_| Renderer::new(true)));
        match created {
            Ok(Ok(mut r)) => {
                r.warm_up();
                if cfg!(debug_assertions) {
                    eprintln!("[Thumbnail] renderer on {}", r.adapter);
                }
                *slot = Some(r);
            }
            Ok(Err(e)) => {
                eprintln!("[Thumbnail] no renderer: {e}");
                return Err(Failure::Renderer);
            }
            Err(_) => return Err(Failure::Renderer),
        }
    }
    let renderer = slot.as_mut().ok_or(Failure::Renderer)?;
    match catch_unwind(AssertUnwindSafe(|| f(renderer))) {
        Ok(result) => result,
        Err(_) => {
            eprintln!("[Thumbnail] renderer crashed; it will be recreated");
            *slot = None;
            Err(Failure::Renderer)
        }
    }
}

/// Creates the renderer ahead of the first request.
pub fn warm_up(software: bool) {
    let _ = with_renderer(software, |_| Ok(()));
}

/// Renders `path` into RGBA8 pixels, `size`×`size`.
pub fn render_rgba(path: &Path, size: u32, software: bool) -> Result<Vec<u8>, Failure> {
    // The device takes longer to create than most models take to load:
    // without one yet, create it while the model loads. (A held lock means
    // another thread is already at it.)
    if let Ok(slot) = renderer_slot().try_lock() {
        if slot.is_none() {
            std::thread::spawn(move || warm_up(software));
        }
    }
    // Loading needs no GPU: done outside the renderer lock, so a server
    // reads several models in parallel.
    let prepared = match catch_unwind(|| dv3_render::prepare(path, size)) {
        Ok(Ok(p)) => p,
        Ok(Err(e)) => {
            if cfg!(debug_assertions) {
                eprintln!("[Thumbnail] {e} ({})", path.display());
            }
            return Err(Failure::NoPicture);
        }
        Err(_) => {
            eprintln!("[Thumbnail] loader crashed on {}", path.display());
            return Err(Failure::NoPicture);
        }
    };
    with_renderer(software, |r| match dv3_render::render(r, &prepared, size) {
        Ok(t) => Ok(t.rgba),
        Err(e) => {
            if cfg!(debug_assertions) {
                eprintln!("[Thumbnail] {e} ({})", path.display());
            }
            match e {
                ThumbnailError::Render(RenderError::EmptyFrame) => Err(Failure::NoPicture),
                _ => Err(Failure::Renderer),
            }
        }
    })
}

/// Renders `path` into a PNG.
pub fn render_png(path: &Path, size: u32, software: bool) -> Result<Vec<u8>, Failure> {
    let rgba = render_rgba(path, size, software)?;
    dv3_render::encode_png(&rgba, size, size).ok_or(Failure::Renderer)
}
