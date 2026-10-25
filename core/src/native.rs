// The native (Rust + wgpu) thumbnail renderer, shared by the 3dv-thumbnail
// CLI, the Linux thumbnail cache and the Windows thumbnail server. Every
// failure returns None and the caller falls back to the WebView renderer,
// which stays the reference for anything the native loaders cannot handle.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use dv3_render::Renderer;

/// Size of the thumbnails the CLI and the freedesktop cache store.
pub const THUMB_SIZE: u32 = 256;

/// One device per process, created on first use and kept warm.
fn renderer_slot() -> &'static Mutex<Option<Renderer>> {
    static SLOT: OnceLock<Mutex<Option<Renderer>>> = OnceLock::new();
    SLOT.get_or_init(|| Mutex::new(None))
}

/// Runs `f` with the process renderer, creating it if needed. A panic
/// inside (a lost device, a driver error surfacing as a wgpu panic) drops
/// the renderer, so the next request starts on a fresh one.
pub fn with_renderer<T>(software: bool, f: impl FnOnce(&mut Renderer) -> Option<T>) -> Option<T> {
    let mut slot = renderer_slot().lock().unwrap_or_else(|e| e.into_inner());
    if slot.is_none() {
        let created = catch_unwind(|| Renderer::new(software).or_else(|_| Renderer::new(true)));
        match created {
            Ok(Ok(mut r)) => {
                r.warm_up();
                if cfg!(debug_assertions) {
                    eprintln!("[Thumbnail] native renderer on {}", r.adapter);
                }
                *slot = Some(r);
            }
            Ok(Err(e)) => {
                eprintln!("[Thumbnail] no native renderer: {e}");
                return None;
            }
            Err(_) => return None,
        }
    }
    let renderer = slot.as_mut()?;
    match catch_unwind(AssertUnwindSafe(|| f(renderer))) {
        Ok(result) => result,
        Err(_) => {
            eprintln!("[Thumbnail] native renderer crashed; it will be recreated");
            *slot = None;
            None
        }
    }
}

/// Creates the renderer ahead of the first request.
#[cfg_attr(not(windows), allow(dead_code))] // only the Windows server warms up
pub fn warm_up(software: bool) {
    with_renderer(software, |_| Some(()));
}

/// Renders `path` into RGBA8 pixels, `size`×`size`, or None.
pub fn render_rgba(path: &Path, size: u32, software: bool) -> Option<Vec<u8>> {
    // Loading needs no GPU: done outside the renderer lock, so a server
    // reads several models in parallel.
    let prepared = match catch_unwind(|| dv3_render::prepare(path, size)) {
        Ok(Ok(p)) => p,
        Ok(Err(e)) => {
            if cfg!(debug_assertions) {
                eprintln!("[Thumbnail] native: {e} ({})", path.display());
            }
            return None;
        }
        Err(_) => {
            eprintln!("[Thumbnail] native loader crashed on {}", path.display());
            return None;
        }
    };
    with_renderer(software, |r| match dv3_render::render(r, &prepared, size) {
        Ok(t) => Some(t.rgba),
        Err(e) => {
            if cfg!(debug_assertions) {
                eprintln!("[Thumbnail] native: {e} ({})", path.display());
            }
            None
        }
    })
}

/// Renders `path` into a PNG, or None.
pub fn render_png(path: &Path, size: u32, software: bool) -> Option<Vec<u8>> {
    let rgba = render_rgba(path, size, software)?;
    dv3_render::encode_png(&rgba, size, size)
}
