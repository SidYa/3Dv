// Texture images for the loaders: decoding, finding the file a model refers
// to, and scaling it down to what the thumbnail can show.
//
// A 256-pixel thumbnail never samples a 4096-pixel texture at its full
// size, so images are reduced right at decode time: JPEGs through the
// decoder's DCT scaling (an 8x smaller decode is ~20x faster), everything
// else by resampling. Each file is decoded once per model, and all of a
// model's images are decoded in parallel once the loader is done.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::scene::{Image, Scene};

/// What happens to a texture that cannot be shown. three's TextureLoader
/// hands out a texture that never receives an image when the file is missing
/// or the browser cannot decode it; WebGL samples such a texture as black,
/// so that is what the viewer shows. GLTFLoader instead drops the map.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Missing {
    Black,
    Skip,
}

enum Source {
    File(PathBuf),
    Bytes(Vec<u8>),
}

/// An image slot handed out to a loader, decoded later by `finish`.
struct Pending {
    image: usize,
    source: Source,
    tga: bool,
    missing: Missing,
}

pub struct Textures {
    /// Model directory, the base of relative texture paths.
    base: PathBuf,
    max_size: u32,
    by_path: HashMap<(PathBuf, bool, bool), Option<usize>>,
    by_key: HashMap<String, Option<usize>>,
    black: Option<usize>,
    pending: Vec<Pending>,
}

impl Textures {
    pub fn new(model: &Path, max_size: u32) -> Textures {
        Textures {
            base: model.parent().map(Path::to_path_buf).unwrap_or_default(),
            max_size: max_size.max(16),
            by_path: HashMap::new(),
            by_key: HashMap::new(),
            black: None,
            pending: Vec::new(),
        }
    }

    /// A texture referenced from the model by a path relative to it, loaded
    /// the way three's TextureLoader would: only formats a browser decodes
    /// (plus TGA where the loader brings TGALoader, `tga`), and black when
    /// that fails.
    pub fn load(&mut self, scene: &mut Scene, reference: &str, tga: bool) -> Option<usize> {
        self.load_with(scene, reference, tga, Missing::Black)
    }

    pub fn load_with(&mut self, scene: &mut Scene, reference: &str, tga: bool, missing: Missing) -> Option<usize> {
        let Some(path) = self.resolve(reference) else { return self.fallback(scene, missing) };
        let key = (path.clone(), tga, missing == Missing::Black);
        if let Some(cached) = self.by_path.get(&key) {
            return *cached;
        }
        let image = self.defer(scene, Source::File(path), tga, missing);
        self.by_path.insert(key, Some(image));
        Some(image)
    }

    /// A texture whose bytes live inside the model (glTF buffer views, FBX
    /// embedded content). `key` identifies it for the cache.
    pub fn load_bytes(&mut self, scene: &mut Scene, key: &str, bytes: &[u8], missing: Missing) -> Option<usize> {
        if let Some(cached) = self.by_key.get(key) {
            return *cached;
        }
        let image = self.defer(scene, Source::Bytes(bytes.to_vec()), false, missing);
        self.by_key.insert(key.to_string(), Some(image));
        Some(image)
    }

    fn defer(&mut self, scene: &mut Scene, source: Source, tga: bool, missing: Missing) -> usize {
        let image = scene.add_image(Image { width: 1, height: 1, rgba: vec![0, 0, 0, 255] });
        self.pending.push(Pending { image, source, tga, missing });
        image
    }

    fn fallback(&mut self, scene: &mut Scene, missing: Missing) -> Option<usize> {
        match missing {
            Missing::Skip => None,
            Missing::Black => Some(*self.black.get_or_insert_with(|| scene.add_image(Image { width: 1, height: 1, rgba: vec![0, 0, 0, 255] }))),
        }
    }

    /// Decodes every image the loader asked for, on all cores. A failed
    /// image stays black, or - for GLTFLoader's semantics - is removed from
    /// the materials that use it.
    pub fn finish(&mut self, scene: &mut Scene) {
        let pending = std::mem::take(&mut self.pending);
        if pending.is_empty() {
            return;
        }
        let max_size = self.max_size;
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(pending.len()).max(1);
        let next = std::sync::atomic::AtomicUsize::new(0);
        let results: Vec<(usize, Option<Image>)> = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..threads)
                .map(|_| {
                    scope.spawn(|| {
                        let mut out = Vec::new();
                        loop {
                            let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            let Some(p) = pending.get(i) else { break };
                            let decoded = std::panic::catch_unwind(|| match &p.source {
                                Source::File(path) => std::fs::read(path).ok().and_then(|bytes| decode_like_browser(&bytes, path, p.tga, max_size)),
                                Source::Bytes(bytes) => decode_like_browser(bytes, Path::new(""), false, max_size),
                            })
                            .ok()
                            .flatten();
                            out.push((i, decoded));
                        }
                        out
                    })
                })
                .collect();
            workers.into_iter().flat_map(|w| w.join().unwrap_or_default()).collect()
        });
        let mut dropped = Vec::new();
        for (i, decoded) in results {
            let p = &pending[i];
            match decoded {
                Some(img) => scene.images[p.image] = img,
                // The slot already holds a black pixel.
                None if p.missing == Missing::Black => {}
                None => dropped.push(p.image),
            }
        }
        if !dropped.is_empty() {
            for m in scene.materials.iter_mut() {
                for slot in [&mut m.map, &mut m.emissive_map, &mut m.metal_rough_map, &mut m.normal_map, &mut m.bump_map, &mut m.ao_map, &mut m.alpha_map, &mut m.specular_map] {
                    if slot.is_some_and(|t| dropped.contains(&t.image)) {
                        *slot = None;
                    }
                }
            }
        }
    }

    /// The file three requests: `<model dir>/<reference>`. The WebView turns
    /// backslashes into slashes and the asset protocol percent-decodes the
    /// path, so both spellings are tried.
    pub fn resolve(&self, reference: &str) -> Option<PathBuf> {
        let reference = reference.trim();
        if reference.is_empty() {
            return None;
        }
        let normalized = reference.replace('\\', "/");
        let decoded = percent_decode(&normalized);
        let found = [normalized.as_str(), decoded.as_str()]
            .into_iter()
            .map(|r| self.base.join(r.trim_start_matches("./")))
            .find(|c| c.is_file());
        found
    }
}

/// Decodes an image if a browser would: PNG, JPEG, GIF, BMP and WebP by
/// their signatures, TGA only when `tga` is set (by extension, it has no
/// signature).
fn decode_like_browser(bytes: &[u8], path: &Path, tga: bool, max_size: u32) -> Option<Image> {
    let browser = bytes.starts_with(&[0x89, b'P', b'N', b'G'])
        || bytes.starts_with(&[0xFF, 0xD8, 0xFF])
        || bytes.starts_with(b"GIF8")
        || bytes.starts_with(b"BM")
        || (bytes.len() > 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP");
    if browser {
        return decode(bytes, max_size);
    }
    let is_tga = path.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("tga"));
    if tga && is_tga {
        let img = image::load_from_memory_with_format(bytes, image::ImageFormat::Tga).ok()?;
        return Some(scaled(img, max_size));
    }
    None
}

fn scaled(img: image::DynamicImage, max_size: u32) -> Image {
    let img = if img.width() > max_size || img.height() > max_size { img.resize(max_size, max_size, image::imageops::FilterType::Triangle) } else { img };
    let rgba = img.to_rgba8();
    let (width, height) = rgba.dimensions();
    Image { width, height, rgba: rgba.into_raw() }
}

fn percent_decode(s: &str) -> String {
    if !s.contains('%') {
        return s.to_string();
    }
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(v) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Decodes any supported image and scales it so neither side exceeds
/// `max_size`.
pub fn decode(bytes: &[u8], max_size: u32) -> Option<Image> {
    let is_jpeg = bytes.len() > 3 && bytes[0] == 0xFF && bytes[1] == 0xD8 && bytes[2] == 0xFF;
    if is_jpeg {
        if let Some(img) = decode_jpeg(bytes, max_size) {
            return Some(img);
        }
    }
    let img = image::load_from_memory(bytes).ok()?;
    let img = if img.width() > max_size || img.height() > max_size {
        img.resize(max_size, max_size, image::imageops::FilterType::Triangle)
    } else {
        img
    };
    let rgba = img.to_rgba8();
    let (width, height) = rgba.dimensions();
    if width == 0 || height == 0 {
        return None;
    }
    Some(Image { width, height, rgba: rgba.into_raw() })
}

/// JPEG through DCT scaling: the decoder emits the smallest of 1/1, 1/2,
/// 1/4 or 1/8 that still covers the requested size.
fn decode_jpeg(bytes: &[u8], max_size: u32) -> Option<Image> {
    let mut decoder = jpeg_decoder::Decoder::new(bytes);
    decoder.read_info().ok()?;
    let info = decoder.info()?;
    let (w, h) = (info.width as u32, info.height as u32);
    if w == 0 || h == 0 {
        return None;
    }
    let scale = (max_size as f32 / w.max(h) as f32).min(1.0);
    let req_w = ((w as f32 * scale).ceil() as u16).max(1);
    let req_h = ((h as f32 * scale).ceil() as u16).max(1);
    let (out_w, out_h) = decoder.scale(req_w, req_h).ok()?;
    let pixels = decoder.decode().ok()?;
    let info = decoder.info()?;
    let (out_w, out_h) = (out_w as u32, out_h as u32);
    let count = (out_w * out_h) as usize;
    let mut rgba = Vec::with_capacity(count * 4);
    match info.pixel_format {
        jpeg_decoder::PixelFormat::RGB24 => {
            for px in pixels.chunks_exact(3).take(count) {
                rgba.extend_from_slice(&[px[0], px[1], px[2], 255]);
            }
        }
        jpeg_decoder::PixelFormat::L8 => {
            for &l in pixels.iter().take(count) {
                rgba.extend_from_slice(&[l, l, l, 255]);
            }
        }
        jpeg_decoder::PixelFormat::L16 => {
            for px in pixels.chunks_exact(2).take(count) {
                let l = px[0];
                rgba.extend_from_slice(&[l, l, l, 255]);
            }
        }
        jpeg_decoder::PixelFormat::CMYK32 => {
            for px in pixels.chunks_exact(4).take(count) {
                let k = 255 - px[3] as u32;
                let conv = |c: u8| ((255 - c as u32) * k / 255) as u8;
                rgba.extend_from_slice(&[conv(px[0]), conv(px[1]), conv(px[2]), 255]);
            }
        }
    }
    if rgba.len() != count * 4 {
        return None;
    }
    let img = Image { width: out_w, height: out_h, rgba };
    if out_w > max_size || out_h > max_size {
        let buffer = image::RgbaImage::from_raw(out_w, out_h, img.rgba)?;
        let resized = image::DynamicImage::ImageRgba8(buffer).resize(max_size, max_size, image::imageops::FilterType::Triangle);
        let rgba = resized.to_rgba8();
        let (width, height) = rgba.dimensions();
        return Some(Image { width, height, rgba: rgba.into_raw() });
    }
    Some(img)
}

/// The mip chain the GPU samples from, as WebGL's generateMipmap would
/// build it: 2x2 box filtering, in linear light for sRGB images.
pub fn mip_chain(image: &Image, srgb: bool) -> Vec<(u32, u32, Vec<u8>)> {
    let mut levels = vec![(image.width, image.height, image.rgba.clone())];
    let to_lin: Vec<f32> = (0..256).map(|i| crate::scene::srgb_to_linear(i as f32 / 255.0)).collect();
    loop {
        let (w, h, ref data) = *levels.last().unwrap();
        if w == 1 && h == 1 {
            break;
        }
        let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
        let mut out = vec![0u8; (nw * nh * 4) as usize];
        for y in 0..nh {
            for x in 0..nw {
                let mut acc = [0f32; 4];
                let mut n = 0.0;
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let sx = (x * 2 + dx).min(w - 1);
                    let sy = (y * 2 + dy).min(h - 1);
                    let i = ((sy * w + sx) * 4) as usize;
                    for c in 0..4 {
                        let v = data[i + c];
                        acc[c] += if srgb && c < 3 { to_lin[v as usize] } else { v as f32 / 255.0 };
                    }
                    n += 1.0;
                }
                let o = ((y * nw + x) * 4) as usize;
                for c in 0..4 {
                    let v = acc[c] / n;
                    let v = if srgb && c < 3 { linear_to_srgb(v) } else { v };
                    out[o + c] = (v * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
                }
            }
        }
        levels.push((nw, nh, out));
    }
    levels
}

fn linear_to_srgb(c: f32) -> f32 {
    if c < 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(0.41666) - 0.055
    }
}
