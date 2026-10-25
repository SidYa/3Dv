// Thumbnail storage: the freedesktop cache, atomic writes and housekeeping.
// The pictures themselves come from the native renderer (native.rs).
//
// 3Dv keeps no thumbnail cache of its own. Pictures live where the OS keeps
// every other thumbnail: Explorer's own cache on Windows (fed by the
// thumbnail-provider DLL through the thumbnail server), the shared
// freedesktop.org cache on Linux, nothing on macOS.

use std::path::{Path, PathBuf};

/// Larger models are not prewarmed into the Linux cache on open.
pub const PREWARM_MAX_BYTES: u64 = 128 * 1024 * 1024;

const META_READ_LIMIT: usize = 64 * 1024;
const PNG_SIGNATURE: [u8; 8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

// --- file URI ------------------------------------------------------------
// Characters glib's g_filename_to_uri() leaves unescaped, so a thumbnail
// lands on exactly the name every other file manager looks for.
const URI_SAFE: &str = "!$&'()*+,-./:;=?@_~";

fn encode_uri_path(native_path: &str) -> String {
    let mut out = String::new();
    for byte in native_path.as_bytes() {
        let b = *byte;
        let alnum = b.is_ascii_alphanumeric();
        if alnum || URI_SAFE.as_bytes().contains(&b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{:02X}", b));
        }
    }
    out
}

fn to_file_uri(file_path: &Path) -> String {
    let abs = std::path::absolute(file_path)
        .unwrap_or_else(|_| file_path.to_path_buf())
        .to_string_lossy()
        .replace('\\', "/");
    let p = if abs.starts_with('/') {
        abs
    } else {
        format!("/{abs}")
    };
    format!("file://{}", encode_uri_path(&p))
}

// --- cache location -------------------------------------------------------
pub fn has_system_thumbnail_cache() -> bool {
    cfg!(target_os = "linux")
}

fn thumbnail_dir() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| dirs_home().map(|h| h.join(".cache")))?;
    Some(base.join("thumbnails").join("large"))
}

fn dirs_home() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("USERPROFILE").map(PathBuf::from))
}

fn thumbnail_path(file_path: &Path) -> Option<PathBuf> {
    let uri = to_file_uri(file_path);
    let hash = format!("{:x}", md5::compute(uri.as_bytes()));
    thumbnail_dir().map(|d| d.join(format!("{hash}.png")))
}

struct ModelIdentity {
    uri: String,
    mtime: u64,
    size: u64,
}

fn model_identity(file_path: &Path) -> Option<ModelIdentity> {
    let meta = std::fs::metadata(file_path).ok()?;
    if !meta.is_file() {
        return None;
    }
    let mtime = meta
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs();
    Some(ModelIdentity {
        uri: to_file_uri(file_path),
        mtime,
        size: meta.len(),
    })
}

// --- PNG tEXt chunks -------------------------------------------------------
fn crc32(data: &[u8]) -> u32 {
    let mut table = [0u32; 256];
    for (n, slot) in table.iter_mut().enumerate() {
        let mut c = n as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 {
                0xedb88320 ^ (c >> 1)
            } else {
                c >> 1
            };
        }
        *slot = c;
    }
    let mut crc: u32 = 0xFFFF_FFFF;
    for byte in data {
        crc = table[((crc ^ (*byte as u32)) & 0xff) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

fn build_text_chunk(keyword: &str, value: &str) -> Vec<u8> {
    let mut body = b"tEXt".to_vec();
    body.extend_from_slice(keyword.as_bytes());
    body.push(0);
    body.extend_from_slice(value.as_bytes());
    let mut out = (body.len() as u32 - 4).to_be_bytes().to_vec();
    let crc = crc32(&body).to_be_bytes();
    out.extend_from_slice(&body);
    out.extend_from_slice(&crc);
    out
}

fn is_png(buffer: &[u8]) -> bool {
    buffer.len() > 8 + 12 && buffer.starts_with(&PNG_SIGNATURE)
}

// Inserts the identity chunks right behind IHDR, where every decoder
// tolerates ancillary chunks.
fn embed_thumb_meta(png: &[u8], identity: &ModelIdentity) -> Vec<u8> {
    let ihdr_len =
        u32::from_be_bytes([png[8], png[9], png[10], png[11]]) as usize;
    let insert_at = 8 + 12 + ihdr_len;
    if insert_at > png.len() {
        return png.to_vec();
    }
    let mut out = Vec::with_capacity(png.len() + 128);
    out.extend_from_slice(&png[..insert_at]);
    out.extend_from_slice(&build_text_chunk("Thumb::URI", &identity.uri));
    out.extend_from_slice(&build_text_chunk(
        "Thumb::MTime",
        &identity.mtime.to_string(),
    ));
    out.extend_from_slice(&build_text_chunk(
        "Thumb::Size",
        &identity.size.to_string(),
    ));
    out.extend_from_slice(&png[insert_at..]);
    out
}

fn parse_thumb_meta(buffer: &[u8]) -> Option<(String, u64, Option<u64>)> {
    if !is_png(buffer) {
        return None;
    }
    let (mut uri, mut mtime, mut size) = (None, None, None);
    let mut offset = 8;
    while offset + 12 <= buffer.len() {
        let length =
            u32::from_be_bytes([buffer[offset], buffer[offset + 1], buffer[offset + 2], buffer[offset + 3]])
                as usize;
        let chunk_type = &buffer[offset + 4..offset + 8];
        if chunk_type == b"IDAT" || chunk_type == b"IEND" {
            break;
        }
        if length > buffer.len() - offset - 12 {
            break;
        }
        if chunk_type == b"tEXt" {
            let data = &buffer[offset + 8..offset + 8 + length];
            if let Some(sep) = data.iter().position(|b| *b == 0) {
                if sep > 0 {
                    let key = String::from_utf8_lossy(&data[..sep]);
                    let value = String::from_utf8_lossy(&data[sep + 1..]);
                    match key.as_ref() {
                        "Thumb::URI" => uri = Some(value.into_owned()),
                        "Thumb::MTime" => mtime = value.parse().ok(),
                        "Thumb::Size" => size = value.parse().ok(),
                        _ => {}
                    }
                }
            }
        }
        offset += 12 + length;
    }
    Some((uri?, mtime?, size))
}

fn read_thumb_meta(png_path: &Path) -> Option<(String, u64, Option<u64>)> {
    use std::io::Read;
    let mut file = std::fs::File::open(png_path).ok()?;
    let mut header = vec![0u8; META_READ_LIMIT];
    let read = file.read(&mut header).ok()?;
    header.truncate(read);
    parse_thumb_meta(&header)
}

// --- writing ---------------------------------------------------------------
// Written aside and renamed into place, so nothing ever reads a
// half-written picture.
pub fn write_file_atomic(target: &Path, data: &[u8]) -> bool {
    let tmp = target.with_extension(format!("{}.tmp", std::process::id()));
    if let Some(parent) = target.parent() {
        if std::fs::create_dir_all(parent).is_err() {
            return false;
        }
    }
    if std::fs::write(&tmp, data).is_err() {
        return false;
    }
    if std::fs::rename(&tmp, target).is_err() {
        let _ = std::fs::remove_file(&tmp);
        return false;
    }
    true
}

// True when the system cache already holds a thumbnail rendered from
// exactly the file that is on disk right now.
pub fn has_fresh_thumbnail(file_path: &Path) -> bool {
    let identity = match model_identity(file_path) {
        Some(i) => i,
        None => return false,
    };
    let thumb = match thumbnail_path(file_path).and_then(|p| read_thumb_meta(&p).map(|m| (p, m))) {
        Some(v) => v,
        None => return false,
    };
    let (_path, (uri, mtime, size)) = thumb;
    uri == identity.uri
        && mtime == identity.mtime
        && size.map(|s| s == identity.size).unwrap_or(true)
}

// Stores a rendered PNG in the system cache, stamped with the identity of
// the model. Without those chunks other applications would treat it as
// stale, so an unstampable picture is not stored at all.
pub fn store_thumbnail(file_path: &Path, png: &[u8]) -> bool {
    let identity = match model_identity(file_path) {
        Some(i) => i,
        None => return false,
    };
    if !is_png(png) {
        return false;
    }
    match thumbnail_path(file_path) {
        Some(path) => write_file_atomic(&path, &embed_thumb_meta(png, &identity)),
        None => false,
    }
}

// Earlier versions kept a private cache of their own, and WebView2 profiles
// for the WebView thumbnail renderer that 1.0 replaced (`profiles`).
// Nothing reads them any more, so they are removed once rather than left
// behind.
pub fn remove_legacy_cache(profiles: Option<PathBuf>) {
    if let Some(p) = profiles.filter(|p| p.exists()) {
        let _ = std::fs::remove_dir_all(p);
    }
    let dir = if cfg!(target_os = "windows") {
        std::env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .or_else(dirs_home)
            .map(|b| b.join("3Dv").join("thumbnails"))
    } else if cfg!(target_os = "macos") {
        dirs_home().map(|h| h.join("Library").join("Caches").join("3Dv").join("thumbnails"))
    } else {
        dirs_home().map(|h| h.join(".3dv").join("thumbnails"))
    };
    let dir = match dir {
        Some(d) => d,
        None => return,
    };
    if !dir.exists() {
        return;
    }
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir(dir.parent().unwrap_or(Path::new(".")));
}
