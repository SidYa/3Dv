// Wire format between the Explorer thumbnail provider (this DLL) and the
// thumbnail server in 3Dv.exe (`3Dv.exe 3dv-thumbnail-server`, see
// core/src/thumb_server.rs, which includes this file). Std only.
//
// One request per pipe connection, all integers little-endian:
//   request   magic u32, size u32, path length u32, path UTF-8
//   response  status u32, width u32, height u32, then width*height BGRA8
//             pixels when status is OK (top row first, opaque)
//
// The pipe name carries the session (thumbnails belong to the user's
// desktop) and a hash of the 3Dv.exe path, so two installations - or an old
// server after an update - never answer each other's requests.

use std::io::{Read, Write};

pub const MAGIC: u32 = 0x3376_5444; // "DTv3"
pub const STATUS_OK: u32 = 0;
/// The native renderer could not draw the model; use the WebView CLI.
pub const STATUS_FALLBACK: u32 = 1;
/// Largest thumbnail the server renders; Explorer scales bigger requests.
pub const MAX_SIZE: u32 = 1024;
pub const MIN_SIZE: u32 = 16;
/// Paths longer than this are refused (Windows' own limit is 32767 UTF-16
/// units).
pub const MAX_PATH_BYTES: u32 = 4 * 32_768;

pub fn pipe_name(exe: &str, session: u32) -> String {
    // FNV-1a over the lower-cased path.
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in exe.to_lowercase().bytes() {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!(r"\\.\pipe\3Dv.Thumbnails.{hash:016x}.{session}")
}

pub fn write_request(w: &mut impl Write, path: &str, size: u32) -> std::io::Result<()> {
    let mut buf = Vec::with_capacity(12 + path.len());
    buf.extend_from_slice(&MAGIC.to_le_bytes());
    buf.extend_from_slice(&size.to_le_bytes());
    buf.extend_from_slice(&(path.len() as u32).to_le_bytes());
    buf.extend_from_slice(path.as_bytes());
    w.write_all(&buf)?;
    w.flush()
}

pub fn read_request(r: &mut impl Read) -> std::io::Result<(String, u32)> {
    let magic = read_u32(r)?;
    if magic != MAGIC {
        return Err(invalid("bad magic"));
    }
    let size = read_u32(r)?.clamp(MIN_SIZE, MAX_SIZE);
    let len = read_u32(r)?;
    if len == 0 || len > MAX_PATH_BYTES {
        return Err(invalid("bad path length"));
    }
    let mut path = vec![0u8; len as usize];
    r.read_exact(&mut path)?;
    let path = String::from_utf8(path).map_err(|_| invalid("path is not UTF-8"))?;
    Ok((path, size))
}

pub fn write_response(w: &mut impl Write, status: u32, width: u32, height: u32, bgra: &[u8]) -> std::io::Result<()> {
    let mut header = [0u8; 12];
    header[0..4].copy_from_slice(&status.to_le_bytes());
    header[4..8].copy_from_slice(&width.to_le_bytes());
    header[8..12].copy_from_slice(&height.to_le_bytes());
    w.write_all(&header)?;
    if status == STATUS_OK {
        w.write_all(bgra)?;
    }
    w.flush()
}

/// (status, width, height, pixels)
pub fn read_response(r: &mut impl Read) -> std::io::Result<(u32, u32, u32, Vec<u8>)> {
    let status = read_u32(r)?;
    let width = read_u32(r)?;
    let height = read_u32(r)?;
    if status != STATUS_OK {
        return Ok((status, 0, 0, Vec::new()));
    }
    if width == 0 || height == 0 || width > MAX_SIZE || height > MAX_SIZE {
        return Err(invalid("bad thumbnail size"));
    }
    let mut pixels = vec![0u8; (width * height * 4) as usize];
    r.read_exact(&mut pixels)?;
    Ok((status, width, height, pixels))
}

fn read_u32(r: &mut impl Read) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn invalid(msg: &str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, msg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let mut buf = Vec::new();
        write_request(&mut buf, "C:\\Models\\ключ.glb", 256).unwrap();
        let (path, size) = read_request(&mut buf.as_slice()).unwrap();
        assert_eq!((path.as_str(), size), ("C:\\Models\\ключ.glb", 256));

        let mut buf = Vec::new();
        write_response(&mut buf, STATUS_OK, 2, 1, &[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        let (status, w, h, px) = read_response(&mut buf.as_slice()).unwrap();
        assert_eq!((status, w, h, px), (STATUS_OK, 2, 1, vec![1, 2, 3, 4, 5, 6, 7, 8]));

        let mut buf = Vec::new();
        write_response(&mut buf, STATUS_FALLBACK, 0, 0, &[]).unwrap();
        assert_eq!(read_response(&mut buf.as_slice()).unwrap().0, STATUS_FALLBACK);
    }

    #[test]
    fn pipe_names_differ_per_install_and_session() {
        let a = pipe_name(r"C:\Program Files\3Dv\3Dv.exe", 1);
        assert_eq!(a, pipe_name(r"c:\program files\3dv\3dv.EXE", 1));
        assert_ne!(a, pipe_name(r"D:\3Dv\3Dv.exe", 1));
        assert_ne!(a, pipe_name(r"C:\Program Files\3Dv\3Dv.exe", 2));
    }

    #[test]
    fn rejects_garbage() {
        assert!(read_request(&mut [0u8; 12].as_slice()).is_err());
    }
}
