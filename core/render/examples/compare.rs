// Side-by-side sheet of reference (three.js) and native thumbnails, with the
// mean absolute difference per pair:
//   cargo run --release -p dv3-render --example compare -- <ref dir> <new dir> <out.png>
// Files are paired by name: <ref dir>/<a>_<b>.<ext>.png ~ <new dir>/<b>_<c>.png
use std::path::{Path, PathBuf};

fn load(p: &Path) -> Option<(u32, u32, Vec<u8>)> {
    let decoder = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(p).ok()?));
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()?];
    let info = reader.next_frame(&mut buf).ok()?;
    let rgba = match info.color_type {
        png::ColorType::Rgba => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgb => buf[..info.buffer_size()].chunks_exact(3).flat_map(|c| [c[0], c[1], c[2], 255]).collect(),
        _ => return None,
    };
    Some((info.width, info.height, rgba))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (refs, news, out) = (PathBuf::from(&args[0]), PathBuf::from(&args[1]), PathBuf::from(&args[2]));
    let mut pairs = Vec::new();
    for e in std::fs::read_dir(&news).unwrap().flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        let model = name.trim_end_matches(".png");
        let found = std::fs::read_dir(&refs).unwrap().flatten().find(|r| r.file_name().to_string_lossy().ends_with(&format!("{model}.png")));
        if let Some(r) = found {
            pairs.push((model.to_string(), r.path(), e.path()));
        }
    }
    pairs.sort();
    let cell = 256u32;
    let cols = 4u32; // 2 pairs per row
    let rows = (pairs.len() as u32).div_ceil(2);
    let (w, h) = (cell * cols, cell * rows.max(1));
    let mut sheet = vec![0u8; (w * h * 4) as usize];
    for (i, (model, r, n)) in pairs.iter().enumerate() {
        let (Some(a), Some(b)) = (load(r), load(n)) else { continue };
        let mut diff = 0u64;
        for (x, y) in a.2.iter().zip(b.2.iter()) {
            diff += (*x as i32 - *y as i32).unsigned_abs() as u64;
        }
        println!("{:>6.2}  {model}", diff as f64 / a.2.len() as f64);
        for (k, img) in [a, b].iter().enumerate() {
            let ox = ((i as u32 % 2) * 2 + k as u32) * cell;
            let oy = (i as u32 / 2) * cell;
            for y in 0..cell.min(img.1) {
                for x in 0..cell.min(img.0) {
                    let s = ((y * img.0 + x) * 4) as usize;
                    let d = (((oy + y) * w + ox + x) * 4) as usize;
                    sheet[d..d + 4].copy_from_slice(&img.2[s..s + 4]);
                }
            }
        }
    }
    let png = dv3_render::encode_png(&sheet, w, h).unwrap();
    std::fs::write(out, png).unwrap();
}
