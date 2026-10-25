// Renders thumbnails of the given models into a directory, with timings:
//   cargo run --release -p dv3-render --example thumb -- <out dir> [--size N] [--software] <model>...
use std::path::{Path, PathBuf};
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let out = PathBuf::from(args.next().expect("output directory"));
    let mut size = 256;
    let mut software = false;
    let mut models = Vec::new();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--size" => size = args.next().and_then(|s| s.parse().ok()).expect("size"),
            "--software" => software = true,
            _ => models.push(PathBuf::from(a)),
        }
    }
    std::fs::create_dir_all(&out).expect("output directory");
    let t = Instant::now();
    let mut renderer = dv3_render::Renderer::new(software).expect("renderer");
    renderer.warm_up();
    println!("device: {} in {:?}", renderer.adapter, t.elapsed());
    for model in models {
        let t = Instant::now();
        let name = model.file_name().unwrap().to_string_lossy().to_string();
        match dv3_render::thumbnail(&mut renderer, &model, size) {
            Ok(thumb) => {
                let png = dv3_render::encode_png(&thumb.rgba, thumb.size, thumb.size).expect("png");
                std::fs::write(out.join(format!("{}.png", stem(&model))), png).expect("write");
                println!("{:>7.1} ms  (load {:>6.1}, render {:>6.1})  {name}", t.elapsed().as_secs_f64() * 1e3, thumb.load_time.as_secs_f64() * 1e3, thumb.render_time.as_secs_f64() * 1e3);
            }
            Err(e) => println!("{:>7.1} ms  FAILED {e}  {name}", t.elapsed().as_secs_f64() * 1e3),
        }
    }
}

fn stem(p: &Path) -> String {
    let parent = p.parent().and_then(|d| d.file_name()).map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    format!("{parent}_{}", p.file_name().unwrap().to_string_lossy()).replace(' ', "_")
}
