// Writes what the viewer receives from load_model, for tests:
//   cargo run --release -p dv3-render --example viewer_blob -- <model> <out.bin>
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let t = std::time::Instant::now();
    match dv3_render::viewer::load(std::path::Path::new(&args[0])) {
        Ok(blob) => {
            std::fs::write(&args[1], &blob).expect("write");
            eprintln!("{:.1} ms, {} bytes", t.elapsed().as_secs_f64() * 1e3, blob.len());
        }
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}
