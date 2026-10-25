// Where the first thumbnail's time goes: device, pipelines, first render.
use std::time::Instant;
fn main() {
    let path = std::env::args().nth(1).expect("model");
    let t = Instant::now();
    let mut r = dv3_render::Renderer::new(false).expect("renderer");
    let device = t.elapsed();
    let t = Instant::now();
    r.warm_up();
    let warm = t.elapsed();
    let t = Instant::now();
    let thumb = dv3_render::thumbnail(&mut r, std::path::Path::new(&path), 256).expect("thumb");
    let first = t.elapsed();
    let t = Instant::now();
    dv3_render::thumbnail(&mut r, std::path::Path::new(&path), 256).expect("thumb");
    let second = t.elapsed();
    println!("device {device:?}  warm-up {warm:?}  first {first:?} (load {:?})  second {second:?}  [{}]", thumb.load_time, r.adapter);
}
