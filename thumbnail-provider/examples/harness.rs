// Drives the DLL the way Explorer does, without registering it:
//   cargo run -p dv3-thumbnail-provider --example harness -- <3DvThumb.dll> <out dir> <size> <model>...
// The DLL must sit in <install>\thumbnail-dll\ next to <install>\3Dv.exe.
use std::time::Instant;

use windows::core::{Interface, GUID, HRESULT, PCWSTR};
use windows::Win32::Graphics::Gdi::{DeleteObject, GetDIBits, GetObjectW, CreateCompatibleDC, DeleteDC, BITMAP, BITMAPINFO, BITMAPINFOHEADER, BI_RGB, DIB_RGB_COLORS, HBITMAP, HGDIOBJ};
use windows::Win32::System::Com::{CoInitializeEx, IClassFactory, COINIT_APARTMENTTHREADED};
use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
use windows::Win32::UI::Shell::PropertiesSystem::IInitializeWithFile;
use windows::Win32::UI::Shell::{IThumbnailProvider, WTS_ALPHATYPE};

const CLSID: GUID = GUID::from_u128(0x3D76A1B0_9F5C_4A2D_8BF1_C24E5A6D7F8E);
type GetClassObject = unsafe extern "system" fn(*const GUID, *const GUID, *mut *mut std::ffi::c_void) -> HRESULT;

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(Some(0)).collect()
}

fn main() -> windows::core::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (dll, out, size) = (&args[0], std::path::PathBuf::from(&args[1]), args[2].parse::<u32>().unwrap());
    std::fs::create_dir_all(&out).unwrap();
    unsafe {
        CoInitializeEx(None, COINIT_APARTMENTTHREADED).ok()?;
        let module = LoadLibraryW(PCWSTR(wide(dll).as_ptr()))?;
        let proc = GetProcAddress(module, windows::core::s!("DllGetClassObject")).expect("export");
        let get: GetClassObject = std::mem::transmute(proc);
        let mut factory: *mut std::ffi::c_void = std::ptr::null_mut();
        get(&CLSID, &IClassFactory::IID, &mut factory).ok()?;
        let factory = IClassFactory::from_raw(factory);
        for model in &args[3..] {
            let t = Instant::now();
            let init: IInitializeWithFile = factory.CreateInstance(None)?;
            init.Initialize(PCWSTR(wide(model).as_ptr()), 0)?;
            let provider: IThumbnailProvider = init.cast()?;
            let mut bitmap = HBITMAP::default();
            let mut alpha = WTS_ALPHATYPE::default();
            match provider.GetThumbnail(size, &mut bitmap, &mut alpha) {
                Ok(()) => {
                    let mut bm = BITMAP::default();
                    GetObjectW(HGDIOBJ(bitmap.0), std::mem::size_of::<BITMAP>() as i32, Some(&mut bm as *mut _ as *mut _));
                    let (w, h) = (bm.bmWidth as u32, bm.bmHeight.unsigned_abs());
                    let mut info = BITMAPINFO::default();
                    info.bmiHeader = BITMAPINFOHEADER { biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32, biWidth: w as i32, biHeight: -(h as i32), biPlanes: 1, biBitCount: 32, biCompression: BI_RGB.0, ..Default::default() };
                    let mut px = vec![0u8; (w * h * 4) as usize];
                    let dc = CreateCompatibleDC(None);
                    GetDIBits(dc, bitmap, 0, h, Some(px.as_mut_ptr() as *mut _), &mut info, DIB_RGB_COLORS);
                    let _ = DeleteDC(dc);
                    let _ = DeleteObject(HGDIOBJ(bitmap.0));
                    for p in px.chunks_exact_mut(4) {
                        p.swap(0, 2);
                    }
                    let name = std::path::Path::new(model).file_name().unwrap().to_string_lossy().replace(' ', "_");
                    let file = std::fs::File::create(out.join(format!("{name}.png"))).unwrap();
                    let mut enc = png::Encoder::new(file, w, h);
                    enc.set_color(png::ColorType::Rgba);
                    enc.write_header().unwrap().write_image_data(&px).unwrap();
                    println!("{:>8.1} ms  {w}x{h} alpha={}  {model}", t.elapsed().as_secs_f64() * 1e3, alpha.0);
                }
                Err(e) => println!("{:>8.1} ms  FAILED {e}  {model}", t.elapsed().as_secs_f64() * 1e3),
            }
        }
    }
    Ok(())
}
