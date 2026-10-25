#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use tauri::{AppHandle, Emitter, Manager, State};
use tauri_plugin_dialog::{DialogExt, FilePath};

mod thumb;
mod assoc;
mod native;
#[cfg(windows)]
mod thumb_server;

// ---- constants (mirrors FILE_TYPES in old main.js) --------------------
const MODEL_EXTS: &[&str] = &["glb", "gltf", "obj", "fbx", "stl", "ply", "dae", "3ds"];

struct PendingFiles(Mutex<Vec<String>>);

fn push_pending_file(app: &AppHandle, file: String) {
    if let Some(state) = app.try_state::<PendingFiles>() {
        if let Ok(mut pending) = state.0.lock() {
            pending.push(file);
        }
    }
}

fn is_supported_model(p: &str) -> bool {
    Path::new(p)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| MODEL_EXTS.contains(&e.to_lowercase().as_str()))
        .unwrap_or(false)
}

fn find_model_arg(args: &[String]) -> Option<String> {
    args.iter()
        .skip(1)
        .find(|a| !a.starts_with('-') && is_supported_model(a))
        .cloned()
}

#[derive(Clone)]
struct ThumbnailRequest {
    input: String,
    output: String,
    software: bool,
    /// --3dv-webview: skip the native renderer (the Explorer provider asks
    /// for this after the thumbnail server already failed on the model).
    webview_only: bool,
}

fn get_thumbnail_request(args: &[String]) -> Option<ThumbnailRequest> {
    let i = args
        .iter()
        .position(|a| a == "3dv-thumbnail" || a == "--3dv-thumbnail")?;
    let input = args.get(i + 1)?.clone();
    let output = args.get(i + 2)?.clone();
    let software = args.iter().any(|a| a == "--3dv-software");
    let webview_only = args.iter().any(|a| a == "--3dv-webview");
    Some(ThumbnailRequest {
        input,
        output,
        software,
        webview_only,
    })
}

// ---- settings (app_data_dir/settings.json, atomic write) --------------
fn settings_path(app: &AppHandle) -> Option<PathBuf> {
    app.path().app_data_dir().ok().map(|d| d.join("settings.json"))
}

fn read_settings_file(path: &Path) -> serde_json::Value {
    let raw = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(_) => return serde_json::json!({}),
    };
    let raw = raw.strip_prefix('\u{feff}').unwrap_or(&raw);
    if raw.trim().is_empty() {
        return serde_json::json!({});
    }
    match serde_json::from_str(&raw) {
        Ok(serde_json::Value::Object(_)) => serde_json::from_str(raw).unwrap(),
        _ => serde_json::json!({}),
    }
}

fn write_settings_atomic(path: &Path, value: &serde_json::Value) -> bool {
    let tmp = path.with_extension(format!("{}.tmp", std::process::id()));
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let data = serde_json::to_string_pretty(value).unwrap_or_default();
    if std::fs::write(&tmp, data).is_err() {
        return false;
    }
    if std::fs::rename(&tmp, path).is_err() {
        let _ = std::fs::remove_file(&tmp);
        return false;
    }
    true
}

#[tauri::command]
fn get_settings(app: AppHandle) -> serde_json::Value {
    match settings_path(&app) {
        Some(p) => read_settings_file(&p),
        None => serde_json::json!({}),
    }
}

#[tauri::command]
fn set_settings(app: AppHandle, patch: serde_json::Value) -> bool {
    let path = match settings_path(&app) {
        Some(p) => p,
        None => return false,
    };
    let mut current = read_settings_file(&path);
    if let (Some(dst), Some(src)) = (current.as_object_mut(), patch.as_object()) {
        for (k, v) in src {
            dst.insert(k.clone(), v.clone());
        }
    }
    write_settings_atomic(&path, &current)
}

#[tauri::command]
fn get_initial_file(state: State<'_, PendingFiles>) -> Option<String> {
    let mut pending = state.0.lock().ok()?;
    if pending.is_empty() {
        None
    } else {
        Some(pending.remove(0))
    }
}

// Second instance / macOS open-file: reuse the main window in single-window
// mode, otherwise open the file in a fresh viewer window.
fn open_in_viewer(app: &AppHandle, file: Option<String>) {
    let single = settings_path(app)
        .map(|p| read_settings_file(&p))
        .and_then(|s| s.get("singleWindow").and_then(|v| v.as_bool()))
        .unwrap_or(true);

    if single {
        if let Some(win) = app.get_webview_window("main") {
            let _ = win.unminimize();
            let _ = win.set_focus();
            let _ = win.show();
            // emit() would broadcast to every viewer window; only main
            // should open the file.
            if let Some(f) = file {
                let _ = win.emit_to("main", "open-file", f);
            }
            return;
        }
    }
    if let Some(f) = file {
        push_pending_file(app, f);
    }
    create_viewer_window(app);
}

fn create_viewer_window(app: &AppHandle) {
    let mut n = 2u32;
    let label = if app.get_webview_window("main").is_none() {
        "main".to_string()
    } else {
        loop {
            let candidate = format!("main-{n}");
            if app.get_webview_window(&candidate).is_none() {
                break candidate;
            }
            n += 1;
        }
    };
    match tauri::WebviewWindowBuilder::new(app, &label, tauri::WebviewUrl::App("index.html".into()))
        .title("3Dv")
        .inner_size(900.0, 680.0)
        .min_inner_size(400.0, 300.0)
        // The viewer's own background, so the window never flashes white
        // before the page paints.
        .background_color(tauri::window::Color(0x1a, 0x1a, 0x1a, 0xff))
        .visible(false)
        .build()
    {
        Ok(win) => {
            // The page shows the window after its first frame (App.svelte,
            // revealWindow); this is the safety net if it is slower or fails.
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(300));
                let _ = win.show();
            });
        }
        Err(e) => eprintln!("[Viewer] window build failed: {e}"),
    }
}

#[tauri::command]
fn show_open_dialog(app: AppHandle, window: tauri::Window) -> Option<String> {
    let settings = match settings_path(&app) {
        Some(p) => read_settings_file(&p),
        None => serde_json::json!({}),
    };
    let start_dir = settings
        .get("lastDir")
        .and_then(|v| v.as_str())
        .filter(|d| Path::new(d).exists());

    let mut picker = window.dialog().file().add_filter("3D Models", MODEL_EXTS);
    if let Some(dir) = start_dir {
        picker = picker.set_directory(dir);
    }
    let picked: Option<FilePath> = picker.blocking_pick_file();
    let path = picked?.as_path()?.to_string_lossy().to_string();
    if let Some(parent) = Path::new(&path).parent() {
        let patch = serde_json::json!({ "lastDir": parent.to_string_lossy() });
        set_settings(app, patch);
    }
    Some(path)
}

#[tauri::command]
fn set_as_default(app: AppHandle) -> bool {
    assoc::set_as_default(&app)
}

// --- thumbnail service commands ----------------------------------------
// The hidden "thumbnailer" window (route #thumbnail in the frontend) calls
// these; the viewer calls generate_thumbnail on Linux to prewarm the
// shared freedesktop cache.

#[tauri::command]
fn thumbnail_ready(app: AppHandle, svc: State<'_, thumb::ThumbService>) {
    thumb::on_ready(&app, &svc);
}

#[tauri::command]
fn submit_thumbnail(
    app: AppHandle,
    svc: State<'_, thumb::ThumbService>,
    id: u32,
    png: Option<Vec<u8>>,
    reason: Option<String>,
) {
    if png.is_none() {
        eprintln!(
            "[Thumbnail] render #{id} failed ({})",
            reason.as_deref().unwrap_or("unknown")
        );
    }
    thumb::on_result(&app, &svc, id, png);
}

// Linux only: puts a freshly opened model's thumbnail into the shared
// freedesktop cache, where every file manager finds it. Windows renders
// thumbnails on demand through the Explorer provider (CLI below) and keeps
// them in its own cache, and macOS has nothing that would read one.
#[tauri::command]
async fn generate_thumbnail(
    app: AppHandle,
    svc: State<'_, thumb::ThumbService>,
    file_path: String,
) -> Result<bool, ()> {
    if !thumb::has_system_thumbnail_cache() || !is_supported_model(&file_path) {
        return Ok(false);
    }
    let path = Path::new(&file_path);
    if thumb::has_fresh_thumbnail(path) {
        return Ok(true);
    }
    let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(u64::MAX);
    if size > thumb::PREWARM_MAX_BYTES {
        return Ok(false);
    }
    // Native renderer first, off the async runtime; the hidden WebView
    // renderer only for what it cannot draw.
    let native_path = path.to_path_buf();
    let native = tauri::async_runtime::spawn_blocking(move || native::render_png(&native_path, native::THUMB_SIZE, false))
        .await
        .ok()
        .flatten();
    if let Some(png) = native {
        return Ok(thumb::store_thumbnail(path, &png));
    }
    match thumb::request_render(&app, &svc, file_path.clone()).await {
        Some(png) => Ok(thumb::store_thumbnail(path, &png)),
        None => Ok(false),
    }
}

// `3Dv.exe 3dv-thumbnail <model> <output.png> [--3dv-software]` — used by
// the Windows Explorer thumbnail provider (thumbnail-provider/). Explorer
// starts this once per thumbnail, so concurrent runs must NOT be blocked
// by the single-instance lock (viewer-only, see run_app).
async fn cli_render(app: &AppHandle, req: &ThumbnailRequest) -> i32 {
    let svc = app.state::<thumb::ThumbService>();
    if !Path::new(&req.input).is_file() {
        return 2;
    }
    match thumb::request_render(app, &svc, req.input.clone()).await {
        Some(png) if thumb::write_file_atomic(Path::new(&req.output), &png) => 0,
        Some(_) => 1,
        None => 2,
    }
}

fn run_thumbnail_cli(req: ThumbnailRequest) -> ! {
    // The native renderer answers almost every model in milliseconds and
    // needs no WebView at all; only what it cannot draw goes on below.
    if !req.webview_only {
        let input = Path::new(&req.input);
        if !input.is_file() {
            std::process::exit(2);
        }
        if let Some(png) = native::render_png(input, native::THUMB_SIZE, req.software) {
            let code = if thumb::write_file_atomic(Path::new(&req.output), &png) { 0 } else { 1 };
            std::process::exit(code);
        }
    }
    if req.software {
        // WebView2 software fallback; the Explorer provider retries with
        // --3dv-software if a GPU run crashes.
        std::env::set_var(
            "WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS",
            "--use-gl=angle --use-angle=swiftshader",
        );
    }
    // run_app diverges; the setup hook renders and exits the process.
    run_app(Some(req), None);
    unreachable!("tauri run() must diverge");
}

fn run_app(cli: Option<ThumbnailRequest>, initial: Option<String>) {
    let pending = Mutex::new(initial.into_iter().collect::<Vec<String>>());
    let builder = tauri::Builder::default()
        .manage(PendingFiles(pending))
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            get_settings,
            set_settings,
            get_initial_file,
            show_open_dialog,
            generate_thumbnail,
            thumbnail_ready,
            submit_thumbnail,
            set_as_default
        ]);

    // Single instance is a viewer concept; CLI renders must run in parallel.
    let builder = match cli.clone() {
        None => builder.plugin(tauri_plugin_single_instance::init(|app, argv, _cwd| {
            let file = find_model_arg(&argv).filter(|f| is_supported_model(f));
            open_in_viewer(app, file);
        })),
        Some(_) => builder,
    };

    builder
        .setup(move |app| {
            let handle = app.handle().clone();
            match cli.clone() {
                Some(req) => {
                    // Private WebView2 profile, see thumb::cli_profile_dir.
                    let profile = thumb::cli_profile_dir(&handle);
                    app.manage(thumb::ThumbService::with_profile(profile));
                    std::thread::spawn(move || {
                        let code = tauri::async_runtime::block_on(cli_render(&handle, &req));
                        std::process::exit(code);
                    });
                }
                None => {
                    app.manage(thumb::ThumbService::new());
                    // The main window is created here rather than from the
                    // config ("create": false), so CLI thumbnail runs do not
                    // boot a hidden viewer next to the thumbnailer.
                    create_viewer_window(&handle);
                    // Housekeeping, off the startup path.
                    std::thread::spawn(thumb::remove_legacy_cache);
                }
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("failed to run 3Dv");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    // Explorer's thumbnail server: no Tauri, no window, see thumb_server.rs.
    #[cfg(windows)]
    if args.iter().any(|a| a == "3dv-thumbnail-server") {
        let software = args.iter().any(|a| a == "--3dv-software");
        std::process::exit(thumb_server::run(software));
    }
    if let Some(req) = get_thumbnail_request(&args) {
        run_thumbnail_cli(req);
    }
    run_app(None, find_model_arg(&args));
}
