// Default-application registration: file associations plus the Explorer
// thumbnail hookup. Port of the old setDefaultWindows/Linux/Mac in main.js.

#[allow(unused_imports)]
use std::path::PathBuf;
use tauri::{AppHandle, Manager};

const FILE_TYPES: &[(&str, &str)] = &[
    ("glb", "model/gltf-binary"),
    ("gltf", "model/gltf+json"),
    ("obj", "model/obj"),
    ("fbx", "application/x-fbx"),
    ("stl", "model/stl"),
    ("ply", "application/x-ply"),
    ("dae", "model/vnd.collada+xml"),
    ("3ds", "application/x-3ds"),
];

const THUMBNAIL_CLSID: &str = "{3D76A1B0-9F5C-4A2D-8BF1-C24E5A6D7F8E}";
const THUMBNAIL_HANDLER_KEY: &str = "ShellEx\\{e357fccd-a995-4576-b01f-234630154e96}";

pub fn set_as_default(app: &AppHandle) -> bool {
    #[cfg(target_os = "windows")]
    return set_default_windows(app);
    #[cfg(target_os = "linux")]
    return set_default_linux();
    #[cfg(target_os = "macos")]
    return set_default_mac();
    #[cfg(not(any(
        target_os = "windows",
        target_os = "linux",
        target_os = "macos"
    )))]
    return false;
}

#[cfg(target_os = "windows")]
fn run_hidden(file: &str, args: &[&str]) -> bool {
    use std::os::windows::process::CommandExt;
    std::process::Command::new(file)
        .args(args)
        .creation_flags(0x08000000)
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[cfg(not(target_os = "windows"))]
fn run_hidden(file: &str, args: &[&str]) -> bool {
    std::process::Command::new(file)
        .args(args)
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

// One `reg import` instead of a reg.exe process per key - the old code
// started close to fifty processes.
#[cfg(target_os = "windows")]
fn set_default_windows(app: &AppHandle) -> bool {
    let exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(_) => return false,
    };
    let dll = app
        .path()
        .resource_dir()
        .map(|d| d.join("thumbnail-dll").join("3DvThumb.dll"))
        .unwrap_or_default();
    let has_dll = !dll.as_os_str().is_empty() && dll.is_file();

    let reg_text = build_assoc_reg(&exe.to_string_lossy(), has_dll);
    let reg_file =
        std::env::temp_dir().join(format!("3dv-assoc-{}.reg", std::process::id()));

    // reg.exe expects UTF-16 with a byte order mark.
    let mut bytes: Vec<u8> = vec![0xFF, 0xFE];
    for unit in reg_text.encode_utf16() {
        bytes.extend_from_slice(&unit.to_le_bytes());
    }
    if std::fs::write(&reg_file, &bytes).is_err() {
        return false;
    }
    let ok = run_hidden("reg", &["import", &reg_file.to_string_lossy()]);
    let _ = std::fs::remove_file(&reg_file);

    if has_dll {
        run_hidden("regsvr32", &["/s", &dll.to_string_lossy()]);
    }
    ok
}

fn reg_quote(value: &str) -> String {
    format!("\"{}\"", value.replace("\\", "\\\\").replace("\"", "\\\""))
}

fn build_assoc_reg(exe_path: &str, has_dll: bool) -> String {
    let mut lines = vec![
        "Windows Registry Editor Version 5.00".to_string(),
        String::new(),
    ];
    for (ext, _mime) in FILE_TYPES {
        let prog_id = format!("3Dv.Viewer.{ext}");
        let entries = [
            (
                format!("HKEY_CURRENT_USER\\Software\\Classes\\.{ext}"),
                prog_id.clone(),
            ),
            (
                format!("HKEY_CURRENT_USER\\Software\\Classes\\{prog_id}"),
                format!("3D Model ({ext})"),
            ),
            (
                format!("HKEY_CURRENT_USER\\Software\\Classes\\{prog_id}\\DefaultIcon"),
                format!("{exe_path},0"),
            ),
            (
                format!("HKEY_CURRENT_USER\\Software\\Classes\\{prog_id}\\shell\\open\\command"),
                format!("\"{exe_path}\" \"%1\""),
            ),
        ];
        for (name, value) in entries {
            lines.push(format!("[{name}]"));
            lines.push(format!("@={}", reg_quote(&value)));
            lines.push(String::new());
        }
        if has_dll {
            // An extension-level handler also overrides dead CLSIDs left
            // behind by uninstalled apps, which silently kill thumbnails.
            // The SystemFileAssociations entry is what the installer wrote;
            // the runtime keeps it too so a button click alone restores
            // thumbnails end to end.
            for scope in [
                format!("HKEY_CURRENT_USER\\Software\\Classes\\.{ext}"),
                format!("HKEY_CURRENT_USER\\Software\\Classes\\{prog_id}"),
                format!("HKEY_CURRENT_USER\\Software\\Classes\\SystemFileAssociations\\.{ext}"),
            ] {
                lines.push(format!("[{scope}\\{THUMBNAIL_HANDLER_KEY}]"));
                lines.push(format!("@={}", reg_quote(THUMBNAIL_CLSID)));
                lines.push(String::new());
            }
        }
    }
    lines.join("\r\n")
}

#[cfg(target_os = "linux")]
fn set_default_linux() -> bool {
    let home = match std::env::var_os("HOME").map(PathBuf::from) {
        Some(h) => h,
        None => return false,
    };
    // An AppImage runs from a mount point that changes on every start; the
    // image itself is the stable path.
    let exe = std::env::var_os("APPIMAGE")
        .map(PathBuf::from)
        .or_else(|| std::env::current_exe().ok())
        .unwrap_or_default();
    let mimes: Vec<&str> = FILE_TYPES.iter().map(|(_, m)| *m).collect();
    let entry = build_desktop_entry(&exe.to_string_lossy(), &mimes.join(";"));
    let desktop_dir = home.join(".local").join("share").join("applications");
    if std::fs::create_dir_all(&desktop_dir).is_err() {
        return false;
    }
    if std::fs::write(desktop_dir.join("3dv.desktop"), entry.as_bytes()).is_err() {
        return false;
    }
    run_hidden(
        "update-desktop-database",
        &[&desktop_dir.to_string_lossy()],
    );
    let mut args: Vec<&str> = vec!["default", "3dv.desktop"];
    args.extend(mimes);
    run_hidden("xdg-mime", &args)
}

// Icon by name, not by path: the .deb and the extracted AppImage both
// install the icon into the theme.
#[allow(dead_code)]
fn build_desktop_entry(exe_path: &str, mime_types: &str) -> String {
    format!(
        "[Desktop Entry]\nName=3Dv\nComment=Powerful minimalist 3D Viewer\nExec=\"{exe_path}\" %f\nIcon=3dv\nTerminal=false\nType=Application\nCategories=Graphics;Viewer;\nMimeType={mime_types};\n"
    )
}

// duti is the only scriptable way to change associations on macOS; when it
// is missing the renderer tells the user to do it from Finder.
#[cfg(target_os = "macos")]
fn set_default_mac() -> bool {
    FILE_TYPES
        .iter()
        .all(|(ext, _)| run_hidden("duti", &["-s", "com.3dv.viewer", &format!(".{ext}"), "all"]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reg_contains_all_extensions() {
        let reg = build_assoc_reg("C:\\Apps\\3Dv.exe", true);
        assert!(reg.starts_with("Windows Registry Editor Version 5.00"));
        for (ext, _) in FILE_TYPES {
            assert!(reg.contains(&format!(".{ext}]")), "missing .{ext}");
            assert!(reg.contains(&format!("3Dv.Viewer.{ext}")));
            assert!(reg.contains(&format!("3D Model ({ext})")));
        }
        assert!(reg.contains(THUMBNAIL_CLSID));
        assert!(reg.contains("SystemFileAssociations\\.glb"));
        // .reg escaping: quotes become \" and backslashes double.
        assert!(reg.contains("\"\\\"C:\\\\Apps\\\\3Dv.exe\\\" \\\"%1\\\"\""));
    }

    #[test]
    fn reg_without_dll_has_no_thumbnail_keys() {
        let reg = build_assoc_reg("C:\\Apps\\3Dv.exe", false);
        assert!(!reg.contains(THUMBNAIL_CLSID));
        assert!(reg.contains("3Dv.Viewer.glb"));
    }

    #[test]
    fn desktop_entry_format() {
        let entry = build_desktop_entry("/app/3Dv", "model/gltf-binary;model/obj");
        assert!(entry.contains("Exec=\"/app/3Dv\" %f"));
        assert!(entry.contains("MimeType=model/gltf-binary;model/obj;"));
        assert!(entry.contains("Icon=3dv"));
    }
}
