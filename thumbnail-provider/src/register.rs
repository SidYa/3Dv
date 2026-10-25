// DllRegisterServer / DllUnregisterServer: the same registry layout the C++
// handler wrote, so updating 3Dv replaces it in place.

use windows_core::{GUID, HRESULT};
use windows_sys::Win32::Foundation::ERROR_SUCCESS;
use windows_sys::Win32::System::Registry as Reg;
use windows_sys::Win32::UI::Shell::{SHChangeNotify, SHCNE_ASSOCCHANGED, SHCNF_IDLIST};

use crate::{wide, CLSID_THUMBNAIL_PROVIDER, E_FAIL, S_OK};

/// The shell's IThumbnailProvider handler interface id.
const THUMBNAIL_HANDLER: &str = "{e357fccd-a995-4576-b01f-234630154e96}";
const EXTENSIONS: &[&str] = &[".glb", ".gltf", ".obj", ".fbx", ".stl", ".ply", ".dae", ".3ds"];
const PROG_IDS: &[&str] = &[
    "3Dv.Viewer.glb",
    "3Dv.Viewer.gltf",
    "3Dv.Viewer.obj",
    "3Dv.Viewer.fbx",
    "3Dv.Viewer.stl",
    "3Dv.Viewer.ply",
    "3Dv.Viewer.dae",
    "3Dv.Viewer.3ds",
];

fn clsid_string(g: &GUID) -> String {
    format!(
        "{{{:08X}-{:04X}-{:04X}-{:02X}{:02X}-{:02X}{:02X}{:02X}{:02X}{:02X}{:02X}}}",
        g.data1, g.data2, g.data3, g.data4[0], g.data4[1], g.data4[2], g.data4[3], g.data4[4], g.data4[5], g.data4[6], g.data4[7]
    )
}

fn hresult(err: u32) -> HRESULT {
    if err == ERROR_SUCCESS {
        S_OK
    } else {
        HRESULT::from_win32(err)
    }
}

fn create(root: Reg::HKEY, sub: &str) -> Result<Reg::HKEY, HRESULT> {
    let sub = wide(sub);
    let mut key: Reg::HKEY = std::ptr::null_mut();
    let err = unsafe { Reg::RegCreateKeyExW(root, sub.as_ptr(), 0, std::ptr::null(), Reg::REG_OPTION_NON_VOLATILE, Reg::KEY_WRITE, std::ptr::null(), &mut key, std::ptr::null_mut()) };
    if err == ERROR_SUCCESS {
        Ok(key)
    } else {
        Err(hresult(err))
    }
}

fn write_str(root: Reg::HKEY, sub: &str, name: Option<&str>, value: &str) -> HRESULT {
    let key = match create(root, sub) {
        Ok(k) => k,
        Err(hr) => return hr,
    };
    let name = name.map(wide);
    let data = wide(value);
    let err = unsafe {
        Reg::RegSetValueExW(key, name.as_ref().map_or(std::ptr::null(), |n| n.as_ptr()), 0, Reg::REG_SZ, data.as_ptr() as *const u8, (data.len() * 2) as u32)
    };
    unsafe { Reg::RegCloseKey(key) };
    hresult(err)
}

fn write_dword(root: Reg::HKEY, sub: &str, name: &str, value: u32) -> HRESULT {
    let key = match create(root, sub) {
        Ok(k) => k,
        Err(hr) => return hr,
    };
    let name = wide(name);
    let err = unsafe { Reg::RegSetValueExW(key, name.as_ptr(), 0, Reg::REG_DWORD, &value as *const u32 as *const u8, 4) };
    unsafe { Reg::RegCloseKey(key) };
    hresult(err)
}

fn delete_both_roots(sub: &str) {
    let w = wide(sub);
    for root in [Reg::HKEY_CURRENT_USER, Reg::HKEY_LOCAL_MACHINE] {
        unsafe {
            Reg::RegDeleteTreeW(root, w.as_ptr());
            Reg::RegDeleteKeyW(root, w.as_ptr());
        }
    }
}

/// Removes a parent key once nothing at all is left in it - no subkeys and
/// no values, its default included - so unregistering does not leave empty
/// keys scattered over the registry.
fn delete_if_empty_both_roots(sub: &str) {
    let w = wide(sub);
    for root in [Reg::HKEY_CURRENT_USER, Reg::HKEY_LOCAL_MACHINE] {
        let mut key: Reg::HKEY = std::ptr::null_mut();
        if unsafe { Reg::RegOpenKeyExW(root, w.as_ptr(), 0, Reg::KEY_READ, &mut key) } != ERROR_SUCCESS {
            continue;
        }
        let (mut subkeys, mut values) = (0u32, 0u32);
        let status = unsafe {
            Reg::RegQueryInfoKeyW(
                key,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null(),
                &mut subkeys,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut values,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        unsafe { Reg::RegCloseKey(key) };
        if status == ERROR_SUCCESS && subkeys == 0 && values == 0 {
            unsafe { Reg::RegDeleteKeyW(root, w.as_ptr()) };
        }
    }
}

fn cleanup(clsid: &str) {
    delete_both_roots(&format!("Software\\Classes\\CLSID\\{clsid}"));
    for (ext, prog_id) in EXTENSIONS.iter().zip(PROG_IDS) {
        delete_both_roots(&format!("Software\\Classes\\SystemFileAssociations\\{ext}\\ShellEx\\{THUMBNAIL_HANDLER}"));
        delete_if_empty_both_roots(&format!("Software\\Classes\\SystemFileAssociations\\{ext}\\ShellEx"));
        delete_if_empty_both_roots(&format!("Software\\Classes\\SystemFileAssociations\\{ext}"));
        delete_both_roots(&format!("Software\\Classes\\{prog_id}\\ShellEx\\{THUMBNAIL_HANDLER}"));
        delete_if_empty_both_roots(&format!("Software\\Classes\\{prog_id}\\ShellEx"));
        delete_if_empty_both_roots(&format!("Software\\Classes\\{prog_id}"));
    }
    delete_both_roots(&format!("Software\\Classes\\3Dv.Viewer\\ShellEx\\{THUMBNAIL_HANDLER}"));
    delete_if_empty_both_roots("Software\\Classes\\3Dv.Viewer\\ShellEx");
    delete_if_empty_both_roots("Software\\Classes\\3Dv.Viewer");
}

pub fn register(module_path: &str) -> HRESULT {
    let clsid = clsid_string(&CLSID_THUMBNAIL_PROVIDER);
    // Clean up any previous registrations first.
    cleanup(&clsid);

    let mut hr = E_FAIL;
    // HKLM first so every account gets thumbnails; without administrator
    // rights those writes fail and HKCU is used instead.
    for root in [Reg::HKEY_LOCAL_MACHINE, Reg::HKEY_CURRENT_USER] {
        let mut first_error = S_OK;
        let mut check = |r: HRESULT| {
            if r.is_err() && first_error.is_ok() {
                first_error = r;
            }
        };
        let key = format!("Software\\Classes\\CLSID\\{clsid}");
        check(write_str(root, &key, None, "3Dv Thumbnail Provider"));
        // In Explorer's process: the handler needs the model's path (see
        // lib.rs); the drawing itself happens in the thumbnail server.
        check(write_dword(root, &key, "DisableProcessIsolation", 1));
        let server = format!("{key}\\InprocServer32");
        check(write_str(root, &server, None, module_path));
        check(write_str(root, &server, Some("ThreadingModel"), "Apartment"));
        for ext in EXTENSIONS {
            // Not under the 3Dv ProgIDs: writing there created them as empty
            // shells whenever they did not exist yet. "Set as Default
            // Viewer" adds the handler to them when it creates them for real.
            check(write_str(root, &format!("Software\\Classes\\SystemFileAssociations\\{ext}\\ShellEx\\{THUMBNAIL_HANDLER}"), None, &clsid));
        }
        hr = first_error;
        if hr.is_ok() {
            break;
        }
    }
    if hr.is_ok() {
        unsafe { SHChangeNotify(SHCNE_ASSOCCHANGED as i32, SHCNF_IDLIST, std::ptr::null(), std::ptr::null()) };
    }
    hr
}

pub fn unregister() -> HRESULT {
    cleanup(&clsid_string(&CLSID_THUMBNAIL_PROVIDER));
    unsafe { SHChangeNotify(SHCNE_ASSOCCHANGED as i32, SHCNF_IDLIST, std::ptr::null(), std::ptr::null()) };
    S_OK
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clsid_matches_the_cpp_handler() {
        assert_eq!(clsid_string(&CLSID_THUMBNAIL_PROVIDER), "{3D76A1B0-9F5C-4A2D-8BF1-C24E5A6D7F8E}");
    }
}
