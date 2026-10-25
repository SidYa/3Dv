// Windows Explorer thumbnail handler for 3D model files (3DvThumb.dll).
//
// Explorer loads this DLL into its own process (DisableProcessIsolation:
// the handler needs the file's path for glTF buffers, OBJ materials and
// textures next to the model, which an isolated handler would not get).
// So nothing heavy happens here. A thumbnail is asked of the thumbnail
// server in 3Dv.exe (thumb_server.rs) over a named pipe; the server is
// started on first use and keeps its GPU device warm while Explorer works
// through a folder. Should the server be unreachable or its renderer fail,
// a one-off process renders the model instead:
//
//     3Dv.exe 3dv-thumbnail "<model>" "<temp.png>"
//
// whose PNG is decoded here. There is deliberately no cache of our own:
// Explorer keeps the result in its thumbnail cache like any other file
// type's.

#![cfg(windows)]
#![allow(non_snake_case)]

#[allow(dead_code)] // shared with the server, which uses the other half
mod protocol;
mod register;

use std::ffi::c_void;
use std::io::{Read, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicIsize, AtomicPtr, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use windows::Win32::Graphics::Gdi::HBITMAP;
use windows::Win32::System::Com::{IClassFactory, IClassFactory_Impl};
use windows::Win32::UI::Shell::PropertiesSystem::{IInitializeWithFile, IInitializeWithFile_Impl};
use windows::Win32::UI::Shell::{IThumbnailProvider, IThumbnailProvider_Impl, WTSAT_ARGB, WTS_ALPHATYPE};
use windows_core::{implement, IUnknown, Interface, BOOL, GUID, HRESULT, PCWSTR};
use windows_sys::Win32::Foundation as F;
use windows_sys::Win32::Graphics::Gdi as Gdi;
use windows_sys::Win32::Storage::FileSystem as Fs;
use windows_sys::Win32::System::JobObjects as Job;
use windows_sys::Win32::System::Threading as Th;
use windows_sys::Win32::System::IO as Io;
use windows_sys::Win32::System::Pipes as Pipes;

/// {3D76A1B0-9F5C-4A2D-8BF1-C24E5A6D7F8E}, unchanged from the C++ handler.
pub const CLSID_THUMBNAIL_PROVIDER: GUID = GUID::from_u128(0x3D76A1B0_9F5C_4A2D_8BF1_C24E5A6D7F8E);

const E_FAIL: HRESULT = HRESULT(0x8000_4005u32 as i32);
const E_POINTER: HRESULT = HRESULT(0x8000_4003u32 as i32);
const E_UNEXPECTED: HRESULT = HRESULT(0x8000_FFFFu32 as i32);
const E_INVALIDARG: HRESULT = HRESULT(0x8007_0057u32 as i32);
const CLASS_E_NOAGGREGATION: HRESULT = HRESULT(0x8004_0110u32 as i32);
const CLASS_E_CLASSNOTAVAILABLE: HRESULT = HRESULT(0x8004_0111u32 as i32);
const ERROR_ALREADY_INITIALIZED: HRESULT = HRESULT(0x8007_04DFu32 as i32);
const S_OK: HRESULT = HRESULT(0);
const S_FALSE: HRESULT = HRESULT(1);

/// A whole request to the server, connecting included.
const SERVER_TIMEOUT: Duration = Duration::from_secs(30);
/// How long a freshly started server may take to open its pipe.
const SERVER_START_TIMEOUT: Duration = Duration::from_secs(10);
/// A folder full of models makes Explorer ask for many thumbnails at once,
/// and every fallback render is a 3Dv process of its own. Two at a time keeps
/// the machine usable; the rest wait their turn.
const MAX_CONCURRENT_CLI: i32 = 2;
const CLI_SEMAPHORE: &str = "Local\\3Dv.ThumbnailRenders";
const CLI_QUEUE_TIMEOUT_MS: u32 = 60_000;
/// The CLI gives up on a model after 25 seconds by itself.
const CLI_TIMEOUT_MS: u32 = 30_000;
/// Exit code of the CLI when the model produced no usable picture;
/// rendering it again in software would not change that.
const EXIT_NO_PICTURE: u32 = 2;

static DLL_MODULE: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());
static DLL_REFS: AtomicIsize = AtomicIsize::new(0);

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(Some(0)).collect()
}

/// Path of this DLL.
fn module_path() -> Option<String> {
    let mut buf = vec![0u16; 32_768];
    let len = unsafe { windows_sys::Win32::System::LibraryLoader::GetModuleFileNameW(DLL_MODULE.load(Ordering::Relaxed), buf.as_mut_ptr(), buf.len() as u32) } as usize;
    if len == 0 || len >= buf.len() {
        return None;
    }
    Some(String::from_utf16_lossy(&buf[..len]))
}

/// The installer puts the DLL at <install>\thumbnail-dll\3DvThumb.dll, so
/// 3Dv.exe is two levels up; older installs kept it three levels up.
fn viewer_executable() -> Option<String> {
    let dll = std::path::PathBuf::from(module_path()?);
    dll.ancestors()
        .skip(2)
        .take(2)
        .map(|dir| dir.join("3Dv.exe"))
        .find(|p| p.is_file())
        .and_then(|p| p.to_str().map(str::to_owned))
}

// --- COM objects ----------------------------------------------------------

struct DllRef;

impl DllRef {
    fn new() -> DllRef {
        DLL_REFS.fetch_add(1, Ordering::SeqCst);
        DllRef
    }
}

impl Drop for DllRef {
    fn drop(&mut self) {
        DLL_REFS.fetch_sub(1, Ordering::SeqCst);
    }
}

#[implement(IThumbnailProvider, IInitializeWithFile)]
struct ThumbnailProvider {
    path: Mutex<Option<String>>,
    _ref: DllRef,
}

impl IInitializeWithFile_Impl for ThumbnailProvider_Impl {
    fn Initialize(&self, pszfilepath: &PCWSTR, _grfmode: u32) -> windows_core::Result<()> {
        let mut path = self.path.lock().unwrap_or_else(|e| e.into_inner());
        if path.is_some() {
            return Err(ERROR_ALREADY_INITIALIZED.into());
        }
        if pszfilepath.is_null() {
            return Err(E_INVALIDARG.into());
        }
        // Heap-backed: shell paths can exceed MAX_PATH.
        let s = unsafe { pszfilepath.to_string() }.map_err(|_| windows_core::Error::from(E_INVALIDARG))?;
        if s.is_empty() {
            return Err(E_INVALIDARG.into());
        }
        *path = Some(s);
        Ok(())
    }
}

impl IThumbnailProvider_Impl for ThumbnailProvider_Impl {
    fn GetThumbnail(&self, cx: u32, phbmp: *mut HBITMAP, pdwalpha: *mut WTS_ALPHATYPE) -> windows_core::Result<()> {
        if phbmp.is_null() || pdwalpha.is_null() {
            return Err(E_POINTER.into());
        }
        unsafe {
            *phbmp = HBITMAP(std::ptr::null_mut());
            *pdwalpha = WTSAT_ARGB;
        }
        let path = self.path.lock().unwrap_or_else(|e| e.into_inner()).clone().ok_or(windows_core::Error::from(E_UNEXPECTED))?;
        // A panic must never unwind into Explorer.
        let bitmap = catch_unwind(AssertUnwindSafe(|| thumbnail(&path, cx))).ok().flatten().ok_or(windows_core::Error::from(E_FAIL))?;
        unsafe { *phbmp = HBITMAP(bitmap) };
        Ok(())
    }
}

#[implement(IClassFactory)]
struct ClassFactory {
    _ref: DllRef,
}

impl IClassFactory_Impl for ClassFactory_Impl {
    fn CreateInstance(&self, punkouter: windows_core::Ref<IUnknown>, riid: *const GUID, ppvobject: *mut *mut c_void) -> windows_core::Result<()> {
        if ppvobject.is_null() || riid.is_null() {
            return Err(E_POINTER.into());
        }
        unsafe { *ppvobject = std::ptr::null_mut() };
        if punkouter.is_some() {
            return Err(CLASS_E_NOAGGREGATION.into());
        }
        let unknown: IUnknown = ThumbnailProvider { path: Mutex::new(None), _ref: DllRef::new() }.into();
        unsafe { unknown.query(riid, ppvobject).ok() }
    }

    fn LockServer(&self, flock: BOOL) -> windows_core::Result<()> {
        if flock.as_bool() {
            DLL_REFS.fetch_add(1, Ordering::SeqCst);
        } else {
            DLL_REFS.fetch_sub(1, Ordering::SeqCst);
        }
        Ok(())
    }
}

// --- DLL exports ----------------------------------------------------------

#[no_mangle]
extern "system" fn DllMain(module: *mut c_void, reason: u32, _reserved: *mut c_void) -> BOOL {
    if reason == windows_sys::Win32::System::SystemServices::DLL_PROCESS_ATTACH {
        DLL_MODULE.store(module, Ordering::Relaxed);
        unsafe { windows_sys::Win32::System::LibraryLoader::DisableThreadLibraryCalls(module) };
    }
    BOOL(1)
}

#[no_mangle]
unsafe extern "system" fn DllGetClassObject(rclsid: *const GUID, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    catch_unwind(|| {
        if rclsid.is_null() || riid.is_null() || ppv.is_null() {
            return E_POINTER;
        }
        unsafe { *ppv = std::ptr::null_mut() };
        if unsafe { *rclsid } != CLSID_THUMBNAIL_PROVIDER {
            return CLASS_E_CLASSNOTAVAILABLE;
        }
        let factory: IClassFactory = ClassFactory { _ref: DllRef::new() }.into();
        unsafe { factory.query(riid, ppv) }
    })
    .unwrap_or(E_FAIL)
}

#[no_mangle]
extern "system" fn DllCanUnloadNow() -> HRESULT {
    if DLL_REFS.load(Ordering::SeqCst) == 0 {
        S_OK
    } else {
        S_FALSE
    }
}

#[no_mangle]
extern "system" fn DllRegisterServer() -> HRESULT {
    catch_unwind(|| match module_path() {
        Some(path) => register::register(&path),
        None => E_FAIL,
    })
    .unwrap_or(E_FAIL)
}

#[no_mangle]
extern "system" fn DllUnregisterServer() -> HRESULT {
    catch_unwind(register::unregister).unwrap_or(E_FAIL)
}

// --- thumbnails -----------------------------------------------------------

/// Renders `path` at `cx` pixels into a top-down 32-bit DIB section.
fn thumbnail(path: &str, cx: u32) -> Option<*mut c_void> {
    let exe = viewer_executable()?;
    let size = cx.clamp(protocol::MIN_SIZE, protocol::MAX_SIZE);
    match ask_server(&exe, path, size) {
        Answer::Picture(w, h, bgra) => return dib_section(w, h, &bgra),
        Answer::NoPicture => return None,
        Answer::Unavailable => {}
    }
    let (w, h, rgba) = run_cli(&exe, path)?;
    let (w, h, rgba) = fit(w, h, rgba, cx.max(1));
    let bgra: Vec<u8> = rgba.chunks_exact(4).flat_map(|p| [p[2], p[1], p[0], p[3]]).collect();
    dib_section(w, h, &bgra)
}

fn dib_section(width: u32, height: u32, bgra: &[u8]) -> Option<*mut c_void> {
    if bgra.len() != (width * height * 4) as usize {
        return None;
    }
    let mut info: Gdi::BITMAPINFO = unsafe { std::mem::zeroed() };
    info.bmiHeader.biSize = std::mem::size_of::<Gdi::BITMAPINFOHEADER>() as u32;
    info.bmiHeader.biWidth = width as i32;
    info.bmiHeader.biHeight = -(height as i32); // top-down
    info.bmiHeader.biPlanes = 1;
    info.bmiHeader.biBitCount = 32;
    info.bmiHeader.biCompression = Gdi::BI_RGB;
    let mut bits: *mut c_void = std::ptr::null_mut();
    let bitmap = unsafe { Gdi::CreateDIBSection(std::ptr::null_mut(), &info, Gdi::DIB_RGB_COLORS, &mut bits, std::ptr::null_mut(), 0) };
    if bitmap.is_null() || bits.is_null() {
        return None;
    }
    unsafe { std::ptr::copy_nonoverlapping(bgra.as_ptr(), bits as *mut u8, bgra.len()) };
    Some(bitmap)
}

// --- the thumbnail server -------------------------------------------------

/// Keeps the server in a job that dies with Explorer: closing the last
/// handle (process exit) kills it, so no orphan outlives the shell.
static SERVER_JOB: Mutex<Option<usize>> = Mutex::new(None);
static SERVER_START: Mutex<()> = Mutex::new(());

fn session_id() -> u32 {
    let mut session = 0u32;
    unsafe { windows_sys::Win32::System::RemoteDesktop::ProcessIdToSessionId(Th::GetCurrentProcessId(), &mut session) };
    session
}

/// An overlapped pipe handle whose reads and writes give up at a deadline.
struct PipeClient {
    handle: F::HANDLE,
    event: F::HANDLE,
    deadline: Instant,
}

impl PipeClient {
    fn remaining_ms(&self) -> u32 {
        self.deadline.saturating_duration_since(Instant::now()).as_millis().min(u32::MAX as u128) as u32
    }

    fn io(&mut self, read: bool, buf: *mut u8, len: u32) -> std::io::Result<usize> {
        let mut ov: Io::OVERLAPPED = unsafe { std::mem::zeroed() };
        ov.hEvent = self.event;
        let mut done = 0u32;
        let ok = unsafe {
            if read {
                Fs::ReadFile(self.handle, buf, len, &mut done, &mut ov)
            } else {
                Fs::WriteFile(self.handle, buf, len, &mut done, &mut ov)
            }
        };
        if ok == 0 {
            let err = unsafe { F::GetLastError() };
            if err != F::ERROR_IO_PENDING {
                return Err(std::io::Error::from_raw_os_error(err as i32));
            }
            let ok = unsafe { Io::GetOverlappedResultEx(self.handle, &ov, &mut done, self.remaining_ms(), 0) };
            if ok == 0 {
                let err = unsafe { F::GetLastError() };
                unsafe {
                    Io::CancelIoEx(self.handle, &ov);
                    Io::GetOverlappedResult(self.handle, &ov, &mut done, 1);
                }
                return Err(std::io::Error::from_raw_os_error(err as i32));
            }
        }
        Ok(done as usize)
    }
}

impl Read for PipeClient {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let len = buf.len().min(1 << 20) as u32;
        self.io(true, buf.as_mut_ptr(), len)
    }
}

impl Write for PipeClient {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let len = buf.len().min(1 << 20) as u32;
        self.io(false, buf.as_ptr() as *mut u8, len)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Drop for PipeClient {
    fn drop(&mut self) {
        unsafe {
            F::CloseHandle(self.handle);
            F::CloseHandle(self.event);
        }
    }
}

fn open_pipe(name: &[u16], deadline: Instant) -> Option<PipeClient> {
    let handle = unsafe {
        Fs::CreateFileW(
            name.as_ptr(),
            F::GENERIC_READ | F::GENERIC_WRITE,
            0,
            std::ptr::null(),
            Fs::OPEN_EXISTING,
            Fs::FILE_FLAG_OVERLAPPED,
            std::ptr::null_mut(),
        )
    };
    if handle == F::INVALID_HANDLE_VALUE {
        return None;
    }
    let event = unsafe { Th::CreateEventW(std::ptr::null(), 1, 0, std::ptr::null()) };
    if event.is_null() {
        unsafe { F::CloseHandle(handle) };
        return None;
    }
    Some(PipeClient { handle, event, deadline })
}

/// Connects to the server, starting it when it is not running.
fn connect(exe: &str, deadline: Instant) -> Option<PipeClient> {
    let name = wide(&protocol::pipe_name(exe, session_id()));
    let mut started = false;
    loop {
        if let Some(pipe) = open_pipe(&name, deadline) {
            return Some(pipe);
        }
        let err = unsafe { F::GetLastError() };
        if Instant::now() >= deadline {
            return None;
        }
        if err == F::ERROR_PIPE_BUSY {
            // All workers busy: wait for one.
            let wait = deadline.saturating_duration_since(Instant::now()).as_millis().min(5_000) as u32;
            unsafe { Pipes::WaitNamedPipeW(name.as_ptr(), wait.max(1)) };
            continue;
        }
        if err != F::ERROR_FILE_NOT_FOUND {
            return None;
        }
        if !started {
            started = true;
            let _guard = SERVER_START.lock().unwrap_or_else(|e| e.into_inner());
            // Another thread may have started it meanwhile.
            if let Some(pipe) = open_pipe(&name, deadline) {
                return Some(pipe);
            }
            if !start_server(exe) {
                return None;
            }
        }
        let start_deadline = (Instant::now() + Duration::from_millis(50)).min(deadline);
        std::thread::sleep(start_deadline.saturating_duration_since(Instant::now()));
    }
}

fn start_server(exe: &str) -> bool {
    let mut command = wide(&format!("\"{exe}\" 3dv-thumbnail-server"));
    let mut job_slot = SERVER_JOB.lock().unwrap_or_else(|e| e.into_inner());
    let job = *job_slot.get_or_insert_with(|| kill_on_close_job() as usize) as F::HANDLE;
    let mut si: Th::STARTUPINFOW = unsafe { std::mem::zeroed() };
    si.cb = std::mem::size_of::<Th::STARTUPINFOW>() as u32;
    let mut pi: Th::PROCESS_INFORMATION = unsafe { std::mem::zeroed() };
    // Below normal priority: a thumbnail must never make the desktop
    // stutter.
    let ok = unsafe {
        Th::CreateProcessW(
            std::ptr::null(),
            command.as_mut_ptr(),
            std::ptr::null(),
            std::ptr::null(),
            0,
            Th::CREATE_NO_WINDOW | Th::CREATE_SUSPENDED | Th::BELOW_NORMAL_PRIORITY_CLASS,
            std::ptr::null(),
            std::ptr::null(),
            &si,
            &mut pi,
        )
    };
    if ok == 0 {
        return false;
    }
    unsafe {
        if !job.is_null() {
            Job::AssignProcessToJobObject(job, pi.hProcess);
        }
        Th::ResumeThread(pi.hThread);
        F::CloseHandle(pi.hThread);
        F::CloseHandle(pi.hProcess);
    }
    true
}

fn kill_on_close_job() -> F::HANDLE {
    let job = unsafe { Job::CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
    if !job.is_null() {
        let mut limits: Job::JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
        limits.BasicLimitInformation.LimitFlags = Job::JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        unsafe {
            Job::SetInformationJobObject(job, Job::JobObjectExtendedLimitInformation, &limits as *const _ as *const c_void, std::mem::size_of_val(&limits) as u32);
        }
    }
    job
}

enum Answer {
    /// width, height, BGRA
    Picture(u32, u32, Vec<u8>),
    /// The model gives no picture; the CLI would not do better.
    NoPicture,
    /// No server, a timeout, or its renderer failed: worth the CLI.
    Unavailable,
}

fn ask_server(exe: &str, path: &str, size: u32) -> Answer {
    let deadline = Instant::now() + SERVER_TIMEOUT;
    let Some(mut pipe) = connect(exe, (Instant::now() + SERVER_START_TIMEOUT).min(deadline)) else { return Answer::Unavailable };
    pipe.deadline = deadline;
    if protocol::write_request(&mut pipe, path, size).is_err() {
        return Answer::Unavailable;
    }
    match protocol::read_response(&mut pipe) {
        Ok((protocol::STATUS_OK, w, h, bgra)) => Answer::Picture(w, h, bgra),
        Ok((protocol::STATUS_NO_PICTURE, ..)) => Answer::NoPicture,
        _ => Answer::Unavailable,
    }
}

// --- the CLI fallback -----------------------------------------------------
// For when the server cannot be reached or its renderer failed: a fresh
// 3Dv process renders the one model, and if the GPU fails it there, once
// more on the software rasterizer.

/// Runs the CLI inside a job object that kills the whole process tree when
/// the handle closes - on timeout, and also if Explorer itself goes away
/// mid-render, so no orphaned 3Dv processes are ever left running.
/// Returns the CLI's exit code, or 1 when it could not run to completion.
fn run_cli_once(exe: &str, model: &str, output: &str, software: bool) -> u32 {
    let mut cmd = format!("\"{exe}\"");
    if software {
        cmd.push_str(" --3dv-software");
    }
    cmd.push_str(&format!(" 3dv-thumbnail \"{model}\" \"{output}\""));
    let mut command = wide(&cmd);
    let job = kill_on_close_job();
    let mut si: Th::STARTUPINFOW = unsafe { std::mem::zeroed() };
    si.cb = std::mem::size_of::<Th::STARTUPINFOW>() as u32;
    let mut pi: Th::PROCESS_INFORMATION = unsafe { std::mem::zeroed() };
    let ok = unsafe {
        Th::CreateProcessW(
            std::ptr::null(),
            command.as_mut_ptr(),
            std::ptr::null(),
            std::ptr::null(),
            0,
            Th::CREATE_NO_WINDOW | Th::CREATE_SUSPENDED | Th::BELOW_NORMAL_PRIORITY_CLASS,
            std::ptr::null(),
            std::ptr::null(),
            &si,
            &mut pi,
        )
    };
    if ok == 0 {
        if !job.is_null() {
            unsafe { F::CloseHandle(job) };
        }
        return 1;
    }
    let mut exit = 1u32;
    unsafe {
        if !job.is_null() {
            Job::AssignProcessToJobObject(job, pi.hProcess);
        }
        Th::ResumeThread(pi.hThread);
        let wait = Th::WaitForSingleObject(pi.hProcess, CLI_TIMEOUT_MS);
        if wait == F::WAIT_TIMEOUT {
            Th::TerminateProcess(pi.hProcess, 1);
        }
        Th::GetExitCodeProcess(pi.hProcess, &mut exit);
        F::CloseHandle(pi.hProcess);
        F::CloseHandle(pi.hThread);
        if !job.is_null() {
            F::CloseHandle(job);
        }
        if wait != F::WAIT_OBJECT_0 {
            exit = 1;
        }
    }
    exit
}

/// A unique, empty file in %TEMP% for the CLI to replace with the PNG.
fn temp_output() -> Option<String> {
    let mut dir = vec![0u16; 1024];
    let len = unsafe { Fs::GetTempPathW(dir.len() as u32, dir.as_mut_ptr()) } as usize;
    if len == 0 || len >= dir.len() {
        return None;
    }
    let mut file = vec![0u16; 1024];
    let prefix = wide("3dv");
    if unsafe { Fs::GetTempFileNameW(dir.as_ptr(), prefix.as_ptr(), 0, file.as_mut_ptr()) } == 0 {
        return None;
    }
    let end = file.iter().position(|&c| c == 0).unwrap_or(file.len());
    Some(String::from_utf16_lossy(&file[..end]))
}

/// The CLI's PNG as RGBA, or None.
fn run_cli(exe: &str, model: &str) -> Option<(u32, u32, Vec<u8>)> {
    let name = wide(CLI_SEMAPHORE);
    let semaphore = unsafe { Th::CreateSemaphoreW(std::ptr::null(), MAX_CONCURRENT_CLI, MAX_CONCURRENT_CLI, name.as_ptr()) };
    if !semaphore.is_null() && unsafe { Th::WaitForSingleObject(semaphore, CLI_QUEUE_TIMEOUT_MS) } != F::WAIT_OBJECT_0 {
        unsafe { F::CloseHandle(semaphore) };
        return None;
    }
    let result = temp_output().and_then(|output| {
        // GPU first; a crashed or hung GPU run gets one more try on the
        // software renderer.
        let mut code = run_cli_once(exe, model, &output, false);
        if code != 0 && code != EXIT_NO_PICTURE {
            code = run_cli_once(exe, model, &output, true);
        }
        let decoded = if code == 0 { std::fs::read(&output).ok().and_then(|png| decode_png(&png)) } else { None };
        let _ = std::fs::remove_file(&output);
        decoded
    });
    if !semaphore.is_null() {
        unsafe {
            Th::ReleaseSemaphore(semaphore, 1, std::ptr::null_mut());
            F::CloseHandle(semaphore);
        }
    }
    result
}

fn decode_png(bytes: &[u8]) -> Option<(u32, u32, Vec<u8>)> {
    let mut decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()?];
    let info = reader.next_frame(&mut buf).ok()?;
    let data = &buf[..info.buffer_size()];
    let rgba = match info.color_type {
        png::ColorType::Rgba => data.to_vec(),
        png::ColorType::Rgb => data.chunks_exact(3).flat_map(|c| [c[0], c[1], c[2], 255]).collect(),
        png::ColorType::GrayscaleAlpha => data.chunks_exact(2).flat_map(|c| [c[0], c[0], c[0], c[1]]).collect(),
        png::ColorType::Grayscale => data.iter().flat_map(|&g| [g, g, g, 255]).collect(),
        png::ColorType::Indexed => return None,
    };
    Some((info.width, info.height, rgba))
}

/// Scales to fit `cx` on the longer side, keeping the aspect: an area
/// average when shrinking, bilinear when enlarging (what WIC's Fant mode
/// amounts to).
fn fit(w: u32, h: u32, rgba: Vec<u8>, cx: u32) -> (u32, u32, Vec<u8>) {
    let scale = cx as f64 / w.max(h) as f64;
    let (nw, nh) = (((w as f64 * scale) as u32).max(1), ((h as f64 * scale) as u32).max(1));
    if (nw, nh) == (w, h) {
        return (w, h, rgba);
    }
    let mut out = vec![0u8; (nw * nh * 4) as usize];
    for y in 0..nh {
        for x in 0..nw {
            let mut acc = [0f64; 4];
            if scale < 1.0 {
                let (x0, x1) = (x as f64 / scale, (x + 1) as f64 / scale);
                let (y0, y1) = (y as f64 / scale, (y + 1) as f64 / scale);
                let mut weight = 0.0;
                let mut sy = y0.floor();
                while sy < y1 {
                    let wy = (y1.min(sy + 1.0) - y0.max(sy)).max(0.0);
                    let mut sx = x0.floor();
                    while sx < x1 {
                        let wx = (x1.min(sx + 1.0) - x0.max(sx)).max(0.0);
                        let i = (((sy as u32).min(h - 1) * w + (sx as u32).min(w - 1)) * 4) as usize;
                        for c in 0..4 {
                            acc[c] += rgba[i + c] as f64 * wx * wy;
                        }
                        weight += wx * wy;
                        sx += 1.0;
                    }
                    sy += 1.0;
                }
                for a in acc.iter_mut() {
                    *a /= weight.max(1e-9);
                }
            } else {
                let fx = ((x as f64 + 0.5) / scale - 0.5).clamp(0.0, (w - 1) as f64);
                let fy = ((y as f64 + 0.5) / scale - 0.5).clamp(0.0, (h - 1) as f64);
                let (x0, y0) = (fx.floor() as u32, fy.floor() as u32);
                let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
                let (tx, ty) = (fx - x0 as f64, fy - y0 as f64);
                for c in 0..4 {
                    let p = |xx: u32, yy: u32| rgba[((yy * w + xx) * 4) as usize + c] as f64;
                    acc[c] = (p(x0, y0) * (1.0 - tx) + p(x1, y0) * tx) * (1.0 - ty) + (p(x0, y1) * (1.0 - tx) + p(x1, y1) * tx) * ty;
                }
            }
            let o = ((y * nw + x) * 4) as usize;
            for c in 0..4 {
                out[o + c] = acc[c].round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    (nw, nh, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_keeps_size_and_aspect() {
        let img = vec![200u8; 256 * 128 * 4];
        assert_eq!(fit(256, 128, img.clone(), 256).0, 256);
        let (w, h, out) = fit(256, 128, img, 64);
        assert_eq!((w, h), (64, 32));
        assert!(out.iter().all(|&v| v == 200));
        let (w, h, _) = fit(10, 10, vec![0u8; 400], 40);
        assert_eq!((w, h), (40, 40));
    }
}
