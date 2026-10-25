// `3Dv.exe 3dv-thumbnail-server`: the process the Explorer thumbnail
// provider (thumbnail-provider/, the 3DvThumb DLL) asks for thumbnails.
//
// Explorer runs the provider inside its own process, so the drawing happens
// here instead: a crash or a hung driver costs this process, never the
// desktop. The server stays up while requests keep coming and quits when
// idle, which keeps the GPU device and pipelines warm across a folder full
// of models - one thumbnail takes milliseconds instead of a WebView start.
//
// Requests arrive over a named pipe (see thumbnail-provider/src/protocol.rs).
// Several workers read and parse models in parallel; drawing is serialised
// on the one device. Whatever the native renderer cannot draw is answered
// with STATUS_FALLBACK and the provider runs the WebView CLI instead.

#[path = "../../thumbnail-provider/src/protocol.rs"]
#[allow(dead_code)]
mod protocol;

use std::io::{Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::native::Failure;
use windows_sys::Win32::Foundation::{CloseHandle, GetLastError, ERROR_PIPE_CONNECTED, HANDLE, INVALID_HANDLE_VALUE};
use windows_sys::Win32::Storage::FileSystem::{FlushFileBuffers, ReadFile, WriteFile, FILE_FLAG_FIRST_PIPE_INSTANCE, PIPE_ACCESS_DUPLEX};
use windows_sys::Win32::System::Pipes::{ConnectNamedPipe, CreateNamedPipeW, DisconnectNamedPipe, PIPE_READMODE_BYTE, PIPE_REJECT_REMOTE_CLIENTS, PIPE_TYPE_BYTE, PIPE_WAIT};
use windows_sys::Win32::System::RemoteDesktop::ProcessIdToSessionId;
use windows_sys::Win32::System::Threading::GetCurrentProcessId;

/// Parallel model loads. Drawing itself is one at a time.
const WORKERS: usize = 4;
/// Quit after this long without a request.
const IDLE_EXIT: Duration = Duration::from_secs(30);

struct Pipe(HANDLE);

// The handle is used by one worker thread at a time.
unsafe impl Send for Pipe {}

impl Read for Pipe {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let mut read = 0u32;
        let len = buf.len().min(u32::MAX as usize) as u32;
        let ok = unsafe { ReadFile(self.0, buf.as_mut_ptr(), len, &mut read, std::ptr::null_mut()) };
        if ok == 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(read as usize)
    }
}

impl Write for Pipe {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let mut written = 0u32;
        let len = buf.len().min(u32::MAX as usize) as u32;
        let ok = unsafe { WriteFile(self.0, buf.as_ptr(), len, &mut written, std::ptr::null_mut()) };
        if ok == 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(written as usize)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        unsafe { FlushFileBuffers(self.0) };
        Ok(())
    }
}

fn create_pipe(name: &[u16], first: bool) -> Option<Pipe> {
    let flags = PIPE_ACCESS_DUPLEX | if first { FILE_FLAG_FIRST_PIPE_INSTANCE } else { 0 };
    // Default security: full access for the user who started the server
    // (the Explorer session's user), SYSTEM and administrators only.
    let handle = unsafe {
        CreateNamedPipeW(
            name.as_ptr(),
            flags,
            PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS,
            WORKERS as u32,
            1 << 20,
            1 << 16,
            0,
            std::ptr::null(),
        )
    };
    if handle == INVALID_HANDLE_VALUE || handle.is_null() {
        None
    } else {
        Some(Pipe(handle))
    }
}

struct State {
    software: bool,
    active: AtomicUsize,
    /// Milliseconds since `started` of the last finished request.
    last_activity: AtomicU64,
    started: Instant,
}

impl State {
    fn touch(&self) {
        self.last_activity.store(self.started.elapsed().as_millis() as u64, Ordering::Relaxed);
    }
}

/// Runs the server; returns the process exit code.
pub fn run(software: bool) -> i32 {
    let Some(exe) = std::env::current_exe().ok().and_then(|p| p.to_str().map(str::to_owned)) else { return 1 };
    let mut session = 0u32;
    unsafe { ProcessIdToSessionId(GetCurrentProcessId(), &mut session) };
    let name: Vec<u16> = protocol::pipe_name(&exe, session).encode_utf16().chain(Some(0)).collect();

    // Only one server per session and installation: the first pipe instance
    // is exclusive, a second server simply leaves.
    let Some(first) = create_pipe(&name, true) else { return 0 };

    let state = Arc::new(State { software, active: AtomicUsize::new(0), last_activity: AtomicU64::new(0), started: Instant::now() });

    // Create the device and compile the pipelines while the first request
    // is still on its way.
    std::thread::spawn(move || crate::native::warm_up(software));

    let mut first = Some(first);
    for _ in 0..WORKERS {
        let pipe = match first.take() {
            Some(p) => p,
            None => match create_pipe(&name, false) {
                Some(p) => p,
                None => continue,
            },
        };
        let state = state.clone();
        std::thread::spawn(move || worker(pipe, &state));
    }

    loop {
        std::thread::sleep(Duration::from_secs(1));
        let idle_ms = state.started.elapsed().as_millis() as u64 - state.last_activity.load(Ordering::Relaxed);
        if state.active.load(Ordering::Relaxed) == 0 && idle_ms > IDLE_EXIT.as_millis() as u64 {
            return 0;
        }
    }
}

fn worker(mut pipe: Pipe, state: &State) {
    loop {
        let connected = unsafe { ConnectNamedPipe(pipe.0, std::ptr::null_mut()) } != 0 || unsafe { GetLastError() } == ERROR_PIPE_CONNECTED;
        if connected {
            state.active.fetch_add(1, Ordering::Relaxed);
            serve(&mut pipe, state);
            unsafe {
                FlushFileBuffers(pipe.0);
                DisconnectNamedPipe(pipe.0);
            }
            state.touch();
            state.active.fetch_sub(1, Ordering::Relaxed);
        } else {
            // The pipe is broken; this worker retires.
            unsafe { CloseHandle(pipe.0) };
            return;
        }
    }
}

fn serve(pipe: &mut Pipe, state: &State) {
    let Ok((path, size)) = protocol::read_request(pipe) else { return };
    match crate::native::render_rgba(Path::new(&path), size, state.software) {
        Ok(mut pixels) => {
            // RGBA → BGRA, what a DIB section wants.
            for px in pixels.chunks_exact_mut(4) {
                px.swap(0, 2);
            }
            let _ = protocol::write_response(pipe, protocol::STATUS_OK, size, size, &pixels);
        }
        Err(Failure::NoPicture) => {
            let _ = protocol::write_response(pipe, protocol::STATUS_NO_PICTURE, 0, 0, &[]);
        }
        Err(Failure::Renderer) => {
            let _ = protocol::write_response(pipe, protocol::STATUS_RENDERER_FAILED, 0, 0, &[]);
        }
    }
}
