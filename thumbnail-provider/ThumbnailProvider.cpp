// Windows Explorer thumbnail handler for 3D model files.
//
// Explorer asks for a thumbnail; this starts
//
//     3Dv.exe 3dv-thumbnail "<model>" "<temp.png>"
//
// decodes the PNG it writes, deletes it again and hands the bitmap back.
// There is deliberately no cache of our own: Explorer keeps the result in
// its thumbnail cache (thumbcache_*.db) next to every photo and video
// thumbnail, invalidates it when the model changes and lets Disk Cleanup
// reclaim it - the same as for any other file type.

#include "ThumbnailProvider.h"
#include <pathcch.h>
#include <strsafe.h>
#include <new>
#include <string>
#include <vector>

#pragma comment(lib, "pathcch.lib")
#pragma comment(lib, "shlwapi.lib")
#pragma comment(lib, "windowscodecs.lib")
#pragma comment(lib, "ole32.lib")

extern HINSTANCE g_hInst;
extern long g_cDllRef;

// A folder full of models makes Explorer ask for many thumbnails at once,
// and every render is a full 3Dv process. Two at a time keeps the machine
// usable; the rest wait their turn.
static const LONG kMaxConcurrentRenders = 2;
static const wchar_t* const kRenderSemaphoreName = L"Local\\3Dv.ThumbnailRenders";
static const DWORD kQueueTimeoutMs = 60000;
// The CLI gives up on a model after 25 seconds by itself.
static const DWORD kRenderTimeoutMs = 30000;

ThumbnailProvider3Dv::ThumbnailProvider3Dv() : m_cRef(1)
{
    InterlockedIncrement(&g_cDllRef);
}

ThumbnailProvider3Dv::~ThumbnailProvider3Dv()
{
    InterlockedDecrement(&g_cDllRef);
}

IFACEMETHODIMP_(ULONG) ThumbnailProvider3Dv::AddRef()
{
    return InterlockedIncrement(&m_cRef);
}

IFACEMETHODIMP_(ULONG) ThumbnailProvider3Dv::Release()
{
    ULONG cRef = InterlockedDecrement(&m_cRef);
    if (cRef == 0) delete this;
    return cRef;
}

IFACEMETHODIMP ThumbnailProvider3Dv::Initialize(LPCWSTR pszFilePath, DWORD)
{
    if (!m_filePath.empty())
        return HRESULT_FROM_WIN32(ERROR_ALREADY_INITIALIZED);
    if (!pszFilePath || !*pszFilePath)
        return E_INVALIDARG;

    try
    {
        m_filePath.assign(pszFilePath);
    }
    catch (...)
    {
        return E_OUTOFMEMORY;
    }
    return S_OK;
}

// --- rendering --------------------------------------------------------

// Exit code of the CLI when the model itself produced no usable picture;
// rendering it again in software would not change that.
static const DWORD kExitNoPicture = 2;

// Runs the CLI inside a job object that kills the whole process tree when
// the handle closes - on timeout, and also if Explorer itself goes away
// mid-render, so no orphaned 3Dv processes are ever left running.
// Returns the CLI's exit code, or 1 when it could not run to completion.
static DWORD RunThumbnailCLI(const std::wstring& exePath, const std::wstring& modelPath,
    const std::wstring& outputPath, bool software)
{
    std::wstring commandLine;
    commandLine.reserve(exePath.size() + modelPath.size() + outputPath.size() + 48);
    commandLine += L'"'; commandLine += exePath; commandLine += L'"';
    if (software) commandLine += L" --3dv-software";
    commandLine += L" 3dv-thumbnail \"";
    commandLine += modelPath; commandLine += L"\" \"";
    commandLine += outputPath; commandLine += L'"';

    std::vector<wchar_t> mutableCommandLine(commandLine.begin(), commandLine.end());
    mutableCommandLine.push_back(L'\0');

    HANDLE job = CreateJobObjectW(nullptr, nullptr);
    if (job)
    {
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits = {};
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof(limits));
    }

    STARTUPINFOW si = { sizeof(si) };
    PROCESS_INFORMATION pi = { 0 };

    // Below normal priority: a thumbnail must never make the desktop stutter.
    if (!CreateProcessW(nullptr, mutableCommandLine.data(), nullptr, nullptr, FALSE,
            CREATE_NO_WINDOW | CREATE_SUSPENDED | BELOW_NORMAL_PRIORITY_CLASS,
            nullptr, nullptr, &si, &pi))
    {
        if (job) CloseHandle(job);
        return 1;
    }

    if (job) AssignProcessToJobObject(job, pi.hProcess);
    ResumeThread(pi.hThread);

    DWORD waitResult = WaitForSingleObject(pi.hProcess, kRenderTimeoutMs);
    if (waitResult == WAIT_TIMEOUT) TerminateProcess(pi.hProcess, 1);

    DWORD exitCode = 1;
    GetExitCodeProcess(pi.hProcess, &exitCode);

    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    if (job) CloseHandle(job);

    return waitResult == WAIT_OBJECT_0 ? exitCode : 1;
}

// The Tauri installer puts the DLL at <install>\thumbnail-dll\3DvThumb.dll,
// so 3Dv.exe is two levels up. Older Electron installs kept it under
// resources\thumbnail-dll\, three levels up; both are probed.
static bool FindViewerExecutable(std::wstring& exePath)
{
    wchar_t dir[MAX_PATH];
    if (GetModuleFileNameW(g_hInst, dir, MAX_PATH) == 0) return false;

    for (int level = 1; level <= 3; level++)
    {
        if (FAILED(PathCchRemoveFileSpec(dir, MAX_PATH))) return false;
        if (level < 2) continue;

        wchar_t candidate[MAX_PATH];
        if (FAILED(PathCchCombine(candidate, MAX_PATH, dir, L"3Dv.exe"))) return false;
        if (PathFileExistsW(candidate))
        {
            exePath.assign(candidate);
            return true;
        }
    }
    return false;
}

// A unique, empty file in %TEMP% for the CLI to replace with the PNG.
static bool CreateTempOutput(std::wstring& outputPath)
{
    wchar_t tempDir[MAX_PATH];
    DWORD length = GetTempPathW(MAX_PATH, tempDir);
    if (length == 0 || length >= MAX_PATH) return false;

    wchar_t tempFile[MAX_PATH];
    if (GetTempFileNameW(tempDir, L"3dv", 0, tempFile) == 0) return false;

    outputPath.assign(tempFile);
    return true;
}

HRESULT ThumbnailProvider3Dv::ConvertToHBITMAP(
    IWICBitmapSource* pSource, IWICImagingFactory* pFactory, UINT cx, HBITMAP* phbmp)
{
    *phbmp = nullptr;

    UINT srcW = 0, srcH = 0;
    HRESULT hr = pSource->GetSize(&srcW, &srcH);
    if (FAILED(hr)) return hr;
    if (srcW == 0 || srcH == 0) return E_FAIL;

    double scale = (double)cx / max(srcW, srcH);
    UINT destW = max(1u, (UINT)(srcW * scale));
    UINT destH = max(1u, (UINT)(srcH * scale));

    IWICBitmapScaler* pScaler = nullptr;
    hr = pFactory->CreateBitmapScaler(&pScaler);
    if (FAILED(hr)) return hr;

    hr = pScaler->Initialize(pSource, destW, destH, WICBitmapInterpolationModeFant);
    if (FAILED(hr)) { pScaler->Release(); return hr; }

    WICPixelFormatGUID pixelFormat;
    hr = pScaler->GetPixelFormat(&pixelFormat);

    IWICBitmapSource* pConverted = nullptr;
    if (SUCCEEDED(hr) && pixelFormat != GUID_WICPixelFormat32bppBGRA)
    {
        IWICFormatConverter* pConverter = nullptr;
        hr = pFactory->CreateFormatConverter(&pConverter);
        if (SUCCEEDED(hr))
        {
            hr = pConverter->Initialize(pScaler, GUID_WICPixelFormat32bppBGRA,
                WICBitmapDitherTypeNone, nullptr, 0, WICBitmapPaletteTypeCustom);
            if (SUCCEEDED(hr))
                hr = pConverter->QueryInterface(&pConverted);
            pConverter->Release();
        }
    }
    else if (SUCCEEDED(hr))
    {
        hr = pScaler->QueryInterface(&pConverted);
    }
    pScaler->Release();

    if (FAILED(hr) || !pConverted) return FAILED(hr) ? hr : E_FAIL;

    hr = pConverted->GetSize(&destW, &destH);
    if (SUCCEEDED(hr))
    {
        BITMAPINFO bmi = {};
        bmi.bmiHeader.biSize = sizeof(bmi.bmiHeader);
        bmi.bmiHeader.biWidth = destW;
        bmi.bmiHeader.biHeight = -(LONG)destH;
        bmi.bmiHeader.biPlanes = 1;
        bmi.bmiHeader.biBitCount = 32;
        bmi.bmiHeader.biCompression = BI_RGB;

        BYTE* pBits = nullptr;
        HBITMAP hbmp = CreateDIBSection(nullptr, &bmi, DIB_RGB_COLORS, (void**)&pBits, nullptr, 0);
        hr = hbmp ? S_OK : E_OUTOFMEMORY;
        if (SUCCEEDED(hr))
        {
            WICRect rect = { 0, 0, (INT)destW, (INT)destH };
            hr = pConverted->CopyPixels(&rect, destW * 4, destW * destH * 4, pBits);
            if (SUCCEEDED(hr)) *phbmp = hbmp;
            else DeleteObject(hbmp);
        }
    }
    pConverted->Release();
    return hr;
}

HRESULT ThumbnailProvider3Dv::DecodePng(const std::wstring& pngPath, UINT cx, HBITMAP* phbmp)
{
    IWICImagingFactory* pFactory = nullptr;
    HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr,
        CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&pFactory));
    if (FAILED(hr)) return hr;

    IWICBitmapDecoder* pDecoder = nullptr;
    hr = pFactory->CreateDecoderFromFilename(
        pngPath.c_str(), nullptr, GENERIC_READ, WICDecodeMetadataCacheOnDemand, &pDecoder);

    if (SUCCEEDED(hr))
    {
        IWICBitmapFrameDecode* pFrame = nullptr;
        hr = pDecoder->GetFrame(0, &pFrame);
        if (SUCCEEDED(hr))
        {
            hr = ConvertToHBITMAP(pFrame, pFactory, cx, phbmp);
            pFrame->Release();
        }
        pDecoder->Release();
    }
    pFactory->Release();
    return hr;
}

IFACEMETHODIMP ThumbnailProvider3Dv::GetThumbnail(UINT cx, HBITMAP* phbmp, WTS_ALPHATYPE* pdwAlpha)
{
    if (!phbmp || !pdwAlpha) return E_POINTER;
    *phbmp = nullptr;
    *pdwAlpha = WTSAT_ARGB;

    if (m_filePath.empty()) return E_UNEXPECTED;

    std::wstring exePath;
    if (!FindViewerExecutable(exePath)) return E_FAIL;

    HANDLE semaphore = CreateSemaphoreW(nullptr, kMaxConcurrentRenders, kMaxConcurrentRenders, kRenderSemaphoreName);
    if (semaphore && WaitForSingleObject(semaphore, kQueueTimeoutMs) != WAIT_OBJECT_0)
    {
        CloseHandle(semaphore);
        return E_FAIL;
    }

    HRESULT hr = E_FAIL;
    std::wstring outputPath;
    if (CreateTempOutput(outputPath))
    {
        // GPU first; a crashed or hung GPU run gets one more try on the
        // software renderer.
        DWORD exitCode = RunThumbnailCLI(exePath, m_filePath, outputPath, false);
        if (exitCode != 0 && exitCode != kExitNoPicture)
            exitCode = RunThumbnailCLI(exePath, m_filePath, outputPath, true);
        if (exitCode == 0)
            hr = DecodePng(outputPath, cx, phbmp);
        DeleteFileW(outputPath.c_str());
    }

    if (semaphore)
    {
        ReleaseSemaphore(semaphore, 1, nullptr);
        CloseHandle(semaphore);
    }
    return hr;
}

STDAPI ThumbnailProvider3Dv_CreateInstance(REFIID riid, void** ppv)
{
    ThumbnailProvider3Dv* p = new (std::nothrow) ThumbnailProvider3Dv();
    if (!p) return E_OUTOFMEMORY;
    HRESULT hr = p->QueryInterface(riid, ppv);
    p->Release();
    return hr;
}
