#include <windows.h>
#include <objbase.h>
#include <shlobj.h>
#include <shlwapi.h>
#include <pathcch.h>
#include <new>
#include <stdio.h>

#pragma comment(lib, "pathcch.lib")
#pragma comment(lib, "advapi32.lib")
#pragma comment(lib, "shell32.lib")

extern HINSTANCE g_hInst;
extern long g_cDllRef;

static const CLSID CLSID_ThumbnailProvider3Dv =
    { 0x3D76A1B0, 0x9F5C, 0x4A2D, { 0x8B, 0xF1, 0xC2, 0x4E, 0x5A, 0x6D, 0x7F, 0x8E } };

HINSTANCE g_hInst = nullptr;
long g_cDllRef = 0;

// The shell's IThumbnailProvider handler interface id.
static const wchar_t* const kThumbnailHandlerGuid = L"{e357fccd-a995-4576-b01f-234630154e96}";

static const wchar_t* const kExtensions[] = {
    L".glb", L".gltf", L".obj", L".fbx", L".stl", L".ply", L".dae", L".3ds"
};
static const wchar_t* const kProgIds[] = {
    L"3Dv.Viewer.glb", L"3Dv.Viewer.gltf", L"3Dv.Viewer.obj", L"3Dv.Viewer.fbx",
    L"3Dv.Viewer.stl", L"3Dv.Viewer.ply", L"3Dv.Viewer.dae", L"3Dv.Viewer.3ds"
};
static const int kTypeCount = ARRAYSIZE(kExtensions);

STDAPI ThumbnailProvider3Dv_CreateInstance(REFIID riid, void** ppv);

class ClassFactory3Dv : public IClassFactory
{
public:
    ClassFactory3Dv() : m_cRef(1) { InterlockedIncrement(&g_cDllRef); }

    IFACEMETHODIMP QueryInterface(REFIID riid, void** ppv)
    {
        static const QITAB qit[] = {
            QITABENT(ClassFactory3Dv, IClassFactory),
            { 0 },
        };
        return QISearch(this, qit, riid, ppv);
    }

    IFACEMETHODIMP_(ULONG) AddRef() { return InterlockedIncrement(&m_cRef); }
    IFACEMETHODIMP_(ULONG) Release()
    {
        ULONG cRef = InterlockedDecrement(&m_cRef);
        if (cRef == 0) delete this;
        return cRef;
    }

    IFACEMETHODIMP CreateInstance(IUnknown* pUnkOuter, REFIID riid, void** ppv)
    {
        if (pUnkOuter) return CLASS_E_NOAGGREGATION;
        return ThumbnailProvider3Dv_CreateInstance(riid, ppv);
    }

    IFACEMETHODIMP LockServer(BOOL fLock)
    {
        if (fLock) InterlockedIncrement(&g_cDllRef);
        else InterlockedDecrement(&g_cDllRef);
        return S_OK;
    }

protected:
    ~ClassFactory3Dv() { InterlockedDecrement(&g_cDllRef); }
    long m_cRef;
};

STDAPI_(BOOL) DllMain(HINSTANCE hInst, DWORD dwReason, LPVOID)
{
    if (dwReason == DLL_PROCESS_ATTACH)
    {
        g_hInst = hInst;
        DisableThreadLibraryCalls(hInst);
    }
    return TRUE;
}

STDAPI DllGetClassObject(REFCLSID rclsid, REFIID riid, void** ppv)
{
    if (!IsEqualCLSID(rclsid, CLSID_ThumbnailProvider3Dv))
        return CLASS_E_CLASSNOTAVAILABLE;

    ClassFactory3Dv* pFactory = new (std::nothrow) ClassFactory3Dv();
    if (!pFactory) return E_OUTOFMEMORY;

    HRESULT hr = pFactory->QueryInterface(riid, ppv);
    pFactory->Release();
    return hr;
}

STDAPI DllCanUnloadNow()
{
    return g_cDllRef == 0 ? S_OK : S_FALSE;
}

static HRESULT RegWriteStr(HKEY root, const wchar_t* subKey, const wchar_t* valueName, const wchar_t* value)
{
    HKEY hKey = nullptr;
    HRESULT hr = HRESULT_FROM_WIN32(RegCreateKeyEx(root, subKey, 0, nullptr,
        REG_OPTION_NON_VOLATILE, KEY_WRITE, nullptr, &hKey, nullptr));
    if (FAILED(hr)) return hr;
    hr = HRESULT_FROM_WIN32(RegSetValueEx(hKey, valueName, 0, REG_SZ,
        (const BYTE*)value, ((DWORD)wcslen(value) + 1) * sizeof(wchar_t)));
    RegCloseKey(hKey);
    return hr;
}

static HRESULT RegWriteDword(HKEY root, const wchar_t* subKey, const wchar_t* valueName, DWORD value)
{
    HKEY hKey = nullptr;
    HRESULT hr = HRESULT_FROM_WIN32(RegCreateKeyEx(root, subKey, 0, nullptr,
        REG_OPTION_NON_VOLATILE, KEY_WRITE, nullptr, &hKey, nullptr));
    if (FAILED(hr)) return hr;
    hr = HRESULT_FROM_WIN32(RegSetValueEx(hKey, valueName, 0, REG_DWORD, (const BYTE*)&value, sizeof(DWORD)));
    RegCloseKey(hKey);
    return hr;
}

static void RegDeleteBothRoots(const wchar_t* subKey)
{
    RegDeleteTree(HKEY_CURRENT_USER, subKey);
    RegDeleteKeyW(HKEY_CURRENT_USER, subKey);
    RegDeleteTree(HKEY_LOCAL_MACHINE, subKey);
    RegDeleteKeyW(HKEY_LOCAL_MACHINE, subKey);
}

// Removes a parent key once nothing at all is left in it - no subkeys and
// no values, its default included - so unregistering does not leave empty
// keys scattered over the registry.
static void RegDeleteIfEmpty(HKEY root, const wchar_t* subKey)
{
    HKEY hKey = nullptr;
    if (RegOpenKeyExW(root, subKey, 0, KEY_READ, &hKey) != ERROR_SUCCESS) return;
    DWORD subKeys = 0, values = 0;
    LSTATUS status = RegQueryInfoKeyW(hKey, nullptr, nullptr, nullptr, &subKeys, nullptr, nullptr,
        &values, nullptr, nullptr, nullptr, nullptr);
    RegCloseKey(hKey);
    if (status == ERROR_SUCCESS && subKeys == 0 && values == 0) RegDeleteKeyW(root, subKey);
}

static void RegDeleteIfEmptyBothRoots(const wchar_t* subKey)
{
    RegDeleteIfEmpty(HKEY_CURRENT_USER, subKey);
    RegDeleteIfEmpty(HKEY_LOCAL_MACHINE, subKey);
}

static void RegCleanup(const wchar_t* clsidStr)
{
    wchar_t subkey[512];

    swprintf_s(subkey, L"Software\\Classes\\CLSID\\%s", clsidStr);
    RegDeleteBothRoots(subkey);

    for (int i = 0; i < kTypeCount; i++)
    {
        swprintf_s(subkey, L"Software\\Classes\\SystemFileAssociations\\%s\\ShellEx\\%s",
            kExtensions[i], kThumbnailHandlerGuid);
        RegDeleteBothRoots(subkey);
        swprintf_s(subkey, L"Software\\Classes\\SystemFileAssociations\\%s\\ShellEx", kExtensions[i]);
        RegDeleteIfEmptyBothRoots(subkey);
        swprintf_s(subkey, L"Software\\Classes\\SystemFileAssociations\\%s", kExtensions[i]);
        RegDeleteIfEmptyBothRoots(subkey);

        swprintf_s(subkey, L"Software\\Classes\\%s\\ShellEx\\%s",
            kProgIds[i], kThumbnailHandlerGuid);
        RegDeleteBothRoots(subkey);
        swprintf_s(subkey, L"Software\\Classes\\%s\\ShellEx", kProgIds[i]);
        RegDeleteIfEmptyBothRoots(subkey);
        swprintf_s(subkey, L"Software\\Classes\\%s", kProgIds[i]);
        RegDeleteIfEmptyBothRoots(subkey);
    }

    swprintf_s(subkey, L"Software\\Classes\\3Dv.Viewer\\ShellEx\\%s", kThumbnailHandlerGuid);
    RegDeleteBothRoots(subkey);
    RegDeleteIfEmptyBothRoots(L"Software\\Classes\\3Dv.Viewer\\ShellEx");
    RegDeleteIfEmptyBothRoots(L"Software\\Classes\\3Dv.Viewer");
}

STDAPI DllRegisterServer()
{
    wchar_t modulePath[MAX_PATH];
    if (GetModuleFileName(g_hInst, modulePath, MAX_PATH) == 0)
        return HRESULT_FROM_WIN32(GetLastError());

    wchar_t clsidStr[64];
    StringFromGUID2(CLSID_ThumbnailProvider3Dv, clsidStr, 64);

    // Clean up any previous registrations first
    RegCleanup(clsidStr);

    wchar_t subkey[512];
    HRESULT hr = E_FAIL;

    // HKLM first so every account gets thumbnails; without administrator
    // rights those writes fail and HKCU is used instead.
    const HKEY roots[] = { HKEY_LOCAL_MACHINE, HKEY_CURRENT_USER };

    for (int r = 0; r < ARRAYSIZE(roots); r++)
    {
        const HKEY root = roots[r];
        HRESULT firstError = S_OK;

        // Every write is checked on its own. OR-ing HRESULTs together, as
        // this used to, merges unrelated facility and code bits into a
        // value that means nothing.
        auto write = [&](const wchar_t* key, const wchar_t* name, const wchar_t* value) {
            const HRESULT result = RegWriteStr(root, key, name, value);
            if (FAILED(result) && SUCCEEDED(firstError)) firstError = result;
        };

        swprintf_s(subkey, L"Software\\Classes\\CLSID\\%s", clsidStr);
        write(subkey, nullptr, L"3Dv Thumbnail Provider");
        const HRESULT isolation = RegWriteDword(root, subkey, L"DisableProcessIsolation", 1);
        if (FAILED(isolation) && SUCCEEDED(firstError)) firstError = isolation;

        swprintf_s(subkey, L"Software\\Classes\\CLSID\\%s\\InprocServer32", clsidStr);
        write(subkey, nullptr, modulePath);
        write(subkey, L"ThreadingModel", L"Apartment");

        for (int i = 0; i < kTypeCount; i++)
        {
            swprintf_s(subkey, L"Software\\Classes\\SystemFileAssociations\\%s\\ShellEx\\%s",
                kExtensions[i], kThumbnailHandlerGuid);
            write(subkey, nullptr, clsidStr);
            // Not under the 3Dv ProgIDs: writing there created them as empty
            // shells whenever they did not exist yet. "Set as Default Viewer"
            // adds the handler to them when it creates them for real.
        }

        hr = firstError;
        if (SUCCEEDED(hr)) break;
    }

    if (SUCCEEDED(hr))
        SHChangeNotify(SHCNE_ASSOCCHANGED, SHCNF_IDLIST, nullptr, nullptr);

    return hr;
}

STDAPI DllUnregisterServer()
{
    wchar_t clsidStr[64];
    StringFromGUID2(CLSID_ThumbnailProvider3Dv, clsidStr, 64);
    RegCleanup(clsidStr);
    SHChangeNotify(SHCNE_ASSOCCHANGED, SHCNF_IDLIST, nullptr, nullptr);
    return S_OK;
}
