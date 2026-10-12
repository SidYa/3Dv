#pragma once
#include <windows.h>
#include <thumbcache.h>
#include <wincodec.h>
#include <shlwapi.h>
#include <new>
#include <string>

class ThumbnailProvider3Dv
    : public IInitializeWithFile
    , public IThumbnailProvider
{
public:
    ThumbnailProvider3Dv();

    IFACEMETHODIMP QueryInterface(REFIID riid, void** ppv)
    {
        static const QITAB qit[] = {
            QITABENT(ThumbnailProvider3Dv, IThumbnailProvider),
            QITABENT(ThumbnailProvider3Dv, IInitializeWithFile),
            { 0 },
        };
        return QISearch(this, qit, riid, ppv);
    }

    IFACEMETHODIMP_(ULONG) AddRef();
    IFACEMETHODIMP_(ULONG) Release();

    IFACEMETHODIMP Initialize(LPCWSTR pszFilePath, DWORD grfMode);
    IFACEMETHODIMP GetThumbnail(UINT cx, HBITMAP* phbmp, WTS_ALPHATYPE* pdwAlpha);

protected:
    ~ThumbnailProvider3Dv();

private:
    long m_cRef;
    // Heap-backed: shell paths can exceed MAX_PATH, and a fixed buffer here
    // used to be filled with wcscpy_s, whose overflow handler terminates the
    // hosting process.
    std::wstring m_filePath;

    HRESULT ConvertToHBITMAP(IWICBitmapSource* pSource, IWICImagingFactory* pFactory, UINT cx, HBITMAP* phbmp);
    HRESULT DecodePng(const std::wstring& pngPath, UINT cx, HBITMAP* phbmp);
};
