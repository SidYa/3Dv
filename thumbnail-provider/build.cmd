@echo off
setlocal enableextensions
pushd "%~dp0"
echo === 3Dv Thumbnail Provider Build ===

:: The provider is a Rust crate of the core workspace (needs the MSVC
:: toolchain, as for 3Dv itself). The installer picks the DLL up from
:: build\3DvThumb.dll (see core\tauri.conf.json).
set "CARGO=cargo"
:: rustup's cargo directly: a malformed PATH entry can hide it from cmd.
if exist "%USERPROFILE%\.cargo\bin\cargo.exe" set "CARGO=%USERPROFILE%\.cargo\bin\cargo.exe"

"%CARGO%" build --release --manifest-path ..\core\Cargo.toml -p dv3-thumbnail-provider
if %errorlevel% neq 0 (
    echo Build failed. Check errors above.
    popd
    exit /b 1
)

if not exist build mkdir build
copy /y ..\core\target\release\3DvThumb.dll build\3DvThumb.dll >nul
if %errorlevel% neq 0 (
    echo Could not copy the DLL.
    popd
    exit /b 1
)

echo.
echo ==========================================
echo  Build OK: %~dp0build\3DvThumb.dll
echo ==========================================
popd
endlocal & exit /b 0
