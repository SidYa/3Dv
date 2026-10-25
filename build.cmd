@echo off
:: Builds the Windows installer: core\target\release\bundle\nsis\3Dv_<version>_x64-setup.exe
::   build.cmd
:: Needs Rust (MSVC toolchain) and Node.js.
setlocal enableextensions
pushd "%~dp0"
set "ROOT=%CD%"

:: rustup's tools first: a malformed PATH entry can hide them from cmd and
:: from the Tauri CLI.
if exist "%USERPROFILE%\.cargo\bin\cargo.exe" set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"

:: Release binaries carry no paths of this machine (user name, checkout
:: location) in their panic messages.
set "RUSTFLAGS=--remap-path-prefix=%USERPROFILE%\.cargo=cargo --remap-path-prefix=%USERPROFILE%\.rustup=rustup --remap-path-prefix=%ROOT%=3Dv"

echo === Thumbnail provider ===
call thumbnail-provider\build.cmd || goto :fail

echo === Frontend ===
pushd app
if not exist node_modules (
    call npm ci --no-audit --no-fund || (popd & goto :fail)
)
call npm run build || (popd & goto :fail)
popd

echo === Installer ===
pushd core
:: The frontend is already built; Tauri would run its build command from
:: the wrong directory.
call ..\app\node_modules\.bin\tauri.cmd build --bundles nsis -c "{\"build\":{\"beforeBuildCommand\":\"\"}}" || (popd & goto :fail)
popd

popd
endlocal & exit /b 0

:fail
echo Build failed.
popd
endlocal & exit /b 1
