@echo off
setlocal enableextensions
pushd "%~dp0"
echo === 3Dv Thumbnail Provider Build ===

:: Try vswhere first (comes with VS installer)
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" set "VSWHERE=%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe"

if exist "%VSWHERE%" (
    for /f "usebackq tokens=*" %%a in (`"%VSWHERE%" -latest -property installationPath`) do (
        if exist "%%a\VC\Auxiliary\Build\vcvars64.bat" (
            echo Found VS: %%a
            call "%%a\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
            goto :check
        )
    )
)

:: Fallback: try known paths
for %%d in (
    "C:\Program Files\Microsoft Visual Studio\2022\BuildTools"
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
    "C:\Program Files\Microsoft Visual Studio\2022\Community"
    "C:\Program Files\Microsoft Visual Studio\2022\Professional"
    "C:\Program Files\Microsoft Visual Studio\2022\Enterprise"
) do (
    if exist "%%~d\VC\Auxiliary\Build\vcvars64.bat" (
        echo Found VS: %%~d
        call "%%~d\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
        goto :check
    )
)

:: Last resort: search for cl.exe
for /f "tokens=*" %%a in ('dir /s /b "C:\Program Files*\Microsoft Visual Studio\2022\*\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe" 2^>nul') do (
    set "CLDIR=%%~dpa"
    goto :havecl
)

echo.
echo Cannot find Visual Studio or cl.exe.
echo Make sure "Desktop development with C++" workload is installed.
exit /b 1

:havecl
set "PATH=%CLDIR%;%PATH%"
echo Found cl.exe at: %CLDIR%

:check
where cl.exe >nul 2>&1
if %errorlevel% neq 0 (
    echo cl.exe not in PATH after vcvars. Trying direct path...
    goto :havecl_fallback
)
goto :build

:havecl_fallback
:: Already tried above, give up
echo Cannot set up MSVC environment. Try running from "x64 Native Tools Command Prompt".
exit /b 1

:build
set "BUILD_DIR=%~dp0build"
for %%i in ("%BUILD_DIR%") do set "BUILD_DIR=%%~fi"
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

cl.exe /nologo /EHsc /O2 /MT /LD ^
    /D_USRDLL /D_WINDLL /DUNICODE /D_UNICODE ^
    /I. /Fo"%BUILD_DIR%\\" ^
    ThumbnailProvider.cpp dllmain.cpp ^
    /link /DEF:3DvThumb.def /OUT:"%BUILD_DIR%\3DvThumb.dll" /IMPLIB:"%BUILD_DIR%\3DvThumb.lib" ^
    pathcch.lib shlwapi.lib windowscodecs.lib shell32.lib advapi32.lib ole32.lib gdi32.lib user32.lib

set "BUILD_ERROR=%errorlevel%"

if "%BUILD_ERROR%"=="0" (
    echo.
    echo ==========================================
    echo  Build OK: %BUILD_DIR%\3DvThumb.dll
    echo  npm build copies it to release\thumbnail-dll.
    echo ==========================================
) else (
    echo Build failed. Check errors above.
)
popd
endlocal & exit /b %BUILD_ERROR%
