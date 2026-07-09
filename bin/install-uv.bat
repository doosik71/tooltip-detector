@echo off
setlocal enabledelayedexpansion

where uv >nul 2>nul
if %ERRORLEVEL%==0 (
    for /f "delims=" %%v in ('uv --version') do set "UV_VERSION=%%v"
    echo uv already installed: !UV_VERSION!
    exit /b 0
)

echo uv not found. Installing...
powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
if errorlevel 1 (
    echo Installation failed >&2
    exit /b 1
)

set "UV_BIN=%USERPROFILE%\.local\bin\uv.exe"
if not exist "%UV_BIN%" (
    echo Installation failed: uv not found at %UV_BIN% >&2
    exit /b 1
)

for /f "delims=" %%v in ('"%UV_BIN%" --version') do set "UV_VERSION=%%v"
echo Installed: !UV_VERSION!
echo.
echo Add uv to your PATH, or restart your shell if %%USERPROFILE%%\.local\bin is already in PATH.
