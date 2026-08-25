@echo off
setlocal

cd /d "%~dp0"

set "script=%~1"
if "%script%"=="" (
    echo Usage: run.bat ^<script^> [args...]
    echo.
    echo Available scripts:
    for %%f in (scripts\*.py) do echo   %%~nf
    exit /b 1
)
shift

set "args="
:loop
if "%~1"=="" goto :run
set "args=%args% %1"
shift
goto :loop

:run
uv run python scripts\%script%.py%args%
