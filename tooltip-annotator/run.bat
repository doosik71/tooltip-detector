@echo off
rem run.bat -- dispatch to a project Python script under scripts\.
rem
rem   run <script-name> [args...]   =^>  uv run python -m scripts.<script-name> [args...]
rem
rem The trailing ".py" is optional, so "run generate_dataset" and
rem "run generate_dataset.py" are equivalent. With no arguments, the available
rem scripts are listed. Invoked as a module (-m) rather than a file path so
rem scripts can import sibling packages at the project root (e.g. tooltip\).
rem
rem Examples:
rem   run generate_dataset --dataset erop
rem   run generate_segmentation --dataset erop --device cuda:0
rem   run pipeline
setlocal EnableExtensions
cd /d "%~dp0"

if "%~1"=="" (
    call :list
    exit /b 0
)

set "NAME=%~1"
if /I "%NAME:~-3%"==".py" set "NAME=%NAME:~0,-3%"
set "TARGET=scripts\%NAME%.py"

if not exist "%TARGET%" (
    echo run: unknown script '%NAME%' ^(no %TARGET%^) 1>&2
    echo. 1>&2
    call :list 1>&2
    exit /b 1
)

rem Pass every argument after the script name through unchanged.
set "REST="
for /f "tokens=1,* delims= " %%a in ("%*") do set "REST=%%b"

uv run python -m scripts.%NAME% %REST%
exit /b %ERRORLEVEL%

:list
echo Usage: run ^<script-name^> [args...]
echo        run ^<script-name^> --help
echo.
echo Available scripts:
for %%f in (scripts\*.py) do echo   %%~nf
exit /b 0
