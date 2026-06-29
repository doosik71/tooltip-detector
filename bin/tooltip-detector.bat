@echo off
setlocal

cd /d "%~dp0.."

uv run python ttd/tooltip-detector.py %*
