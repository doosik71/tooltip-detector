@echo off
setlocal

cd /d "%~dp0.."

uv run python ttd/dataset-browser.py %*
