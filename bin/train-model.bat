@echo off
setlocal

cd /d "%~dp0.."

uv run python -m ttd.train ^
  --data-root  "data/dataset" ^
  --epochs     30 ^
  --batch-size 16 ^
  --lr         1e-4 ^
  --workers    4 ^
  %*
