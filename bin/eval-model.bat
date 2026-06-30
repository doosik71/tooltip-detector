@echo off
setlocal

cd /d "%~dp0.."

uv run python -m ttd.eval ^
  --data-root  "data/dataset" ^
  --threshold  0.5 ^
  --nms-radius 20 ^
  --batch-size 16 ^
  --workers    4 ^
  %*
