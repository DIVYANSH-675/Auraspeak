@echo off
setlocal enabledelayedexpansion

REM One-click startup for Windows: creates/updates conda env, installs deps, runs the app.

REM Ensure we're at repo root
pushd %~dp0

if not exist environment.yml (
  echo [ERROR] environment.yml not found in %~dp0
  exit /b 1
)

REM Check conda
where conda >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Conda not found in PATH. Please install Anaconda/Miniconda and relaunch.
  exit /b 1
)

echo [INFO] Creating/updating conda environment "auraspeak"...
conda env create --file environment.yml --name auraspeak --force
if errorlevel 1 (
  echo [ERROR] Failed to create/update conda environment.
  exit /b 1
)

call conda activate auraspeak
if errorlevel 1 (
  echo [ERROR] Failed to activate conda environment.
  exit /b 1
)

echo [INFO] Installing/updating Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
  echo [ERROR] Pip install failed.
  exit /b 1
)

echo [INFO] Starting server at http://127.0.0.1:8000 ...
pushd code
python server.py
popd

popd
endlocal
