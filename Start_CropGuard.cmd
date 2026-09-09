@echo off
setlocal
set "ROOT=%~dp0"
set "PYTHON=%ROOT%.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
  echo CropGuard virtual environment was not found.
  echo Run the installation steps in README.md first.
  pause
  exit /b 1
)

echo Starting CropGuard API and dashboard...
start "CropGuard API - keep open" cmd /k "cd /d \"%ROOT%\" ^&^& \"%PYTHON%\" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000"
timeout /t 8 /nobreak >nul
start "CropGuard Dashboard - keep open" cmd /k "cd /d \"%ROOT%\" ^&^& \"%PYTHON%\" -m streamlit run frontend/app.py --server.address 127.0.0.1 --server.port 8501"
timeout /t 4 /nobreak >nul
start "" "http://127.0.0.1:8501"
echo CropGuard is opening in your browser.
exit /b 0
