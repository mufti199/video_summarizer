@echo off
cd /d "%~dp0"

REM Launch the YouTube Audio Summarizer and open it in your browser.

if not exist ".venv\Scripts\python.exe" (
  echo First-time setup needed. Running setup.bat ...
  call setup.bat
  if not exist ".venv\Scripts\python.exe" exit /b 1
)

REM Open the browser a few seconds after the server starts.
start "" /min cmd /c "timeout /t 4 >nul & start "" http://localhost:5000"

echo Starting the summarizer. Keep this window open while you use the app.
echo Close this window (or press Ctrl+C) to stop the server.
echo.
".venv\Scripts\python.exe" app.py

pause