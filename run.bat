@echo off
rem Set up (first run only) and start Text LLM Studio. Extra options go to run.py, e.g. run.bat --port 8000
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
  py -3 run.py %*
) else (
  where python >nul 2>nul
  if errorlevel 1 (
    echo Python 3 was not found. Install it from https://www.python.org/downloads/ and tick "Add python.exe to PATH".
  ) else (
    python run.py %*
  )
)
pause
