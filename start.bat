@echo off
REM Double-click to start Neural on Windows.
cd /d "%~dp0"
if not exist .venv (
  echo First run: setting things up, this takes a few minutes...
  python -m venv .venv
  .venv\Scripts\python -m pip install --upgrade pip
  .venv\Scripts\pip install -r requirements.txt
)
.venv\Scripts\python app.py
pause
