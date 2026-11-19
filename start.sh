#!/usr/bin/env bash
# Start Neural on Mac/Linux:  ./start.sh
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "First run: setting things up, this takes a few minutes..."
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
fi
.venv/bin/python app.py
