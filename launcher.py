"""Starts Neural and starts it again after an update (start.bat / start.sh use this)."""

import os
import subprocess
import sys
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)
env = dict(os.environ, NEURAL_LAUNCHER="1")

while True:
    code = subprocess.call([sys.executable, "app.py"], env=env)
    if code != 3:          # 3 means "restart me" (after an update)
        sys.exit(code)
    print("\n  Neural updated - restarting...\n")
    env["NEURAL_RESTARTED"] = "1"  # don't open a second browser tab
