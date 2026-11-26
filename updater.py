"""
Updates from GitHub, without ever needing the internet to run.

When the computer is online, Neural checks the GitHub repo every few hours.
If any file there is different from your copy, the app asks if you want to
update. If you say yes, it:

  1. downloads the new version,
  2. backs up every file it's about to change (saved/backups/),
  3. replaces only the files that changed,
  4. installs new Python packages if requirements.txt changed,
  5. restarts itself.

It compares the real file contents (the same fingerprints Git uses), not a
version number, so no change can be missed. Your chats, learned data, model
and downloaded training text are never touched. If you edited a file yourself
and GitHub didn't change it, your edit is kept.

Offline? The check just fails quietly and Neural works as normal.
"""

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
import zipfile

from config import BASE_DIR, SAVE_DIR

REPO = os.environ.get("NEURAL_UPDATE_REPO", "ninjasam111411/trial_AI")
BRANCH = os.environ.get("NEURAL_UPDATE_BRANCH", "main")
API = os.environ.get("NEURAL_UPDATE_API", "https://api.github.com")
IS_GIT = (BASE_DIR / ".git").exists()
# Off for Git checkouts (use `git pull`) and for hosted servers.
ENABLED = os.environ.get("NEURAL_UPDATES", "0" if IS_GIT else "1") == "1"
CHECK_EVERY = 6 * 3600

MANIFEST_FILE = SAVE_DIR / "update_manifest.json"
BACKUP_DIR = SAVE_DIR / "backups"
NEVER_TOUCH = ("saved/", "data/pretrain/", "data/chats/", ".venv/", ".git/", "__pycache__/")
GENERATED = {"data/training.txt"}  # rebuilt locally by build_dataset.py
LAUNCHER_FILES = ("start.bat", "start.sh")  # written last (they may be running)

BOOT_ID = uuid.uuid4().hex  # changes on every start, so the page can see a restart
_lock = threading.Lock()
state = {
    "enabled": ENABLED, "is_git": IS_GIT, "available": False, "checking": False,
    "applying": False, "stage": "", "error": None, "last_check": None,
    "online": None, "commit": None, "message": "", "date": "", "changed": [],
    "installed": None, "can_undo": False, "boot_id": BOOT_ID,
}
_pending = {}


# ---- Fingerprints --------------------------------------------------------
def blob_sha(data):
    """Git's fingerprint for a file's contents."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def local_sha(path):
    p = BASE_DIR / path
    if not p.is_file():
        return None
    data = p.read_bytes()
    sha = blob_sha(data)
    if b"\r\n" in data:  # Windows line endings: compare like GitHub stores it
        lf = blob_sha(data.replace(b"\r\n", b"\n"))
        return {sha, lf}
    return {sha}


def tracked(path):
    return not path.startswith(NEVER_TOUCH) and path not in GENERATED


def load_manifest():
    try:
        return json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save_manifest(m):
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_FILE.write_text(json.dumps(m, indent=1), encoding="utf-8")


# ---- GitHub --------------------------------------------------------------
def _get(url, raw=False, timeout=10):
    req = urllib.request.Request(url, headers={
        "User-Agent": "Neural-updater", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    return data if raw else json.loads(data)


def check():
    """Ask GitHub what changed. Safe to call any time; fails quietly offline."""
    if not ENABLED or state["applying"]:
        return state
    with _lock:
        state.update(checking=True, error=None)
        try:
            api = f"{API}/repos/{REPO}"
            commit = _get(f"{api}/commits/{BRANCH}")
            sha = commit["sha"]
            tree = _get(f"{api}/git/trees/{sha}?recursive=1")
            remote = {e["path"]: e["sha"] for e in tree["tree"]
                      if e["type"] == "blob" and tracked(e["path"])}

            manifest = load_manifest()
            if manifest is None:
                # First run: remember what this copy looks like right now.
                files = {}
                for path, rsha in remote.items():
                    shas = local_sha(path)
                    if shas:
                        files[path] = rsha if rsha in shas else next(iter(shas))
                manifest = {"commit": None, "files": files, "time": int(time.time())}
                save_manifest(manifest)
            base = manifest["files"]

            changed, deleted = [], []
            for path, rsha in remote.items():
                shas = local_sha(path) or set()
                if rsha in shas:
                    continue                      # already the same
                if base.get(path) == rsha:
                    continue                      # GitHub unchanged; keep your edit
                edited = bool(shas) and base.get(path) not in shas
                changed.append({"path": path, "new": not shas, "edited": edited})
            for path, bsha in base.items():
                if path not in remote and bsha in (local_sha(path) or set()):
                    deleted.append(path)

            msg = commit["commit"]["message"].split("\n")[0]
            _pending.update(sha=sha, changed=changed, deleted=deleted)
            state.update(online=True, commit=sha[:7], message=msg,
                         date=commit["commit"]["author"]["date"],
                         changed=[c["path"] for c in changed] + [f"{d} (removed)" for d in deleted],
                         available=bool(changed or deleted))
        except Exception as e:  # noqa: BLE001 - offline, rate limited, etc.
            state.update(online=False, error=_friendly(e))
        finally:
            state.update(checking=False, last_check=int(time.time()))
            _refresh_local()
    return state


def _friendly(e):
    text = str(e)
    if "403" in text or "rate limit" in text.lower():
        return "GitHub is busy right now; will try again later."
    if "404" in text:
        return f"Couldn't find {REPO} ({BRANCH}) on GitHub."
    return "Offline. Neural works fine without the internet."


def _refresh_local():
    m = load_manifest() or {}
    state["installed"] = (m.get("commit") or "")[:7] or None
    state["can_undo"] = bool(_backups())


# ---- Applying ------------------------------------------------------------
def apply_async(restart_fn):
    if state["applying"] or not state["available"]:
        return False
    state.update(applying=True, stage="Downloading", error=None)
    threading.Thread(target=_apply, args=(restart_fn,), daemon=True).start()
    return True


def _apply(restart_fn):
    backup = None
    try:
        sha = _pending["sha"]
        data = _get(f"{API}/repos/{REPO}/zipball/{sha}", raw=True, timeout=120)
        zf = zipfile.ZipFile(io.BytesIO(data))
        files = {}
        for info in zf.infolist():
            if info.is_dir():
                continue
            path = info.filename.split("/", 1)[1]  # drop "owner-repo-sha/"
            if tracked(path):
                files[path] = zf.read(info)

        state["stage"] = "Backing up"
        backup = BACKUP_DIR / time.strftime("%Y%m%d-%H%M%S")
        backup.mkdir(parents=True, exist_ok=True)
        to_write = [c["path"] for c in _pending["changed"] if c["path"] in files]
        to_write.sort(key=lambda p: p in LAUNCHER_FILES)
        old_manifest = load_manifest()
        added = []
        for path in to_write + _pending["deleted"]:
            src = BASE_DIR / path
            if src.exists():
                (backup / path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, backup / path)
            else:
                added.append(path)
        (backup / "info.json").write_text(json.dumps({
            "from": (old_manifest or {}).get("commit"), "to": sha,
            "replaced": [p for p in to_write if p not in added],
            "added": added, "deleted": _pending["deleted"],
            "manifest": old_manifest}, indent=1), encoding="utf-8")

        state["stage"] = "Installing update"
        for path in to_write:
            dest = BASE_DIR / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".new")
            tmp.write_bytes(files[path])
            os.replace(tmp, dest)
            if path.endswith(".sh"):
                dest.chmod(0o755)
        for path in _pending["deleted"]:
            (BASE_DIR / path).unlink(missing_ok=True)

        if "requirements.txt" in to_write:
            state["stage"] = "Installing new packages"
            r = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                               cwd=BASE_DIR, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError("Installing packages failed: " + r.stderr[-300:])

        save_manifest({"commit": sha, "time": int(time.time()),
                       "files": {p: blob_sha(d) for p, d in files.items()}})
        state.update(stage="Restarting", available=False)
        threading.Timer(1.0, restart_fn).start()
    except Exception as e:  # noqa: BLE001
        if backup and (backup / "info.json").exists():
            _restore(backup)
        state.update(applying=False, stage="", error=f"Update failed, nothing changed. ({e})")


def _restore(backup):
    info = json.loads((backup / "info.json").read_text(encoding="utf-8"))
    for path in info["replaced"] + info["deleted"]:
        if (backup / path).exists():
            (BASE_DIR / path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup / path, BASE_DIR / path)
    for path in info["added"]:
        (BASE_DIR / path).unlink(missing_ok=True)
    if info.get("manifest"):
        save_manifest(info["manifest"])
    else:
        MANIFEST_FILE.unlink(missing_ok=True)
    return info


def _backups():
    if not BACKUP_DIR.exists():
        return []
    return sorted(b for b in BACKUP_DIR.iterdir()
                  if (b / "info.json").exists() and not b.name.endswith("-undone"))


def undo_last(restart_fn):
    """Put back the files from before the last update."""
    backups = _backups()
    if not backups:
        return False
    _restore(backups[-1])
    shutil.move(str(backups[-1]), str(backups[-1]) + "-undone")
    state.update(stage="Restarting", applying=True)
    threading.Timer(1.0, restart_fn).start()
    return True


def start_background_checks():
    _refresh_local()
    if not ENABLED:
        return

    def loop():
        time.sleep(3)
        while True:
            check()
            time.sleep(CHECK_EVERY)
    threading.Thread(target=loop, daemon=True).start()
