"""
Neural web app.

    python app.py            # opens http://127.0.0.1:5000 in your browser

Environment variables (optional, mainly for hosting it online):
    PORT                 port to listen on (default 5000)
    HOST                 address to listen on (default 127.0.0.1)
    NEURAL_ADMIN_KEY     if set, teaching and retraining need this password
    NEURAL_DATA_DIR      where chats, learned data and the model are saved
"""

import os
import re
import subprocess
import sys
import threading
import webbrowser

import json
import uuid

from flask import Flask, Response, g, jsonify, render_template, request

import memory
import updater
from brain import Brain
from config import BASE_DIR, MODEL_FILE, TRAIN_INFO_FILE
from learning import Speller, _read_json

app = Flask(__name__)
brain = Brain()
ADMIN_KEY = os.environ.get("NEURAL_ADMIN_KEY", "")


COOKIE = "neural_uid"
UID_RE = re.compile(r"^[a-f0-9]{32}$")


@app.before_request
def identify_browser():
    """Give each browser its own id, remembered in a cookie for a year."""
    uid = request.cookies.get(COOKIE, "")
    g.new_uid = not UID_RE.match(uid)
    g.uid = uuid.uuid4().hex if g.new_uid else uid


@app.after_request
def save_cookie(resp):
    if getattr(g, "new_uid", False) or request.path == "/":
        resp.set_cookie(COOKIE, g.uid, max_age=365 * 24 * 3600, httponly=True,
                        samesite="Lax", secure=request.is_secure)
    return resp


def is_admin():
    return not ADMIN_KEY or request.headers.get("X-Admin-Key") == ADMIN_KEY


# ---- Pages ---------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# ---- Conversations -------------------------------------------------------
@app.get("/api/conversations")
def conversations():
    return jsonify(memory.list_conversations(g.uid))


@app.post("/api/conversations")
def new_conversation():
    return jsonify({"id": memory.create_conversation(g.uid)})


@app.get("/api/conversations/<cid>")
def get_conversation(cid):
    convo = memory.get_conversation(cid, g.uid)
    return (jsonify(convo), 200) if convo else (jsonify({"error": "Not found"}), 404)


@app.delete("/api/conversations/<cid>")
def delete_conversation(cid):
    memory.delete_conversation(cid, g.uid)
    return jsonify({"ok": True})


@app.patch("/api/conversations/<cid>")
def rename_conversation(cid):
    title = str((request.get_json(silent=True) or {}).get("title", "")).strip()
    memory.rename_conversation(cid, g.uid, title)
    return jsonify({"ok": True})


@app.get("/api/export")
def export_chats():
    """Download all of this browser's chats as a file."""
    body = json.dumps(memory.all_conversations(g.uid), indent=2, ensure_ascii=False)
    return Response(body, mimetype="application/json",
                    headers={"Content-Disposition": "attachment; filename=neural-chats.json"})


# ---- Chat ----------------------------------------------------------------
@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    cid = str(data.get("conversation_id") or "")
    if not cid or not memory.get_conversation(cid, g.uid):
        cid = memory.create_conversation(g.uid)

    if not message:
        return jsonify({"error": "Please enter a message."}), 400
    if len(message) > 500:
        return jsonify({"error": "Message is too long (500 characters max)."}), 400

    style = "plain" if data.get("style") == "plain" else "casual"
    result = brain.respond(message, memory.history(cid), style=style)
    entry_id = brain.chat_log.add(cid, message, result["answer"],
                                  result["source"], result["corrected"])

    memory.add_message(cid, "User", message, corrected=result["corrected"])
    memory.add_message(cid, "Bot", result["answer"], source=result["source"],
                       id=entry_id, rating=None)

    return jsonify({**result, "id": entry_id, "conversation_id": cid})


@app.post("/api/feedback")
def feedback():
    data = request.get_json(silent=True) or {}
    rating = data.get("rating")
    if rating not in ("good", "bad", None):
        return jsonify({"error": "rating must be good or bad"}), 400
    entry = brain.chat_log.rate(str(data.get("id")), rating)
    if not entry:
        return jsonify({"error": "Message not found"}), 404
    memory.set_rating(entry["conversation"], entry["id"], rating)
    return jsonify({"ok": True})


@app.post("/api/teach")
def teach():
    if not is_admin():
        return jsonify({"error": "Admin password needed to teach."}), 403
    data = request.get_json(silent=True) or {}
    question = str(data.get("question", "")).strip()
    answer = str(data.get("answer", "")).strip()
    if not question or not answer:
        return jsonify({"error": "Both a question and an answer are needed."}), 400
    if len(question) > 300 or len(answer) > 1000:
        return jsonify({"error": "That's too long."}), 400
    brain.teach(question, answer)
    if data.get("id"):
        brain.chat_log.rate(str(data["id"]), "bad")
    return jsonify({"ok": True})


# ---- Brain stats + retraining --------------------------------------------
training = {"running": False, "stage": "", "progress": 0, "log": [], "error": None}


FRESH_STEPS = {"tiny": 800, "small": 3000, "medium": 5000}


def run_training(fresh, preset):
    py = sys.executable
    cmds = [[py, "-u", "build_dataset.py"],
            [py, "-u", "train.py", "--eval-every", "100"]]
    if fresh:
        cmds[1] += ["--preset", preset, "--steps", str(FRESH_STEPS[preset]), "--dropout", "0.1"]
    else:
        # Keep learning from where it left off, gently, so it doesn't forget.
        cmds[1] += ["--resume", "--steps", "400", "--lr", "3e-4"]

    try:
        for i, cmd in enumerate(cmds):
            training["stage"] = "Building dataset" if i == 0 else "Training"
            proc = subprocess.Popen(cmd, cwd=BASE_DIR, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1,
                                    encoding="utf-8", errors="replace")
            training["proc"] = proc
            for line in proc.stdout:
                line = line.strip()
                training["log"] = (training["log"] + [line])[-8:]
                m = re.match(r"Step (\d+)/(\d+)", line)
                if m:
                    training["progress"] = int(m.group(1)) / int(m.group(2))
                if "(saved)" in line and brain.neural is None:
                    brain.reload_model()  # chat with it while it keeps training
            if training.get("stopping"):
                break
            if proc.wait() != 0:
                last = training["log"][-1] if training["log"] else ""
                raise RuntimeError(f"{cmd[2]} failed: {last}")
        brain.speller = Speller()  # pick up newly learned words
        brain.reload_model()
        training["stage"] = "Done"
    except Exception as e:  # noqa: BLE001
        training["error"] = str(e)
        training["stage"] = "Failed"
    finally:
        training["running"] = False
        training.pop("proc", None)


@app.post("/api/retrain")
def retrain():
    if not is_admin():
        return jsonify({"error": "Admin password needed to retrain."}), 403
    if training["running"]:
        return jsonify({"error": "Already training."}), 409
    body = request.get_json(silent=True) or {}
    fresh = brain.neural is None or bool(body.get("fresh"))
    preset = body.get("preset") if body.get("preset") in FRESH_STEPS else "tiny"
    training.update(running=True, stage="Starting", progress=0, log=[], error=None, stopping=False)
    threading.Thread(target=run_training, args=(fresh, preset), daemon=True).start()
    return jsonify({"ok": True, "fresh": fresh, "preset": preset})


@app.get("/api/stats")
def stats():
    return jsonify({
        **brain.stats(),
        "last_training": _read_json(TRAIN_INFO_FILE, None),
        "training": {k: v for k, v in training.items() if k != "proc"},
        "admin_required": bool(ADMIN_KEY),
    })


# ---- Updates from GitHub -------------------------------------------------
def stop_training():
    """Stop training (the best version so far is already saved)."""
    proc = training.get("proc")
    if proc and proc.poll() is None:
        training["stopping"] = True
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def restart():
    """Restart Neural so the update takes effect."""
    stop_training()
    if os.environ.get("NEURAL_LAUNCHER") == "1":
        os._exit(3)  # launcher.py starts us again
    os.execv(sys.executable, [sys.executable] + sys.argv)


@app.get("/api/update")
def update_status():
    return jsonify(updater.state)


@app.post("/api/update/check")
def update_check():
    return jsonify(updater.check())


@app.post("/api/update/apply")
def update_apply():
    if not is_admin():
        return jsonify({"error": "Admin password needed to update."}), 403
    if not updater.apply_async(restart):
        return jsonify({"error": "No update to apply."}), 409
    return jsonify({"ok": True})


@app.post("/api/update/undo")
def update_undo():
    if not is_admin():
        return jsonify({"error": "Admin password needed."}), 403
    if not updater.undo_last(restart):
        return jsonify({"error": "Nothing to undo."}), 409
    return jsonify({"ok": True})


def autotrain_if_needed():
    """First run with no model yet: train a tiny starter model in the background.
    Neural still answers from its knowledge while it trains."""
    # Only when there's no model file at all. If one exists (even from a
    # training run that was stopped early), use it instead of starting over.
    if not MODEL_FILE.exists() and not training["running"] \
            and os.environ.get("NEURAL_AUTOTRAIN", "1") == "1":
        print("  No neural network yet - training a starter model in the background...")
        training.update(running=True, stage="Starting", progress=0, log=[], error=None, stopping=False)
        threading.Thread(target=run_training, args=(True, "tiny"), daemon=True).start()


if __name__ == "__main__":
    updater.start_background_checks()
    autotrain_if_needed()
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 5000))
    if host == "127.0.0.1" and not os.environ.get("NO_BROWSER") and not os.environ.get("NEURAL_RESTARTED"):
        threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    print(f"\n  Neural is running at http://127.0.0.1:{port}\n")
    app.run(host=host, port=port, debug=False, threaded=True)
