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

from flask import Flask, jsonify, render_template, request

import memory
from brain import Brain
from config import BASE_DIR, MODEL_FILE, TRAIN_INFO_FILE
from learning import Speller, _read_json

app = Flask(__name__)
brain = Brain()
ADMIN_KEY = os.environ.get("NEURAL_ADMIN_KEY", "")


def is_admin():
    return not ADMIN_KEY or request.headers.get("X-Admin-Key") == ADMIN_KEY


# ---- Pages ---------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# ---- Conversations -------------------------------------------------------
@app.get("/api/conversations")
def conversations():
    return jsonify(memory.list_conversations())


@app.post("/api/conversations")
def new_conversation():
    return jsonify({"id": memory.create_conversation()})


@app.get("/api/conversations/<cid>")
def get_conversation(cid):
    convo = memory.get_conversation(cid)
    return (jsonify(convo), 200) if convo else (jsonify({"error": "Not found"}), 404)


@app.delete("/api/conversations/<cid>")
def delete_conversation(cid):
    memory.delete_conversation(cid)
    return jsonify({"ok": True})


# ---- Chat ----------------------------------------------------------------
@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    cid = str(data.get("conversation_id") or "") or memory.create_conversation()

    if not message:
        return jsonify({"error": "Please enter a message."}), 400
    if len(message) > 500:
        return jsonify({"error": "Message is too long (500 characters max)."}), 400

    result = brain.respond(message, memory.history(cid))
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


def run_training(fresh):
    py = sys.executable
    steps = 1500 if fresh else 400
    cmds = [[py, "-u", "build_dataset.py"],
            [py, "-u", "train.py", "--steps", str(steps), "--eval-every", "100"]]
    if fresh:
        # A small, quick model that trains in minutes on a normal computer.
        cmds[1] += ["--context-size", "128", "--embedding-size", "128",
                    "--heads", "4", "--layers", "3"]
    else:
        cmds[1] += ["--resume", "--lr", "3e-4"]

    try:
        for i, cmd in enumerate(cmds):
            training["stage"] = "Building dataset" if i == 0 else "Training"
            proc = subprocess.Popen(cmd, cwd=BASE_DIR, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                line = line.strip()
                training["log"] = (training["log"] + [line])[-8:]
                m = re.match(r"Step (\d+)/(\d+)", line)
                if m:
                    training["progress"] = int(m.group(1)) / int(m.group(2))
            if proc.wait() != 0:
                raise RuntimeError(f"{cmd[2]} failed")
        brain.speller = Speller()  # pick up newly learned words
        brain.reload_model()
        training["stage"] = "Done"
    except Exception as e:  # noqa: BLE001
        training["error"] = str(e)
        training["stage"] = "Failed"
    finally:
        training["running"] = False


@app.post("/api/retrain")
def retrain():
    if not is_admin():
        return jsonify({"error": "Admin password needed to retrain."}), 403
    if training["running"]:
        return jsonify({"error": "Already training."}), 409
    fresh = not MODEL_FILE.exists() or bool((request.get_json(silent=True) or {}).get("fresh"))
    training.update(running=True, stage="Starting", progress=0, log=[], error=None)
    threading.Thread(target=run_training, args=(fresh,), daemon=True).start()
    return jsonify({"ok": True, "fresh": fresh})


@app.get("/api/stats")
def stats():
    return jsonify({
        **brain.stats(),
        "last_training": _read_json(TRAIN_INFO_FILE, None),
        "training": training,
        "admin_required": bool(ADMIN_KEY),
    })


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 5000))
    if host == "127.0.0.1" and not os.environ.get("NO_BROWSER"):
        threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    print(f"\n  Neural is running at http://127.0.0.1:{port}\n")
    app.run(host=host, port=port, debug=False, threaded=True)
