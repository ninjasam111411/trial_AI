"""Saved conversations (like the chat list in other AI apps)."""

import json
import threading
import time
import uuid

from config import CONVERSATIONS_FILE

_lock = threading.Lock()


def _load():
    try:
        return json.loads(CONVERSATIONS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save(data):
    CONVERSATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CONVERSATIONS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(CONVERSATIONS_FILE)


def list_conversations():
    data = _load()
    items = [{"id": cid, "title": c["title"], "updated": c["updated"]}
             for cid, c in data.items()]
    return sorted(items, key=lambda c: c["updated"], reverse=True)


def create_conversation():
    with _lock:
        data = _load()
        cid = uuid.uuid4().hex[:12]
        data[cid] = {"title": "New chat", "updated": time.time(), "messages": []}
        _save(data)
    return cid


def get_conversation(cid):
    return _load().get(cid)


def delete_conversation(cid):
    with _lock:
        data = _load()
        data.pop(cid, None)
        _save(data)


def add_message(cid, role, text, **extra):
    with _lock:
        data = _load()
        if cid not in data:
            data[cid] = {"title": "New chat", "updated": time.time(), "messages": []}
        convo = data[cid]
        convo["messages"].append({"role": role, "message": text, **extra})
        convo["messages"] = convo["messages"][-200:]
        if convo["title"] == "New chat" and role == "User":
            convo["title"] = text[:40] + ("..." if len(text) > 40 else "")
        convo["updated"] = time.time()
        _save(data)


def set_rating(cid, entry_id, rating):
    with _lock:
        data = _load()
        for m in data.get(cid, {}).get("messages", []):
            if m.get("id") == entry_id:
                m["rating"] = rating
        _save(data)


def history(cid, turns=4):
    convo = get_conversation(cid)
    return convo["messages"][-turns:] if convo else []
