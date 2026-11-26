"""Saved conversations (like the chat list in other AI apps).

Each browser gets its own id (saved in a cookie), so everyone who uses
Neural sees only their own chats, and they're still there next visit."""

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


def list_conversations(owner):
    data = _load()
    items = [{"id": cid, "title": c["title"], "updated": c["updated"]}
             for cid, c in data.items() if c.get("owner") == owner]
    return sorted(items, key=lambda c: c["updated"], reverse=True)


def create_conversation(owner):
    with _lock:
        data = _load()
        cid = uuid.uuid4().hex[:12]
        data[cid] = {"title": "New chat", "updated": time.time(), "messages": [],
                     "owner": owner}
        _save(data)
    return cid


def get_conversation(cid, owner=None):
    convo = _load().get(cid)
    if convo and owner is not None and convo.get("owner") != owner:
        return None  # someone else's chat
    return convo


def all_conversations(owner):
    return [{"id": cid, **c} for cid, c in _load().items() if c.get("owner") == owner]


def delete_conversation(cid, owner):
    with _lock:
        data = _load()
        if data.get(cid, {}).get("owner") == owner:
            data.pop(cid)
        _save(data)


def rename_conversation(cid, owner, title):
    with _lock:
        data = _load()
        if data.get(cid, {}).get("owner") == owner:
            data[cid]["title"] = title[:60] or "New chat"
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
