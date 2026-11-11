
import json
from pathlib import Path

MEMORY_FILE = Path("saved/memory.json")


def load_memory():
    if not MEMORY_FILE.exists():
        return []

    with open(MEMORY_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def save_memory(memory):
    MEMORY_FILE.parent.mkdir(exist_ok=True)

    with open(MEMORY_FILE, "w", encoding="utf-8") as file:
        json.dump(memory, file, indent=2)


def add_message(role, message):
    memory = load_memory()

    memory.append({
        "role": role,
        "message": message
    })

    # Keep the latest 20 messages
    memory = memory[-20:]

    save_memory(memory)


def get_context():
    memory = load_memory()

    return "\n".join(
        f"{item['role']}: {item['message']}"
        for item in memory
    )
