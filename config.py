"""Shared file locations. Set NEURAL_DATA_DIR to keep learned data elsewhere
(for example /data on a hosted server with persistent storage)."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SAVE_DIR = Path(os.environ.get("NEURAL_DATA_DIR", BASE_DIR / "saved"))

INTENTS_FILE = DATA_DIR / "intents.json"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"            # more topic packs (*.json)


def load_intents():
    """All built-in knowledge: intents.json plus every pack in data/knowledge/."""
    import json
    intents = []
    for path in [INTENTS_FILE] + sorted(KNOWLEDGE_DIR.glob("*.json")):
        if path.exists():
            intents += json.loads(path.read_text(encoding="utf-8"))["intents"]
    return intents
TRAINING_FILE = DATA_DIR / "training.txt"
EXTRA_FILE = DATA_DIR / "extra.txt"

MODEL_FILE = SAVE_DIR / "chatbot.pt"
CONVERSATIONS_FILE = SAVE_DIR / "conversations.json"

LEARNED_DIR = SAVE_DIR / "learned"
CHAT_LOG_FILE = LEARNED_DIR / "chats.jsonl"      # every exchange + rating
TAUGHT_FILE = LEARNED_DIR / "taught.json"         # answers people taught it
WORDS_FILE = LEARNED_DIR / "words.json"           # words seen in chats
TYPOS_FILE = LEARNED_DIR / "typos.json"           # misspelling -> correct word
TRAIN_INFO_FILE = LEARNED_DIR / "train_info.json"  # last training result

TOKENIZER_FILE = SAVE_DIR / "tokenizer.json"      # Neural's own trained tokenizer
PRETRAIN_DIR = DATA_DIR / "pretrain"              # general text (download_data.py)
CHAT_DATA_DIR = DATA_DIR / "chats"                # extra chat datasets (download_data.py)
TOKEN_CACHE_DIR = SAVE_DIR / "token_cache"        # pre-tokenized text for fast training
