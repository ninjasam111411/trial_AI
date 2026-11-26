"""
How Neural learns from the chats people send it.

  Speller     - knows every word in its training data plus words people
                type often. It fixes misspellings ("waht is teh sun") and
                remembers which misspellings map to which words.
  ChatLog     - saves every exchange with its 👍/👎 rating.
  TaughtStore - answers people teach it. They work right away, and they
                go into the training data on the next retrain.

Only USER messages are used for spelling, and only replies that were
taught or rated 👍 are used as answers. The bot's own unrated replies are
never trained on, otherwise it would learn from its own mistakes.
"""

import json
import re
import threading
import time
import uuid
from collections import Counter

from config import (CHAT_LOG_FILE, LEARNED_DIR, TAUGHT_FILE,
                    TYPOS_FILE, WORDS_FILE, load_intents)

_lock = threading.Lock()
WORD_RE = re.compile(r"[a-z']+")
# Words typed this many times in chats become "known" words.
LEARN_AFTER = 3

# Chat-speak people use a lot, expanded before anything else.
SLANG = {
    "u": "you", "r": "are", "ur": "your", "y": "why", "pls": "please",
    "plz": "please", "thx": "thanks", "ty": "thank you", "wat": "what",
    "wut": "what", "im": "i am", "ive": "i have", "dont": "don't",
    "cant": "can't", "wont": "won't", "didnt": "didn't", "doesnt": "doesn't",
    "isnt": "isn't", "whats": "what is", "hows": "how is", "wheres": "where is",
    "whos": "who is", "ya": "you", "yea": "yeah", "hii": "hi", "hiii": "hi",
    "heyy": "hey", "helo": "hello", "hru": "how are you", "wbu": "what about you",
    "idk": "i don't know", "rn": "right now", "bc": "because", "cuz": "because",
    "becuz": "because", "abt": "about", "gonna": "going to", "wanna": "want to",
    "gimme": "give me", "tmrw": "tomorrow", "2day": "today", "b4": "before",
}


def _read_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _write_json(path, data):
    LEARNED_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def words_in(text):
    return WORD_RE.findall(text.lower())


# ---- Spelling ------------------------------------------------------------
class Speller:
    LETTERS = "abcdefghijklmnopqrstuvwxyz'"

    def __init__(self):
        # Words from Neural's clean knowledge (not training.txt, which
        # contains deliberate typos) plus answers people taught it.
        self.base = Counter()
        for intent in load_intents():
            self.base.update(words_in(" ".join(intent["patterns"] + intent["responses"])))
        if TAUGHT_FILE.exists():
            self.base.update(words_in(TAUGHT_FILE.read_text(encoding="utf-8")))
        # Plus a real English dictionary (the 50,000 most common words).
        self.dictionary = {}
        try:
            from spellchecker import SpellChecker
            counts = SpellChecker().word_frequency.dictionary
            self.dictionary = dict(sorted(counts.items(), key=lambda x: -x[1])[:50000])
        except ImportError:
            print("[speller] pip install pyspellchecker for a full English dictionary")
        self.chat_words = Counter(_read_json(WORDS_FILE, {}))
        self.typos = _read_json(TYPOS_FILE, {})  # {typo: {correct: count}}

    # Words it trusts: training words + words people use a lot.
    def known(self, word):
        return (self.base[word] > 0 or word in self.dictionary
                or self.chat_words[word] >= LEARN_AFTER)

    def freq(self, word):
        # Words Neural was trained on and words people use with it come first.
        return (self.dictionary.get(word, 0) + self.base[word] * 100000
                + self.chat_words[word] * 10000)

    def _edits1(self, w):
        splits = [(w[:i], w[i:]) for i in range(len(w) + 1)]
        deletes = [a + b[1:] for a, b in splits if b]
        swaps = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
        replaces = [a + c + b[1:] for a, b in splits if b for c in self.LETTERS]
        inserts = [a + c + b for a, b in splits for c in self.LETTERS]
        return set(deletes + swaps + replaces + inserts)

    def correct_word(self, word):
        if len(word) < 3 or self.known(word):
            return word
        # 1. A misspelling it has seen before.
        if word in self.typos:
            return max(self.typos[word], key=self.typos[word].get)
        # 2. The most common known word one or two edits away.
        e1 = self._edits1(word)
        candidates = [w for w in e1 if self.known(w)]
        if not candidates and len(word) > 4:
            candidates = [w2 for w1 in e1 for w2 in self._edits1(w1) if self.known(w2)]
        return max(candidates, key=self.freq) if candidates else word

    def correct(self, text):
        """Return (corrected text, {typo: correction})."""
        changes = {}

        def fix(m):
            original = m.group()
            lower = original.lower()
            if lower not in SLANG and any(c.isdigit() for c in lower):
                return original  # leave numbers alone
            fixed = SLANG.get(lower) or self.correct_word(lower)
            if fixed == lower:
                return original
            changes[lower] = fixed
            return fixed.capitalize() if original[0].isupper() else fixed

        return re.sub(r"[A-Za-z0-9']+", fix, text), changes

    def observe(self, text):
        """Count the words a person typed (so new words/slang get learned)."""
        with _lock:
            self.chat_words.update(words_in(text))
            _write_json(WORDS_FILE, dict(self.chat_words))

    def learn_typos(self, changes):
        """Remember misspellings that led to a good answer."""
        changes = {t: r for t, r in changes.items() if t not in SLANG}
        if not changes:
            return
        with _lock:
            for typo, right in changes.items():
                self.typos.setdefault(typo, {})
                self.typos[typo][right] = self.typos[typo].get(right, 0) + 1
            _write_json(TYPOS_FILE, self.typos)

    def stats(self):
        return {
            "known_words": len(set(self.base) | set(self.dictionary)
                               | {w for w, c in self.chat_words.items() if c >= LEARN_AFTER}),
            "words_from_chats": sum(1 for w, c in self.chat_words.items()
                                    if c >= LEARN_AFTER and not self.base[w]
                                    and w not in self.dictionary),
            "typos_learned": len(self.typos),
        }


# ---- Chat log ------------------------------------------------------------
class ChatLog:
    def add(self, conversation_id, user, bot, source, corrected=None):
        entry = {
            "id": uuid.uuid4().hex[:12],
            "time": int(time.time()),
            "conversation": conversation_id,
            "user": user,
            "corrected": corrected,
            "bot": bot,
            "source": source,
            "rating": None,
        }
        with _lock:
            LEARNED_DIR.mkdir(parents=True, exist_ok=True)
            with open(CHAT_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry["id"]

    def entries(self):
        if not CHAT_LOG_FILE.exists():
            return []
        out = []
        for line in CHAT_LOG_FILE.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return out

    def rate(self, entry_id, rating):
        with _lock:
            entries = self.entries()
            found = None
            for e in entries:
                if e["id"] == entry_id:
                    e["rating"] = rating
                    found = e
            if found:
                CHAT_LOG_FILE.write_text(
                    "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries),
                    encoding="utf-8")
        return found

    def stats(self):
        entries = self.entries()
        return {
            "messages_logged": len(entries),
            "rated_good": sum(e["rating"] == "good" for e in entries),
            "rated_bad": sum(e["rating"] == "bad" for e in entries),
        }


# ---- Taught answers ------------------------------------------------------
class TaughtStore:
    def all(self):
        return _read_json(TAUGHT_FILE, [])

    def add(self, question, answer):
        with _lock:
            items = self.all()
            items.append({"question": question.strip(), "answer": answer.strip(),
                          "time": int(time.time())})
            _write_json(TAUGHT_FILE, items)

    def as_intents(self):
        return [{"tag": f"taught_{i}", "patterns": [t["question"]],
                 "responses": [t["answer"]]} for i, t in enumerate(self.all())]
