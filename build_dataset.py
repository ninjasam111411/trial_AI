"""
Builds data/training.txt from everything Neural knows:

  - data/intents.json           built-in topics
  - answers people taught it    (saved/learned/taught.json)
  - chats rated 👍              (saved/learned/chats.jsonl)
  - misspelled messages that it understood after fixing the spelling
  - fake typos + real typos people made, so it learns what misspelled
    words look like and still answers correctly
  - data/extra.txt              optional hand-written conversations

Run:  python build_dataset.py
"""

import json
import random
import re

from config import EXTRA_FILE, TRAINING_FILE, load_intents
from learning import ChatLog, Speller, TaughtStore
from style import humanize, kind_for

random.seed(42)

KEYBOARD = {
    "q": "wa", "w": "qes", "e": "wrd", "r": "etf", "t": "ryg", "y": "tuh",
    "u": "yij", "i": "uok", "o": "ipl", "p": "ol", "a": "qsz", "s": "adwx",
    "d": "sfec", "f": "dgrv", "g": "fhtb", "h": "gjyn", "j": "hkum",
    "k": "jli", "l": "kop", "z": "asx", "x": "zcs", "c": "xvd", "v": "cbf",
    "b": "vng", "n": "bmh", "m": "nj",
}


def make_typo(word):
    """One realistic typing mistake: swap, drop, double, or hit a neighbor key."""
    if len(word) < 4:
        return word
    i = random.randrange(1, len(word) - 1)
    kind = random.choice(["swap", "drop", "double", "neighbor"])
    if kind == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if kind == "drop":
        return word[:i] + word[i + 1:]
    if kind == "double":
        return word[:i] + word[i] + word[i:]
    near = KEYBOARD.get(word[i].lower())
    return word[:i] + random.choice(near) + word[i + 1:] if near else word


def typo_sentence(text, real_typos):
    words = text.split()
    candidates = [i for i, w in enumerate(words) if len(w.strip("?!.,")) >= 4]
    if not candidates:
        return None
    for i in random.sample(candidates, k=min(len(candidates), random.choice([1, 1, 2]))):
        core = words[i].strip("?!.,").lower()
        # Prefer a misspelling a real person actually made.
        if core in real_typos:
            words[i] = random.choice(real_typos[core])
        else:
            words[i] = make_typo(words[i].strip("?!.,"))
    out = " ".join(words)
    return out if out.lower() != text.lower() else None


def one_block(text):
    """Conversations are separated by blank lines, so messages can't contain any."""
    return re.sub(r"\n\s*\n+", "\n", str(text).strip())


def variants(text):
    out = {text, text.lower()}
    stripped = text.translate(str.maketrans("", "", "?!.,'")).strip()
    out.update({stripped, stripped.lower()})
    return [v for v in out if v]


def build():
    intents = load_intents()
    taught = [{**t, "patterns": [one_block(x) for x in t["patterns"]],
               "responses": [one_block(x) for x in t["responses"]]}
              for t in TaughtStore().as_intents()]
    all_intents = intents + taught

    # real typos: correct word -> [misspellings people typed]
    real_typos = {}
    for typo, fixes in Speller().typos.items():
        for right in fixes:
            real_typos.setdefault(right, []).append(typo)

    def talk(intent, answer):
        """Casual, human-sounding version of an answer (taught answers stay exact)."""
        if intent["tag"].startswith("taught_"):
            return answer
        return humanize(answer, kind_for(intent))

    examples = []
    for intent in all_intents:
        repeat = 3 if intent["tag"].startswith("taught_") else 1
        for pattern in intent["patterns"]:
            for question in variants(pattern):
                for answer in intent["responses"]:
                    examples.extend([f"User: {question}\nBot: {talk(intent, answer)}"] * repeat)
            # Misspelled versions of the question -> same correct answer.
            for _ in range(3):
                typo = typo_sentence(pattern, real_typos)
                if typo:
                    examples.append(f"User: {typo}\nBot: {talk(intent, random.choice(intent['responses']))}")

    # Learn from real chats.
    good = confirmed = 0
    for e in ChatLog().entries():
        if e.get("rating") == "bad":
            continue
        if e.get("rating") == "good":
            examples.extend([f"User: {one_block(e['user'])}\nBot: {one_block(e['bot'])}"] * 3)
            good += 1
        elif e.get("corrected") and e.get("source") == "knowledge":
            # A misspelled message it understood: keep the original spelling
            # so the model sees what real typos look like.
            examples.append(f"User: {one_block(e['user'])}\nBot: {one_block(e['bot'])}")
            confirmed += 1

    # Multi-turn conversations so it learns to use chat history.
    pairs = [(random.choice(i["patterns"]), talk(i, random.choice(i["responses"])))
             for i in all_intents for _ in range(3)]
    for _ in range(600):
        turns = random.sample(pairs, k=min(len(pairs), random.randint(2, 3)))
        examples.append("\n".join(f"User: {q}\nBot: {a}" for q, a in turns))

    if EXTRA_FILE.exists():
        extra = EXTRA_FILE.read_text(encoding="utf-8").strip()
        examples.extend(e.strip() for e in extra.split("\n\n") if e.strip())

    random.shuffle(examples)
    TRAINING_FILE.write_text("\n\n".join(examples) + "\n", encoding="utf-8")

    print(f"Topics: {len(intents)}  |  taught: {len(taught)}  |  "
          f"rated-good chats: {good}  |  corrected chats: {confirmed}")
    print(f"Examples: {len(examples)}  |  size: {TRAINING_FILE.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    build()
