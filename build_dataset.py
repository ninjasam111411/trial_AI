"""
Builds data/training.txt from data/intents.json.

Every question pattern is paired with every answer, and extra copies are
made in lowercase and without punctuation so the model copes with the way
people actually type. Short multi-turn conversations are added too, so the
bot learns to use chat history.

Run:  python build_dataset.py
"""

import json
import random
from pathlib import Path

INTENTS_FILE = Path("data/intents.json")
OUTPUT_FILE = Path("data/training.txt")
EXTRA_FILE = Path("data/extra.txt")  # optional hand-written conversations

random.seed(42)


def variants(text):
    """Return the text plus a few realistic typing variations."""
    out = {text}
    out.add(text.lower())
    stripped = text.translate(str.maketrans("", "", "?!.,'"))
    out.add(stripped.strip())
    out.add(stripped.lower().strip())
    return [v for v in out if v]


def build():
    intents = json.loads(INTENTS_FILE.read_text(encoding="utf-8"))["intents"]
    examples = []

    for intent in intents:
        for pattern in intent["patterns"]:
            for question in variants(pattern):
                for answer in intent["responses"]:
                    examples.append(f"User: {question}\nBot: {answer}")

    # Multi-turn conversations: chain random exchanges together.
    pairs = [
        (random.choice(i["patterns"]), random.choice(i["responses"]))
        for i in intents
        for _ in range(3)
    ]
    for _ in range(600):
        turns = random.sample(pairs, k=random.randint(2, 3))
        examples.append(
            "\n".join(f"User: {q}\nBot: {a}" for q, a in turns)
        )

    if EXTRA_FILE.exists():
        extra = EXTRA_FILE.read_text(encoding="utf-8").strip()
        examples.extend(e.strip() for e in extra.split("\n\n") if e.strip())

    random.shuffle(examples)
    OUTPUT_FILE.write_text("\n\n".join(examples) + "\n", encoding="utf-8")

    print(f"Intents:  {len(intents)}")
    print(f"Examples: {len(examples)}")
    print(f"Size:     {OUTPUT_FILE.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    build()
