"""
Download free, public training text for Neural. This is text only, never a
pretrained model: Neural still learns everything itself.

    python download_data.py                  # recommended starter set (~400 MB)
    python download_data.py --stories-mb 1000 --wiki-mb 1000   # more = smarter, slower

What it gets:
  data/pretrain/tinystories.txt   simple English stories       (teaches grammar)
  data/pretrain/wikipedia.txt     Simple English Wikipedia     (teaches facts)
  data/chats/dolly.txt            15k human-written Q&A        (teaches answering)
  data/chats/alpaca.txt           52k instruction answers      (teaches answering)

Licenses: TinyStories CDLA-Sharing-1.0, Wikipedia CC BY-SA, Dolly CC BY-SA 3.0,
Alpaca-cleaned CC BY-NC 4.0 (fine for personal use; skip with --no-alpaca if
you ever sell this).
"""

import argparse
import re

from config import CHAT_DATA_DIR, PRETRAIN_DIR

p = argparse.ArgumentParser()
p.add_argument("--stories-mb", type=int, default=200)
p.add_argument("--wiki-mb", type=int, default=200)
p.add_argument("--max-answer", type=int, default=1500, help="skip answers longer than this (characters)")
p.add_argument("--no-alpaca", action="store_true")
args = p.parse_args()

try:
    from datasets import load_dataset
except ImportError:
    raise SystemExit("Run: pip install datasets")

PRETRAIN_DIR.mkdir(parents=True, exist_ok=True)
CHAT_DATA_DIR.mkdir(parents=True, exist_ok=True)


def clean(text):
    text = text.replace("\r", "").strip()
    return re.sub(r"\n{3,}", "\n\n", text)


def write_docs(name, rows, field, limit_mb):
    path = PRETRAIN_DIR / f"{name}.txt"
    limit, size, n = limit_mb * 1_000_000, 0, 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            text = clean(row[field])
            if len(text) < 100:
                continue
            f.write(text + "\n<|end|>\n")
            size += len(text)
            n += 1
            if n % 20000 == 0:
                print(f"  {name}: {size / 1e6:.0f} / {limit_mb} MB", flush=True)
            if size >= limit:
                break
    print(f"  saved {n:,} documents to {path}")


def one_line(text):
    # Chat files use blank lines between conversations, so keep each
    # message free of blank lines.
    return re.sub(r"\n\s*\n+", "\n", clean(text))


def write_chats(name, rows, to_pair):
    path = CHAT_DATA_DIR / f"{name}.txt"
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            q, a = to_pair(row)
            if not q or not a or len(a) > args.max_answer or len(q) > 1500:
                continue
            f.write(f"User: {one_line(q)}\nBot: {one_line(a)}\n\n")
            n += 1
    print(f"  saved {n:,} conversations to {path}")


def with_context(instruction, context):
    return f"{instruction}\n{context}" if context and context.strip() else instruction


if args.stories_mb:
    print("Downloading TinyStories...")
    write_docs("tinystories", load_dataset("roneneldan/TinyStories", split="train", streaming=True),
               "text", args.stories_mb)

if args.wiki_mb:
    print("Downloading Simple English Wikipedia...")
    write_docs("wikipedia", load_dataset("wikimedia/wikipedia", "20231101.simple",
                                         split="train", streaming=True), "text", args.wiki_mb)

print("Downloading Dolly 15k...")
write_chats("dolly", load_dataset("databricks/databricks-dolly-15k", split="train"),
            lambda r: (with_context(r["instruction"], r["context"]), r["response"]))

if not args.no_alpaca:
    print("Downloading Alpaca (cleaned)...")
    write_chats("alpaca", load_dataset("yahma/alpaca-cleaned", split="train"),
                lambda r: (with_context(r["instruction"], r["input"]), r["output"]))

print("\nDone! Next:\n"
      "  python train.py --preset small --stage pretrain --steps 20000\n"
      "  python train.py --stage chat --resume --steps 4000")
