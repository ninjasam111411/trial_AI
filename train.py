"""
Train Neural's language model from scratch. No downloaded model, no Ollama:
every weight is learned here.

Two stages, the same way big models are made:

  1. pretrain  learns English from lots of general text      (data/pretrain/)
  2. chat      learns to answer like an assistant              (data/training.txt
                                                                + data/chats/)

Examples:
    python train.py --preset tiny                     # quick: chat only, minutes
    python download_data.py                           # get free training text
    python train.py --preset small --stage pretrain --steps 20000
    python train.py --stage chat --resume --steps 3000
    python train.py --resume --steps 400 --lr 3e-4    # keep learning new chats

Presets (see model.py): tiny ~5M, small ~33M, medium ~110M, large ~335M, xl ~1.3B
"""

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from config import (CHAT_DATA_DIR, LEARNED_DIR, MODEL_FILE, PRETRAIN_DIR,
                    TOKEN_CACHE_DIR, TOKENIZER_FILE, TRAIN_INFO_FILE, TRAINING_FILE)
from model import PRESETS, NeuralLM
from tokenizer import Tokenizer, parse_chat

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--preset", default="tiny", choices=PRESETS)
p.add_argument("--stage", default="chat", choices=["chat", "pretrain"])
p.add_argument("--dim", type=int)
p.add_argument("--layers", type=int)
p.add_argument("--heads", type=int)
p.add_argument("--context", type=int)
p.add_argument("--vocab", type=int)
p.add_argument("--steps", type=int, default=2000)
p.add_argument("--batch-size", type=int, default=16)
p.add_argument("--grad-accum", type=int, default=1, help="bigger effective batch on small GPUs")
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--dropout", type=float, default=0.0)
p.add_argument("--eval-every", type=int, default=200)
p.add_argument("--resume", action="store_true", help="keep training the saved model")
p.add_argument("--new-tokenizer", action="store_true", help="retrain the tokenizer (fresh models only)")
p.add_argument("--compile", action="store_true", help="torch.compile for speed (Linux + GPU)")
p.add_argument("--out", default=str(MODEL_FILE))
args = p.parse_args()

torch.manual_seed(1337)
random.seed(1337)
np.random.seed(1337)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
amp_dtype = None
if device == "cuda":
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)


def log(msg):
    print(msg, flush=True)


def chat_texts():
    for f in [TRAINING_FILE] + sorted(CHAT_DATA_DIR.glob("*.txt")):
        if f.exists():
            for ex in f.read_text(encoding="utf-8").split("\n\n"):
                if ex.strip():
                    yield ex.strip()


def tokenizer_corpus():
    yield from chat_texts()
    budget = 200_000_000  # bytes of general text to learn word pieces from
    for f in sorted(PRETRAIN_DIR.glob("*.txt")):
        with open(f, encoding="utf-8", errors="ignore") as fh:
            while budget > 0 and (chunk := fh.read(1_000_000)):
                budget -= len(chunk)
                yield chunk


# ---- Model config + tokenizer --------------------------------------------
checkpoint = None
if args.resume:
    if not Path(args.out).exists():
        sys.exit(f"--resume: no saved model at {args.out}. Train one first.")
    checkpoint = torch.load(args.out, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != 2:
        sys.exit("The saved model is the old letter-by-letter kind. Train a new one (no --resume).")
    config = checkpoint["config"]
    TOKENIZER_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKENIZER_FILE.write_text(checkpoint["tokenizer"], encoding="utf-8")
    tokenizer = Tokenizer(TOKENIZER_FILE)
else:
    preset = dict(PRESETS[args.preset])
    for k in ("dim", "layers", "heads", "context", "vocab"):
        if getattr(args, k):
            preset[k] = getattr(args, k)
    tokenizer = Tokenizer(TOKENIZER_FILE) if TOKENIZER_FILE.exists() else None
    if args.new_tokenizer or tokenizer is None or tokenizer.vocab_size != preset["vocab"]:
        log(f"Training Neural's own tokenizer (vocab {preset['vocab']:,})...")
        tokenizer = Tokenizer.train(tokenizer_corpus(), TOKENIZER_FILE, preset["vocab"])
    config = dict(vocab=tokenizer.vocab_size, dim=preset["dim"], layers=preset["layers"],
                  heads=preset["heads"], context=preset["context"])

# ---- Data ----------------------------------------------------------------
tok_hash = hashlib.md5(TOKENIZER_FILE.read_bytes()).hexdigest()[:8]


def load_chat():
    examples = list(chat_texts())
    random.shuffle(examples)
    ids, mask = [], []
    for ex in examples:
        turns = parse_chat(ex)
        if turns:
            i, m = tokenizer.encode_chat(turns)
            ids += i
            mask += m
    return np.array(ids, dtype=np.int32), np.array(mask, dtype=np.bool_), len(examples)


def read_docs(fh):
    buf = []
    for line in fh:
        if line.strip() == "<|end|>":
            if buf:
                yield "".join(buf).strip()
            buf = []
        else:
            buf.append(line)
    if buf:
        yield "".join(buf).strip()


def encode_docs(docs):
    out = []
    for ids in tokenizer.encode_batch(docs):
        out.extend(ids)
        out.append(tokenizer.end_id)
    return out


def load_pretrain():
    files = sorted(PRETRAIN_DIR.glob("*.txt"))
    if not files:
        sys.exit("No text in data/pretrain/. Run `python download_data.py` first.")
    TOKEN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    arrays = []
    for f in files:
        size = f.stat().st_size
        cache = TOKEN_CACHE_DIR / f"{f.stem}-{size}-{tok_hash}.bin"
        if not cache.exists():
            log(f"Tokenizing {f.name} ({size / 1e6:.0f} MB, one time only)...")
            with open(cache, "wb") as out, open(f, encoding="utf-8", errors="ignore") as fh:
                docs, chunk = [], 0
                for doc in read_docs(fh):
                    docs.append(doc)
                    chunk += len(doc)
                    if chunk > 20_000_000:
                        np.array(encode_docs(docs), dtype=np.uint32).tofile(out)
                        docs, chunk = [], 0
                np.array(encode_docs(docs), dtype=np.uint32).tofile(out)
        arrays.append(np.memmap(cache, dtype=np.uint32, mode="r"))
    return (np.concatenate(arrays) if len(arrays) > 1 else arrays[0]), None, None


data, mask, n_examples = load_chat() if args.stage == "chat" else load_pretrain()
ctx = config["context"]
split = int(len(data) * 0.97)
if split <= ctx + 1 or len(data) - split <= ctx + 1:
    sys.exit("Not enough training data for this context size.")
log(f"Device: {device}{' (' + str(amp_dtype).split('.')[-1] + ')' if amp_dtype else ''}"
    f"  |  stage: {args.stage}  |  tokens: {len(data):,}"
    + (f"  |  chats: {n_examples:,}" if n_examples else ""))


def get_batch(part):
    lo, hi = (0, split) if part == "train" else (split, len(data))
    starts = np.random.randint(lo, hi - ctx - 1, args.batch_size)
    x = np.stack([data[s:s + ctx] for s in starts]).astype(np.int64)
    y = np.stack([data[s + 1:s + ctx + 1] for s in starts]).astype(np.int64)
    if mask is not None:
        m = np.stack([mask[s + 1:s + ctx + 1] for s in starts])
        y[~m] = -100
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    if device == "cuda":
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


# ---- Model ---------------------------------------------------------------
model = NeuralLM(config["vocab"], config["dim"], config["layers"], config["heads"],
                 config["context"], args.dropout).to(device)
if checkpoint:
    model.load_state_dict(checkpoint["model"])
n_params = model.num_params()
log(f"Model: {n_params / 1e6:.1f}M parameters  "
    f"(dim {config['dim']}, {config['layers']} layers, {config['heads']} heads, "
    f"context {ctx}, vocab {config['vocab']:,})" + ("  |  resuming" if checkpoint else ""))

decay = [p for p in model.parameters() if p.dim() >= 2]
no_decay = [p for p in model.parameters() if p.dim() < 2]
optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                               {"params": no_decay, "weight_decay": 0.0}],
                              lr=args.lr, betas=(0.9, 0.95))
train_model = torch.compile(model) if args.compile else model
warmup = max(1, min(500, args.steps // 20))


def lr_at(step):
    if step < warmup:
        return args.lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, args.steps - warmup)
    return args.lr * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))


def autocast():
    if amp_dtype:
        return torch.autocast("cuda", dtype=amp_dtype)
    return torch.autocast("cpu", enabled=False)


@torch.no_grad()
def evaluate(iters=20):
    model.eval()
    out = {}
    for part in ("train", "val"):
        losses = []
        for _ in range(iters):
            x, y = get_batch(part)
            with autocast():
                losses.append(train_model(x, y)[1].item())
        out[part] = sum(losses) / len(losses)
    model.train()
    return out


def save(val_loss):
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(args.out).with_suffix(".tmp")
    torch.save({
        "model": model.state_dict(),
        "config": config,
        "tokenizer": TOKENIZER_FILE.read_text(encoding="utf-8"),
        "params": n_params,
        "stage": args.stage,
        "val_loss": val_loss,
        "format": 2,
    }, tmp)
    tmp.replace(args.out)  # never leave a half-written model behind


# ---- Training loop -------------------------------------------------------
best = float("inf")
start = time.time()
quiet = not sys.stdout.isatty()
model.train()

for step in range(args.steps + 1):
    for g in optimizer.param_groups:
        g["lr"] = lr_at(step)

    if step % args.eval_every == 0 or step == args.steps:
        losses = evaluate()
        saved = losses["val"] < best
        if saved:
            best = losses["val"]
            save(best)
        elapsed = time.time() - start
        eta = elapsed / max(step, 1) * (args.steps - step)
        log(f"Step {step}/{args.steps} | train {losses['train']:.3f} | val {losses['val']:.3f} | "
            f"{elapsed / 60:.1f}m elapsed, ~{eta / 60:.0f}m left{'  (saved)' if saved else ''}")
    if step == args.steps:
        break
    if quiet and step % 10 == 0 and step % args.eval_every:
        log(f"Step {step}/{args.steps}")  # progress for the app's Retrain bar

    for _ in range(args.grad_accum):
        x, y = get_batch("train")
        with autocast():
            loss = train_model(x, y)[1] / args.grad_accum
        scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

LEARNED_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_INFO_FILE.write_text(json.dumps({
    "time": int(time.time()), "val_loss": round(best, 4), "steps": args.steps,
    "stage": args.stage, "params": n_params, "examples": n_examples or 0,
    "tokens": int(len(data)), "resumed": bool(checkpoint),
}), encoding="utf-8")
log(f"Training complete! Best val loss: {best:.4f}  ({n_params / 1e6:.1f}M parameters)")
log(f"Model saved to {args.out}")
