"""
Train the Neural chatbot.

    python build_dataset.py     # (re)build data/training.txt
    python train.py             # train with defaults
    python train.py --steps 6000 --layers 6   # bigger / longer
    python train.py --resume --steps 500      # keep learning from new data

The best checkpoint (lowest validation loss) is saved to saved/chatbot.pt.
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from config import LEARNED_DIR, MODEL_FILE, TRAIN_INFO_FILE, TRAINING_FILE
from model import NeuralChatbot
from tokenizer import Tokenizer, EOS

parser = argparse.ArgumentParser()
parser.add_argument("--data", default=str(TRAINING_FILE))
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--context-size", type=int, default=256)
parser.add_argument("--embedding-size", type=int, default=192)
parser.add_argument("--heads", type=int, default=6)
parser.add_argument("--layers", type=int, default=4)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--eval-every", type=int, default=250)
parser.add_argument("--out", default=str(MODEL_FILE))
parser.add_argument("--resume", action="store_true",
                    help="continue training the saved model instead of starting over")
args = parser.parse_args()

torch.manual_seed(1337)
random.seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Data ----------------------------------------------------------------
text = Path(args.data).read_text(encoding="utf-8")
examples = [e.strip() for e in text.split("\n\n") if e.strip()]
random.shuffle(examples)

checkpoint = None
if args.resume and Path(args.out).exists():
    checkpoint = torch.load(args.out, map_location=device, weights_only=True)

# Fixed keyboard alphabet, so new data never changes the vocabulary.
tokenizer = Tokenizer("".join(checkpoint["characters"])) if checkpoint else Tokenizer()


def to_tensor(items):
    # Each conversation ends with the end-of-response token, so the
    # model learns when to STOP talking.
    ids = []
    for e in items:
        ids.extend(tokenizer.encode(e))
        ids.append(EOS)
    return torch.tensor(ids, dtype=torch.long)


split = int(len(examples) * 0.95)
train_data = to_tensor(examples[:split])
val_data = to_tensor(examples[split:])

if len(train_data) <= args.context_size + 1 or len(val_data) <= args.context_size + 1:
    raise ValueError("Add more training text!")

print(f"Device: {device}")
print(f"Examples: {len(examples)}  |  vocab: {tokenizer.vocab_size}")
print(f"Train tokens: {len(train_data):,}  |  val tokens: {len(val_data):,}")


def get_batch(data):
    starts = torch.randint(0, len(data) - args.context_size - 1, (args.batch_size,))
    x = torch.stack([data[i:i + args.context_size] for i in starts])
    y = torch.stack([data[i + 1:i + args.context_size + 1] for i in starts])
    return x.to(device), y.to(device)


# ---- Model ---------------------------------------------------------------
if checkpoint and checkpoint.get("config"):
    config = checkpoint["config"]
    args.context_size = config["context_size"]
else:
    config = dict(
        vocab_size=tokenizer.vocab_size,
        embedding_size=args.embedding_size,
        heads=args.heads,
        layers=args.layers,
        context_size=args.context_size,
        dropout=args.dropout,
    )
    checkpoint = None
model = NeuralChatbot(**config).to(device)
if checkpoint:
    model.load_state_dict(checkpoint["model"])
    print("Resuming from the saved model")
print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
warmup = max(1, args.steps // 20)


def lr_at(step):
    # Warm up, then cosine decay to 10% of the peak learning rate.
    if step < warmup:
        return args.lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, args.steps - warmup)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def estimate_loss(iters=20):
    model.eval()
    result = {}
    for name, data in (("train", train_data), ("val", val_data)):
        losses = []
        for _ in range(iters):
            x, y = get_batch(data)
            logits = model(x)
            losses.append(F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1)
            ).item())
        result[name] = sum(losses) / len(losses)
    model.train()
    return result


def save():
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "characters": tokenizer.characters,
        "config": config,
        "context_size": args.context_size,
    }, args.out)


# ---- Training loop -------------------------------------------------------
best_val = float("inf")
start = time.time()

for step in range(args.steps + 1):
    for group in optimizer.param_groups:
        group["lr"] = lr_at(step)

    if step % args.eval_every == 0 or step == args.steps:
        losses = estimate_loss()
        marker = ""
        if losses["val"] < best_val:
            best_val = losses["val"]
            save()
            marker = "  (saved)"
        print(
            f"Step {step}/{args.steps} | train {losses['train']:.4f} | "
            f"val {losses['val']:.4f} | {time.time() - start:.0f}s{marker}"
        )

    if step == args.steps:
        break
    if step % 10 == 0 and step % args.eval_every and not sys.stdout.isatty():
        print(f"Step {step}/{args.steps}", flush=True)  # progress for the app

    x, y = get_batch(train_data)
    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

LEARNED_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_INFO_FILE.write_text(json.dumps({
    "time": int(time.time()), "val_loss": round(best_val, 4),
    "steps": args.steps, "examples": len(examples), "resumed": bool(checkpoint),
}), encoding="utf-8")
print(f"Training complete! Best val loss: {best_val:.4f}")
print(f"Model saved to {args.out}")
