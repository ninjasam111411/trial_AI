---
title: Neural
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Neural (trial_AI)

A small language model (SLM) **trained completely from scratch**: its own
tokenizer, its own transformer and every weight learned here. No Ollama, no
downloaded model. It comes with a chat app that runs in your browser and
**learns from the chats people send it**.

- Chat app in the browser: saved chats, dark/light mode, works on phones
- Each browser gets its own cookie, so your chats are saved and still there next time
- Fixes spelling and chat-speak ("waht is teh sun", "hru", "r u ok")
- 👍 / 👎 on every answer, and you can **teach it** the right answer
- One-click **Retrain**, and model sizes from 5 million to 1.3 billion parameters

## Start it

**Windows:** double-click `start.bat`
**Mac / Linux:** run `./start.sh`

The first run installs everything (a few minutes) and opens Neural at
http://127.0.0.1:5000. If there's no model yet, it starts training a tiny one
in the background (you can chat with its built-in knowledge meanwhile). Watch
progress in **🧠 Brain & training**.

## Works offline, updates itself

After the first run installs everything, Neural needs **no internet**: the
model, tokenizer, spell checker and app all run on your computer.

When you are online, it checks this GitHub repo (the `main` branch) every
few hours. If anything changed, a banner asks **"An update is ready. Update
now?"**. Say yes and Neural:

1. downloads the new version,
2. backs up every file it's about to replace (`saved/backups/`),
3. replaces only the files that changed (and installs new packages if needed),
4. restarts itself and reloads the page.

It compares the actual file contents, not a version number, so no change is
missed. Your chats, learned data, trained model and downloaded training text
are never touched. If you edited a file and GitHub didn't change it, your
edit stays. If both changed, GitHub's version is used and yours is kept in
the backup. Changed your mind? **🧠 Brain & training → Undo last update**.

Settings (optional): `NEURAL_UPDATES=0` turns updates off,
`NEURAL_UPDATE_REPO` / `NEURAL_UPDATE_BRANCH` point at a different repo or
branch. Copies made with `git clone` skip this (use `git pull`).

## Make it smarter (bigger model + real training data)

The tiny starter model only learns from Neural's own Q&A, so it's good at those
and weak at everything else. To get a model that writes real sentences:

```bash
python download_data.py                                     # free public text, ~400 MB
python train.py --preset small --stage pretrain --steps 20000   # learns English
python train.py --stage chat --resume --steps 4000              # learns to answer
```

Then restart the app. Pretraining reads stories and Wikipedia to learn grammar
and facts. The chat stage teaches it to answer questions (from Neural's
knowledge plus ~65k human-written Q&A pairs). Big AI models are made the same
way.

### Picking a size

| Preset | Parameters | What you need | Training time (rough) |
|---|---|---|---|
| `tiny` | ~5M | any computer | minutes |
| `small` | ~33M | any computer; GPU helps a lot | hours on CPU, under an hour on a GPU |
| `medium` | ~110M | NVIDIA gaming GPU (8GB+) | several hours to a day |
| `large` | ~335M | strong GPU (12-16GB+) | days |
| `xl` | ~1.3B | 24GB GPU | weeks |

Rule of thumb: a model needs about **20 tokens of text per parameter** to
train well. `small` wants ~700M tokens (about 3 GB of text:
`python download_data.py --stories-mb 1500 --wiki-mb 1500`). A bigger model on
too little text won't be smarter. Useful flags: `--batch-size`, `--grad-accum`
(if your GPU runs out of memory, halve batch size and double grad-accum),
`--compile` (faster on Linux GPUs). Training uses NVIDIA GPUs (CUDA) and Apple
Silicon (MPS) automatically. On Windows with an NVIDIA card, install the GPU
version of PyTorch first (pick your setup at pytorch.org → Get Started); the
default install is CPU-only.

### Why not 8 billion parameters?

An 8B model trained from scratch isn't possible on a home computer:

- **Memory:** training needs about 16 bytes per parameter (weights, gradients
  and optimizer state), so ~130 GB of GPU memory, which means a rack of data-center GPUs.
- **Data:** it needs trillions of words of training text to be good (~20 tokens per parameter
  is the bare minimum; modern 8B models use far more).
- **Time and cost:** big labs train 8B models on thousands of GPUs, using more
  than a million GPU-hours. On one gaming GPU that would take centuries.

Even *running* an 8B model needs 16 GB of memory. Small models trained well
are surprisingly capable for their size, so start with `small` or `medium`.
The code scales: `--preset xl` or custom `--dim/--layers/--heads` if you ever
get access to big hardware.

## Put it online (a website anyone can use)

Free option: **Hugging Face Spaces**.

1. Make an account at huggingface.co, then **New Space** → SDK **Docker** → Blank.
2. Upload this repo's files to the Space. The settings at the top of this
   README tell it how to run.
3. Under **Settings → Variables and secrets**, add a secret `NEURAL_ADMIN_KEY`
   with a password, so only you can teach or retrain it.

It builds and trains a starter model automatically, then gives you a link.
Each visitor's chats are kept separately by their own cookie. On the free tier,
saved data resets when the Space restarts. With persistent storage, set
`NEURAL_DATA_DIR=/data` to keep it.

## How it answers

1. **Spelling** - fixes typos and chat-speak, answers "how do you spell ___?"
2. **Calculator** - exact math like `48 * 12`
3. **Knowledge** - 140+ built-in topics plus everything people taught it
4. **Neural network** - its own language model writes a reply

Each answer in the app shows which one it came from.

## How it learns from chats

| What happens | What Neural learns |
|---|---|
| Someone types a word a lot | It becomes a known word (slang, names, game words) |
| A misspelled message gets understood | It remembers that misspelling |
| 👎 + teach an answer | It knows that answer **right away** |
| 👍 | That chat goes into the next training run |
| **Retrain** | Taught answers, 👍 chats and real typos go into the training data; the model keeps learning from where it left off |

It never trains on its own unrated replies, so it doesn't learn its own mistakes.

## Your data

Chats, learned words and the model are saved in `saved/` on the computer
running Neural (not uploaded to GitHub). Your browser keeps a cookie
(`neural_uid`) so it knows which chats are yours, plus your theme setting.
**⬇️ Download my chats** saves all your chats as a file.

## Files

| File | What it does |
|---|---|
| `app.py`, `templates/`, `static/` | The chat app |
| `brain.py` | Spelling + calculator + knowledge + neural network |
| `model.py` | The transformer (RoPE, SwiGLU, RMSNorm, KV cache) and size presets |
| `tokenizer.py` | Neural's own byte-level BPE tokenizer |
| `train.py` | Pretraining and chat training (`--resume` keeps learning) |
| `download_data.py` | Free public training text |
| `build_dataset.py` | Chat data from knowledge, learned chats and typos |
| `learning.py`, `memory.py` | Spell checker, ratings, taught answers, saved chats |
| `data/intents.json` | Built-in knowledge |
| `updater.py`, `launcher.py` | Update checks from GitHub, restarting after an update |
| `Dockerfile` | Runs it as a website |
