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

A small AI chatbot built from scratch with Python and PyTorch, with a full
chat app and the ability to **learn from the chats people send it**.

- Chat app like other AIs: saved chats, dark/light mode, works on phones
- Fixes spelling and understands chat-speak ("waht is teh sun", "hru", "r u ok")
- 👍 / 👎 on every answer, and you can **teach it** the right answer
- One-click **Retrain** button so the neural network learns from new chats

## Start it

**Windows:** double-click `start.bat`
**Mac / Linux:** run `./start.sh`

The first run installs everything (a few minutes). After that, Neural opens in
your browser at http://127.0.0.1:5000. If the neural network hasn't been
trained yet, open **🧠 Brain & training** and press **Train the neural network**.

Prefer the command line?

```bash
pip install -r requirements.txt
python app.py             # the chat app
python main.py            # terminal chat
python test_chat.py       # quick test with sample questions
```

## Put it online (a real website anyone can use)

The easiest free option is **Hugging Face Spaces**:

1. Make a free account at huggingface.co, then **New Space** → SDK **Docker** → Blank.
2. Upload this repo's files to the Space (or push with git). The settings at the
   top of this README tell Hugging Face how to run it.
3. In the Space's **Settings → Variables and secrets**, add a secret named
   `NEURAL_ADMIN_KEY` with a password. Then only people with the password can
   teach it or retrain it, and strangers can't teach it bad answers.

It builds and trains a starter model automatically (about 10 minutes), then
gives you a link. Note: on the free tier, learned chats reset when the Space
restarts. Turning on persistent storage and setting `NEURAL_DATA_DIR=/data`
keeps them.

The `Dockerfile` also works on Render, Railway, or any server that runs Docker.

## How it answers

Every message goes through these steps (`brain.py`):

1. **Spelling** - fixes typos and chat-speak, and answers "how do you spell ___?"
2. **Calculator** - exact answers for math like `48 * 12`
3. **Knowledge** - fuzzy-matches the question against 140+ built-in topics plus
   everything people have taught it
4. **Neural network** - if nothing matches, the transformer writes a reply

Each answer in the app shows which step it came from.

## How it learns from chats

| What happens | What Neural learns |
|---|---|
| Someone types a word a lot | It becomes a known word (slang, names, game words) |
| A misspelled message gets understood | It remembers that misspelling for next time |
| Someone presses 👎 and teaches an answer | It knows that answer **right away** |
| Someone presses 👍 | That chat goes into the next training run |
| You press **Retrain** | Taught answers, 👍 chats and real typos are added to the training data, and the neural network keeps learning from where it left off |

The training data also includes made-up typos of every question (swapped,
missing, doubled letters and nearby keys), so the network learns what
misspelled words look like.

It never trains on its own unrated replies; otherwise it would learn its own
mistakes. Learned data is saved in `saved/learned/` (not uploaded to GitHub).

## Making it smarter

- **Add knowledge:** add entries to `data/intents.json`, or just teach it in the app.
- **Add conversations:** put example chats in `data/extra.txt`:
  ```
  User: What is your favorite planet?
  Bot: Saturn, because of its beautiful rings.
  ```
- **Bigger model:** `python build_dataset.py && python train.py --steps 6000`
  (uses a bigger model than the app's quick training; better, but slower on
  a normal computer). Lower `val` loss = better.

## Files

| File | What it does |
|---|---|
| `app.py` | The chat app and its API |
| `templates/`, `static/` | The app's page, styles and code |
| `brain.py` | Spelling + calculator + knowledge + neural net working together |
| `learning.py` | Spell checker, chat log, taught answers |
| `memory.py` | Saved conversations |
| `data/intents.json` | Built-in knowledge |
| `build_dataset.py` | Builds training data from knowledge + learned chats + typos |
| `train.py` | Trains the transformer (`--resume` keeps learning) |
| `model.py`, `tokenizer.py` | The neural network and text-to-numbers |
| `Dockerfile` | Runs it as a website |
