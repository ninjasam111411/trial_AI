# Project Neural (trial_AI)

A small chatbot built from scratch with Python and PyTorch: a character-level
transformer neural network, plus a knowledge base and calculator that make it
answer reliably.

## How it answers

Every message goes through three steps (see `brain.py`):

1. **Calculator** - exact answers for math like `12 * 7` or `100 divided by 4`.
2. **Knowledge base** - fuzzy-matches your message against `data/intents.json`
   (140+ topics: science, space, animals, computers, AI, geography, history,
   jokes, riddles, fun facts...). Typos and different wording still match.
3. **Neural network** - if nothing matches, the trained transformer writes a reply.

The web page labels each answer with where it came from.

## Setup

```bash
pip install -r requirements.txt
python build_dataset.py   # builds data/training.txt from data/intents.json
python train.py           # trains the model -> saved/chatbot.pt
```

Then chat in the terminal or the browser:

```bash
python main.py            # terminal chat (type /debug to see answer sources)
python app.py             # web chat at http://127.0.0.1:5000
python test_chat.py       # quick test with sample questions
```

## Making it smarter

- **Add knowledge:** add a new entry to `data/intents.json` with a few ways to
  ask the question (`patterns`) and one or more answers (`responses`). It works
  in the knowledge base right away; run `build_dataset.py` and `train.py` again
  so the neural net learns it too.
- **Add conversations:** put extra example chats in `data/extra.txt`
  (format below, blank line between conversations) and rebuild.
- **Bigger model / longer training:** `python train.py --steps 6000 --layers 6`.
  On a slow computer use a smaller model:
  `python train.py --context-size 128 --embedding-size 128 --heads 4 --layers 3`.
  Watch the `val` loss - lower is better. If it stops going down, more data
  helps more than more steps.

```
User: What is your favorite planet?
Bot: Saturn, because of its beautiful rings.
```

## Files

| File | What it does |
|---|---|
| `data/intents.json` | The knowledge: questions and answers by topic |
| `build_dataset.py` | Turns intents into training conversations |
| `train.py` | Trains the transformer (validation, learning-rate schedule, saves best) |
| `model.py` | The transformer neural network |
| `tokenizer.py` | Turns characters into numbers and back |
| `brain.py` | Calculator + knowledge base + neural net working together |
| `memory.py` | Remembers the last 20 messages |
| `main.py` / `app.py` | Terminal and web chat |
