"""
The chatbot's "brain", shared by main.py, app.py and test_chat.py.

Answering a message goes through these steps:
  0. Spelling    - fix typos first (and answer "how do you spell X?").
  1. Calculator  - exact answers for arithmetic like "12 * 7".
  2. Knowledge   - fuzzy-match the message against data/intents.json and
                   reply with a known answer when the match is strong.
  3. Neural net  - otherwise, the trained transformer writes a reply.

This hybrid approach makes a small model feel much smarter: it is reliable
on things it was taught, and still creative on everything else.
"""

import ast
import difflib
import json
import operator
import random
import re
from pathlib import Path

from config import MODEL_FILE, load_intents
from learning import ChatLog, Speller, TaughtStore, words_in
from style import humanize, kind_for

MATCH_THRESHOLD = 0.72


# ---- 1. Calculator -------------------------------------------------------
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported")


def calculate(message):
    text = message.lower()
    for word, symbol in (("plus", "+"), ("minus", "-"), ("times", "*"),
                         ("multiplied by", "*"), ("divided by", "/"),
                         ("x", "*"), ("^", "**")):
        text = re.sub(rf"(?<=[\d\s]){re.escape(word)}(?=[\d\s])", f" {symbol} ", text)

    candidates = [
        c.strip() for c in re.findall(r"[\d\.\s\+\-\*/\(\)%]+", text)
        if re.search(r"\d\s*[\+\-\*/%]+\s*[\d\(]", c)
    ]
    if not candidates:
        return None
    expr = " ".join(max(candidates, key=len).split())

    try:
        value = _eval(ast.parse(expr, mode="eval").body)
    except (ValueError, SyntaxError, ZeroDivisionError, TypeError):
        return None

    if isinstance(value, float) and value.is_integer():
        value = int(value)
    elif isinstance(value, float):
        value = round(value, 6)
    return f"{expr} = {value}"


# ---- 2. Knowledge base ---------------------------------------------------
def _normalize(text):
    text = text.lower().replace("what's", "what is").replace("i'm", "i am")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


STOPWORDS = set("""a an the is are was were am be do does did you your i me my we our it
its to of in on at for with about what whats who how why when where which can could would
should will tell give say please hey hi hello this that there and or so just some any
very really much many more most tell u ur r""".split())


KEY_STOPWORDS = STOPWORDS - {"you", "your", "u", "ur", "i", "me", "my"}


class KnowledgeBase:
    def __init__(self, extra_intents=()):
        self.entries = []  # (normalized pattern, word set, intent)
        intents = load_intents()
        self.topic_count = len(intents)
        for intent in list(intents) + list(extra_intents):
            for pattern in intent["patterns"]:
                norm = _normalize(pattern)
                self.entries.append((norm, set(norm.split()), intent))

    def match(self, message):
        """Return (score, intent) for the closest known question."""
        norm = _normalize(message)
        words = set(norm.split())
        best_score, best_intent = 0.0, None

        for pattern, pattern_words, intent in self.entries:
            ratio = difflib.SequenceMatcher(None, norm, pattern).ratio()
            union = words | pattern_words
            overlap = len(words & pattern_words) / len(union) if union else 0
            score = max(ratio, 0.5 * ratio + 0.5 * overlap)
            # The important words must agree: "tell me about frogs" is not
            # "tell me about dogs", even though the sentences look alike.
            # "you"/"I" count here: "how smart are YOU" is not about octopuses.
            key_msg, key_pat = words - KEY_STOPWORDS, pattern_words - KEY_STOPWORDS
            shared = key_msg & key_pat
            if key_msg and key_pat and not shared:
                score *= 0.6
            elif key_msg - key_pat and key_pat - key_msg:
                score *= 0.7   # a key word was swapped (spain vs japan)
            elif key_msg - key_pat:
                # you mentioned something extra it doesn't cover
                score *= 0.85 if len(shared) < 2 else 0.95
            elif key_pat - key_msg:
                # the known question has an extra detail ("abraham" lincoln)
                score *= 0.9
            # One- or two-word topics ("macbeth", "ww1"): if every important
            # word is in the known question, that's a match.
            topic = words - STOPWORDS
            if len(words) <= 2 and topic and all(len(w) >= 3 for w in topic) \
                    and topic <= (pattern_words - STOPWORDS):
                score = max(score, 0.8)
            if intent["tag"].startswith("taught_"):
                score += 0.06  # what people taught it beats the built-in answer
            if score > best_score:
                best_score, best_intent = score, intent

        return best_score, best_intent


# ---- 3. Neural network ---------------------------------------------------
class NeuralModel:
    """Neural's own self-trained language model (see model.py / train.py)."""

    def __init__(self, path=MODEL_FILE):
        import tempfile

        import torch
        from model import NeuralLM
        from tokenizer import Tokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if ckpt.get("format") != 2:
            raise ValueError("old model format - press Train in the Brain panel to make a new one")

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            f.write(ckpt["tokenizer"])
        self.tokenizer = Tokenizer(f.name)
        Path(f.name).unlink(missing_ok=True)

        c = ckpt["config"]
        self.model = NeuralLM(c["vocab"], c["dim"], c["layers"], c["heads"], c["context"]).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.info = {"params": ckpt.get("params", self.model.num_params()),
                     "stage": ckpt.get("stage"), **c}
        self.stop = {self.tokenizer.end_id, self.tokenizer.user_id}

    def generate(self, message, history=(), max_tokens=200, temperature=0.7):
        turns = [("user" if m["role"] == "User" else "bot", m["message"]) for m in history]
        ids = self.tokenizer.chat_prompt(turns, message)
        out = self.model.generate(ids, max_tokens=max_tokens, temperature=temperature,
                                  stop_ids=self.stop)
        return self.tokenizer.decode(out).strip()


# ---- Learning from what people tell it ----------------------------------
TEACH_INVITE = "tell me and I'll remember"
QUESTION_STARTS = ("what", "who", "why", "how", "when", "where", "which", "whats", "whos",
                   "is", "are", "can", "could", "do", "does", "did", "will", "would",
                   "should", "tell", "explain", "give", "show", "say", "define")
CORRECTION_RE = re.compile(
    r"^(?:actually|no+|nope|wrong|that'?s (?:wrong|not right)|not quite)[\s,.!:-]+(?:it'?s\s+)?(.+)",
    re.I)


def is_statement(text):
    t = text.strip().lower()
    words = t.split()
    return (len(words) >= 3 and not t.endswith("?")
            and words[0].strip(",.!") not in QUESTION_STARTS)


def tidy(text):
    text = text.strip()
    text = text[0].upper() + text[1:] if text else text
    return text if text.endswith((".", "!", "?")) else text + "."


# ---- Putting it together -------------------------------------------------
SPELL_RE = re.compile(r"(?:how|hw|hwo) (?:do you|do u|do i|to|would you) spell ([a-z']+)", re.I)


class Brain:
    def __init__(self, model_path=MODEL_FILE):
        self.model_path = model_path
        self.speller = Speller()
        self.chat_log = ChatLog()
        self.taught = TaughtStore()
        self.knowledge = None
        self.neural = None
        self.last_reply = None
        self.reload_knowledge()
        self.reload_model()

    def reload_knowledge(self):
        self.knowledge = KnowledgeBase(self.taught.as_intents())
        # Facts it knows, to spot when the neural net just pastes in a
        # memorized fact that doesn't fit the question.
        self.known_facts = [_normalize(r) for i in load_intents() if kind_for(i) == "fact"
                            for r in i["responses"] if len(r) > 25]

    def reload_model(self):
        self.neural = None
        if Path(self.model_path).exists():
            try:
                self.neural = NeuralModel(self.model_path)
            except Exception as e:  # noqa: BLE001
                print(f"[brain] Could not load the model: {e}")
        else:
            print(f"[brain] No trained model at {self.model_path} yet. "
                  "Using knowledge base only until it is trained.")

    def teach(self, question, answer):
        self.taught.add(question, answer)
        self.speller.base.update(words_in(question + " " + answer))
        self.reload_knowledge()

    def respond(self, message, history=(), learn=True, style="casual"):
        """Returns a dict: answer, source, corrected (or None).
        learn=False answers without saving anything (used by tests)."""
        corrected, changes = self.speller.correct(message)
        if learn:
            self.speller.observe(message)
        answer, source = None, None

        # Did they just answer a question I couldn't, or correct me?
        learned = self.learn_from_chat(message, corrected, history) if learn else None
        if learned:
            return {"answer": learned, "source": "learned",
                    "corrected": corrected if changes else None}

        spell = SPELL_RE.search(message) or SPELL_RE.search(corrected)
        if spell:
            # Use the word as the person typed it, not the auto-corrected one.
            raw = SPELL_RE.search(message)
            word = (raw or spell).group(1).lower()
            right = self.speller.correct_word(word)
            if right != word or self.speller.known(word):
                answer = f"It's spelled {right}: {'-'.join(right.upper())}."
            else:
                answer = f"I don't know the word \"{word}\" yet."
            source = "spelling"

        if answer is None:
            answer = calculate(message)
            source = "calculator"

        if answer is None:
            score, intent = self.knowledge.match(message)
            if changes:
                fixed_score, fixed_intent = self.knowledge.match(corrected)
                if fixed_score > score:
                    score, intent = fixed_score, fixed_intent
                    if score >= MATCH_THRESHOLD and learn:
                        # The fix led to a confident answer: remember the typos.
                        self.speller.learn_typos(changes)
            if intent and score >= MATCH_THRESHOLD:
                options = [r for r in intent["responses"] if r != self.last_reply]
                answer = random.choice(options or intent["responses"])
                self.last_reply = answer
                if style == "casual" and not intent["tag"].startswith("taught_"):
                    answer = humanize(answer, kind_for(intent))
                source = "knowledge"

        if answer is None and self.neural:
            guess = self.neural.generate(corrected, history)
            g = _normalize(guess)
            # The question didn't match that fact, so a pasted-in memorized
            # fact is almost certainly the wrong answer. Admit it instead.
            if guess and not any(f in g for f in self.known_facts):
                answer, source = guess, "neural"

        if not answer:
            if self.neural is None:
                answer = ("I don't know that one yet, and my neural network is still being "
                          f"trained (check the Brain panel). If you know the answer, {TEACH_INVITE} it!")
            else:
                answer = f"Hmm, I don't know that one yet. If you know the answer, {TEACH_INVITE} it!"
            source = "fallback"

        if source != "knowledge":
            self.last_reply = answer
        return {"answer": answer, "source": source,
                "corrected": corrected if changes else None}

    def learn_from_chat(self, message, corrected, history):
        """Two natural ways to teach Neural just by chatting:
        1. It said it didn't know, and you reply with the answer.
        2. You correct it: "actually, it's ..." / "no, ..."
        Returns a reply if it learned something, else None."""
        history = list(history)
        bot = next((m for m in reversed(history) if m["role"] == "Bot"), None)
        if not bot:
            return None
        idx = len(history) - 1 - history[::-1].index(bot)
        question = next((m["message"] for m in reversed(history[:idx]) if m["role"] == "User"), None)
        if not question:
            return None

        fact = None
        m = CORRECTION_RE.match(corrected.strip())
        if TEACH_INVITE in bot["message"] and is_statement(message):
            fact = m.group(1) if m else corrected
        else:
            # Only correct guesses (neural/fallback); built-in answers are checked
            # facts, and 👎 + teach still works for those.
            if m and len(corrected.split()) >= 2 and bot.get("source") in ("neural", "fallback"):
                fact = m.group(1)
        if not fact:
            return None

        # Save the spelling-fixed question so it matches however it's typed later.
        self.teach(self.speller.correct(question)[0], tidy(fact))
        return (f"Ooh, got it, thanks for teaching me! Next time someone asks "
                f"\"{question}\", I'll tell them that.")

    def stats(self):
        return {
            "topics": self.knowledge.topic_count,
            "taught_answers": len(self.taught.all()),
            "model_ready": self.neural is not None,
            "model": self.neural.info if self.neural else None,
            **self.speller.stats(),
            **self.chat_log.stats(),
        }
