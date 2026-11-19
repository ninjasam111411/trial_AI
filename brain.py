"""
The chatbot's "brain", shared by main.py, app.py and test_chat.py.

Answering a message goes through three steps:
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

from memory import load_memory

INTENTS_FILE = Path("data/intents.json")
MODEL_FILE = Path("saved/chatbot.pt")
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


class KnowledgeBase:
    def __init__(self, path=INTENTS_FILE):
        self.entries = []  # (normalized pattern, word set, intent)
        if not Path(path).exists():
            return
        for intent in json.loads(Path(path).read_text(encoding="utf-8"))["intents"]:
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
            if score > best_score:
                best_score, best_intent = score, intent

        return best_score, best_intent


# ---- 3. Neural network ---------------------------------------------------
class NeuralModel:
    def __init__(self, path=MODEL_FILE):
        import torch
        from model import NeuralChatbot
        from tokenizer import Tokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.tokenizer = Tokenizer("".join(checkpoint["characters"]))

        config = checkpoint.get("config") or {
            "vocab_size": self.tokenizer.vocab_size,
            "context_size": checkpoint["context_size"],
        }
        self.model = NeuralChatbot(**config).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.stop_ids = self.tokenizer.encode("\nUser:")

    def generate(self, prompt, max_tokens=200, temperature=0.7):
        encoded = self.tokenizer.encode(prompt)
        # Keep only what fits in the model's context window.
        encoded = encoded[-(self.model.context_size - 1):]
        x = self.torch.tensor([encoded], dtype=self.torch.long, device=self.device)
        out = self.model.generate(
            x, max_tokens=max_tokens, temperature=temperature,
            top_k=20, stop_text_ids=self.stop_ids,
        )
        reply = self.tokenizer.decode(out[0, len(encoded):].tolist())
        return reply.split("\nUser:")[0].split("\n\n")[0].strip()


# ---- Putting it together -------------------------------------------------
class Brain:
    def __init__(self, model_path=MODEL_FILE):
        self.knowledge = KnowledgeBase()
        self.neural = None
        self.last_reply = None
        if Path(model_path).exists():
            self.neural = NeuralModel(model_path)
        else:
            print(f"[brain] No trained model at {model_path} - "
                  "run `python train.py`. Using knowledge base only.")

    def build_prompt(self, message, history_turns=4):
        history = load_memory()[-history_turns:]
        lines = [f"{m['role']}: {m['message']}" for m in history]
        lines.append(f"User: {message}")
        return "\n".join(lines) + "\nBot:"

    def respond(self, message, use_history=True):
        answer = calculate(message)
        source = "calculator"

        if answer is None:
            score, intent = self.knowledge.match(message)
            if intent and score >= MATCH_THRESHOLD:
                options = [r for r in intent["responses"] if r != self.last_reply]
                answer = random.choice(options or intent["responses"])
                source = "knowledge"

        if answer is None and self.neural:
            prompt = self.build_prompt(message) if use_history else f"User: {message}\nBot:"
            answer = self.neural.generate(prompt)
            source = "neural"

        if not answer:
            answer = "I'm still learning about that. Try asking me something else!"
            source = "fallback"

        self.last_reply = answer
        return answer, source
