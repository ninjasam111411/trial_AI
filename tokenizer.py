"""
Neural's own tokenizer: byte-level BPE, trained on Neural's data.

It learns common chunks of text ("the", "ing", " planet") from the training
data, so the model reads words and word pieces instead of single letters.
That makes the model much faster and smarter than reading letter by letter.
Byte-level means any text at all can be encoded, even emoji.

Chats are written with special tokens:
    <|user|>question<|bot|>answer<|end|>
"""

from pathlib import Path

from tokenizers import Tokenizer as HFTokenizer
from tokenizers import decoders, models, pre_tokenizers, trainers

END, USER, BOT = "<|end|>", "<|user|>", "<|bot|>"
SPECIAL = [END, USER, BOT]


def parse_chat(text):
    """'User: hi\\nBot: hello' -> [('user', 'hi'), ('bot', 'hello')]"""
    turns = []
    for line in text.split("\n"):
        if line.startswith("User: "):
            turns.append(["user", line[6:]])
        elif line.startswith("Bot: "):
            turns.append(["bot", line[5:]])
        elif turns:  # multi-line message continues
            turns[-1][1] += "\n" + line
    return [tuple(t) for t in turns]


class Tokenizer:
    def __init__(self, path=None, tok=None):
        self.tok = tok or HFTokenizer.from_file(str(path))
        self.vocab_size = self.tok.get_vocab_size()
        self.end_id, self.user_id, self.bot_id = (self.tok.token_to_id(s) for s in SPECIAL)

    @classmethod
    def from_string(cls, json_text):
        """Load a tokenizer saved inside a model file (no temp files needed)."""
        return cls(tok=HFTokenizer.from_str(json_text))

    @staticmethod
    def train(texts, path, vocab_size=8000):
        """Train a new tokenizer from an iterator of strings and save it."""
        tok = HFTokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size, special_tokens=SPECIAL, min_frequency=2,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
        tok.train_from_iterator(texts, trainer)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tok.save(str(path))
        return Tokenizer(path)

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts):
        return [e.ids for e in self.tok.encode_batch(texts, add_special_tokens=False)]

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True)

    def encode_chat(self, turns):
        """Return (ids, train_mask). Only the bot's words are trained on,
        so the model learns to answer instead of learning to ask."""
        ids, mask = [], []
        for role, text in turns:
            piece = self.encode(text)
            if role == "user":
                ids += [self.user_id] + piece
                mask += [0] * (len(piece) + 1)
            else:
                ids += [self.bot_id] + piece + [self.end_id]
                mask += [0] + [1] * (len(piece) + 1)
        return ids, mask

    def chat_prompt(self, history, message):
        """Prompt for generating the next bot reply."""
        ids = []
        for role, text in history:
            ids += [self.user_id if role == "user" else self.bot_id] + self.encode(text)
            if role != "user":
                ids.append(self.end_id)
        return ids + [self.user_id] + self.encode(message) + [self.bot_id]
