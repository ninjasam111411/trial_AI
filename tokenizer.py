import string

EOS = 0  # end-of-response token
UNK = 1  # unknown character token

# A fixed alphabet (all normal keyboard characters) means the vocabulary
# never changes when new data is added, so the model can keep learning
# from where it left off instead of starting over.
DEFAULT_CHARACTERS = sorted(set(string.printable) - set("\t\r\x0b\x0c"))

_REPLACE = {"‘": "'", "’": "'", "“": '"', "”": '"',
            "–": "-", "—": "-", "…": "...", "\t": " "}


def clean(text):
    for a, b in _REPLACE.items():
        text = text.replace(a, b)
    return text


class Tokenizer:
    def __init__(self, text=None):
        self.characters = sorted(set(text)) if text else list(DEFAULT_CHARACTERS)

        # 0 = end of response, 1 = unknown character, real characters from 2.
        self.char_to_id = {c: i + 2 for i, c in enumerate(self.characters)}
        self.id_to_char = {i + 2: c for i, c in enumerate(self.characters)}
        self.vocab_size = len(self.characters) + 2

    def encode(self, text):
        return [self.char_to_id.get(c, UNK) for c in clean(text)]

    def decode(self, tokens):
        return "".join(self.id_to_char.get(t, "") for t in tokens if t > UNK)
