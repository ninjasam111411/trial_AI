EOS = 0  # end-of-response token
UNK = 1  # unknown character token


class Tokenizer:
    def __init__(self, text):
        self.characters = sorted(set(text))

        # 0 = end of response, 1 = unknown character.
        # Real characters start at 2, so a character the model has never
        # seen no longer gets mistaken for "stop talking".
        self.char_to_id = {
            char: i + 2
            for i, char in enumerate(self.characters)
        }

        self.id_to_char = {
            i + 2: char
            for i, char in enumerate(self.characters)
        }

        self.vocab_size = len(self.characters) + 2

    def encode(self, text):
        return [
            self.char_to_id.get(char, UNK)
            for char in text
        ]

    def decode(self, tokens):
        return "".join(
            self.id_to_char.get(token, "")
            for token in tokens
            if token > UNK
        )
