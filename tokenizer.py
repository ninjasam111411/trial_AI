
class Tokenizer:
    def __init__(self, text):
        self.characters = sorted(set(text))

        # Zero is reserved for the end-of-response token.
        self.char_to_id = {
            char: i + 1
            for i, char in enumerate(self.characters)
        }

        self.id_to_char = {
            i + 1: char
            for i, char in enumerate(self.characters)
        }

        self.vocab_size = len(self.characters) + 1

    def encode(self, text):
        return [
            self.char_to_id.get(char, 0)
            for char in text
        ]

    def decode(self, tokens):
        return "".join(
            self.id_to_char.get(token, "")
            for token in tokens
            if token != 0
        )
