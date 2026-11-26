"""
Makes Neural's answers sound like a person talking, not a textbook:
contractions, casual openers, and the odd follow-up question.

    humanize("It is the capital of France.", "fact")
    -> "So basically, it's the capital of France."

Used when answering (if the casual style is on) and when building the
training data, so the neural network learns to write this way too.
"""

import random
import re

CONTRACTIONS = [
    ("I am", "I'm"), ("I have", "I've"), ("I will", "I'll"), ("I would", "I'd"),
    ("you are", "you're"), ("you have", "you've"), ("you will", "you'll"),
    ("we are", "we're"), ("they are", "they're"), ("it is", "it's"),
    ("that is", "that's"), ("there is", "there's"), ("here is", "here's"),
    ("what is", "what's"), ("who is", "who's"), ("he is", "he's"), ("she is", "she's"),
    ("do not", "don't"), ("does not", "doesn't"), ("did not", "didn't"),
    ("is not", "isn't"), ("are not", "aren't"), ("was not", "wasn't"),
    ("were not", "weren't"), ("cannot", "can't"), ("can not", "can't"),
    ("could not", "couldn't"), ("would not", "wouldn't"), ("should not", "shouldn't"),
    ("will not", "won't"), ("have not", "haven't"), ("has not", "hasn't"),
    ("let us", "let's"),
]
_PATTERNS = [(re.compile(rf"\b{a}\b", re.IGNORECASE), b) for a, b in CONTRACTIONS]

OPENERS = ["So basically, ", "Okay so, ", "Short version: ", "Good question! ", "Ooh, fun one. ",
           "Here's the deal: ", "Alright, ", "So, "]
FOLLOW_UPS = {
    "fact": ["Want an example?", "Want me to go deeper on that?", "Pretty interesting, right?",
             "Make sense?", "Want to know more?", "Need it explained another way?"],
    "chat": [],
}
# Words that can be lowercased after an opener ("So, the ..."); anything
# else (names, places) keeps its capital letter.
LOWER_OK = {"the", "a", "an", "it", "it's", "there", "there's", "this", "that", "that's",
            "they", "they're", "these", "those", "use", "most", "some", "in", "on", "at",
            "when", "if", "each", "every", "both", "one", "many", "start", "make", "put",
            "think", "try", "pick", "write", "restate", "take", "lots", "all"}


def contract(text):
    for pat, short in _PATTERNS:
        def repl(m, short=short):
            word = m.group()
            return short[0].upper() + short[1:] if word[0].isupper() else short
        text = pat.sub(repl, text)
    return text


def humanize(text, kind="fact", rng=random):
    """kind: 'fact' for knowledge answers, 'chat' for small talk."""
    if not text or "```" in text or "\n- " in text:
        return contract(text)  # leave code and lists tidy
    text = contract(text)
    if kind != "fact":
        return text

    if rng.random() < 0.45:
        first = text.split(" ", 1)[0].lower().strip(",.:")
        if first in LOWER_OK:
            opener = rng.choice(OPENERS)
            if opener.endswith((", ", ": ")):
                text = text[0].lower() + text[1:]
        else:
            # Names and other capitalized words: use an opener that ends a sentence.
            opener = rng.choice([o for o in OPENERS if o.endswith(("! ", ". "))])
        text = opener + text
    if FOLLOW_UPS[kind] and rng.random() < 0.25 and not text.endswith("?"):
        text += " " + rng.choice(FOLLOW_UPS[kind])
    return text


# Topics that are small talk get contractions only, no "So basically,".
CHAT_TAGS = ("greeting", "good_", "how_are_you", "how_smart", "user_", "thanks", "sorry",
             "goodbye", "yes", "no", "compliment", "insult", "joke", "riddle", "test",
             "repeat", "dont_understand", "name", "what_are_you", "capabilities",
             "who_made_you", "how_do_you_work", "can_learn", "age", "where_live", "feelings",
             "favorite_", "weather", "time", "internet", "chatgpt", "secret", "story",
             "taught_")
CHAT_PACKS = {"smalltalk"}


def kind_for(intent):
    if intent.get("_pack") in CHAT_PACKS or intent["tag"].startswith(CHAT_TAGS):
        return "chat"
    return "fact"
