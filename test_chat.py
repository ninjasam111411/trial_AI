"""
Quick check of the chatbot. Runs a set of questions and shows each answer
and where it came from (calculator, knowledge, neural).

    python test_chat.py            # run the built-in questions
    python test_chat.py -i         # interactive, without saving memory
"""

import sys

from brain import Brain

QUESTIONS = [
    "hi",
    "whats your name",
    "why is the sky blue",
    "tell me a joke",
    "how many planets are there",
    "what is 12 * 7",
    "who invented the telephone",
    "what is a neural network",
    "I like turtles",
    "what should I do this weekend?",
    "bye",
]

brain = Brain()

if "-i" in sys.argv:
    print("Project Neural test mode. Type 'quit' to stop.")
    while True:
        message = input("\nYou: ").strip()
        if message.lower() == "quit":
            break
        answer, source = brain.respond(message, use_history=False)
        print(f"Neural [{source}]: {answer}")
else:
    for q in QUESTIONS:
        answer, source = brain.respond(q, use_history=False)
        print(f"You: {q}\nBot [{source}]: {answer}\n")
