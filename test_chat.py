"""
Quick check of the chatbot. Runs a set of questions and shows each answer
and where it came from (spelling, calculator, knowledge, neural).
Note: this does not save anything to Neural's learning data.

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
    "waht is teh biggest planit",
    "how do you spell recieve",
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
        r = brain.respond(message, learn=False)
        if r["corrected"]:
            print(f"  (read as: {r['corrected']})")
        print(f"Neural [{r['source']}]: {r['answer']}")
else:
    for q in QUESTIONS:
        r = brain.respond(q, learn=False)
        read_as = f"  (read as: {r['corrected']})" if r["corrected"] else ""
        print(f"You: {q}{read_as}\nBot [{r['source']}]: {r['answer']}\n")
