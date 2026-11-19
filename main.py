from brain import Brain
from memory import add_message

brain = Brain()

print("============================")
print("       PROJECT NEURAL")
print("============================")
print("Chatbot online. Type exit to quit.")
print("Tip: type /debug to see where each answer comes from.\n")

debug = False

while True:
    message = input("You: ").strip()

    if message.lower() in ["exit", "quit"]:
        print("Bot: Goodbye!")
        break

    if not message:
        continue

    if message == "/debug":
        debug = not debug
        print(f"(debug {'on' if debug else 'off'})")
        continue

    # Answer first, then save, so the message isn't in the prompt twice.
    answer, source = brain.respond(message)
    print("Bot:", answer + (f"  [{source}]" if debug else ""))

    add_message("User", message)
    add_message("Bot", answer)
