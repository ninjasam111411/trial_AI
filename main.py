"""Chat with Neural in the terminal.  (For the full app, run: python app.py)"""

import memory
from brain import Brain

brain = Brain()
conversation = memory.create_conversation()

print("============================")
print("       PROJECT NEURAL")
print("============================")
print("Chatbot online. Type exit to quit.")
print("Commands: /debug (show answer sources), /good, /bad, /teach <answer>\n")

debug = False
last_id = None
last_question = None

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
    if message in ("/good", "/bad") and last_id:
        brain.chat_log.rate(last_id, message[1:])
        print("(thanks for the feedback!)")
        continue
    if message.startswith("/teach ") and last_question:
        brain.teach(last_question, message[7:])
        print("(learned it!)")
        continue

    result = brain.respond(message, memory.history(conversation))
    if result["corrected"] and debug:
        print(f"  (read as: {result['corrected']})")
    print("Bot:", result["answer"] + (f"  [{result['source']}]" if debug else ""))

    last_id = brain.chat_log.add(conversation, message, result["answer"],
                                 result["source"], result["corrected"])
    last_question = message
    memory.add_message(conversation, "User", message)
    memory.add_message(conversation, "Bot", result["answer"])
