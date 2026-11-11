
import torch
from pathlib import Path

from model import NeuralChatbot
from tokenizer import Tokenizer
from memory import add_message, get_context

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load trained model
checkpoint = torch.load(
    "saved/chatbot.pt",
    map_location=device,
    weights_only=True
)

tokenizer = Tokenizer("".join(checkpoint["characters"]))

model = NeuralChatbot(
    vocab_size=tokenizer.vocab_size,
    context_size=checkpoint["context_size"]
).to(device)

model.load_state_dict(checkpoint["model"])
model.eval()

print("============================")
print("       PROJECT NEURAL")
print("============================")
print("Chatbot online. Type exit to quit.\n")


def respond(message):

    history = get_context()

    prompt = (
        history
        + "\nUser: " + message
        + "\nBot:"
    )

    encoded = tokenizer.encode(prompt)

    input_tensor = torch.tensor(
        [encoded],
        dtype=torch.long,
        device=device
    )

    with torch.no_grad():
        generated = model.generate(
            input_tensor,
            max_tokens=250,
            temperature=0.8
        )

    # Decode only newly generated characters
    new_tokens = generated[0, len(encoded):].tolist()

    response = tokenizer.decode(new_tokens)

    # Stop at the next conversation turn
    response = response.split("\nUser:")[0]
    response = response.strip()

    return response or "I don't know what to say yet."


while True:

    message = input("You: ").strip()

    if message.lower() in ["exit", "quit"]:
        print("Bot: Goodbye!")
        break

    if not message:
        continue

    add_message("User", message)

    answer = respond(message)

    print("Bot:", answer)

    add_message("Bot", answer)
