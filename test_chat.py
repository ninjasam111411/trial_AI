
import torch
from model import NeuralChatbot
from tokenizer import Tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

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

print("Project Neural is ready! Type 'quit' to stop.")

while True:
    message = input("\nYou: ")

    if message.lower() == "quit":
        break

    encoded = tokenizer.encode(message + "\nBot:")
    x = torch.tensor([encoded], dtype=torch.long, device=device)

    with torch.no_grad():
        output = model.generate(x, max_tokens=150, temperature=0.8)

    answer = tokenizer.decode(output[0, len(encoded):].tolist())
    print("Neural:", answer.split("\n")[0])
