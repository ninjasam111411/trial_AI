
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from model import NeuralChatbot
from tokenizer import Tokenizer

# Settings
EPOCHS = 1000
BATCH_SIZE = 16
CONTEXT_SIZE = 128
LEARNING_RATE = 0.001

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load training text
text = Path("data/training.txt").read_text(
    encoding="utf-8"
)

tokenizer = Tokenizer(text)
data = torch.tensor(
    tokenizer.encode(text),
    dtype=torch.long
)

if len(data) <= CONTEXT_SIZE:
    raise ValueError("Add more training text!")

# Create model
model = NeuralChatbot(
    vocab_size=tokenizer.vocab_size,
    context_size=CONTEXT_SIZE
).to(device)

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

loss_function = nn.CrossEntropyLoss()

# Training loop
for step in range(EPOCHS):

    starts = torch.randint(
        0,
        len(data) - CONTEXT_SIZE - 1,
        (BATCH_SIZE,)
    )

    x = torch.stack([
        data[i:i + CONTEXT_SIZE]
        for i in starts
    ]).to(device)

    y = torch.stack([
        data[i + 1:i + CONTEXT_SIZE + 1]
        for i in starts
    ]).to(device)

    predictions = model(x)

    loss = loss_function(
        predictions.reshape(-1, tokenizer.vocab_size),
        y.reshape(-1)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(
            f"Step {step}/{EPOCHS} | Loss: {loss.item():.4f}"
        )

# Save everything
Path("saved").mkdir(exist_ok=True)

torch.save({
    "model": model.state_dict(),
    "characters": tokenizer.characters,
    "context_size": CONTEXT_SIZE
}, "saved/chatbot.pt")

print("Training complete!")
print("Model saved!")
