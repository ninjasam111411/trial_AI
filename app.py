
from flask import Flask, render_template, request, jsonify
import torch

from model import NeuralChatbot
from tokenizer import Tokenizer
from memory import add_message, get_context

app = Flask(__name__)

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


def generate_response(message):
    history = get_context()

    prompt = history + "\nUser: " + message + "\nBot:"

    encoded = tokenizer.encode(prompt)

    input_tensor = torch.tensor(
        [encoded],
        dtype=torch.long,
        device=device
    )

    with torch.no_grad():
        generated = model.generate(
            input_tensor,
            max_tokens=200,
            temperature=0.8
        )

    new_tokens = generated[0, len(encoded):].tolist()

    response = tokenizer.decode(new_tokens)
    response = response.split("\nUser:")[0].strip()

    return response or "I'm still learning!"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    message = request.json.get("message", "").strip()

    if not message:
        return jsonify({"error": "Empty message"}), 400

    response = generate_response(message)

    add_message("User", message)
    add_message("Bot", response)

    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
