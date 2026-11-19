from flask import Flask, render_template, request, jsonify

from brain import Brain
from memory import add_message, save_memory

app = Flask(__name__)
brain = Brain()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({"error": "Please enter a message."}), 400
    if len(message) > 500:
        return jsonify({"error": "Message is too long (500 characters max)."}), 400

    response, source = brain.respond(message)

    add_message("User", message)
    add_message("Bot", response)

    return jsonify({"response": response, "source": source})


@app.route("/reset", methods=["POST"])
def reset():
    save_memory([])
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
