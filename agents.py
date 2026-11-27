"""
Neural's agents: one specialist per subject, plus Neural, the main agent,
who decides which specialists to call for each question.

Each agent owns some of the knowledge (by topic pack or topic tag). When you
ask something, Neural asks every agent how well it can answer, picks the best,
and asks you a quick question if two agents both think they know.

To add an agent: add an entry to AGENTS below with the packs/tags it owns.
"""

AGENTS = [
    {"id": "neural", "name": "Neural", "emoji": "🧠", "color": "#7c5cff",
     "about": "The main agent. Reads your question, calls the right specialists, and writes when no one knows."},
    {"id": "english", "name": "English", "emoji": "📚", "color": "#f472b6",
     "about": "Grammar, writing, essays, reading skills, poetry and literature.",
     "packs": ["english_grammar", "english_reading_writing", "literature", "language"],
     "tags": ["books"]},
    {"id": "history", "name": "History", "emoji": "🏛️", "color": "#f59e0b",
     "about": "US and world history, famous people, wars, inventions and civics.",
     "packs": ["history", "history_us", "history_world", "civics"],
     "tags": ["first_president", "pyramids", "moon_landing", "light_bulb", "telephone",
              "airplane", "einstein", "newton"]},
    {"id": "science", "name": "Science", "emoji": "🔬", "color": "#22d3ee",
     "about": "Physics, chemistry, biology, space, animals, nature and the human body.",
     "packs": ["science", "space_nature", "health"],
     "tags": ["sky_blue", "sunset_colors", "rainbow", "gravity", "water", "water_boil", "atom",
              "photosynthesis", "electricity", "light_speed", "sound", "dna", "cells", "volcano",
              "earthquake", "dinosaurs", "sun", "moon", "planets", "biggest_planet", "pluto",
              "mars", "black_hole", "galaxy", "stars", "earth", "seasons", "dog", "cat",
              "fastest_animal", "biggest_animal", "tallest_animal", "shark", "octopus",
              "penguin", "bees", "healthy", "sleep", "heart", "brain", "bones"]},
    {"id": "geography", "name": "Geography", "emoji": "🌍", "color": "#34d399",
     "about": "Countries, capitals, states, continents, oceans and landmarks.",
     "packs": ["capitals"],
     "tags": ["continents", "oceans", "tallest_mountain", "longest_river", "biggest_country",
              "capital_", "us_states"]},
    {"id": "math", "name": "Math", "emoji": "🧮", "color": "#a3e635",
     "about": "Arithmetic (exact, with a calculator), math ideas and how-tos.",
     "packs": ["math"], "tags": ["math_"]},
    {"id": "tech", "name": "Tech", "emoji": "💻", "color": "#60a5fa",
     "about": "Computers, coding (Python, Lua, web), AI and staying safe online.",
     "packs": ["tech_coding"],
     "tags": ["computers", "cpu", "gpu", "ram", "binary", "internet_what", "programming",
              "python", "hello_world", "variable", "loop", "function", "bug", "ai", "ml",
              "neural_network", "transformer", "training", "chatgpt", "robots", "minecraft",
              "roblox"]},
    {"id": "life", "name": "Life", "emoji": "🌱", "color": "#fb923c",
     "about": "Study tips, habits, cooking, money, feelings and everyday advice.",
     "packs": ["everyday", "arts"],
     "tags": ["school", "user_bad", "user_bored", "user_tired", "user_angry", "user_scared",
              "chess", "sports", "soccer", "music", "movies", "meaning_of_life"]},
    {"id": "friend", "name": "Friend", "emoji": "💬", "color": "#c084fc",
     "about": "Small talk, jokes, riddles, stories, and questions about Neural itself.",
     "packs": ["smalltalk", "intents"]},  # anything else in intents.json lands here
    {"id": "memory", "name": "Memory", "emoji": "💡", "color": "#fbbf24",
     "about": "Everything people have taught Neural in chat."},
    {"id": "files", "name": "Files", "emoji": "📎", "color": "#94a3b8",
     "about": "Reads files you attach with ➕ and answers questions about them."},
]
BY_ID = {a["id"]: a for a in AGENTS}


def agent_for(intent):
    """Which agent owns this topic."""
    tag, pack = intent["tag"], intent.get("_pack", "")
    if tag.startswith("taught_"):
        return "memory"
    for agent in AGENTS:
        for t in agent.get("tags", []):
            if (t.endswith("_") and tag.startswith(t)) or tag == t:
                return agent["id"]
    for agent in AGENTS:
        if pack in agent.get("packs", []):
            return agent["id"]
    return "friend"


def public_list(counts):
    """Agents for the app's graph view, with how many topics each knows."""
    return [{**{k: a[k] for k in ("id", "name", "emoji", "color", "about")},
             "topics": counts.get(a["id"], 0)} for a in AGENTS]
