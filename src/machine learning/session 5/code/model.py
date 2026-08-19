import random
import json
from pathlib import Path

data_path = Path(__file__).parent / "responses.json"
if not data_path.exists():
    raise FileNotFoundError(f"responses.json not found next to {__file__}: {data_path}")

with data_path.open("r", encoding="utf-8") as file:
    responses = json.load(file)


def get_response(user_input):
    user_input = user_input.lower()
    for key in responses:
        if key in user_input:
            return random.choice(responses[key])
    return random.choice(responses.get("default", ["Sorry, I don't understand."]))

