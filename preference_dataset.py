import json

def create_preference_pairs(data):
    preference_pairs = []
    for item in data:
        preference_pairs.append({
            "prompt": item["injected_prompt"],
            "chosen": item["secure_output"],
            "rejected": item["insecure_output"]
        })
    return preference_pairs

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    data = load_json("evaluation_data.json")
    preference_pairs = create_preference_pairs(data)
    save_json(preference_pairs, "preference_pairs.json")
