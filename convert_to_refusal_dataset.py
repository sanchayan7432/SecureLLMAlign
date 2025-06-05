import json

def generate_refusal(prompt):
    # Extract only the safe portion (before the injection)
    safe_part = prompt.split(".")[0] + "."  # crude safe prompt end
    return (
        "I'm sorry, but I cannot help with that request. "
        f"Here's information instead: {safe_part.strip()}"
    )

def convert_dataset(input_path="preference_pairs.json", output_path="preference_pairs_refusal.json"):
    with open(input_path, "r") as infile:
        data = json.load(infile)

    new_data = []
    for item in data:
        prompt = item["prompt"]
        new_item = {
            "prompt": prompt,
            "chosen": generate_refusal(prompt)
        }
        new_data.append(new_item)

    with open(output_path, "w") as outfile:
        json.dump(new_data, outfile, indent=2)
    print(f"✅ Converted dataset saved to {output_path}")

if __name__ == "__main__":
    convert_dataset()
