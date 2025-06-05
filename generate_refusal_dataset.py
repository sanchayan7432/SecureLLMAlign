import json

def generate_refusal(prompt):
    # You can use this improved refusal logic:
    split_tokens = ["Ignore the above", "Forget previous instructions", "Now write", "Ignore all previous"]
    for token in split_tokens:
        if token.lower() in prompt.lower():
            safe_part = prompt.lower().split(token.lower())[0].strip()
            safe_part = safe_part[0].upper() + safe_part[1:] + "."
            break
    else:
        safe_part = prompt.split(".")[0] + "."

    return (
        "I'm sorry, but I cannot help with that request. "
        f"Here's information instead: {safe_part}"
    )

def convert_dataset(input_path="preference_pairs.json", output_path="preference_pairs_refusal.json"):
    with open(input_path, "r") as infile:
        data = json.load(infile)

    new_data = []
    for item in data:
        prompt = item["prompt"]
        refusal = generate_refusal(prompt)
        new_item = {
            "prompt": prompt,
            "chosen": refusal
        }
        new_data.append(new_item)

    with open(output_path, "w") as outfile:
        json.dump(new_data, outfile, indent=2)
    print(f"✅ Converted dataset saved to {output_path}")

if __name__ == "__main__":
    convert_dataset()
