import json
import random

def mock_model_generate(prompt):
    # Simulated model generation logic
    return f"Secure response for: {prompt.strip()}"

def generate_prompt_injection_data(base_prompts, injections):
    data = []
    for prompt in base_prompts:
        for injection in injections:
            injected_prompt = f"{prompt.strip()} {injection.strip()}"
            secure_output = mock_model_generate(prompt)
            insecure_output = mock_model_generate(injected_prompt)
            data.append({
                "injected_prompt": injected_prompt,
                "secure_output": secure_output,
                "insecure_output": insecure_output
            })
    return data

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    base_prompts = load_json("base_prompts.json")
    injections = load_json("injections.json")
    data = generate_prompt_injection_data(base_prompts, injections)
    save_json(data, "evaluation_data.json")