import json
import re
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

DEFAULT_REFUSAL = "I'm sorry, but I can't help with that request."

def sanitize_prompt(prompt):
    """Removes common jailbreak phrases from prompts using regex (case-insensitive)."""
    banned_phrases = [
        "ignore previous", "disregard above", "forget earlier", "now write",
        "override instructions", "forget what i said", "please pretend",
        "please act as", "disregard the above", "ignore all previous instructions"
    ]
    prompt_clean = prompt
    for phrase in banned_phrases:
        prompt_clean = re.sub(re.escape(phrase), "", prompt_clean, flags=re.IGNORECASE)
    return prompt_clean.strip()

def load_json(path):
    """Loads JSON data from the specified file."""
    with open(path, "r") as f:
        return json.load(f)

class PreferenceDataset(Dataset):
    """Custom dataset for instruction preference fine-tuning."""
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = sanitize_prompt(item.get("prompt", ""))
        chosen = item.get("chosen", "").strip() or DEFAULT_REFUSAL
        prompt_with_nl = prompt + "\n"
        full_text = prompt_with_nl + chosen

        tokens = self.tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = tokens["input_ids"].squeeze()
        attention_mask = tokens["attention_mask"].squeeze()
        labels = input_ids.clone()

        # Mask prompt portion in loss
        prompt_len = self.tokenizer(
            prompt_with_nl,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )["input_ids"].size(1)
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

def fine_tune(model_name="gpt2", data_path="preference_pairs_refusal.json", output_dir="tuned_model"):
    """Fine-tunes a causal language model on refusal-preference pairs."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Ensure pad token is defined and configured
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    data = load_json(data_path)
    dataset = PreferenceDataset(data, tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=70,
        per_device_train_batch_size=2,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    fine_tune()