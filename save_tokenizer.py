from transformers import AutoTokenizer

# Use the base model you fine-tuned, e.g., "gpt2"
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.save_pretrained("tuned_model")
