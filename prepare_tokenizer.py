from transformers import AutoTokenizer

model_name = "gpt2"  # or your base pretrained model name
save_dir = "/data1/SANCHAYANghosh01/SecureLLMAlign/tuned_model"  # replace with your actual model directory

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(save_dir)

print(f"Tokenizer saved to {save_dir}")
