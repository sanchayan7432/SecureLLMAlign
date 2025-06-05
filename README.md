# SecureLLMAlign

SecureLLMAlign enhances the robustness of Large Language Models (LLMs) against prompt injection attacks using preference optimization techniques.

## Modules

- `data_preparation.py`: Generates prompt-injected data with corresponding secure and insecure outputs.
- `preference_dataset.py`: Constructs preference pairs indicating preferred (secure) outputs.
- `fine_tuning.py`: Fine-tunes the LLM using the preference dataset.
- `evaluation.py`: Evaluates the fine-tuned model's robustness against prompt injections.

## Usage

1. Prepare base prompts and injection phrases in `base_prompts.json` and `injections.json`.
2. Run `data_preparation.py` to generate injected data.
3. Use `preference_dataset.py` to create preference pairs.
4. Fine-tune the model with `fine_tuning.py`.
5. Evaluate the model using `evaluation.py`.

## Requirements

- Python 3.10+
- Transformers Library
- PyTorch

## License

NONE

## Run commands
>>python data_preparation.py
👉 Output: evaluation_data.json

>>python preference_dataset.py
👉 Output: preference_pairs.json

>>python prepare_tokenizer.py
>>python convert_to_refusal_dataset.py
>>python generate_secure_data.py
>>python fine_tuning.py
✅ Output saved to: tuned_model/

>>python evaluation.py
✅ It prints the accuracy like: Accuracy on prompt-injected inputs: 91.67%.


for further query please contact to sanchayan7432@gmail.com
