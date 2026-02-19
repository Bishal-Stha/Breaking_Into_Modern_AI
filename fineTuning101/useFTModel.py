"""
useFTModel.py

Simple helper to load your fine-tuned model from ./my_special_model
and run interactive generation while automatically handling tokenizer
pad token, context window (truncation), device placement and generation
lengths.

Run: python useFTModel.py
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_dir="./my_special_model"):
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)

    # Ensure tokenizer has a pad token (GPT-2 usually doesn't)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine context window (max tokens model accepts)
    max_len = None
    if hasattr(tokenizer, "model_max_length") and tokenizer.model_max_length and tokenizer.model_max_length < 1e9:
        max_len = int(tokenizer.model_max_length)
    elif hasattr(model.config, "max_position_embeddings"):
        max_len = int(model.config.max_position_embeddings)
    elif hasattr(model.config, "n_ctx"):
        max_len = int(model.config.n_ctx)
    else:
        max_len = 1024

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device) # type: ignore
    model.eval()

    return model, tokenizer, device, max_len


def generate_from_prompt(prompt, model, tokenizer, device, max_len, max_new_tokens_default=200):
    if not prompt or not prompt.strip():
        return ""

    # Tokenize without truncation so we can handle it manually
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"]

    input_len = input_ids.shape[1]

    # If the prompt is longer than model context, keep the last tokens (most recent context)
    if input_len >= max_len:
        keep = max_len - 1
        input_ids = input_ids[:, -keep:]
        input_len = input_ids.shape[1]

    # How many new tokens can we generate while staying within model's window
    allowed_new = max_len - input_len
    if allowed_new <= 0:
        allowed_new = 50
    allowed_new = min(allowed_new, max_new_tokens_default)
    if allowed_new < 1:
        allowed_new = 1

    input_ids = input_ids.to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=allowed_new,
            do_sample=True,
            top_p=0.95,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id,
        )

    # We want only the newly generated portion appended to the prompt
    gen_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return gen_text


def main():
    model_dir = "./my_special_model"
    try:
        model, tokenizer, device, max_len = load_model(model_dir)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    print(f"Loaded model from {model_dir} on device {device}. Context window: {max_len} tokens.")
    print("Type a prompt and press Enter. Type 'exit' or 'quit' to stop.")

    while True:
        try:
            prompt = input("Prompt: ")
        except (EOFError, KeyboardInterrupt):
            print()  # newline
            break

        if prompt.strip().lower() in ("exit", "quit"):
            break

        out = generate_from_prompt(prompt, model, tokenizer, device, max_len)
        # If tokenizer decodes both prompt+continuation, try to show only continuation
        if out.startswith(prompt):
            continuation = out[len(prompt):].strip()
            print("\n=== Generated continuation ===")
            print(continuation)
            print("=== End ===\n")
        else:
            print("\n=== Generated text ===")
            print(out)
            print("=== End ===\n")


if __name__ == "__main__":
    main()
