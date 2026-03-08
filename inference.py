"""
Run inference with the LoRA-fine-tuned Llama 3.2 1B (C4).

Usage:
  python inference.py "Your prompt here"
  python inference.py --prompt "The weather today is"
"""

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from config import MODEL_ID, OUTPUT_DIR, get_device, configure_mps


def load_model_and_tokenizer(adapter_path: str = None):
    adapter_path = adapter_path or OUTPUT_DIR
    # Resolve to absolute path so Hugging Face loads from disk, not hub
    adapter_path = os.path.abspath(adapter_path)
    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(
            f"Adapter path not found: {adapter_path}. Run train.py first to save the LoRA adapter."
        )
    
    # Configure MPS if available
    configure_mps()
    
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    device = get_device()
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device == "mps":
        base = base.to("mps")
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model, tokenizer, device


def generate(prompt: str, max_new_tokens: int = 128, adapter_path: str = None):
    model, tokenizer, device = load_model_and_tokenizer(adapter_path)
    inputs = tokenizer(prompt, return_tensors="pt")
    if device != "cpu":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--adapter-path", 
        type=str, 
        default=OUTPUT_DIR,
        help="Path to LoRA adapter. Use 'output/llama32-1b-lora-c4' for standard LoRA or 'output/llama32-1b-lora-galore-c4' for GaLore"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["lora", "galore"],
        help="Shortcut: 'lora' loads standard LoRA model, 'galore' loads GaLore-optimized model"
    )
    args = parser.parse_args()
    
    # Handle model-type shortcut
    if args.model_type:
        if args.model_type == "lora":
            args.adapter_path = "output/llama32-1b-lora-c4"
            print(f"Loading standard LoRA model from: {args.adapter_path}")
        elif args.model_type == "galore":
            args.adapter_path = "output/llama32-1b-lora-galore-c4"
            print(f"Loading GaLore-optimized model from: {args.adapter_path}")
    
    print(generate(args.prompt, args.max_new_tokens, args.adapter_path))
