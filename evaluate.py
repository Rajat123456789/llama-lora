"""
Evaluate the LoRA-fine-tuned model on a test set.

Computes perplexity and loss on held-out C4 data.
"""

import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from prepare_data import load_c4_subset
from tqdm import tqdm

from config import MODEL_ID, OUTPUT_DIR, MAX_SEQ_LENGTH, get_device, configure_mps


def evaluate_model(adapter_path: str = None, num_test_samples: int = 200):
    """Evaluate model on test set and compute perplexity."""
    adapter_path = adapter_path or OUTPUT_DIR
    configure_mps()
    device = get_device()
    
    print(f"Loading model from {adapter_path}...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
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
    
    print(f"\nLoading {num_test_samples} test samples from C4...")
    # Use different samples than training (skip first 2000)
    from datasets import load_dataset
    stream = load_dataset(
        "allenai/c4",
        "en",
        split="validation",  # Use validation split for testing
        streaming=True,
    )
    
    test_texts = []
    for i, item in enumerate(stream):
        if i >= num_test_samples:
            break
        text = item.get("text", "").strip()[:512]
        if text:
            test_texts.append(text)
    
    print(f"Evaluating on {len(test_texts)} samples...")
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for text in tqdm(test_texts, desc="Evaluating"):
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=MAX_SEQ_LENGTH,
                truncation=True,
            )
            if device != "cpu":
                inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Forward pass
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            total_loss += loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Test samples:     {len(test_texts)}")
    print(f"Total tokens:     {total_tokens:,}")
    print(f"Average loss:     {avg_loss:.4f}")
    print(f"Perplexity:       {perplexity:.4f}")
    print("="*60)
    
    return {
        "test_samples": len(test_texts),
        "total_tokens": total_tokens,
        "avg_loss": avg_loss,
        "perplexity": perplexity,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", type=str, default=OUTPUT_DIR)
    parser.add_argument("--num-test-samples", type=int, default=200)
    args = parser.parse_args()
    
    evaluate_model(args.adapter_path, args.num_test_samples)
