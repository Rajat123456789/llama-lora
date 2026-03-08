"""
LoRA fine-tuning of Llama 3.2 1B on C4 (clean) for interview demo.

Uses PEFT (LoRA) + TRL SFTTrainer. C4 is used as continuous text for
causal language modeling (next-token prediction).
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
from prepare_data import load_c4_subset

from config import (
    MODEL_ID,
    OUTPUT_DIR,
    MAX_SEQ_LENGTH,
    BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    NUM_EPOCHS,
    LEARNING_RATE,
    WARMUP_RATIO,
    LOGGING_STEPS,
    SAVE_STRATEGY,
    BF16,
    FP16,
    LOAD_IN_4BIT,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_TARGET_MODULES,
    get_device,
    configure_mps,
)

# Memory tracking
from memory_tracking import MemoryTrackingCallback

# Optional: Low-rank optimizer for additional memory savings
try:
    from config import USE_LOWRANK_OPTIMIZER, LOWRANK_OPTIMIZER_RANK, LOWRANK_PROJECTION_FREQ
    from lowrank_optimizer import LowRankAdamW
except ImportError:
    USE_LOWRANK_OPTIMIZER = False


def main():
    # Configure PyTorch for MPS if available
    mps_configured = configure_mps()
    
    # Device: cuda > mps (Apple GPU) > cpu
    device = get_device()
    print(f"Using device: {device}")
    if mps_configured:
        print("Apple Silicon (MPS) backend configured with CPU fallback enabled")

    # Load tokenizer
    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )
    # Ensure pad token for training
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantization config (saves VRAM)
    bnb_config = None
    if LOAD_IN_4BIT and device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if BF16 else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    # Load base model
    print(f"Loading model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.bfloat16 if BF16 else torch.float16,
        trust_remote_code=True,
    )
    if device == "mps":
        model = model.to("mps")

    if LOAD_IN_4BIT and device == "cuda":
        model = prepare_model_for_kbit_training(model)

    # LoRA
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # C4 subset
    train_dataset = load_c4_subset()

    # SFT config (TRL 0.29+ uses SFTConfig and processing_class)
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=LOGGING_STEPS,
        save_strategy=SAVE_STRATEGY,
        bf16=BF16,
        fp16=FP16,
        report_to="none",
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        packing=False,
        dataloader_pin_memory=(device != "mps"),  # MPS doesn't support pin_memory
    )

    # Optional: Use low-rank optimizer for additional memory savings
    optimizers = (None, None)  # Default: let Trainer create optimizer
    if USE_LOWRANK_OPTIMIZER:
        print(f"Using LowRankAdamW optimizer (rank={LOWRANK_OPTIMIZER_RANK})")
        optimizer = LowRankAdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=LEARNING_RATE,
            rank=LOWRANK_OPTIMIZER_RANK,
            projection_update_freq=LOWRANK_PROJECTION_FREQ,
        )
        optimizers = (optimizer, None)

    # Memory tracking callback
    memory_callback = MemoryTrackingCallback(device=device, log_every_n_steps=10)

    # SFTTrainer with text field (causal LM on C4)
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        optimizers=optimizers,
        callbacks=[memory_callback],
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved LoRA adapter and tokenizer to {OUTPUT_DIR}")
    
    # Generate memory visualization
    print("\nGenerating memory usage plot...")
    from memory_tracking import visualize_memory_tracking
    memory_json_path = os.path.join(OUTPUT_DIR, "memory_tracking.json")
    if os.path.exists(memory_json_path):
        visualize_memory_tracking(memory_json_path)


if __name__ == "__main__":
    main()

