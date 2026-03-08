# LoRA Fine-Tuning of Llama 3.2 1B on C4 (Clean)

A small, self-contained project that fine-tunes **Llama 3.2 1B** with **LoRA** (Low-Rank Adaptation) on a subset of the **C4 clean English** dataset, suitable to run locally.

## What This Project Demonstrates

- **LoRA (PEFT)**: Parameter-efficient fine-tuning so only a small set of adapter weights is trained (~0.1% of parameters), reducing memory and time.
- **C4 dataset (clean)**: Uses the **cleaned English** variant of C4 (`allenai/c4`, config `en`):
  - Built from Common Crawl (April 2019).
  - Filtered, deduplicated, and designed for pretraining/fine-tuning language models.
  - We use a **small subset** (e.g. 2000 samples) to avoid long preprocessing and to keep the demo fast.
- **Local run**: Scripts are set up to run on a single machine (GPU recommended; 4-bit quantization helps on smaller VRAM).

## Project Structure

```
llama-lora/
├── config.py         # All hyperparameters and paths (easy to tweak)
├── prepare_data.py   # Load C4 clean (en), take small subset
├── train.py          # LoRA fine-tuning with PEFT + TRL
├── inference.py      # Generate text with the fine-tuned adapter
├── requirements.txt
└── README.md
```

## Setup

1. **Create a virtual environment** (recommended):
  ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   # or: .venv\Scripts\activate  # Windows
  ```
2. **Install dependencies**:
  ```bash
   pip install -r requirements.txt
  ```
3. **Hugging Face (for Llama 3.2)**
  Llama 3.2 is gated. Log in so the scripts can download the model:
   Accept the [Llama 3.2 terms](https://huggingface.co/meta-llama/Llama-3.2-1B) on the model page if required.

## Quick Start

- **Train** (loads C4 clean, takes a small subset, runs LoRA training):
  ```bash
  python train.py
  ```
- **Inference** (uses the saved LoRA adapter in `output/llama32-1b-lora-c4` by default):
  ```bash
  python inference.py --prompt "The capital of France is"
  ```

## Training Results

Trained successfully on **Apple Silicon (MPS)**:


| Metric                   | Value                       |
| ------------------------ | --------------------------- |
| **Total Training Time**  | 18 minutes 55 seconds       |
| **Device**               | Apple Silicon (MPS backend) |
| **Training Samples**     | 2,000                       |
| **Total Steps**          | 125                         |
| **Final Loss**           | 2.875                       |
| **Mean Token Accuracy**  | 45.76%                      |
| **Trainable Parameters** | 5,636,096 (0.45% of 1.24B)  |
| **Throughput**           | ~9.09 seconds/step          |
| **Samples per Second**   | 1.761                       |


### Training Progress

- Starting loss: ~3.2
- Final loss: 2.875
- Loss reduction demonstrates successful learning on C4 text continuation

### Key Achievements

✅ Successfully trained a 1.2B parameter model using only **0.45%** trainable params (LoRA)  
✅ Leveraged **Apple Silicon GPU** (MPS) for ~19 min training  
✅ Used **C4 clean** dataset (standard pretraining corpus)  
✅ Full reproducible pipeline: data loading → training → inference  

## Configuration (`config.py`)

- **Dataset**: `C4_NUM_SAMPLES` (default 2000), `C4_MAX_TEXT_LENGTH` (default 512). Increase for a longer run.
- **LoRA**: `LORA_R = 8`, `LORA_ALPHA = 16`, `LORA_TARGET_MODULES` (7 attention/FFN layers).
- **Training**: `BATCH_SIZE = 2`, `GRADIENT_ACCUMULATION_STEPS = 8`, `NUM_EPOCHS = 1`, `LEARNING_RATE = 2e-5`, `MAX_SEQ_LENGTH = 256`.
- **Memory**: `LOAD_IN_4BIT = True` reduces VRAM; set to `False` if you have enough GPU memory.
- **Apple Silicon**: Automatically uses MPS backend with CPU fallback enabled for unsupported ops.

## C4 Dataset (Clean)

- **Source**: Common Crawl (April 2019).
- **Variant used**: `allenai/c4` with config `**en`** = cleaned English (filtered, deduplicated).
- **Purpose**: Pretraining and fine-tuning language models and word representations.
- **Why “clean”**: The full C4 pipeline does heavy filtering and deduplication; the `en` config is the cleaned version so we avoid implementing that preprocessing ourselves.



1. **Why LoRA?** Fewer trainable parameters (0.45% vs 100%), less memory, faster training, easier to swap adapters for different tasks.
2. **Why C4 clean?** Standard, large-scale text corpus already cleaned; good for demonstrating pretraining-style continuation without building a custom pipeline.
3. **Why a small subset?** Keeps the demo short and runnable locally (~19 min on Apple Silicon) while still showing the full pipeline (data → training → inference).
4. **Apple Silicon optimization**: Utilized MPS (Metal Performance Shaders) backend for GPU acceleration on Mac, with automatic CPU fallback for unsupported operations.
5. **Real-world applicability**: This approach scales to larger datasets and can be used for domain adaptation, instruction tuning, or task-specific fine-tuning with minimal compute.

## License and Attribution

- **Llama 3.2**: Meta’s license applies; ensure you comply with [Llama 3.2 terms](https://huggingface.co/meta-llama/Llama-3.2-1B).
- **C4**: [Allen AI / Hugging Face](https://huggingface.co/datasets/allenai/c4); Common Crawl terms apply to the underlying content.

