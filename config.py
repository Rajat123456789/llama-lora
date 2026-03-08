import torch


def configure_mps():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():

        import os
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

        return True
    return False


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --- Model ---
MODEL_ID = "meta-llama/Llama-3.2-1B"

C4_CONFIG = "en"
C4_DATASET_ID = "allenai/c4"
# Small subset for fast demo (interview); increase for real runs
C4_NUM_SAMPLES = 2000
# Max chars per document to avoid OOM; C4 docs can be long
C4_MAX_TEXT_LENGTH = 512

# --- LoRA ---
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# --- Training ---
OUTPUT_DIR = "output/llama32-1b-lora-c4"
MAX_SEQ_LENGTH = 256
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
NUM_EPOCHS = 1
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.03
LOGGING_STEPS = 10
SAVE_STRATEGY = "epoch"
BF16 = True
FP16 = False

# --- Quantization (saves VRAM; set False if you have 16GB+ GPU) ---
LOAD_IN_4BIT = True

# --- Advanced Optimization ---
# Use low-rank optimizer that compresses optimizer states (momentum, variance)
# Standard: stores O(m×n) optimizer states
# LowRank: stores O((m+n)×r) optimizer states where r << min(m,n)
USE_LOWRANK_OPTIMIZER = False  # Set True for additional memory savings
LOWRANK_OPTIMIZER_RANK = 128
LOWRANK_PROJECTION_FREQ = 100  # Update projection matrices every N steps

# Auto-adjust output directory based on optimizer choice
if USE_LOWRANK_OPTIMIZER:
    OUTPUT_DIR = "output/llama32-1b-lora-galore-c4"
else:
    OUTPUT_DIR = "output/llama32-1b-lora-c4"
