"""
Load a small subset of C4 (clean English) for LoRA fine-tuning demo.

C4 (Colossal Clean Crawled Corpus):
- Clean version of Common Crawl (April 2019)
- allenai/c4 config "en" = cleaned, filtered, deduplicated English
- Used for pretraining/fine-tuning language models

We use streaming and take the first N samples to avoid long preprocessing.
"""

from datasets import load_dataset
from config import (
    C4_DATASET_ID,
    C4_CONFIG,
    C4_NUM_SAMPLES,
    C4_MAX_TEXT_LENGTH,
)


def load_c4_subset(num_samples: int = None, max_text_len: int = None):

    num_samples = num_samples or C4_NUM_SAMPLES
    max_text_len = max_text_len or C4_MAX_TEXT_LENGTH

    print(f"Loading C4 config='{C4_CONFIG}' (clean English) in streaming mode...")
    stream = load_dataset(
        C4_DATASET_ID,
        C4_CONFIG,
        split="train",
        streaming=True,
    )

    texts = []
    for i, item in enumerate(stream):
        if i >= num_samples:
            break
        text = item.get("text", "").strip()
        if not text:
            continue
        # Truncate to limit memory and sequence length
        if len(text) > max_text_len:
            text = text[:max_text_len]
        texts.append({"text": text})

    print(f"Collected {len(texts)} samples.")
    from datasets import Dataset
    return Dataset.from_list(texts)


if __name__ == "__main__":
    dataset = load_c4_subset()
    print(dataset)
    print("Example:", dataset[0]["text"][:200], "...")
