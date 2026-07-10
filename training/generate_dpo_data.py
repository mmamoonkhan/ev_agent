"""
=============================================================
  GreenEnergy — Generate DPO Preference Data (Fixed)
=============================================================
"""

import json
import random
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
INPUT_FILE = PROCESSED_DIR / "master_dataset.json"
OUTPUT_FILE = PROCESSED_DIR / "dpo_data.json"

print("=" * 60)
print("  GreenEnergy — Generating DPO Preference Data")
print("=" * 60)

# ── Generic weak responses the model should AVOID ──────────
GENERIC_WEAK = [
    "This electric vehicle has some good features.",
    "The EV offers reasonable performance for its class.",
    "This vehicle is a decent option in the EV market.",
    "The car provides adequate range and charging options.",
    "This EV model has competitive specifications.",
    "The vehicle meets the needs of most EV drivers.",
    "This is a solid electric vehicle choice.",
    "The EV has good technology and performance.",
    "This model offers decent value for money.",
    "The vehicle has satisfactory electric range.",
]


def make_rejected(chosen: str, question: str) -> str:
    """
    Generate a clearly weaker answer using one of 4 strategies.
    The rejected answer should be noticeably worse than chosen.
    """
    strategy = random.randint(1, 4)

    if strategy == 1:
        # Strategy 1: Completely generic — no specific info at all
        return random.choice(GENERIC_WEAK)

    elif strategy == 2:
        # Strategy 2: Remove ALL numbers and replace with vague words
        result = re.sub(
            r'\d+\.?\d*\s*(km|kWh|kW|km/h|Nm|USD|\$|mph|kg|litres?|L|hrs?|hours?|mins?|seconds?|s\b|%|cells?)',
            lambda m: random.choice(
                ["some", "adequate", "reasonable", "sufficient", "decent"]),
            chosen,
            flags=re.IGNORECASE
        )
        # Keep only first sentence
        first_sentence = result.split(".")[0].strip() + "."
        return first_sentence

    elif strategy == 3:
        # Strategy 3: One vague sentence + wrong conclusion
        first = chosen.split(".")[0].strip()
        # Remove brand/model specific info
        words = first.split()[:8]
        vague_ending = random.choice([
            " It is okay.",
            " Performance varies.",
            " Results may differ.",
            " It depends on usage.",
            " Individual results vary.",
        ])
        return " ".join(words) + "..." + vague_ending

    else:
        # Strategy 4: Answer a different/wrong aspect of the question
        wrong_answers = [
            "You should consult your local EV dealer for more information about this.",
            "EV specifications can vary by region and trim level.",
            "It is recommended to test drive the vehicle before making a decision.",
            "Please check the manufacturer's website for the latest specifications.",
            "EV technology is constantly evolving and specifications may change.",
        ]
        return random.choice(wrong_answers)


def generate_dpo_pairs():
    print(f"\nLoading master dataset...")
    with open(INPUT_FILE, encoding="utf-8") as f:
        master_data = json.load(f)
    print(f"  Loaded {len(master_data):,} samples")

    dpo_pairs = []
    skipped = 0

    for sample in master_data:
        chosen = sample["assistant"]

        # Skip very short answers
        if len(chosen.split()) < 10:
            skipped += 1
            continue

        rejected = make_rejected(chosen, sample["user"])

        # Make sure rejected is meaningfully different
        # If too similar (>80% of chosen length and starts same), regenerate
        if (len(rejected) > len(chosen) * 0.8 and
                rejected[:30] == chosen[:30]):
            rejected = random.choice(GENERIC_WEAK)

        dpo_pairs.append({
            "system":   sample["system"],
            "prompt":   sample["user"],
            "chosen":   chosen,
            "rejected": rejected
        })

    print(f"  Skipped   : {skipped} samples (too short)")
    print(f"  Generated : {len(dpo_pairs):,} DPO pairs ✅")

    # Show 3 clear examples
    print("\n📌 Example DPO pairs (notice the clear difference):")
    examples = random.sample(dpo_pairs, min(3, len(dpo_pairs)))
    for i, pair in enumerate(examples, 1):
        print(f"\n  Example {i}:")
        print(f"  QUESTION : {pair['prompt']}")
        print(f"  CHOSEN   : {pair['chosen'][:140]}...")
        print(f"  REJECTED : {pair['rejected']}")

    return dpo_pairs


def save_dpo_data(dpo_pairs):
    random.seed(42)
    random.shuffle(dpo_pairs)
    split = int(len(dpo_pairs) * 0.9)
    train_dpo = dpo_pairs[:split]
    val_dpo = dpo_pairs[split:]

    for filename, data in [
        ("dpo_data.json",  dpo_pairs),
        ("dpo_train.json", train_dpo),
        ("dpo_val.json",   val_dpo),
    ]:
        with open(PROCESSED_DIR / filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ DPO full dataset : {len(dpo_pairs):,} pairs → dpo_data.json")
    print(f"✅ DPO training set : {len(train_dpo):,} pairs → dpo_train.json")
    print(f"✅ DPO validation   : {len(val_dpo):,} pairs  → dpo_val.json")
    print(f"\n📂 Saved to: {PROCESSED_DIR}")
    print(f"\n▶  Next step: run training/finetune_sft.py on GPU server")
    print("=" * 60)


if __name__ == "__main__":
    dpo_pairs = generate_dpo_pairs()
    save_dpo_data(dpo_pairs)
