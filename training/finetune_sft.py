"""
=============================================================
  GreenEnergy — Smart EV Assistant
  Stage A: Supervised Fine-Tuning (SFT)
=============================================================
  What this script does:
    1. Loads LLaMA 3 8B (the base model)
    2. Applies LoRA (efficient fine-tuning method)
    3. Trains on your 38,047 EV training samples
    4. Saves the trained model to models/sft_model/

  Plain English:
    This teaches the base model everything about EVs.
    After this, it knows battery tech, charging, specs,
    grid optimization — all from your 4 datasets.

  How to run:
    python3 training/finetune_sft.py

  Time estimate on A100 GPU:
    ~2-4 hours for 3 epochs on 38,047 samples
=============================================================
"""

import json
import yaml
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

# ── Load config ────────────────────────────────────────────
CONFIG_PATH = Path(__file__).resolve().parent / "train_config.yaml"
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

BASE_DIR   = Path(__file__).resolve().parent.parent
MODEL_NAME = config["model"]["name"]
TRAIN_FILE = BASE_DIR / config["data"]["train_file"]
VAL_FILE   = BASE_DIR / config["data"]["val_file"]
OUTPUT_DIR = BASE_DIR / config["sft"]["output_dir"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  GreenEnergy — Stage A: SFT Training")
print("=" * 60)
print(f"  Base model : {MODEL_NAME}")
print(f"  Train data : {TRAIN_FILE}")
print(f"  Output dir : {OUTPUT_DIR}")
print("=" * 60)


# ══════════════════════════════════════════════════════════
#  STEP 1 — Load and format the training data
#  We convert our JSON into a single text string
#  that the model can learn from
# ══════════════════════════════════════════════════════════
def format_sample(sample: dict) -> str:
    """
    Convert one training sample into the chat format
    that LLaMA 3 was trained to understand.

    LLaMA 3 uses special tokens to separate parts:
    <|system|>    = the system prompt (GreenEnergy identity)
    <|user|>      = the user's question
    <|assistant|> = the model's answer

    The model learns to predict the assistant part
    given the system + user parts.
    """
    return (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n"
        f"{sample['system']}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{sample['user']}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{sample['assistant']}<|eot_id|>"
    )


def load_data(filepath: Path) -> Dataset:
    """Load JSON file and format each sample."""
    print(f"\nLoading data from {filepath.name}...")
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    # Format each sample into the LLaMA 3 chat format
    formatted = [{"text": format_sample(s)} for s in data]
    dataset   = Dataset.from_list(formatted)
    print(f"  ✅ Loaded {len(dataset):,} samples")
    return dataset


# ══════════════════════════════════════════════════════════
#  STEP 2 — Load the base model (LLaMA 3 8B)
#  We load it in 4-bit to save GPU memory
#  8B parameters × 4 bits = ~4GB instead of ~16GB
# ══════════════════════════════════════════════════════════
def load_base_model():
    """
    Load LLaMA 3 8B in 4-bit quantization.

    4-bit quantization: instead of storing each number
    with full precision (32 bits), we compress to 4 bits.
    This reduces memory by 8x with minimal quality loss.
    """
    print(f"\nLoading base model: {MODEL_NAME}")
    print("  (This may take a few minutes on first run...)")

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_use_double_quant = True,    # Extra compression
        bnb_4bit_quant_type       = "nf4",   # Best quantization type
        bnb_4bit_compute_dtype    = torch.float16
    )

    # Load the tokenizer
    # Tokenizer converts text into numbers the model understands
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("  ✅ Tokenizer loaded")

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config = bnb_config,
        device_map          = "auto",   # Automatically use available GPUs
        torch_dtype         = torch.float16,
    )
    model.config.use_cache             = False
    model.config.pretraining_tp        = 1
    model.enable_input_require_grads()
    print("  ✅ Base model loaded in 4-bit")

    return model, tokenizer


# ══════════════════════════════════════════════════════════
#  STEP 3 — Apply LoRA
#  Instead of training ALL 8 billion parameters
#  (expensive), we add small trainable matrices
#  on top of the frozen base model (cheap)
# ══════════════════════════════════════════════════════════
def apply_lora(model):
    """
    Apply LoRA to the model.

    Imagine the model has a huge wall of knowledge (8B params).
    LoRA adds small sticky notes on specific parts of the wall.
    During training, only the sticky notes change.
    This is much faster and uses much less memory.
    """
    lora_cfg = config["lora"]

    lora_config = LoraConfig(
        task_type        = TaskType.CAUSAL_LM,
        r                = lora_cfg["r"],
        lora_alpha       = lora_cfg["alpha"],
        lora_dropout     = lora_cfg["dropout"],
        target_modules   = lora_cfg["target_modules"],
        bias             = "none",
    )

    model = get_peft_model(model, lora_config)

    # Show how many parameters we are actually training
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"\n  LoRA applied:")
    print(f"  Trainable parameters : {trainable:,}")
    print(f"  Total parameters     : {total:,}")
    print(f"  Training only        : {100 * trainable / total:.2f}% of the model ✅")

    return model


# ══════════════════════════════════════════════════════════
#  STEP 4 — Set up training arguments
#  These control HOW the training happens
# ══════════════════════════════════════════════════════════
def get_training_args():
    sft_cfg = config["sft"]
    hw_cfg  = config["hardware"]

    return TrainingArguments(
        output_dir                  = str(OUTPUT_DIR),
        num_train_epochs            = sft_cfg["num_epochs"],
        per_device_train_batch_size = sft_cfg["batch_size"],
        gradient_accumulation_steps = sft_cfg["gradient_accumulation"],
        learning_rate               = sft_cfg["learning_rate"],
        warmup_ratio                = sft_cfg["warmup_ratio"],
        save_steps                  = sft_cfg["save_steps"],
        logging_steps               = sft_cfg["logging_steps"],
        eval_steps                  = sft_cfg["eval_steps"],
        evaluation_strategy         = "steps",
        save_strategy               = "steps",
        load_best_model_at_end      = True,
        fp16                        = hw_cfg["fp16"],
        gradient_checkpointing      = hw_cfg["gradient_checkpointing"],
        report_to                   = "none",
        optim                       = "paged_adamw_32bit",
        lr_scheduler_type           = "cosine",
        dataloader_pin_memory       = False,
        logging_dir                 = str(OUTPUT_DIR / "logs"),
    )


# ══════════════════════════════════════════════════════════
#  STEP 5 — Train!
# ══════════════════════════════════════════════════════════
def train():
    # Load data
    train_dataset = load_data(TRAIN_FILE)
    val_dataset   = load_data(VAL_FILE)

    # Load model + tokenizer
    model, tokenizer = load_base_model()

    # Apply LoRA
    model = apply_lora(model)

    # Training arguments
    training_args = get_training_args()

    # Create the SFT trainer
    # SFTTrainer handles all the training loop details for us
    trainer = SFTTrainer(
        model           = model,
        tokenizer       = tokenizer,
        train_dataset   = train_dataset,
        eval_dataset    = val_dataset,
        args            = training_args,
        dataset_text_field = "text",
        max_seq_length  = config["data"]["max_length"],
        packing         = False,
    )

    print("\n" + "=" * 60)
    print("  Starting SFT Training...")
    print(f"  Epochs     : {config['sft']['num_epochs']}")
    print(f"  Batch size : {config['sft']['batch_size']}")
    print(f"  Samples    : {len(train_dataset):,}")
    print("=" * 60 + "\n")

    # Run training
    trainer.train()

    # Save the final model
    print("\n  Saving trained model...")
    trainer.model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print("\n" + "=" * 60)
    print("  ✅ SFT Training Complete!")
    print(f"  Model saved to: {OUTPUT_DIR}")
    print("  Next step: run training/finetune_dpo.py")
    print("=" * 60)


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Check GPU is available
    if not torch.cuda.is_available():
        print("⚠️  WARNING: No GPU detected!")
        print("   This script requires a CUDA GPU to run.")
        print("   Please run on the supervisor's GPU server.")
        exit(1)

    print(f"\n  GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"  GPU memory  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    train()
