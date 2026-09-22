"""
=============================================================
  GreenEnergy — Smart EV Assistant
  Stage A: Supervised Fine-Tuning (SFT) — FULLY FIXED
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


def format_sample(sample: dict) -> str:
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
    print(f"\nLoading data from {filepath.name}...")
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    formatted = [{"text": format_sample(s)} for s in data]
    dataset   = Dataset.from_list(formatted)
    print(f"  Loaded {len(dataset):,} samples")
    return dataset


def load_base_model():
    print(f"\nLoading base model: {MODEL_NAME}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_use_double_quant = True,
        bnb_4bit_quant_type       = "nf4",
        bnb_4bit_compute_dtype    = torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("  Tokenizer loaded")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config = bnb_config,
        device_map          = "auto",
        torch_dtype         = torch.float16,
    )
    model.config.use_cache      = False
    model.config.pretraining_tp = 1
    model.enable_input_require_grads()
    print("  Base model loaded in 4-bit")
    return model, tokenizer


def apply_lora(model):
    lora_cfg    = config["lora"]
    lora_config = LoraConfig(
        task_type      = TaskType.CAUSAL_LM,
        r              = lora_cfg["r"],
        lora_alpha     = lora_cfg["alpha"],
        lora_dropout   = lora_cfg["dropout"],
        target_modules = lora_cfg["target_modules"],
        bias           = "none",
    )
    model     = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"\n  LoRA applied:")
    print(f"  Trainable : {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    return model


def get_training_args():
    sft_cfg = config["sft"]
    hw_cfg  = config["hardware"]
    return TrainingArguments(
        output_dir                  = str(OUTPUT_DIR),
        num_train_epochs            = sft_cfg["num_epochs"],
        per_device_train_batch_size = sft_cfg["batch_size"],
        gradient_accumulation_steps = sft_cfg["gradient_accumulation"],
        learning_rate               = sft_cfg["learning_rate"],
        warmup_steps                = 100,
        save_steps                  = sft_cfg["save_steps"],
        logging_steps               = sft_cfg["logging_steps"],
        eval_steps                  = sft_cfg["eval_steps"],
        eval_strategy               = "steps",
        fp16                        = hw_cfg["fp16"],
        gradient_checkpointing      = hw_cfg["gradient_checkpointing"],
        report_to                   = "none",
        optim                       = "paged_adamw_32bit",
        lr_scheduler_type           = "cosine",
    )


def train():
    train_dataset    = load_data(TRAIN_FILE)
    val_dataset      = load_data(VAL_FILE)
    model, tokenizer = load_base_model()
    model            = apply_lora(model)
    training_args    = get_training_args()

    trainer = SFTTrainer(
        model              = model,
        tokenizer          = tokenizer,
        train_dataset      = train_dataset,
        eval_dataset       = val_dataset,
        args               = training_args,
        dataset_text_field = "text",
        max_seq_length     = config["data"]["max_length"],
        packing            = False,
    )

    print("\n" + "=" * 60)
    print("  Starting SFT Training...")
    print(f"  Epochs     : {config['sft']['num_epochs']}")
    print(f"  Batch size : {config['sft']['batch_size']}")
    print(f"  Samples    : {len(train_dataset):,}")
    print("=" * 60 + "\n")

    trainer.train()

    print("\n  Saving trained model...")
    trainer.model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print("\n" + "=" * 60)
    print("  SFT Training Complete!")
    print(f"  Model saved to: {OUTPUT_DIR}")
    print("  Next: run training/finetune_dpo.py")
    print("=" * 60)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("WARNING: No GPU detected!")
        exit(1)
    print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    train()
