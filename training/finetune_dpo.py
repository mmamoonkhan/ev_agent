"""
=============================================================
  GreenEnergy — Smart EV Assistant
  Stage B: Direct Preference Optimization (DPO)
=============================================================
  What this script does:
    1. Loads the SFT model (already trained on EV data)
    2. Applies DPO training on chosen/rejected pairs
    3. Makes the model prefer detailed, accurate answers
    4. Saves the final GreenEnergy model

  Plain English:
    After SFT, the model knows EV facts.
    DPO teaches it HOW to answer well —
    preferring detailed, specific answers
    over vague, generic ones.

    This is the DeepSeek-inspired step!

  IMPORTANT: Run finetune_sft.py FIRST!

  How to run:
    python3 training/finetune_dpo.py

  Time estimate on A100 GPU:
    ~1-2 hours for 1 epoch
=============================================================
"""

import json
import yaml
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, LoraConfig
from trl import DPOTrainer, DPOConfig

# ── Load config ────────────────────────────────────────────
CONFIG_PATH = Path(__file__).resolve().parent / "train_config.yaml"
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

BASE_DIR   = Path(__file__).resolve().parent.parent
SFT_MODEL  = BASE_DIR / config["sft"]["output_dir"]
OUTPUT_DIR = BASE_DIR / config["dpo"]["output_dir"]
DPO_TRAIN  = BASE_DIR / "data" / "processed" / "dpo_train.json"
DPO_VAL    = BASE_DIR / "data" / "processed" / "dpo_val.json"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  GreenEnergy — Stage B: DPO Training")
print("=" * 60)
print(f"  SFT model  : {SFT_MODEL}")
print(f"  DPO data   : {DPO_TRAIN}")
print(f"  Output dir : {OUTPUT_DIR}")
print("=" * 60)


# ══════════════════════════════════════════════════════════
#  STEP 1 — Load DPO preference data
# ══════════════════════════════════════════════════════════
def load_dpo_data(filepath: Path) -> Dataset:
    """
    Load the DPO preference pairs.
    Each sample has: system, prompt, chosen, rejected
    """
    print(f"\nLoading DPO data from {filepath.name}...")
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    # DPOTrainer expects this exact format
    formatted = []
    for item in data:
        formatted.append({
            "prompt":   (
                f"<|begin_of_text|>"
                f"<|start_header_id|>system<|end_header_id|>\n\n"
                f"{item['system']}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"{item['prompt']}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            ),
            "chosen":   item["chosen"]  + "<|eot_id|>",
            "rejected": item["rejected"] + "<|eot_id|>",
        })

    dataset = Dataset.from_list(formatted)
    print(f"  ✅ Loaded {len(dataset):,} preference pairs")
    return dataset


# ══════════════════════════════════════════════════════════
#  STEP 2 — Load the SFT model
#  We load the model we already trained in Stage A
# ══════════════════════════════════════════════════════════
def load_sft_model():
    """
    Load the fine-tuned SFT model.
    This is our GreenEnergy model after Stage A —
    it already knows EV facts, now we refine HOW it answers.
    """
    print(f"\nLoading SFT model from {SFT_MODEL.name}...")

    # Check SFT model exists
    if not SFT_MODEL.exists():
        print(f"❌ ERROR: SFT model not found at {SFT_MODEL}")
        print("   Please run training/finetune_sft.py first!")
        exit(1)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_use_double_quant = True,
        bnb_4bit_quant_type       = "nf4",
        bnb_4bit_compute_dtype    = torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(str(SFT_MODEL))
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"   # DPO needs left padding

    model = AutoModelForCausalLM.from_pretrained(
        str(SFT_MODEL),
        quantization_config = bnb_config,
        device_map          = "auto",
        torch_dtype         = torch.float16,
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

    print("  ✅ SFT model loaded")
    return model, tokenizer


# ══════════════════════════════════════════════════════════
#  STEP 3 — Apply LoRA for DPO
#  We add a new set of LoRA weights for DPO training
# ══════════════════════════════════════════════════════════
def apply_dpo_lora(model):
    from peft import get_peft_model
    lora_cfg = config["lora"]

    lora_config = LoraConfig(
        r              = lora_cfg["r"],
        lora_alpha     = lora_cfg["alpha"],
        lora_dropout   = lora_cfg["dropout"],
        target_modules = lora_cfg["target_modules"],
        bias           = "none",
        task_type      = "CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    print("  ✅ LoRA applied for DPO")
    return model


# ══════════════════════════════════════════════════════════
#  STEP 4 — Train with DPO
# ══════════════════════════════════════════════════════════
def train():
    # Load data
    train_dataset = load_dpo_data(DPO_TRAIN)
    val_dataset   = load_dpo_data(DPO_VAL)

    # Load model
    model, tokenizer = load_sft_model()
    model = apply_dpo_lora(model)

    dpo_cfg = config["dpo"]
    hw_cfg  = config["hardware"]

    # DPO training config
    dpo_config = DPOConfig(
        output_dir                  = str(OUTPUT_DIR),
        num_train_epochs            = dpo_cfg["num_epochs"],
        per_device_train_batch_size = dpo_cfg["batch_size"],
        gradient_accumulation_steps = dpo_cfg["gradient_accumulation"],
        learning_rate               = dpo_cfg["learning_rate"],
        beta                        = dpo_cfg["beta"],
        warmup_ratio                = dpo_cfg["warmup_ratio"],
        save_steps                  = dpo_cfg["save_steps"],
        logging_steps               = dpo_cfg["logging_steps"],
        eval_steps                  = dpo_cfg["eval_steps"],
        evaluation_strategy         = "steps",
        fp16                        = hw_cfg["fp16"],
        gradient_checkpointing      = hw_cfg["gradient_checkpointing"],
        report_to                   = "none",
        optim                       = "paged_adamw_32bit",
        remove_unused_columns       = False,
        logging_dir                 = str(OUTPUT_DIR / "logs"),
    )

    # DPO Trainer
    trainer = DPOTrainer(
        model           = model,
        ref_model       = None,   # None = use the model itself as reference
        tokenizer       = tokenizer,
        train_dataset   = train_dataset,
        eval_dataset    = val_dataset,
        args            = dpo_config,
    )

    print("\n" + "=" * 60)
    print("  Starting DPO Training...")
    print(f"  Epochs     : {dpo_cfg['num_epochs']}")
    print(f"  Batch size : {dpo_cfg['batch_size']}")
    print(f"  Beta       : {dpo_cfg['beta']}")
    print(f"  Pairs      : {len(train_dataset):,}")
    print("=" * 60 + "\n")

    trainer.train()

    # Save final model
    print("\n  Saving final GreenEnergy model...")
    trainer.model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print("\n" + "=" * 60)
    print("  ✅ DPO Training Complete!")
    print(f"  Final model saved to: {OUTPUT_DIR}")
    print("  Next step: run agent/agent.py")
    print("=" * 60)


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("⚠️  WARNING: No GPU detected!")
        print("   This script requires a CUDA GPU to run.")
        print("   Please run on the supervisor's GPU server.")
        exit(1)

    print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    train()
