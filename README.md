# GreenEnergy — Smart EV Assistant
### Master Thesis Project | ECE Paris | Ref #Mas25-R-1860
**Author:** Muhammad Mamoon | **Supervisor:** Muhammad Farrukh Khan

---

## What this project does
GreenEnergy is a fine-tuned LLM-based Agentic AI system that answers
any question about Electric Vehicles — battery technology, charging
infrastructure, vehicle specifications, grid optimization, costs,
and sustainability. It serves as the practical proof of the thesis
research on Agentic AI systems orchestrated around LLMs.

---

## Project Structure
```
ev_agent/
│
├── data/
│   ├── raw/                        # Downloaded datasets (auto-created)
│   │   ├── ev_analytics/           # EV Analytics (Kaggle)
│   │   ├── ev_specs_trends/        # EV Specs & Trends (Kaggle)
│   │   ├── ev_grid/                # EV Grid Optimization (Kaggle)
│   │   └── ev_huggingface/         # EV Specs 2025 (Hugging Face)
│   │
│   └── processed/                  # Cleaned + combined training data
│       ├── master_dataset.json     # 42,275 combined samples
│       ├── train.json              # 38,047 SFT training samples
│       ├── val.json                # 4,228 SFT validation samples
│       ├── dpo_data.json           # 42,275 DPO preference pairs
│       ├── dpo_train.json          # 38,047 DPO training pairs
│       └── dpo_val.json            # 4,228 DPO validation pairs
│
├── preprocessing/
│   ├── download_datasets.py        # Step 1: Download all 4 datasets
│   └── preprocess.py               # Step 2: Clean, merge, generate samples
│
├── training/
│   ├── train_config.yaml           # All training settings in one place
│   ├── generate_dpo_data.py        # Step 3: Generate DPO preference pairs
│   ├── finetune_sft.py             # Step 4: SFT training (Stage A)
│   └── finetune_dpo.py             # Step 5: DPO training (Stage B)
│
├── agent/
│   └── agent.py                    # Step 6: Agentic system (coming soon)
│
├── api/
│   └── main.py                     # Step 7: FastAPI backend (coming soon)
│
├── models/
│   ├── sft_model/                  # Saved after SFT training
│   └── dpo_model/                  # Final GreenEnergy model
│
├── requirements.txt
└── README.md
```

---

## Datasets Used
| # | Dataset | Source | Samples Generated |
|---|---------|--------|------------------|
| 1 | EV Analytics | Kaggle | 29,994 |
| 2 | EV Specs & Trends 2025 | Kaggle | 1,558 |
| 3 | EV Charging Grid Optimization | Kaggle | 7,005 |
| 4 | EV Specs 2025 | Hugging Face | 3,718 |
| | **Total** | | **42,275** |

---

## Training Strategy — DeepSeek-Inspired
- **Base model:** LLaMA 3 8B (meta-llama/Meta-Llama-3-8B-Instruct)
- **Stage A:** Supervised Fine-Tuning (SFT) — teaches EV domain knowledge
- **Stage B:** Direct Preference Optimization (DPO) — improves answer quality
- **Technique:** LoRA / QLoRA for parameter-efficient fine-tuning
- **Framework:** HuggingFace TRL + PEFT

---

## How to Run — Step by Step

### Step 1 — Install dependencies
```bash
# On Mac (preprocessing only)
pip install kaggle datasets huggingface_hub pandas numpy PyYAML tqdm requests

# On GPU server (full install)
pip install -r requirements.txt
```

### Step 2 — Setup Kaggle API
```bash
# Get token from kaggle.com → Account → API → Create New Token
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### Step 3 — Download all 4 datasets
```bash
python3 preprocessing/download_datasets.py
```

### Step 4 — Preprocess and combine datasets
```bash
python3 preprocessing/preprocess.py
# Output: 42,275 training samples in data/processed/
```

### Step 5 — Generate DPO preference pairs (Mac, no GPU needed)
```bash
python3 training/generate_dpo_data.py
# Output: 42,275 preference pairs in data/processed/
```

### Step 6 — SFT Training (GPU server required)
```bash
python3 training/finetune_sft.py
# Time: ~2-4 hours on A100 GPU
# Output: models/sft_model/
```

### Step 7 — DPO Training (GPU server required)
```bash
python3 training/finetune_dpo.py
# Time: ~1-2 hours on A100 GPU
# Output: models/dpo_model/  ← Final GreenEnergy model
```

### Step 8 — Run the API (coming soon)
```bash
uvicorn api.main:app --reload
```

---

## Training Configuration
All settings are in `training/train_config.yaml`:

| Setting | Value |
|---|---|
| Base model | meta-llama/Meta-Llama-3-8B-Instruct |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| SFT epochs | 3 |
| DPO epochs | 1 |
| Quantization | 4-bit (QLoRA) |
| Estimated GPU time | 4-6 hours on A100 |

---

## Thesis Connection
This project directly demonstrates the thesis research question:
> *"To what extent do Agentic AI systems, particularly those orchestrated
> around LLMs, improve the autonomy, performance, and robustness of
> intelligent agents in complex tasks?"*

GreenEnergy is a live, working proof that fine-tuning + preference
optimization significantly improves agent performance on domain-specific
tasks — the core argument of the thesis.

---

## Tech Stack
Python • LLaMA 3 8B • LoRA/QLoRA • HuggingFace Transformers •
TRL • PEFT • FastAPI • Kaggle API • SFT + DPO
