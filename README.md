# EV Agent — Agentic AI System for Electric Vehicles
### Master Thesis Project | ECE Paris | Ref #Mas25-R-1860
**Author:** Muhammad Mamoon | **Supervisor:** Muhammad Farrukh Khan

---

## What this project does
A fine-tuned LLM-based Agentic AI system that answers any question
about Electric Vehicles — battery technology, charging, specs,
grid optimization, costs, and sustainability.

---

## Project Structure
```
ev_agent/
│
├── data/
│   ├── raw/                   # Downloaded datasets (auto-created)
│   └── processed/             # Cleaned + combined training data
│
├── preprocessing/
│   ├── download_datasets.py   # Step 1: Download all 4 datasets
│   └── preprocess.py          # Step 2: Convert to training format
│
├── training/
│   └── finetune.py            # Step 3: Fine-tune LLM (coming next)
│
├── agent/
│   └── agent.py               # Step 4: Agentic system (coming next)
│
├── api/
│   └── main.py                # Step 5: FastAPI backend (coming next)
│
├── requirements.txt
└── README.md
```

---

## How to run — Step by Step

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup Kaggle API
- Go to kaggle.com → Account → API → Create New Token
- Save `kaggle.json` to `~/.kaggle/kaggle.json`
- Run: `chmod 600 ~/.kaggle/kaggle.json`

### 3. Download all datasets
```bash
python preprocessing/download_datasets.py
```

### 4. Preprocess & combine datasets
```bash
python preprocessing/preprocess.py
```

### 5. Fine-tune the model (on GPU)
```bash
python training/finetune.py
```

### 6. Run the API
```bash
uvicorn api.main:app --reload
```

---

## Datasets Used
| # | Dataset | Source |
|---|---------|--------|
| 1 | EV Analytics | Kaggle |
| 2 | EV Specs & Trends 2025 | Kaggle |
| 3 | EV Charging Grid Optimization | Kaggle |
| 4 | EV Specs 2025 | Hugging Face |

---

## Training Method
- **Base model:** LLaMA 3 8B or Mistral 7B
- **Fine-tuning:** SFT (Supervised Fine-Tuning) + DPO
- **Technique:** LoRA / QLoRA for efficiency
- **Framework:** HuggingFace TRL + PEFT
