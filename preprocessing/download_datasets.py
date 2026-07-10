"""
=============================================================
  EV Agent — Step 1: Download All 4 Datasets
=============================================================
  Datasets:
    1. EV Analytics        (Kaggle)
    2. EV Specs & Trends   (Kaggle)
    3. EV Grid Optimization(Kaggle)
    4. EV Specs 2025       (Hugging Face)
=============================================================
  How to run:
    pip install kaggle datasets huggingface_hub
    python download_datasets.py
=============================================================
"""

import os
import json
import shutil
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
RAW_DIR    = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  EV Agent — Downloading All 4 Datasets")
print("=" * 60)


# ══════════════════════════════════════════════════════════
#  KAGGLE SETUP
#  Before running, place your kaggle.json in ~/.kaggle/
#  Get it from: kaggle.com → Account → API → Create Token
# ══════════════════════════════════════════════════════════
def setup_kaggle():
    """Check Kaggle credentials exist."""
    kaggle_path = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_path.exists():
        print("\n❌ ERROR: kaggle.json not found!")
        print("   1. Go to kaggle.com → Your Account → API")
        print("   2. Click 'Create New Token'")
        print("   3. Save kaggle.json to ~/.kaggle/kaggle.json")
        print("   4. Run: chmod 600 ~/.kaggle/kaggle.json")
        raise FileNotFoundError("kaggle.json missing")
    print("✅ Kaggle credentials found!")


# ══════════════════════════════════════════════════════════
#  DATASET 1 — EV Analytics
#  Contains: performance, charging, cost, battery metrics
# ══════════════════════════════════════════════════════════
def download_ev_analytics():
    print("\n[1/4] Downloading EV Analytics Dataset...")
    import kaggle
    save_path = RAW_DIR / "ev_analytics"
    save_path.mkdir(exist_ok=True)

    kaggle.api.dataset_download_files(
        dataset   = "khushikyad001/electric-vehicle-analytics-dataset",
        path      = str(save_path),
        unzip     = True,
        quiet     = False
    )
    print(f"✅ EV Analytics saved to: {save_path}")
    list_files(save_path)


# ══════════════════════════════════════════════════════════
#  DATASET 2 — EV Specs & Trends 2025
#  Contains: 470+ EV models, range, battery, charging ports
# ══════════════════════════════════════════════════════════
def download_ev_specs_trends():
    print("\n[2/4] Downloading EV Specs & Trends Dataset...")
    import kaggle
    save_path = RAW_DIR / "ev_specs_trends"
    save_path.mkdir(exist_ok=True)

    kaggle.api.dataset_download_files(
        dataset   = "vakatarun/electric-vehicle-specifications-and-trends-dataset",
        path      = str(save_path),
        unzip     = True,
        quiet     = False
    )
    print(f"✅ EV Specs & Trends saved to: {save_path}")
    list_files(save_path)


# ══════════════════════════════════════════════════════════
#  DATASET 3 — EV Charging Grid Optimization
#  Contains: grid routing, smart charging, V2G optimization
# ══════════════════════════════════════════════════════════
def download_ev_grid():
    print("\n[3/4] Downloading EV Grid Optimization Dataset...")
    import kaggle
    save_path = RAW_DIR / "ev_grid"
    save_path.mkdir(exist_ok=True)

    kaggle.api.dataset_download_files(
        dataset   = "ziya07/ev-charging-grid-optimization-dataset",
        path      = str(save_path),
        unzip     = True,
        quiet     = False
    )
    print(f"✅ EV Grid Optimization saved to: {save_path}")
    list_files(save_path)


# ══════════════════════════════════════════════════════════
#  DATASET 4 — EV Specs 2025 (Hugging Face)
#  Contains: latest 2025 EV specs from EV-Database.org
# ══════════════════════════════════════════════════════════
def download_ev_huggingface():
    print("\n[4/4] Downloading EV Specs 2025 from Hugging Face...")
    from datasets import load_dataset

    save_path = RAW_DIR / "ev_huggingface"
    save_path.mkdir(exist_ok=True)

    # Load from Hugging Face
    ds = load_dataset("UrvishAhir1/Electric-Vehicle-Specs-Dataset-2025")

    # Save as CSV for easy processing later
    for split_name, split_data in ds.items():
        output_file = save_path / f"ev_specs_2025_{split_name}.csv"
        split_data.to_csv(str(output_file))
        print(f"   Saved split '{split_name}': {len(split_data)} rows → {output_file.name}")

    print(f"✅ HuggingFace EV Specs saved to: {save_path}")


# ══════════════════════════════════════════════════════════
#  HELPER — list downloaded files
# ══════════════════════════════════════════════════════════
def list_files(path: Path):
    files = list(path.glob("*"))
    if files:
        print("   Files downloaded:")
        for f in files:
            size_kb = f.stat().st_size / 1024
            print(f"   📄 {f.name}  ({size_kb:.1f} KB)")
    else:
        print("   ⚠️  No files found — check dataset name!")


# ══════════════════════════════════════════════════════════
#  SUMMARY — show what was downloaded
# ══════════════════════════════════════════════════════════
def print_summary():
    print("\n" + "=" * 60)
    print("  DOWNLOAD COMPLETE — Summary")
    print("=" * 60)
    total_files = 0
    for folder in RAW_DIR.iterdir():
        if folder.is_dir():
            files = list(folder.glob("*.csv")) + list(folder.glob("*.json"))
            total_files += len(files)
            print(f"\n📁 {folder.name}/")
            for f in files:
                size_kb = f.stat().st_size / 1024
                print(f"   📄 {f.name}  ({size_kb:.1f} KB)")

    print(f"\n✅ Total files ready: {total_files}")
    print(f"📂 All data in: {RAW_DIR}")
    print("\n▶  Next step: run  preprocessing/preprocess.py")
    print("=" * 60)


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    try:
        setup_kaggle()
        download_ev_analytics()
        download_ev_specs_trends()
        download_ev_grid()
        download_ev_huggingface()
        print_summary()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("  pip install kaggle datasets huggingface_hub pandas")
        print("  Make sure ~/.kaggle/kaggle.json exists")
