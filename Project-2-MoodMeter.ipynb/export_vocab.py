"""
export_vocab.py
Run this once from the project root to rebuild the vocab and save it as vocab.json.
The vocab is needed by the HF Space app.
"""
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader import get_dataloaders

print("Rebuilding vocabulary from IMDB dataset (this may take a minute)...")
_, _, vocab = get_dataloaders(batch_size=1, max_length=100)

out_path = os.path.join(os.path.dirname(__file__), "hf-space", "vocab.json")
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False)

print(f"✅  Vocab saved → {out_path}  ({len(vocab):,} tokens)")
