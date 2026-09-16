#!/usr/bin/env python3
"""
Predict substrate probabilities for all unlabeled PULDB PULs using the
fine-tuned PULTransformer, then return top-N candidates for a given substrate.

Usage:
    python predict_substrate.py --substrate alginate --topn 20 --device cuda
    python predict_substrate.py --substrate fucoidan  --topn 10 --device cuda
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import percentileofscore
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import PULTransformer
from src.dataset import PULDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--substrate",     required=True,
                   help="Target substrate class (e.g. alginate, fucoidan)")
    p.add_argument("--topn",          type=int, default=20)
    p.add_argument("--finetune_ckpt", default="checkpoints/finetune/final_model.pt")
    p.add_argument("--label_encoder", default="checkpoints/finetune/label_encoder.pkl")
    p.add_argument("--puldb_xlsx",    default="data/raw/PULDB.xlsx")
    p.add_argument("--protein_npz",   default="data/embeddings/pul_esm2_t30_150M_mean_fp16.npz")
    p.add_argument("--device",        default="cuda")
    p.add_argument("--batch_size",    type=int, default=512)
    p.add_argument("--out_csv",       default=None,
                   help="Optional: save results to CSV")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── 1. Label encoder → find target class index ────────────────────────────
    with open(args.label_encoder, "rb") as f:
        le = pickle.load(f)

    substrate = args.substrate.lower().strip()
    if substrate not in le.classes_:
        print(f"Unknown substrate '{substrate}'. Available classes:")
        for c in sorted(le.classes_):
            print(f"  {c}")
        sys.exit(1)

    target_idx = int(le.transform([substrate])[0])
    num_classes = len(le.classes_)
    print(f"Target: '{substrate}'  (class index {target_idx} / {num_classes})\n")

    # ── 2. Load fine-tuned model ──────────────────────────────────────────────
    model = PULTransformer(input_dim=640, d_model=512, nhead=8, num_layers=4)
    model.finetune_head = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, num_classes))
    model.load_state_dict(torch.load(args.finetune_ckpt, map_location=device))
    model = model.to(device).eval()

    # ── 3. Load all PULDB PULs ────────────────────────────────────────────────
    print("Loading PULDB PULDataset …")
    ds = PULDataset(
        xlsx_path=args.puldb_xlsx,
        npz_path=args.protein_npz,
        max_proteins=32,
        min_proteins=2,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # ── 4. Batch inference → collect target class probabilities ───────────────
    print(f"Running inference on {len(ds):,} PULs …")
    all_logits = []
    all_top1   = []

    with torch.no_grad():
        for batch in loader:
            emb  = batch["embeddings"].to(device)
            mask = batch["attention_mask"].to(device)
            _, _, logits = model(emb, mask)          # [B, num_classes]
            all_logits.append(logits[:, target_idx].cpu().numpy())
            all_top1.append(logits.argmax(dim=-1).cpu().numpy())

    all_logits = np.concatenate(all_logits)          # [N_puls]  raw logit
    all_top1   = np.concatenate(all_top1)

    # Percentile of each PUL's logit among all PULs (0–100 score)
    pct_scores = np.array([percentileofscore(all_logits, v) for v in all_logits])

    print(f"\nLogit distribution for '{substrate}':")
    for p in [50, 90, 95, 99, 99.9]:
        print(f"  {p:5.1f}th pct: logit={np.percentile(all_logits, p):.1f}")
    print(f"  Top-{args.topn} threshold: logit={np.sort(all_logits)[::-1][args.topn-1]:.1f}\n")

    # ── 5. Rank and display top-N ─────────────────────────────────────────────
    ranked = np.argsort(all_logits)[::-1][:args.topn]

    rows = []
    for rank, idx in enumerate(ranked, 1):
        pul_name   = ds.pul_names[idx]
        n_prots    = len(ds.pul_indices[idx])
        logit_val  = all_logits[idx]
        pct        = pct_scores[idx]
        top1_label = le.inverse_transform([all_top1[idx]])[0]
        rows.append({
            "rank":       rank,
            "pul_name":   pul_name,
            "n_proteins": n_prots,
            "logit":      round(float(logit_val), 2),
            "percentile": round(float(pct), 1),
            "top1_pred":  top1_label,
            "is_top1":    top1_label == substrate,
        })

    df_out = pd.DataFrame(rows)
    print(f"── Top {args.topn} predicted '{substrate}' PULs ──────────────────────────")
    print(df_out.to_string(index=False))

    n_top1 = df_out["is_top1"].sum()
    print(f"\n{n_top1}/{args.topn} have '{substrate}' as top-1 prediction")

    if args.out_csv:
        df_out.to_csv(args.out_csv, index=False)
        print(f"Saved → {args.out_csv}")


if __name__ == "__main__":
    main()
