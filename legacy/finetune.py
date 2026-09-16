#!/usr/bin/env python3
"""
Fine-tune the pre-trained PULTransformer for substrate specificity prediction.

Two-phase training:
  Phase 1 (frozen encoder): train classification head only — fast convergence
  Phase 2 (full fine-tune): unfreeze all parameters at low LR

Evaluation: stratified 5-fold CV → macro F1, per-class F1
Final model trained on all data and saved to checkpoints/finetune/.

Usage:
    python finetune.py \\
        --checkpoint   checkpoints/pretrain/best_model.pt \\
        --xlsx         data/raw/CANPUL_substrates.xlsx \\
        --faa_dir      data/raw/dbcan_pul \\
        --cache_npz    data/embeddings/labelled_protein_esm2.npz \\
        --device       cuda
"""

import argparse
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Bio import SeqIO
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from src.model import PULTransformer


# ── ESM2 helpers ──────────────────────────────────────────────────────────────

ESM2_VOCAB = frozenset("ACDEFGHIKLMNOPQRSTUVWXYZB")

def to_esm2(seq: str) -> str:
    return "".join(c if c in ESM2_VOCAB else "X" for c in seq.upper())

def mean_pool(token_repr, seq_lens):
    B, L, D = token_repr.shape
    positions = torch.arange(L, device=token_repr.device).unsqueeze(0)
    mask = (positions >= 1) & (positions <= seq_lens.unsqueeze(1))
    summed = (token_repr * mask.unsqueeze(-1).float()).sum(dim=1)
    return summed / seq_lens.float().unsqueeze(-1).clamp(min=1)

def compute_esm2_embeddings(sequences: dict, device, cache_path: Path,
                             batch_size: int = 64, trunc: int = 1022):
    """Return {protein_id: np.float16 [640]}, loading from cache if available."""
    if cache_path.exists():
        print(f"Loading ESM2 cache from {cache_path} …")
        d = np.load(cache_path, allow_pickle=True)
        return dict(zip(d["protein_id"].tolist(), d["embedding"]))

    import esm
    print("Loading ESM2 model …")
    esm_model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    esm_model = esm_model.to(device).eval()

    items   = list(sequences.items())
    results = {}

    for i in tqdm(range(0, len(items), batch_size), desc="ESM2", unit="batch"):
        batch = [(pid, to_esm2(seq)[:trunc]) for pid, seq in items[i:i+batch_size]]
        batch = [(pid, s) for pid, s in batch if s]
        if not batch:
            continue
        try:
            labels, _, tokens = batch_converter(batch)
            tokens = tokens.to(device)
            with torch.no_grad():
                out = esm_model(tokens, repr_layers=[30], return_contacts=False)
            repr_ = out["representations"][30]
            lens  = (tokens != alphabet.padding_idx).sum(1) - 2
            embs  = mean_pool(repr_, lens.clamp(min=0))
            for j, pid in enumerate(labels):
                results[pid] = embs[j].cpu().to(torch.float16).numpy()
        except Exception as e:
            print(f"[warn] batch error: {e}", file=sys.stderr)

    # Cache to disk
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(cache_path).removesuffix(".npz") + "_tmp"
    np.savez_compressed(tmp,
                        protein_id=np.array(list(results.keys()), dtype=object),
                        embedding=np.stack(list(results.values())))
    os.replace(tmp + ".npz", cache_path)
    print(f"ESM2 cache saved → {cache_path}")
    return results


# ── Dataset ───────────────────────────────────────────────────────────────────

class PULFinetuneDataset(Dataset):
    def __init__(self, pul_ids, labels, pul_protein_map, prot_embeddings, max_proteins=32):
        self.pul_ids        = pul_ids
        self.labels         = labels
        self.pul_protein_map = pul_protein_map
        self.prot_embeddings = prot_embeddings
        self.max_proteins   = max_proteins
        self.emb_dim        = 640

    def __len__(self):
        return len(self.pul_ids)

    def __getitem__(self, idx):
        pul_id = self.pul_ids[idx]
        prots  = [p for p in self.pul_protein_map[pul_id]
                  if p in self.prot_embeddings][:self.max_proteins]
        n = len(prots)

        if n == 0:
            emb  = torch.zeros(self.max_proteins, self.emb_dim)
            mask = torch.zeros(self.max_proteins, dtype=torch.bool)
        else:
            stacked = torch.tensor(
                np.stack([self.prot_embeddings[p].astype(np.float32) for p in prots])
            )
            pad  = torch.zeros(self.max_proteins - n, self.emb_dim)
            emb  = torch.cat([stacked, pad], dim=0)
            mask = torch.zeros(self.max_proteins, dtype=torch.bool)
            mask[:n] = True

        return {
            "embeddings":     emb,
            "attention_mask": mask,
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Training helpers ──────────────────────────────────────────────────────────

def make_model(checkpoint: str, num_classes: int, device) -> PULTransformer:
    model = PULTransformer(input_dim=640, d_model=512, nhead=8, num_layers=4)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    # Replace head with Dropout → Linear for regularisation
    model.finetune_head = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.d_model, num_classes),
    )
    nn.init.trunc_normal_(model.finetune_head[1].weight, std=0.02)
    nn.init.zeros_(model.finetune_head[1].bias)
    return model.to(device)


def train_epoch(model, loader, optimizer, criterion, device, clip_norm=1.0):
    model.train()
    total_loss, n = 0.0, 0
    for batch in loader:
        emb    = batch["embeddings"].to(device)
        mask   = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        _, _, logits = model(emb, mask)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        n          += len(labels)
    return total_loss / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n = 0.0, 0
    for batch in loader:
        emb    = batch["embeddings"].to(device)
        mask   = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        _, _, logits = model(emb, mask)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        total_loss += loss.item() * len(labels)
        n          += len(labels)

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / n, macro_f1, np.array(all_preds), np.array(all_labels)


def run_fold(fold_idx, train_idx, val_idx,
             pul_ids, y, pul_protein_map, prot_embeddings,
             checkpoint, num_classes, class_weights,
             args, device):

    tr_ids  = [pul_ids[i] for i in train_idx]
    val_ids = [pul_ids[i] for i in val_idx]
    tr_y    = y[train_idx]
    val_y   = y[val_idx]

    tr_set  = PULFinetuneDataset(tr_ids,  tr_y,  pul_protein_map, prot_embeddings, args.max_proteins)
    val_set = PULFinetuneDataset(val_ids, val_y, pul_protein_map, prot_embeddings, args.max_proteins)
    tr_ldr  = DataLoader(tr_set,  batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_ldr = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model     = make_model(checkpoint, num_classes, device)
    w_tensor  = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=w_tensor)

    best_f1, best_preds, best_labels, patience_cnt = 0.0, None, None, 0

    # ── Phase 1: frozen encoder ───────────────────────────────────────────────
    model.freeze_encoder()
    opt1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.phase1_lr, weight_decay=0.01,
    )
    print(f"  [Fold {fold_idx+1}] Phase 1 — frozen encoder ({args.phase1_epochs} epochs)")
    for ep in range(1, args.phase1_epochs + 1):
        tr_loss = train_epoch(model, tr_ldr, opt1, criterion, device)
        val_loss, val_f1, preds, labels = eval_epoch(model, val_ldr, criterion, device)
        if ep % 5 == 0:
            print(f"    ep {ep:3d}  tr_loss {tr_loss:.4f}  val_loss {val_loss:.4f}  val_f1 {val_f1:.3f}")
        if val_f1 > best_f1:
            best_f1, best_preds, best_labels = val_f1, preds, labels
            patience_cnt = 0
        else:
            patience_cnt += 1

    # ── Phase 2: full fine-tune ───────────────────────────────────────────────
    model.unfreeze_encoder()
    opt2 = torch.optim.AdamW(model.parameters(), lr=args.phase2_lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=args.phase2_epochs)

    patience_cnt = 0
    print(f"  [Fold {fold_idx+1}] Phase 2 — full fine-tune ({args.phase2_epochs} epochs)")
    for ep in range(1, args.phase2_epochs + 1):
        tr_loss = train_epoch(model, tr_ldr, opt2, criterion, device)
        val_loss, val_f1, preds, labels = eval_epoch(model, val_ldr, criterion, device)
        scheduler.step()
        if ep % 10 == 0:
            print(f"    ep {ep:3d}  tr_loss {tr_loss:.4f}  val_loss {val_loss:.4f}  val_f1 {val_f1:.3f}")
        if val_f1 > best_f1:
            best_f1, best_preds, best_labels = val_f1, preds, labels
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"    Early stop at epoch {ep}")
                break

    print(f"  [Fold {fold_idx+1}] best val macro-F1 = {best_f1:.3f}")
    return best_f1, best_preds, best_labels


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    default="checkpoints/pretrain/best_model.pt")
    p.add_argument("--xlsx",          default="data/raw/CANPUL_substrates.xlsx")
    p.add_argument("--faa_dir",       default="data/raw/dbcan_pul")
    p.add_argument("--cache_npz",     default="data/embeddings/labelled_protein_esm2.npz")
    p.add_argument("--out_dir",       default="checkpoints/finetune")
    p.add_argument("--device",        default="cuda")
    p.add_argument("--min_samples",   type=int,   default=5)
    p.add_argument("--max_proteins",  type=int,   default=32)
    p.add_argument("--n_folds",       type=int,   default=5)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--phase1_epochs", type=int,   default=20)
    p.add_argument("--phase2_epochs", type=int,   default=60)
    p.add_argument("--phase1_lr",     type=float, default=1e-3)
    p.add_argument("--phase2_lr",     type=float, default=2e-5)
    p.add_argument("--patience",      type=int,   default=20)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}\n")

    # ── 1. Load labels ────────────────────────────────────────────────────────
    df = pd.read_excel(args.xlsx, usecols=["ID", "degradation_biosynthesis", "substrate_final"])
    df["degradation_biosynthesis"] = df["degradation_biosynthesis"].str.strip()
    df["substrate_final"]          = df["substrate_final"].str.strip().str.lower()
    df = df[df["degradation_biosynthesis"] == "degradation"].dropna(subset=["substrate_final"])
    df = df[df["substrate_final"] != ""]

    counts = Counter(df["substrate_final"])
    df = df[df["substrate_final"].map(counts) >= args.min_samples]
    print(f"Labelled PULs: {len(df):,}  ({df['substrate_final'].nunique()} substrates ≥ {args.min_samples} samples)")

    # ── 2. Load .faa files ────────────────────────────────────────────────────
    faa_dir = Path(args.faa_dir)
    pul_protein_map: dict[str, list[str]] = {}
    sequences:       dict[str, str]       = {}
    missing = 0

    for pul_id in df["ID"]:
        faa = faa_dir / f"{pul_id}.faa"
        if not faa.exists():
            missing += 1
            continue
        prots = []
        for rec in SeqIO.parse(faa, "fasta"):
            sequences[rec.id] = str(rec.seq)
            prots.append(rec.id)
        pul_protein_map[pul_id] = prots

    df = df[df["ID"].isin(pul_protein_map)]
    print(f"FAA found: {len(pul_protein_map):,}  (missing: {missing})")
    print(f"Unique proteins: {len(sequences):,}\n")

    # ── 3. ESM2 embeddings (cached) ───────────────────────────────────────────
    prot_embeddings = compute_esm2_embeddings(
        sequences, device, Path(args.cache_npz)
    )

    # ── 4. Label encode (merge rare → skipped already above) ─────────────────
    pul_ids    = df["ID"].tolist()
    substrates = df["substrate_final"].tolist()
    le         = LabelEncoder()
    y          = le.fit_transform(substrates)
    num_classes = len(le.classes_)

    class_weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=y)
    print(f"Classes: {num_classes}  |  class weight range: {class_weights.min():.2f}–{class_weights.max():.2f}\n")

    # ── 5. Cross-validation ───────────────────────────────────────────────────
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    fold_f1s   = []
    all_preds_cv  = np.full(len(y), -1, dtype=int)
    all_labels_cv = y.copy()

    for fold, (tr_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        best_f1, preds, labels = run_fold(
            fold, tr_idx, val_idx,
            pul_ids, y, pul_protein_map, prot_embeddings,
            args.checkpoint, num_classes, class_weights, args, device,
        )
        fold_f1s.append(best_f1)
        all_preds_cv[val_idx] = preds

    print(f"\n── CV Results ────────────────────────────────────────────────────")
    print(f"  macro-F1 per fold: {[f'{f:.3f}' for f in fold_f1s]}")
    print(f"  mean: {np.mean(fold_f1s):.3f}  std: {np.std(fold_f1s):.3f}")

    # ── 6. Per-class breakdown from CV predictions ────────────────────────────
    report = classification_report(
        all_labels_cv, all_preds_cv,
        target_names=le.classes_, zero_division=0,
    )
    print("\n── Per-class CV report ───────────────────────────────────────────")
    print(report)

    report_path = Path("results/finetune_cv_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(f"CV macro-F1: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}\n\n")
        f.write(report)
    print(f"Report saved → {report_path}")

    # ── 7. Final model on all data ────────────────────────────────────────────
    print("\n── Training final model on all data ──────────────────────────────")
    full_set = PULFinetuneDataset(pul_ids, y, pul_protein_map, prot_embeddings, args.max_proteins)
    full_ldr = DataLoader(full_set, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model     = make_model(args.checkpoint, num_classes, device)
    w_tensor  = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=w_tensor)

    model.freeze_encoder()
    opt1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.phase1_lr, weight_decay=0.01,
    )
    for ep in range(1, args.phase1_epochs + 1):
        train_epoch(model, full_ldr, opt1, criterion, device)

    model.unfreeze_encoder()
    opt2 = torch.optim.AdamW(model.parameters(), lr=args.phase2_lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=args.phase2_epochs)
    for ep in range(1, args.phase2_epochs + 1):
        loss = train_epoch(model, full_ldr, opt2, criterion, device)
        sched.step()
        if ep % 20 == 0:
            print(f"  ep {ep:3d}  loss {loss:.4f}")

    final_path = out_dir / "final_model.pt"
    torch.save(model.state_dict(), final_path)

    le_path = out_dir / "label_encoder.pkl"
    with open(le_path, "wb") as f:
        pickle.dump(le, f)

    print(f"\nFinal model → {final_path}")
    print(f"Label encoder → {le_path}")


if __name__ == "__main__":
    main()
