#!/usr/bin/env python3
"""
Pre-train PULTransformer with masked protein modelling.

For each PUL, randomly mask 15% of its proteins and train the model
to reconstruct the original ESM2 embeddings at those positions (MSE loss).

Usage:
    python pretrain.py \
        --xlsx  data/raw/PULDB.xlsx \
        --npz   data/embeddings/pul_esm2_t30_150M_mean_fp16.npz \
        --out_dir checkpoints/pretrain \
        --epochs 100 \
        --batch_size 256

Resume:
    Add --resume checkpoints/pretrain/checkpoint_latest.pt
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from src.dataset import PULDataset
from src.model import PULTransformer


# ── Args ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--xlsx",        default="data/raw/PULDB.xlsx")
    p.add_argument("--npz",         default="data/embeddings/pul_esm2_t30_150M_mean_fp16.npz")
    p.add_argument("--out_dir",     default="checkpoints/pretrain")
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch_size",  type=int,   default=256)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--weight_decay",type=float, default=0.01)
    p.add_argument("--mask_ratio",  type=float, default=0.15)
    p.add_argument("--max_proteins",type=int,   default=32)
    p.add_argument("--d_model",     type=int,   default=512)
    p.add_argument("--num_layers",  type=int,   default=4)
    p.add_argument("--nhead",       type=int,   default=8)
    p.add_argument("--dropout",     type=float, default=0.1)
    p.add_argument("--device",      default="cuda")
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--save_every",  type=int,   default=10)
    p.add_argument("--resume",      default=None, help="Path to checkpoint to resume from")
    return p.parse_args()


# ── Masking ────────────────────────────────────────────────────────────────────

def sample_masked_positions(
    attention_mask: torch.Tensor,
    mask_ratio: float,
) -> torch.Tensor:
    """
    Randomly mask mask_ratio of real (non-padding) protein positions.

    Args:
        attention_mask: [B, L] bool — True = real protein
        mask_ratio    : fraction of real proteins to mask

    Returns:
        masked_positions: [B, L] bool — True = this position is masked
    """
    rand = torch.rand_like(attention_mask, dtype=torch.float)
    return (rand < mask_ratio) & attention_mask


# ── Training step ──────────────────────────────────────────────────────────────

def step(model, batch, mask_ratio, device):
    embeddings    = batch["embeddings"].to(device)      # [B, L, 640]
    attention_mask = batch["attention_mask"].to(device)  # [B, L]

    masked_positions = sample_masked_positions(attention_mask, mask_ratio)

    # Keep targets before masking
    targets = embeddings.clone()

    _, pred_emb, _ = model(embeddings, attention_mask, masked_positions)

    # MSE only on masked positions
    mask_exp = masked_positions.unsqueeze(-1).expand_as(pred_emb)
    loss = F.mse_loss(pred_emb[mask_exp], targets[mask_exp])

    # Fraction of tokens actually masked (for logging)
    n_masked = masked_positions.sum().item()
    n_real   = attention_mask.sum().item()

    return loss, n_masked, n_real


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    dataset = PULDataset(
        xlsx_path=args.xlsx,
        npz_path=args.npz,
        max_proteins=args.max_proteins,
    )

    n_val   = max(1000, int(0.05 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Train : {n_train:,}   Val : {n_val:,}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = PULTransformer(
        input_dim=640,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters : {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 1
    best_val_loss = float("inf")

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {ckpt['epoch']}  (best val={best_val_loss:.4f})")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):

        # Train
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            loss, _, _ = step(model, batch, args.mask_ratio, device)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                loss, _, _ = step(model, batch, args.mask_ratio, device)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            print(f"  → New best saved ({best_val_loss:.4f})")

        # Periodic checkpoint (resumable)
        if epoch % args.save_every == 0:
            ckpt_path = out_dir / "checkpoint_latest.pt"
            torch.save({
                "epoch":         epoch,
                "model":         model.state_dict(),
                "optimizer":     optimizer.state_dict(),
                "scheduler":     scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "args":          vars(args),
            }, ckpt_path)
            print(f"  → Checkpoint saved to {ckpt_path}")

    print(f"\nDone.  Best val loss : {best_val_loss:.4f}")
    print(f"Best model          : {out_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
