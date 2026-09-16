import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PULDataset(Dataset):
    """
    PyTorch Dataset for PUL pre-training.

    Each item is a PUL represented as a variable-length ordered sequence of
    protein ESM2 embeddings, padded to max_proteins.

    Args:
        xlsx_path    : path to PULDB.xlsx
        npz_path     : path to pul_esm2_*_mean_fp16.npz (per-protein embeddings)
        max_proteins : maximum number of proteins per PUL (longer PULs are truncated)
        min_proteins : PULs with fewer embeddable proteins are excluded
    """

    def __init__(
        self,
        xlsx_path: str,
        npz_path: str,
        max_proteins: int = 32,
        min_proteins: int = 2,
    ):
        super().__init__()
        self.max_proteins = max_proteins

        # ── Load per-protein embeddings ────────────────────────────────────────
        data = np.load(npz_path, allow_pickle=True)
        prot_ids  = data["protein_id"].tolist()
        prot_embs = data["embedding"]           # [N, 640] float16
        self.emb_dim   = prot_embs.shape[1]
        self.prot_embs = prot_embs              # kept as float16, cast in __getitem__
        self.prot_index: dict[str, int] = {pid: i for i, pid in enumerate(prot_ids)}

        # ── Load PULDB and group proteins by PUL ───────────────────────────────
        df = pd.read_excel(xlsx_path, sheet_name=0, dtype=str)
        df["protein_id"] = df["protein_id"].fillna("").str.strip()

        sort_cols = [c for c in ("contig", "start") if c in df.columns]
        if "start" in df.columns:
            df["start"] = pd.to_numeric(df["start"], errors="coerce")

        self.pul_names: list[str] = []
        self.pul_indices: list[list[int]] = []   # embedding row indices per PUL

        for name, group in df.groupby("Name", sort=False):
            if sort_cols:
                group = group.sort_values(sort_cols, na_position="last")

            indices = [
                self.prot_index[p]
                for p in group["protein_id"]
                if p and p in self.prot_index
            ]
            if len(indices) < min_proteins:
                continue

            self.pul_names.append(name)
            self.pul_indices.append(indices[:max_proteins])

        print(
            f"PULDataset: {len(self.pul_names):,} PULs "
            f"(≥{min_proteins} proteins, max {max_proteins})"
        )

    def __len__(self) -> int:
        return len(self.pul_names)

    def __getitem__(self, idx: int) -> dict:
        indices = self.pul_indices[idx]
        n = len(indices)

        # Cast float16 → float32 here (GPU ops need float32)
        embs = torch.tensor(
            self.prot_embs[indices].astype(np.float32)
        )  # [n, emb_dim]

        # Pad to max_proteins with zeros
        pad = torch.zeros(self.max_proteins - n, self.emb_dim)
        embs = torch.cat([embs, pad], dim=0)    # [max_proteins, emb_dim]

        # Attention mask: True = real token, False = padding
        attn_mask = torch.zeros(self.max_proteins, dtype=torch.bool)
        attn_mask[:n] = True

        return {
            "embeddings":    embs,       # [max_proteins, emb_dim]  float32
            "attention_mask": attn_mask, # [max_proteins]            bool
            "n_proteins":    n,
        }


class LabelledPULDataset(Dataset):
    """
    Wraps PULDataset with substrate labels for fine-tuning.

    Args:
        base_dataset : a PULDataset instance (already built)
        labels_df    : DataFrame with columns ['Name', 'label'] where label is int
    """

    def __init__(self, base_dataset: PULDataset, labels_df: pd.DataFrame):
        self.base = base_dataset

        label_map: dict[str, int] = dict(
            zip(labels_df["Name"], labels_df["label"].astype(int))
        )

        self.indices: list[int] = []
        self.labels:  list[int] = []

        for i, name in enumerate(base_dataset.pul_names):
            if name in label_map:
                self.indices.append(i)
                self.labels.append(label_map[name])

        print(f"LabelledPULDataset: {len(self.indices):,} labelled PULs")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        item = self.base[self.indices[idx]]
        item["label"] = self.labels[idx]
        return item
