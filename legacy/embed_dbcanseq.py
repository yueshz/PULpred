"""
embed_dbcanseq.py

Compute PULTransformer CLS embeddings for all substrate-labeled CGCs in
dbCAN-seq (Human Gut, Cow Rumen, Human Oral, Marine).

Pipeline:
  1. Parse substrate files → labeled CGC IDs + substrate labels
  2. Read protein sequences from per-CGC FASTA files
  3. Batch ESM2 computation across all proteins (sorted by length)
  4. Group per-CGC protein embeddings → PULTransformer → CLS embeddings
  5. Save NPZ with cgc_id / substrate / environment / embedding arrays

Output:
  data/dbCAN_seq/embeddings/dbcanseq_cls_embeddings.npz
    cgc_id      : str   [N]
    substrate   : str   [N]
    environment : str   [N]
    embedding   : fp16  [N, 512]

Usage:
  cd /work3/zhayu/PULpred
  conda run -p /work3/zhayu/envs/pulpred \\
      python embed_dbcanseq.py --device cuda
"""

import argparse
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import PULTransformer


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    default="data/dbCAN_seq")
    p.add_argument("--checkpoint",  default="checkpoints/pretrain/best_model.pt")
    p.add_argument("--out_dir",     default="data/dbCAN_seq/embeddings")
    p.add_argument("--device",      default="cuda")
    p.add_argument("--esm_batch",   type=int, default=32,
                   help="proteins per ESM2 batch")
    p.add_argument("--cls_batch",   type=int, default=512,
                   help="CGCs per PULTransformer batch")
    p.add_argument("--max_proteins",type=int, default=32)
    p.add_argument("--min_proteins",type=int, default=2)
    return p.parse_args()


ENVIRONMENTS = ["HUMAN_GUT", "COW_RUMEN", "HUMAN_ORAL", "MARINE"]


# ── Step 1: parse substrate files ────────────────────────────────────────────

def parse_substrates(data_dir: Path) -> dict[str, dict]:
    """
    Returns {cgc_id -> {"substrate": str, "environment": str}}.
    Skips duplicate CGC IDs (keeps first occurrence).
    """
    cgc_meta: dict[str, dict] = {}

    for env in ENVIRONMENTS:
        env_dir = data_dir / env
        if not env_dir.exists():
            print(f"  [warn] {env} directory not found, skipping")
            continue

        n_before = len(cgc_meta)
        for f in sorted(env_dir.iterdir()):
            # substrate files have no extension and are not tar archives
            if f.suffix or not f.is_file():
                continue
            substrate = f.name
            for line in f.read_text(errors="replace").splitlines():
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                cgc_id = parts[1].strip()
                if cgc_id and cgc_id not in cgc_meta:
                    cgc_meta[cgc_id] = {"substrate": substrate,
                                        "environment": env}

        n_new = len(cgc_meta) - n_before
        print(f"  {env}: {n_new:,} labeled CGCs")

    print(f"  Total: {len(cgc_meta):,} unique labeled CGCs")
    return cgc_meta


# ── Step 2: read FASTA sequences ─────────────────────────────────────────────

def cgc_id_to_fasta_path(cgc_id: str) -> str:
    """
    'MGYG000296075_1|CGC11'
    → 'MGYG000296075/dbcan_out/CGC_fasta/MGYG000296075_1#CGC11.fasta'
    """
    genome_contig = cgc_id.split("|")[0]                     # MGYG000296075_1
    genome = re.sub(r"_\d+$", "", genome_contig)             # MGYG000296075
    fname = cgc_id.replace("|", "#") + ".fasta"              # MGYG000296075_1#CGC11.fasta
    return f"{genome}/dbcan_out/CGC_fasta/{fname}"


def read_fasta(path: Path) -> list[tuple[str, str]]:
    seqs, cur_id, cur_seq = [], None, []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith(">"):
            if cur_id is not None:
                seqs.append((cur_id, "".join(cur_seq)))
            # Header: GENOME_CONTIG|CGCX|PROTEIN_ID|TYPE
            parts = line[1:].split("|")
            cur_id = parts[2] if len(parts) >= 3 else line[1:].split()[0]
            cur_seq = []
        else:
            cur_seq.append(line.strip())
    if cur_id is not None:
        seqs.append((cur_id, "".join(cur_seq)))
    return seqs


def collect_sequences(cgc_meta: dict, data_dir: Path,
                      min_proteins: int) -> dict[str, list[tuple[str, str]]]:
    """
    Returns {cgc_id -> [(protein_id, sequence), ...]} for all labeled CGCs
    that have enough proteins and whose FASTA file exists.
    """
    # Group CGC IDs by environment
    env_cgcs: dict[str, list[str]] = defaultdict(list)
    for cgc_id, meta in cgc_meta.items():
        env_cgcs[meta["environment"]].append(cgc_id)

    cgc_seqs: dict[str, list] = {}
    missing = 0

    for env, cgc_ids in env_cgcs.items():
        env_dir = data_dir / env
        print(f"  Reading sequences: {env} ({len(cgc_ids):,} CGCs)…")
        for cgc_id in tqdm(cgc_ids, desc=env, unit="CGC", leave=False):
            rel_path = cgc_id_to_fasta_path(cgc_id)
            fasta_path = env_dir / rel_path
            if not fasta_path.exists():
                missing += 1
                continue
            seqs = read_fasta(fasta_path)
            if len(seqs) >= min_proteins:
                cgc_seqs[cgc_id] = seqs

    print(f"  Loaded {len(cgc_seqs):,} CGCs  ({missing:,} FASTA files not found)")
    return cgc_seqs


# ── Step 3: ESM2 embeddings ───────────────────────────────────────────────────

def compute_esm2(cgc_seqs: dict[str, list], device,
                 batch_size: int = 32) -> dict[str, np.ndarray]:
    """
    Returns {protein_id -> np.ndarray [640] float32}.
    Batches globally across all proteins (sorted by length for efficiency).
    """
    import esm as esm_lib

    print("Loading ESM2 (esm2_t30_150M_UR50D)…")
    esm_model, alphabet = esm_lib.pretrained.esm2_t30_150M_UR50D()
    esm_model = esm_model.to(device).eval()
    bc = alphabet.get_batch_converter()

    # Flatten all proteins; deduplicate by protein_id
    all_proteins: dict[str, str] = {}
    for seqs in cgc_seqs.values():
        for pid, seq in seqs:
            if pid not in all_proteins:
                all_proteins[pid] = seq[:1022]      # ESM2 max length

    print(f"  {len(all_proteins):,} unique proteins to embed")

    # Sort by length for efficient padding
    sorted_pids = sorted(all_proteins, key=lambda p: len(all_proteins[p]))

    pid2emb: dict[str, np.ndarray] = {}

    for i in tqdm(range(0, len(sorted_pids), batch_size),
                  desc="ESM2", unit="batch"):
        batch_pids = sorted_pids[i:i + batch_size]
        batch_data = [(pid, all_proteins[pid]) for pid in batch_pids]
        _, _, tokens = bc(batch_data)
        with torch.no_grad():
            out = esm_model(tokens.to(device), repr_layers=[30],
                            return_contacts=False)
        reps = out["representations"][30]       # [B, L+2, 640]
        for j, (pid, seq) in enumerate(batch_data):
            slen = len(seq)
            emb = reps[j, 1:slen + 1].mean(0).cpu().float().numpy()
            pid2emb[pid] = emb

    del esm_model
    torch.cuda.empty_cache()
    return pid2emb


# ── Step 4: PULTransformer CLS embeddings ────────────────────────────────────

class CGCDataset(Dataset):
    """Per-CGC protein embedding matrix dataset for PULTransformer inference."""

    def __init__(self, cgc_ids: list[str], cgc_seqs: dict,
                 pid2emb: dict[str, np.ndarray],
                 max_proteins: int = 32, emb_dim: int = 640):
        self.emb_dim = emb_dim
        self.max_proteins = max_proteins
        self.items: list[tuple[str, np.ndarray, int]] = []  # (cgc_id, emb, n)

        for cgc_id in cgc_ids:
            seqs = cgc_seqs.get(cgc_id, [])
            vecs = [pid2emb[pid] for pid, _ in seqs if pid in pid2emb]
            if not vecs:
                continue
            n = min(len(vecs), max_proteins)
            mat = np.zeros((max_proteins, emb_dim), dtype=np.float32)
            mat[:n] = np.stack(vecs[:n])
            self.items.append((cgc_id, mat, n))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        cgc_id, mat, n = self.items[idx]
        embs = torch.tensor(mat)                        # [max_proteins, 640]
        mask = torch.zeros(self.max_proteins, dtype=torch.bool)
        mask[:n] = True
        return {"cgc_id": cgc_id, "embeddings": embs, "attention_mask": mask}


@torch.no_grad()
def extract_cls(model, dataset: CGCDataset, device,
                batch_size: int = 512) -> tuple[list[str], np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=2)
    model.eval()
    cgc_ids_out, cls_list = [], []

    for batch in tqdm(loader, desc="CLS", unit="batch"):
        embs = batch["embeddings"].to(device)
        mask = batch["attention_mask"].to(device)
        cls_token = model.cls_token.expand(embs.size(0), -1, -1)
        x = model.input_proj(embs)
        x = torch.cat([cls_token, x], dim=1)
        cls_mask = torch.ones(embs.size(0), 1, dtype=torch.bool, device=device)
        full_mask = torch.cat([cls_mask, mask], dim=1)
        key_pad = ~full_mask
        out = model.encoder(x, src_key_padding_mask=key_pad)
        cls_out = out[:, 0, :]                          # [B, 512]
        cls_list.append(cls_out.cpu().float().numpy())
        cgc_ids_out.extend(batch["cgc_id"])

    return cgc_ids_out, np.concatenate(cls_list, axis=0)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Parse substrates ───────────────────────────────────────────────────
    print("\n[1/4] Parsing substrate files…")
    cgc_meta = parse_substrates(data_dir)

    # ── 2. Collect sequences ──────────────────────────────────────────────────
    print("\n[2/4] Collecting CGC protein sequences…")
    cgc_seqs = collect_sequences(cgc_meta, data_dir, args.min_proteins)

    # Filter meta to only CGCs we actually have sequences for
    valid_ids = list(cgc_seqs.keys())
    print(f"  {len(valid_ids):,} CGCs with sequences (≥{args.min_proteins} proteins)")

    # ── 3. ESM2 embeddings ────────────────────────────────────────────────────
    print("\n[3/4] Computing ESM2 protein embeddings…")
    pid2emb = compute_esm2(cgc_seqs, device, batch_size=args.esm_batch)

    # ── 4. PULTransformer CLS ─────────────────────────────────────────────────
    print("\n[4/4] Extracting PULTransformer CLS embeddings…")
    print(f"  Loading checkpoint: {args.checkpoint}")
    model = PULTransformer()
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()

    dataset = CGCDataset(valid_ids, cgc_seqs, pid2emb,
                         max_proteins=args.max_proteins)
    print(f"  Dataset: {len(dataset):,} CGCs")

    cgc_ids_out, cls_embeddings = extract_cls(model, dataset, device,
                                              batch_size=args.cls_batch)

    # ── Save ──────────────────────────────────────────────────────────────────
    substrates   = np.array([cgc_meta[c]["substrate"]    for c in cgc_ids_out])
    environments = np.array([cgc_meta[c]["environment"]  for c in cgc_ids_out])
    cgc_ids_arr  = np.array(cgc_ids_out)

    out_path = out_dir / "dbcanseq_cls_embeddings.npz"
    np.savez(
        out_path,
        cgc_id      = cgc_ids_arr,
        substrate   = substrates,
        environment = environments,
        embedding   = cls_embeddings.astype(np.float16),
    )
    print(f"\nSaved {len(cgc_ids_out):,} CLS embeddings → {out_path}")

    # Summary
    from collections import Counter
    env_counts = Counter(environments)
    sub_counts = Counter(substrates)
    print("\nPer-environment:")
    for env, n in sorted(env_counts.items()):
        print(f"  {env}: {n:,}")
    print(f"\nTop substrates:")
    for sub, n in sub_counts.most_common(10):
        print(f"  {sub}: {n:,}")


if __name__ == "__main__":
    main()
