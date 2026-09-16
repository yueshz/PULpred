"""
embed_dbcanseq_unannotated.py

Compute ESM2 embeddings for all UNANNOTATED CGCs in dbCAN-seq (i.e., CGCs
present in CGC_fasta directories but absent from substrate files).

Two modes, selected by --mean_only:
  --mean_only   ACTIVE / production path. Skips PULTransformer entirely
                (no checkpoint load) and saves ESM2 mean-pooled embeddings
                only. This is what PULpredSVM/01_score_svm.py and
                multitaskSVM/predict_esm2mean_multitask.py actually consume.
                (This is how embed_dbcanseq_unannotated.sh invokes it.)
  (default)     Legacy path. Also runs the pretrained PULTransformer to
                produce CLS embeddings — kept only for the archived
                CLS-vs-ESM2mean comparison (see legacy/, ablation/).

Output:
  data/dbCAN_seq/embeddings/dbcanseq_unannotated_esm2mean_embeddings.npz  (--mean_only)
    cgc_id      : str   [N]
    environment : str   [N]
    embedding   : fp16  [N, 640]

  data/dbCAN_seq/embeddings/dbcanseq_unannotated_cls_embeddings.npz  (legacy, default mode)
    cgc_id      : str   [N]
    environment : str   [N]
    embedding   : fp16  [N, 512]

Usage:
  cd /work3/zhayu/PULpred
  conda run -p /work3/zhayu/envs/pulpred \\
      python -u embeddings/embed_dbcanseq_unannotated.py --device cuda --mean_only
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

ENVIRONMENTS = ["HUMAN_GUT", "COW_RUMEN", "HUMAN_ORAL", "MARINE"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",     default="data/dbCAN_seq")
    p.add_argument("--checkpoint",   default="checkpoints/pretrain/best_model.pt")
    p.add_argument("--out_dir",      default="data/dbCAN_seq/embeddings")
    p.add_argument("--device",       default="cuda")
    p.add_argument("--esm_batch",    type=int, default=32)
    p.add_argument("--cls_batch",    type=int, default=512)
    p.add_argument("--max_proteins", type=int, default=32)
    p.add_argument("--min_proteins", type=int, default=2)
    p.add_argument("--mean_only",    action="store_true",
                   help="save ESM2 mean pooling only, skip PULTransformer")
    return p.parse_args()


# ── Step 1: collect annotated CGC IDs to skip ────────────────────────────────

def load_annotated_ids(data_dir: Path) -> set[str]:
    annotated = set()
    for env in ENVIRONMENTS:
        env_dir = data_dir / env
        if not env_dir.exists():
            continue
        for f in env_dir.iterdir():
            if f.is_file() and not f.suffix:
                for line in f.read_text(errors="replace").splitlines():
                    parts = line.split("\t")
                    if len(parts) >= 2 and parts[1].strip():
                        annotated.add(parts[1].strip())
    print(f"  Annotated CGC IDs to skip: {len(annotated):,}")
    return annotated


# ── Step 2: scan all CGC FASTA files, skip annotated ────────────────────────

def fasta_path_to_cgc_id(fasta_path: Path, env_dir: Path) -> str:
    """
    '.../MGYG000296075/dbcan_out/CGC_fasta/MGYG000296075_1#CGC11.fasta'
    → 'MGYG000296075_1|CGC11'
    """
    stem = fasta_path.stem          # MGYG000296075_1#CGC11
    return stem.replace("#", "|")


def read_fasta(path: Path) -> list[tuple[str, str]]:
    seqs, cur_id, cur_seq = [], None, []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith(">"):
            if cur_id is not None:
                seqs.append((cur_id, "".join(cur_seq)))
            parts = line[1:].split("|")
            cur_id = parts[2] if len(parts) >= 3 else line[1:].split()[0]
            cur_seq = []
        else:
            cur_seq.append(line.strip())
    if cur_id is not None:
        seqs.append((cur_id, "".join(cur_seq)))
    return seqs


def collect_unannotated(data_dir: Path, annotated: set[str],
                        min_proteins: int) -> tuple[dict, dict]:
    """
    Returns:
      cgc_seqs  : {cgc_id -> [(protein_id, seq), ...]}
      cgc_env   : {cgc_id -> environment}
    """
    cgc_seqs: dict[str, list] = {}
    cgc_env:  dict[str, str]  = {}
    missing, skipped_ann, skipped_few = 0, 0, 0

    for env in ENVIRONMENTS:
        env_dir = data_dir / env
        if not env_dir.exists():
            continue
        fasta_files = list(env_dir.rglob("CGC_fasta/*.fasta"))
        print(f"  {env}: {len(fasta_files):,} CGC FASTA files", flush=True)

        for fp in tqdm(fasta_files, desc=env, unit="CGC", leave=False):
            cgc_id = fasta_path_to_cgc_id(fp, env_dir)
            if cgc_id in annotated:
                skipped_ann += 1
                continue
            seqs = read_fasta(fp)
            if len(seqs) < min_proteins:
                skipped_few += 1
                continue
            cgc_seqs[cgc_id] = seqs
            cgc_env[cgc_id]  = env

    print(f"  Skipped annotated: {skipped_ann:,}")
    print(f"  Skipped (<{min_proteins} proteins): {skipped_few:,}")
    print(f"  Unannotated CGCs to embed: {len(cgc_seqs):,}")
    return cgc_seqs, cgc_env


# ── Step 3: ESM2 embeddings ──────────────────────────────────────────────────

def compute_esm2(cgc_seqs: dict, device, batch_size: int = 32) -> dict:
    import esm as esm_lib
    print("Loading ESM2…")
    esm_model, alphabet = esm_lib.pretrained.esm2_t30_150M_UR50D()
    esm_model = esm_model.to(device).eval()
    bc = alphabet.get_batch_converter()

    all_proteins: dict[str, str] = {}
    for seqs in cgc_seqs.values():
        for pid, seq in seqs:
            if pid not in all_proteins:
                all_proteins[pid] = seq[:1022]

    print(f"  {len(all_proteins):,} unique proteins to embed")
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
        reps = out["representations"][30]
        for j, (pid, seq) in enumerate(batch_data):
            pid2emb[pid] = reps[j, 1:len(seq)+1].mean(0).cpu().float().numpy()

    del esm_model
    torch.cuda.empty_cache()
    return pid2emb


# ── Step 4: PULTransformer CLS ───────────────────────────────────────────────

class CGCDataset(Dataset):
    def __init__(self, cgc_ids, cgc_seqs, pid2emb, max_proteins=32, emb_dim=640):
        self.emb_dim = emb_dim
        self.max_proteins = max_proteins
        self.items = []
        for cgc_id in cgc_ids:
            seqs = cgc_seqs.get(cgc_id, [])
            vecs = [pid2emb[pid] for pid, _ in seqs if pid in pid2emb]
            if not vecs:
                continue
            n = min(len(vecs), max_proteins)
            mat = np.zeros((max_proteins, emb_dim), dtype=np.float32)
            mat[:n] = np.stack(vecs[:n])
            self.items.append((cgc_id, mat, n))

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        cgc_id, mat, n = self.items[idx]
        embs = torch.tensor(mat)
        mask = torch.zeros(self.max_proteins, dtype=torch.bool)
        mask[:n] = True
        return {"cgc_id": cgc_id, "embeddings": embs, "attention_mask": mask}


@torch.no_grad()
def extract_cls(model, dataset, device, batch_size=512):
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
        cls_mask  = torch.ones(embs.size(0), 1, dtype=torch.bool, device=device)
        full_mask = torch.cat([cls_mask, mask], dim=1)
        out = model.encoder(x, src_key_padding_mask=~full_mask)
        cls_list.append(out[:, 0, :].cpu().float().numpy())
        cgc_ids_out.extend(batch["cgc_id"])
    return cgc_ids_out, np.concatenate(cls_list, axis=0)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}\n")

    print("[1/4] Loading annotated CGC IDs to skip…")
    annotated = load_annotated_ids(data_dir)

    print("\n[2/4] Scanning CGC FASTA files…")
    cgc_seqs, cgc_env = collect_unannotated(data_dir, annotated, args.min_proteins)

    print("\n[3/4] Computing ESM2 protein embeddings…")
    pid2emb = compute_esm2(cgc_seqs, device, batch_size=args.esm_batch)

    # ── ESM2 mean pooling ─────────────────────────────────────────────────────
    print("\nComputing per-CGC ESM2 mean pooling…")
    valid_ids = list(cgc_seqs.keys())
    mean_ids, mean_embs = [], []
    for cgc_id in valid_ids:
        vecs = [pid2emb[pid] for pid, _ in cgc_seqs[cgc_id] if pid in pid2emb]
        if vecs:
            mean_ids.append(cgc_id)
            mean_embs.append(np.mean(vecs, axis=0))

    mean_embs    = np.stack(mean_embs)
    environments = np.array([cgc_env[c] for c in mean_ids])
    mean_path    = out_dir / "dbcanseq_unannotated_esm2mean_embeddings.npz"
    np.savez(mean_path,
             cgc_id      = np.array(mean_ids),
             environment = environments,
             embedding   = mean_embs.astype(np.float16))
    print(f"Saved {len(mean_ids):,} ESM2mean embeddings → {mean_path}")

    if args.mean_only:
        from collections import Counter
        for env, n in sorted(Counter(environments).items()):
            print(f"  {env}: {n:,}")
        return

    print("\n[4/4] Extracting PULTransformer CLS embeddings…")
    model = PULTransformer()
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model = model.to(device).eval()

    dataset = CGCDataset(valid_ids, cgc_seqs, pid2emb,
                         max_proteins=args.max_proteins)
    print(f"  Dataset: {len(dataset):,} CGCs")

    cgc_ids_out, cls_embeddings = extract_cls(model, dataset, device,
                                              batch_size=args.cls_batch)

    environments = np.array([cgc_env[c] for c in cgc_ids_out])
    out_path = out_dir / "dbcanseq_unannotated_cls_embeddings.npz"
    np.savez(out_path,
             cgc_id      = np.array(cgc_ids_out),
             environment = environments,
             embedding   = cls_embeddings.astype(np.float16))

    print(f"\nSaved {len(cgc_ids_out):,} CLS embeddings → {out_path}")
    from collections import Counter
    for env, n in sorted(Counter(environments).items()):
        print(f"  {env}: {n:,}")


if __name__ == "__main__":
    main()
