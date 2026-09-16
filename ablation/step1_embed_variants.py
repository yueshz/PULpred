"""
step1_embed_variants.py  [GPU required]

For all annotated dbCAN-seq CGCs, compute:
  1. ESM2 mean pooling  — average of per-protein ESM2 embeddings (640-dim)
  2. Random-CLS         — CLS token from randomly-initialised PULTransformer (512-dim)

Both share the same ESM2 inference pass to avoid redundant computation.
CLS-SVM embeddings are already in data/dbCAN_seq/embeddings/dbcanseq_cls_embeddings.npz.

Output:
  ablation/esm2mean_embeddings.npz   keys: cgc_id, substrate, environment, embedding
  ablation/random_cls_embeddings.npz keys: cgc_id, substrate, environment, embedding

Usage:
  cd /work3/zhayu/PULpred
  conda run -p /work3/zhayu/envs/pulpred python ablation/step1_embed_variants.py --device cuda
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model import PULTransformer

DBCAN_NPZ = Path("data/dbCAN_seq/embeddings/dbcanseq_cls_embeddings.npz")
DBCAN_DIR = Path("data/dbCAN_seq")
OUT_DIR   = Path("ablation")
ENVIRONMENTS = ["HUMAN_GUT", "COW_RUMEN", "HUMAN_ORAL", "MARINE"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device",       default="cuda")
    p.add_argument("--esm_batch",    type=int, default=32)
    p.add_argument("--cls_batch",    type=int, default=256)
    p.add_argument("--max_proteins", type=int, default=32)
    p.add_argument("--min_proteins", type=int, default=2)
    return p.parse_args()


def cgc_id_to_fasta(cgc_id: str, env: str) -> Path:
    stem   = cgc_id.replace("|", "#")
    genome = stem.split("#")[0].rsplit("_", 1)[0]
    return DBCAN_DIR / env / genome / "dbcan_out" / "CGC_fasta" / f"{stem}.fasta"


def read_fasta(path: Path) -> list[tuple[str, str]]:
    seqs, cur_id, cur_seq = [], None, []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith(">"):
            if cur_id is not None:
                seqs.append((cur_id, "".join(cur_seq)))
            parts = line[1:].split("|")
            cur_id = parts[2].split()[0] if len(parts) >= 3 else line[1:].split()[0]
            cur_seq = []
        else:
            cur_seq.append(line.strip())
    if cur_id is not None:
        seqs.append((cur_id, "".join(cur_seq)))
    return seqs


def compute_esm2(cgc_seqs: dict, device, batch_size: int) -> dict[str, np.ndarray]:
    import esm as esm_lib
    print("  Loading ESM2…")
    esm_model, alphabet = esm_lib.pretrained.esm2_t30_150M_UR50D()
    esm_model = esm_model.to(device).eval()
    bc = alphabet.get_batch_converter()

    all_proteins: dict[str, str] = {}
    for seqs in cgc_seqs.values():
        for pid, seq in seqs:
            if pid not in all_proteins:
                all_proteins[pid] = seq[:1022]

    print(f"  {len(all_proteins):,} unique proteins")
    sorted_pids = sorted(all_proteins, key=lambda p: len(all_proteins[p]))
    pid2emb: dict[str, np.ndarray] = {}

    for i in tqdm(range(0, len(sorted_pids), batch_size), desc="ESM2", unit="batch"):
        batch_pids = sorted_pids[i:i + batch_size]
        batch_data = [(p, all_proteins[p]) for p in batch_pids]
        _, _, tokens = bc(batch_data)
        with torch.no_grad():
            out = esm_model(tokens.to(device), repr_layers=[30], return_contacts=False)
        reps = out["representations"][30]
        for j, (pid, seq) in enumerate(batch_data):
            pid2emb[pid] = reps[j, 1:len(seq)+1].mean(0).cpu().float().numpy()

    del esm_model
    torch.cuda.empty_cache()
    return pid2emb


class CGCDataset(Dataset):
    def __init__(self, cgc_ids, cgc_seqs, pid2emb, max_proteins=32, emb_dim=640):
        self.max_proteins = max_proteins
        self.emb_dim = emb_dim
        self.items = []
        for cgc_id in cgc_ids:
            vecs = [pid2emb[pid] for pid, _ in cgc_seqs.get(cgc_id, []) if pid in pid2emb]
            if not vecs:
                continue
            n   = min(len(vecs), max_proteins)
            mat = np.zeros((max_proteins, emb_dim), dtype=np.float32)
            mat[:n] = np.stack(vecs[:n])
            self.items.append((cgc_id, mat, n, np.mean(vecs[:n], axis=0)))

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        cgc_id, mat, n, mean_emb = self.items[idx]
        embs = torch.tensor(mat)
        mask = torch.zeros(self.max_proteins, dtype=torch.bool)
        mask[:n] = True
        return {"cgc_id": cgc_id,
                "embeddings": embs,
                "attention_mask": mask,
                "esm2_mean": torch.tensor(mean_emb)}


@torch.no_grad()
def extract_random_cls(model, dataset, device, batch_size=256):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    cgc_ids_out, cls_list, mean_list = [], [], []
    for batch in tqdm(loader, desc="Random-CLS", unit="batch"):
        embs = batch["embeddings"].to(device)
        mask = batch["attention_mask"].to(device)
        cls_token = model.cls_token.expand(embs.size(0), -1, -1)
        x = model.input_proj(embs)
        x = torch.cat([cls_token, x], dim=1)
        cls_mask  = torch.ones(embs.size(0), 1, dtype=torch.bool, device=device)
        full_mask = torch.cat([cls_mask, mask], dim=1)
        out = model.encoder(x, src_key_padding_mask=~full_mask)
        cls_list.append(out[:, 0, :].cpu().float().numpy())
        mean_list.append(batch["esm2_mean"].numpy())
        cgc_ids_out.extend(batch["cgc_id"])
    return (cgc_ids_out,
            np.concatenate(cls_list, axis=0),
            np.concatenate(mean_list, axis=0))


def main():
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}\n")

    # ── Load annotated CGC IDs ────────────────────────────────────────────────
    print("[1/4] Loading annotated CGC metadata…")
    npz        = np.load(DBCAN_NPZ, allow_pickle=True)
    cgc_ids    = npz["cgc_id"].tolist()
    substrates = npz["substrate"].tolist()
    environments = npz["environment"].tolist()
    print(f"  {len(cgc_ids):,} annotated CGCs\n")

    # ── Read FASTA files ──────────────────────────────────────────────────────
    print("[2/4] Reading CGC FASTA files…")
    cgc_seqs: dict[str, list] = {}
    missing = 0
    for cgc_id, env in zip(cgc_ids, environments):
        fp = cgc_id_to_fasta(cgc_id, env)
        if fp.exists():
            seqs = read_fasta(fp)
            if len(seqs) >= args.min_proteins:
                cgc_seqs[cgc_id] = seqs
            else:
                missing += 1
        else:
            missing += 1
    print(f"  {len(cgc_seqs):,} CGCs with FASTA  ({missing} missing/skipped)\n")

    valid_ids = [c for c in cgc_ids if c in cgc_seqs]

    # ── ESM2 inference ────────────────────────────────────────────────────────
    print("[3/4] Computing ESM2 embeddings…")
    pid2emb = compute_esm2(cgc_seqs, device, args.esm_batch)

    # ── Random-CLS + ESM2mean ─────────────────────────────────────────────────
    print("\n[4/4] Extracting Random-CLS and ESM2 mean pooling…")
    model = PULTransformer()   # random weights — no checkpoint loaded
    model = model.to(device).eval()

    dataset = CGCDataset(valid_ids, cgc_seqs, pid2emb,
                         max_proteins=args.max_proteins)
    print(f"  Dataset: {len(dataset):,} CGCs")

    ids_out, random_cls, esm2_mean = extract_random_cls(
        model, dataset, device, batch_size=args.cls_batch)

    # build metadata arrays aligned to ids_out
    id2sub = dict(zip(cgc_ids, substrates))
    id2env = dict(zip(cgc_ids, environments))
    subs_out = np.array([id2sub[c] for c in ids_out])
    envs_out = np.array([id2env[c] for c in ids_out])

    # ── Save ──────────────────────────────────────────────────────────────────
    esm2_path = OUT_DIR / "esm2mean_embeddings.npz"
    np.savez(esm2_path,
             cgc_id      = np.array(ids_out),
             substrate   = subs_out,
             environment = envs_out,
             embedding   = esm2_mean.astype(np.float16))
    print(f"\nESM2mean saved → {esm2_path}  shape={esm2_mean.shape}")

    rcls_path = OUT_DIR / "random_cls_embeddings.npz"
    np.savez(rcls_path,
             cgc_id      = np.array(ids_out),
             substrate   = subs_out,
             environment = envs_out,
             embedding   = random_cls.astype(np.float16))
    print(f"Random-CLS saved → {rcls_path}  shape={random_cls.shape}")


if __name__ == "__main__":
    main()
