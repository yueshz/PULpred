"""
check_seqid_leakage.py

Rigorous sequence-identity leakage check using MMseqs2 clustering.

Pipeline:
  1. Collect all protein sequences from annotated CGC FASTA files (~344k proteins)
  2. Run MMseqs2 easy-cluster: --min-seq-id 0.30 -c 0.80 --cov-mode 0 (reciprocal)
  3. Map protein clusters → CGC groups (connected components of the CGC-cluster graph)
  4. Compare standard StratifiedKFold vs sequence-cluster GroupKFold AUC for 31 classes

Reference: DefensePredictor clustering criteria (30% identity, 80% reciprocal coverage)

Usage:
  cd /work3/zhayu/PULpred
  conda run -p /work3/zhayu/envs/pulpred python check_seqid_leakage.py [--skip_collect] [--skip_cluster]
"""

import argparse
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score

# ── Paths ──────────────────────────────────────────────────────────────────────
TRAIN_NPZ    = Path("ablation/esm2mean_embeddings.npz")
WORK_DIR     = Path("results/seqid_leakage_check")
COMBINED_FA  = WORK_DIR / "all_annotated_proteins.fasta"
CLUSTER_TSV  = WORK_DIR / "clusters_cluster.tsv"    # MMseqs2 output
MMSEQS_TMP   = WORK_DIR / "mmseqs_tmp"
RESULTS_CSV  = WORK_DIR / "seqid_leakage_results.csv"

EXCLUDE      = {"host glycan"}
MIN_SAMPLES  = 100
SVM_WEIGHT   = 10

# MMseqs2 clustering parameters (DefensePredictor style)
MIN_SEQ_ID   = 0.30   # 30% sequence identity
COVERAGE     = 0.80   # 80% reciprocal coverage


# ── SVM helper ─────────────────────────────────────────────────────────────────

def make_pipe():
    return Pipeline([
        ("sc",  StandardScaler()),
        ("clf", LinearSVC(class_weight={1: SVM_WEIGHT, 0: 1},
                          max_iter=2000, random_state=42)),
    ])


# ── Step 1: Collect protein sequences ─────────────────────────────────────────

def collect_proteins(ids, envs, out_fa: Path) -> dict[str, list[str]]:
    """
    Write all proteins from all annotated CGC FASTA files into one combined FASTA.
    Protein IDs are prefixed with CGC ID for traceability.

    Returns: {cgc_id: [protein_ids...]}
    """
    print(f"[1/4] Collecting protein sequences from {len(ids):,} CGCs…")
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    cgc_to_prots: dict[str, list[str]] = defaultdict(list)
    n_written = 0
    n_missing = 0

    with open(out_fa, "w") as fout:
        for cgc_id, env in zip(ids, envs):
            stem   = cgc_id.replace("|", "#")
            genome = stem.split("#")[0].rsplit("_", 1)[0]
            fa_path = Path(f"data/dbCAN_seq/{env}/{genome}/dbcan_out/CGC_fasta/{stem}.fasta")

            if not fa_path.exists():
                n_missing += 1
                continue

            for line in fa_path.read_text(errors="replace").splitlines():
                if line.startswith(">"):
                    # Unique protein ID: CGC_ID::original_header
                    orig = line[1:].split()[0]
                    pid  = f"{cgc_id}::{orig}"
                    fout.write(f">{pid}\n")
                    cgc_to_prots[cgc_id].append(pid)
                    n_written += 1
                else:
                    fout.write(line + "\n")

    print(f"  Written : {n_written:,} proteins")
    print(f"  Missing FASTAs: {n_missing}")
    return dict(cgc_to_prots)


# ── Step 2: MMseqs2 clustering ─────────────────────────────────────────────────

def run_mmseqs(fa: Path, out_prefix: Path, tmp: Path, threads: int = 8):
    """Run MMseqs2 easy-cluster and return path to cluster TSV."""
    MMSEQS_TMP.mkdir(parents=True, exist_ok=True)
    cmd = [
        "mmseqs", "easy-cluster",
        str(fa), str(out_prefix), str(tmp),
        "--min-seq-id", str(MIN_SEQ_ID),
        "-c",           str(COVERAGE),
        "--cov-mode",   "0",        # reciprocal coverage
        "--cluster-mode", "0",      # greedy set cover
        "--threads",    str(threads),
        "-v",           "2",
    ]
    print(f"\n[2/4] Running MMseqs2 easy-cluster…")
    print(f"  cmd: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-2000:])
        raise RuntimeError("MMseqs2 failed")
    print(result.stdout[-500:] if result.stdout else "  (no stdout)")
    cluster_tsv = Path(str(out_prefix) + "_cluster.tsv")
    if not cluster_tsv.exists():
        raise FileNotFoundError(f"MMseqs2 output not found: {cluster_tsv}")
    print(f"  Cluster TSV → {cluster_tsv}")
    return cluster_tsv


# ── Step 3: Build CGC groups from protein clusters ─────────────────────────────

def build_cgc_groups(cluster_tsv: Path, cgc_to_prots: dict) -> dict[str, int]:
    """
    Parse MMseqs2 cluster TSV (rep_seq, member_seq) and build CGC groups
    using connected components: two CGCs are linked if they share ≥1 protein cluster.

    Returns: {cgc_id: group_id}
    """
    print(f"\n[3/4] Building CGC groups from protein clusters…")

    # protein_id → cluster_representative
    prot_to_rep: dict[str, str] = {}
    n_clusters = 0
    with open(cluster_tsv) as f:
        for line in f:
            rep, member = line.rstrip().split("\t")
            prot_to_rep[member] = rep
            if rep == member:
                n_clusters += 1

    print(f"  Protein clusters: {n_clusters:,}")
    print(f"  Total proteins in TSV: {len(prot_to_rep):,}")

    # Build reverse map: cluster_rep → set of CGC IDs
    rep_to_cgcs: dict[str, set] = defaultdict(set)
    for cgc_id, prots in cgc_to_prots.items():
        for pid in prots:
            rep = prot_to_rep.get(pid)
            if rep is not None:
                rep_to_cgcs[rep].add(cgc_id)

    # Union-Find for connected components
    parent: dict[str, str] = {}

    def find(x):
        if x not in parent:
            parent[x] = x
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    all_cgcs = set(cgc_to_prots.keys())
    for c in all_cgcs:
        find(c)  # initialise

    for rep, cgcs in rep_to_cgcs.items():
        cgcs = list(cgcs)
        for i in range(1, len(cgcs)):
            union(cgcs[0], cgcs[i])

    # Assign integer group IDs
    root_to_gid: dict[str, int] = {}
    gid_counter = 0
    cgc_to_group: dict[str, int] = {}
    for cgc_id in all_cgcs:
        root = find(cgc_id)
        if root not in root_to_gid:
            root_to_gid[root] = gid_counter
            gid_counter += 1
        cgc_to_group[cgc_id] = root_to_gid[root]

    n_groups = gid_counter
    group_sizes = Counter(cgc_to_group.values())
    singletons  = sum(1 for s in group_sizes.values() if s == 1)
    max_size    = max(group_sizes.values())

    print(f"  CGC groups (connected components): {n_groups:,}")
    print(f"  Singletons (unique to one group)  : {singletons:,} ({100*singletons/n_groups:.1f}%)")
    print(f"  Largest group                     : {max_size:,} CGCs")

    return cgc_to_group


# ── Step 4: CV comparison ──────────────────────────────────────────────────────

def run_cv_comparison(ids, subs, X, cgc_to_group, classes):
    print(f"\n[4/4] Comparing Std-5fold vs SeqID-cluster GroupKFold ({len(classes)} classes)…")
    print(f"\n{'Class':<45} {'N':>5}  {'Std-5fold':>9}  {'SeqID-GKF':>9}  "
          f"{'Drop':>6}  {'n_groups':>8}")
    print("-" * 88)

    subs_arr   = np.array(subs)
    groups_arr = np.array([cgc_to_group.get(i, -1) for i in ids])

    rows = []
    for cls in classes:
        y = (subs_arr == cls).astype(int)

        # Standard 5-fold
        cv_std = cross_val_predict(
            make_pipe(), X, y,
            cv=StratifiedKFold(5, shuffle=True, random_state=42),
            method="decision_function")
        auc_std = roc_auc_score(y, expit(cv_std))

        # Sequence-identity-aware GroupKFold
        # Use 5 folds, fall back if not enough groups
        n_pos_groups = len(set(groups_arr[y == 1]))
        k = min(5, n_pos_groups)
        cv_gkf = cross_val_predict(
            make_pipe(), X, y,
            cv=GroupKFold(n_splits=k), groups=groups_arr,
            method="decision_function")
        auc_gkf = roc_auc_score(y, expit(cv_gkf))

        drop = auc_std - auc_gkf
        n_groups_pos = n_pos_groups

        rows.append({"class": cls, "n_pos": int(y.sum()),
                     "n_groups": n_groups_pos,
                     "auc_std": auc_std, "auc_seqid": auc_gkf, "drop": drop})

        flag = " <-- LARGE DROP" if drop > 0.05 else ""
        print(f"  {cls:<45} {y.sum():>5}  {auc_std:>9.4f}  {auc_gkf:>9.4f}  "
              f"{drop:>+6.3f}  {n_groups_pos:>8}{flag}")

    df = pd.DataFrame(rows).sort_values("drop", ascending=False)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nResults → {RESULTS_CSV}")

    print(f"\n── Summary ──────────────────────────────────────────────────")
    print(f"Mean std-CV AUC       : {df['auc_std'].mean():.4f}")
    print(f"Mean seqID-cluster AUC: {df['auc_seqid'].mean():.4f}")
    print(f"Mean AUC drop         : {df['drop'].mean():+.4f}")
    print(f"\nClasses with drop > 0.05:")
    big = df[df["drop"] > 0.05]
    if len(big):
        print(big[["class","n_pos","n_groups","auc_std","auc_seqid","drop"]].to_string(index=False))
    else:
        print("  None — sequence-identity leakage is not inflating CV scores.")

    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_collect",  action="store_true",
                    help="Skip protein collection if combined FASTA already exists")
    ap.add_argument("--skip_cluster",  action="store_true",
                    help="Skip MMseqs2 if cluster TSV already exists")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    # Load embeddings
    print("Loading annotated CGC embeddings…")
    npz  = np.load(TRAIN_NPZ, allow_pickle=True)
    ids  = npz["cgc_id"].tolist()
    subs = npz["substrate"].tolist()
    envs = npz["environment"].tolist()
    X    = npz["embedding"].astype(np.float32)

    counts  = Counter(subs)
    classes = sorted(s for s, n in counts.items()
                     if n >= MIN_SAMPLES and s not in EXCLUDE)
    print(f"{len(ids):,} CGCs, {len(classes)} classes with ≥{MIN_SAMPLES} samples\n")

    # Step 1: collect proteins
    cgc_map_path = WORK_DIR / "cgc_to_prots.npy"
    if args.skip_collect and COMBINED_FA.exists() and cgc_map_path.exists():
        print("[1/4] Skipping collection (files exist).")
        cgc_to_prots = np.load(cgc_map_path, allow_pickle=True).item()
    else:
        cgc_to_prots = collect_proteins(ids, envs, COMBINED_FA)
        np.save(cgc_map_path, cgc_to_prots)

    # Step 2: MMseqs2
    cluster_prefix = WORK_DIR / "clusters"
    if args.skip_cluster and CLUSTER_TSV.exists():
        print(f"[2/4] Skipping MMseqs2 (cluster TSV exists: {CLUSTER_TSV})")
    else:
        run_mmseqs(COMBINED_FA, cluster_prefix, MMSEQS_TMP, threads=args.threads)

    # Step 3: build CGC groups
    cgc_to_group = build_cgc_groups(CLUSTER_TSV, cgc_to_prots)

    # Step 4: CV comparison
    run_cv_comparison(ids, subs, X, cgc_to_group, classes)


if __name__ == "__main__":
    main()
