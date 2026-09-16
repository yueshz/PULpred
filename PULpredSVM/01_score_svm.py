"""
01_score_svm.py

GAG / Alginate novel PUL discovery using ESM2mean SVM.

Inference flow:
  1. Load protein-count cache (or build it once from FASTA files)
  2. Filter unannotated CGCs to those with >= min_proteins  [NEW: before SVM]
  3. Score filtered CGCs with trained SVM (RBF, probability=True)
  4. Rank by SVM probability; flag above CV-optimal F1 threshold
  5. DIAMOND vs CAZyDB — exclude CGCs with known GAG/alginate families
  6. Output top 50 novel candidates

SVM confidence:
  Threshold = CV F1-optimal probability (saved in svm_dir/threshold.npy).
  All top-50 candidates are reported ranked by score; 'above_threshold' marks
  high-confidence hits. The score distribution itself guides where to draw the
  line — a natural gap in scores indicates the boundary.

Usage:
  cd /work3/zhayu/PULpred
  conda run -p /work3/zhayu/envs/pulpred \\
      python PULpredSVM/01_score_svm.py --task gag
      python PULpredSVM/01_score_svm.py --task alginate
      python PULpredSVM/01_score_svm.py --task gag --min_proteins 3
      python PULpredSVM/01_score_svm.py --task gag --retrain
"""

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score, roc_auc_score

# ── Config ────────────────────────────────────────────────────────────────────

TRAIN_NPZ  = Path("ablation/esm2mean_embeddings.npz")
SCORE_NPZ  = Path("data/dbCAN_seq/embeddings/dbcanseq_unannotated_esm2mean_embeddings.npz")
PROT_COUNT_CACHE = Path("data/dbCAN_seq/embeddings/unannotated_protein_counts.npz")

TASKS = {
    "gag": {
        "substrate":      "glycosaminoglycan",
        "exclude_train":  {"glycosaminoglycan", "host glycan"},
        "families":       {"PL8","PL12","PL13","PL15","PL21","PL23",
                           "PL29","PL30","PL33","PL35","GH88"},
        "exclude_filter": {"PL8","PL12","PL13","PL15","PL21","PL23",
                           "PL29","PL30","PL33","PL35"},
    },
    "alginate": {
        "substrate":      "alginate",
        "exclude_train":  set(),
        "families":       None,
        "exclude_filter": {"PL5","PL6","PL7","PL14","PL15","PL17","PL18",
                           "PL31","PL32","PL34","PL36","PL38","PL39"},
    },
}

DBCAN_DIR = Path("data/dbCAN_seq")
CAZY_DMND = Path("data/raw/CAZyDB_local.dmnd")
CAZY_FA   = Path("data/raw/CAZyDB_local.fa")

SVM_WEIGHT = 10
TOP_N      = 50    # final novel candidates after DIAMOND
THREADS    = 8


# ── Helpers ───────────────────────────────────────────────────────────────────

def base_family(s):
    return re.sub(r"_\d+$", "", s.strip())


def parse_dbcanseq_contents(data_dir):
    cgc2contents = {}
    for env in ["HUMAN_GUT", "COW_RUMEN", "HUMAN_ORAL", "MARINE"]:
        env_dir = data_dir / env
        if not env_dir.exists():
            continue
        for f in env_dir.iterdir():
            if f.suffix or not f.is_file():
                continue
            for line in f.read_text(errors="replace").splitlines():
                parts = line.split("\t")
                if len(parts) >= 3 and parts[1].strip():
                    cgc2contents.setdefault(parts[1].strip(), parts[2].strip())
    return cgc2contents


def contents_has_family(contents, families):
    return any(base_family(tok) in families
               for tok in re.split(r"[,|]", contents))


def find_best_threshold(y_true, y_prob):
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.001, 0.99, 500):
        f = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t


def cgc_id_to_fasta(cgc_id, env):
    stem   = cgc_id.replace("|", "#")
    genome = stem.split("#")[0].rsplit("_", 1)[0]
    return DBCAN_DIR / env / genome / "dbcan_out" / "CGC_fasta" / f"{stem}.fasta"


def read_fasta(path):
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


def load_or_build_protein_counts(cgc_ids, envs):
    """Return protein counts for all unannotated CGCs, using cache if available."""
    if PROT_COUNT_CACHE.exists():
        cached = np.load(PROT_COUNT_CACHE, allow_pickle=True)
        if list(cached["cgc_id"]) == cgc_ids:
            print(f"  Protein count cache loaded ({len(cgc_ids):,} CGCs)")
            return cached["n_proteins"].tolist()
        print("  Cache mismatch — rebuilding protein counts…")

    print(f"  Building protein count cache for {len(cgc_ids):,} CGCs "
          f"(one-time, ~5 min)…")
    counts = []
    for i, (cid, env) in enumerate(zip(cgc_ids, envs)):
        fp = cgc_id_to_fasta(cid, env)
        n  = sum(1 for l in fp.read_text(errors="replace").splitlines()
                 if l.startswith(">")) if fp.exists() else 0
        counts.append(n)
        if (i + 1) % 10000 == 0:
            print(f"    {i+1:,}/{len(cgc_ids):,}…")

    np.savez(PROT_COUNT_CACHE,
             cgc_id=np.array(cgc_ids),
             n_proteins=np.array(counts))
    print(f"  Saved → {PROT_COUNT_CACHE}")
    return counts


def find_diamond():
    for p in [shutil.which("diamond"),
              "/zhome/68/5/210030/anaconda3/bin/diamond",
              "/usr/bin/diamond"]:
        if p and Path(p).exists():
            return p
    raise FileNotFoundError("diamond not found")


def run_diamond(fasta_str, dmnd_db, threads, out_tsv):
    with tempfile.NamedTemporaryFile("w", suffix=".faa", delete=False) as tmp:
        tmp.write(fasta_str)
        tmp_path = tmp.name
    subprocess.run([find_diamond(), "blastp",
                    "-q", tmp_path, "-d", str(dmnd_db),
                    "-o", str(out_tsv),
                    "--outfmt", "6", "qseqid", "sseqid", "pident", "evalue",
                    "-p", str(threads), "--sensitive", "-e", "1e-5", "--quiet"],
                   check=True)
    Path(tmp_path).unlink(missing_ok=True)


def build_cazy_lookup(fa_path):
    lookup = {}
    with open(fa_path) as f:
        for line in f:
            if not line.startswith(">"):
                continue
            header = line[1:].rstrip()
            acc = header.split()[0].split("|")[0]
            m   = re.search(r"\|([^|]+)\|", header)
            if m:
                lookup[acc] = frozenset(
                    base_family(tok)
                    for tok in re.split(r"[;,]", m.group(1)) if tok.strip()
                )
    return lookup


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task",         required=True, choices=["gag", "alginate"])
    p.add_argument("--min_proteins", type=int, default=5,
                   help="Minimum proteins per CGC before SVM scoring (default 5)")
    p.add_argument("--retrain",      action="store_true")
    args = p.parse_args()

    cfg     = TASKS[args.task]
    out_dir = Path(f"PULpredSVM/results/{args.task}_minprot{args.min_proteins}")
    out_dir.mkdir(parents=True, exist_ok=True)

    svm_dir   = Path(f"PULpredSVM/results/{args.task}_svm")
    svm_dir.mkdir(parents=True, exist_ok=True)
    svm_cache = svm_dir / "svm.joblib"
    thr_cache = svm_dir / "threshold.npy"

    # ── Step 1: Train SVM ────────────────────────────────────────────────────
    if svm_cache.exists() and thr_cache.exists() and not args.retrain:
        print(f"[1/5] Loading cached SVM…")
        clf       = joblib.load(svm_cache)
        threshold = float(np.load(thr_cache))
        print(f"  threshold = {threshold:.4f}\n")
    else:
        print(f"[1/5] Training {args.task.upper()} SVM (ESM2mean, RBF)…")
        npz  = np.load(TRAIN_NPZ, allow_pickle=True)
        ids  = npz["cgc_id"].tolist()
        subs = npz["substrate"].tolist()
        X    = npz["embedding"].astype(np.float32)

        if cfg["families"]:
            cgc2contents = parse_dbcanseq_contents(DBCAN_DIR)

        labels, keep = [], []
        for i, (cid, sub) in enumerate(zip(ids, subs)):
            if sub == cfg["substrate"]:
                if cfg["families"]:
                    if contents_has_family(cgc2contents.get(cid, ""), cfg["families"]):
                        labels.append(1); keep.append(i)
                else:
                    labels.append(1); keep.append(i)
            elif sub not in cfg["exclude_train"]:
                labels.append(0); keep.append(i)

        labels = np.array(labels)
        X_tr   = X[np.array(keep)]
        print(f"  {labels.sum()} positives / {(labels==0).sum():,} negatives")

        clf = Pipeline([("sc",  StandardScaler()),
                        ("svm", SVC(kernel="rbf", probability=True,
                                    class_weight={1: SVM_WEIGHT, 0: 1},
                                    random_state=42))])
        cv_probs  = cross_val_predict(clf, X_tr, labels,
                                      cv=5, method="predict_proba")[:, 1]
        threshold = find_best_threshold(labels, cv_probs)
        auc = roc_auc_score(labels, cv_probs)
        f1  = f1_score(labels, (cv_probs >= threshold).astype(int))
        print(f"  CV  AUC={auc:.4f}  F1@thr={f1:.4f}  threshold={threshold:.4f}")
        clf.fit(X_tr, labels)
        joblib.dump(clf, svm_cache)
        np.save(thr_cache, threshold)
        print(f"  Saved → {svm_cache}\n")

    # ── Step 2: Load unannotated CGCs & pre-filter by protein count ──────────
    print(f"[2/5] Loading unannotated ESM2mean embeddings…")
    npz     = np.load(SCORE_NPZ, allow_pickle=True)
    cgc_ids = npz["cgc_id"].tolist()
    envs    = npz["environment"].tolist()
    X_all   = npz["embedding"].astype(np.float32)
    print(f"  {len(cgc_ids):,} total unannotated CGCs")

    print(f"\n[3/5] Filtering CGCs with < {args.min_proteins} proteins…")
    prot_counts = load_or_build_protein_counts(cgc_ids, envs)
    prot_counts = np.array(prot_counts)
    mask        = prot_counts >= args.min_proteins
    cgc_ids_f   = [cgc_ids[i] for i in range(len(cgc_ids)) if mask[i]]
    envs_f      = [envs[i]    for i in range(len(cgc_ids)) if mask[i]]
    counts_f    = prot_counts[mask]
    X_f         = X_all[mask]
    print(f"  {mask.sum():,} / {len(cgc_ids):,} CGCs pass (≥{args.min_proteins} proteins)\n")

    # ── Step 3: Score filtered CGCs ──────────────────────────────────────────
    print(f"[4/5] Scoring {len(cgc_ids_f):,} filtered CGCs…")
    scores = clf.predict_proba(X_f)[:, 1]
    ranked = np.argsort(scores)[::-1]

    top_ids    = [cgc_ids_f[i] for i in ranked[:200]]
    top_envs   = [envs_f[i]    for i in ranked[:200]]
    top_scores = scores[ranked[:200]]
    top_counts = counts_f[ranked[:200]]

    above = int((top_scores >= threshold).sum())
    print(f"  Score range (top 200): {top_scores[-1]:.4f} – {top_scores[0]:.4f}")
    print(f"  Above CV threshold ({threshold:.4f}): {above}")
    print(f"  Score at rank 50 : {top_scores[49]:.4f}")
    print(f"  Score at rank 100: {top_scores[99]:.4f}\n")

    # Save full score table (filtered)
    pd.DataFrame({
        "cgc_id": cgc_ids_f, "environment": envs_f,
        "n_proteins": counts_f, "svm_score": scores,
    }).sort_values("svm_score", ascending=False).to_csv(
        out_dir / f"all_cgc_{args.task}_scores.csv", index=False)

    # ── Step 4: DIAMOND on top 200 ───────────────────────────────────────────
    print(f"[5/5] DIAMOND on top 200 candidates…")
    cgc2seqs = {}
    for cid, env in zip(top_ids, top_envs):
        fp = cgc_id_to_fasta(cid, env)
        cgc2seqs[cid] = read_fasta(fp) if fp.exists() else []

    all_seqs  = {pid: seq for seqs in cgc2seqs.values() for pid, seq in seqs}
    fasta_str = "\n".join(f">{pid}\n{seq}" for pid, seq in all_seqs.items()) + "\n"

    dmnd_out = out_dir / "diamond_hits.tsv"
    if dmnd_out.exists():
        print("  DIAMOND cached")
    else:
        run_diamond(fasta_str, CAZY_DMND, THREADS, dmnd_out)

    cazy_lookup = build_cazy_lookup(CAZY_FA)
    df_dmnd     = pd.read_csv(dmnd_out, sep="\t", header=None,
                               names=["qseqid", "sseqid", "pident", "evalue"])
    pid2cazy: dict[str, set] = {}
    for _, row in df_dmnd.iterrows():
        acc = str(row["sseqid"]).split()[0].split("|")[0]
        pid2cazy.setdefault(row["qseqid"], set())
        pid2cazy[row["qseqid"]] |= cazy_lookup.get(acc, frozenset())
    print(f"  {len(pid2cazy):,} proteins with CAZy hits")

    # ── Filter known families & output ───────────────────────────────────────
    results, n_excl = [], 0
    for cid, env, score, n_prot in zip(top_ids, top_envs, top_scores, top_counts):
        pids      = [pid for pid, _ in cgc2seqs.get(cid, [])]
        cazy_hits = set().union(*(pid2cazy.get(pid, set()) for pid in pids))
        if cazy_hits & cfg["exclude_filter"]:
            n_excl += 1
            continue
        pl_fams  = sorted(f for f in cazy_hits if f.startswith("PL"))
        gh_fams  = sorted(f for f in cazy_hits if f.startswith("GH"))
        cbm_fams = sorted(f for f in cazy_hits if f.startswith("CBM"))
        results.append({
            "rank":             len(results) + 1,
            "cgc_id":           cid,
            "environment":      env,
            "n_proteins":       int(n_prot),
            "svm_score":        round(float(score), 4),
            "above_threshold":  score >= threshold,
            "PL_families":      ";".join(pl_fams) or "—",
            "GH_families":      ";".join(gh_fams) or "—",
            "CBM_families":     ";".join(cbm_fams) or "—",
        })
        if len(results) == TOP_N:
            break

    print(f"  Excluded {n_excl} with known families → {len(results)} novel candidates\n")

    df_out = pd.DataFrame(results)
    print(df_out[["rank", "cgc_id", "environment", "n_proteins",
                  "svm_score", "above_threshold", "PL_families", "GH_families"]
                 ].to_string(index=False))

    csv_out = out_dir / f"novel_{args.task}_candidates.csv"
    df_out.to_csv(csv_out, index=False)
    print(f"\nSaved → {csv_out}")


if __name__ == "__main__":
    main()
