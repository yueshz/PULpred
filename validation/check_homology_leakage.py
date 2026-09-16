"""
check_homology_leakage.py

Diagnose sequence homology leakage in the current 5-fold CV:
  1. Within-class cosine similarity distribution (already done, shown here compactly)
  2. Leave-Genome-Out (LGO) CV vs standard 5-fold CV for all 31 classes

Genome is extracted from CGC ID: "GENOME_X|CGCN" → genome = GENOME_X
 e.g. MGYG000000949_143|CGC1 → MGYG000000949_143

Usage:
  cd /work3/zhayu/PULpred
  conda run -p /work3/zhayu/envs/pulpred python validation/check_homology_leakage.py
  conda run -p /work3/zhayu/envs/pulpred python validation/check_homology_leakage.py --class gag
"""

import argparse
from collections import Counter

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold, LeaveOneGroupOut, GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

TRAIN_NPZ    = "ablation/esm2mean_embeddings.npz"
EXCLUDE      = {"host glycan"}
MIN_SAMPLES  = 100
SVM_WEIGHT   = 10


def make_pipe():
    return Pipeline([
        ("sc",  StandardScaler()),
        ("clf", LinearSVC(class_weight={1: SVM_WEIGHT, 0: 1},
                          max_iter=2000, random_state=42)),
    ])


def genome_from_cgc(cgc_id: str) -> str:
    return cgc_id.split("|")[0]   # GENOME_X part, e.g. MGYG000000949_143


def lgo_cv(X, y, groups, n_splits=5):
    """
    Grouped k-fold: ensure CGCs from the same genome stay in the same fold.
    Falls back to fewer folds if not enough unique-genome groups.
    """
    n_groups = len(set(groups))
    k = min(n_splits, n_groups)
    cv = GroupKFold(n_splits=k)
    scores = cross_val_predict(make_pipe(), X, y,
                               cv=cv, groups=groups,
                               method="decision_function")
    return expit(scores)


def standard_cv(X, y):
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    scores = cross_val_predict(make_pipe(), X, y,
                               cv=cv, method="decision_function")
    return expit(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--class", dest="only_class", default=None,
                    help="Only run for this one class (e.g. 'glycosaminoglycan')")
    ap.add_argument("--top_n", type=int, default=10,
                    help="Print top N classes by AUC drop (default 10)")
    args = ap.parse_args()

    print("Loading embeddings…")
    npz  = np.load(TRAIN_NPZ, allow_pickle=True)
    ids  = npz["cgc_id"].tolist()
    subs = npz["substrate"].tolist()
    X    = npz["embedding"].astype(np.float32)

    genomes = [genome_from_cgc(i) for i in ids]

    counts  = Counter(subs)
    classes = sorted(s for s, n in counts.items()
                     if n >= MIN_SAMPLES and s not in EXCLUDE)

    if args.only_class:
        if args.only_class not in counts:
            # try partial match
            matches = [c for c in classes if args.only_class.lower() in c.lower()]
            if not matches:
                print(f"Class '{args.only_class}' not found. Available:\n  " +
                      "\n  ".join(classes))
                return
            classes = matches
        else:
            classes = [args.only_class]

    subs_arr   = np.array(subs)
    groups_arr = np.array(genomes)

    print(f"\n{'Class':<45} {'N':>5}  {'Std-5fold AUC':>13}  {'LGO-5fold AUC':>13}  "
          f"{'Drop':>6}  {'n_genomes':>9}")
    print("-" * 100)

    # Load existing checkpoint
    done = set()
    rows = []
    if RESULTS_CSV.exists():
        existing = pd.read_csv(RESULTS_CSV)
        for _, r in existing.iterrows():
            done.add(r["class"])
            rows.append(r.to_dict())
        print(f"  Resuming: {len(done)} classes already done\n")

    for cls in classes:
        if cls in done:
            flag = " <-- cached"
            r = next(r for r in rows if r["class"] == cls)
            print(f"  {cls:<45} {r['n_pos']:>5}  {r['std_auc']:>13.4f}  {r['lgo_auc']:>13.4f}  "
                  f"{r['drop']:>+6.3f}  {r['n_genomes']:>9}{flag}")
            continue

        y      = (subs_arr == cls).astype(int)
        n_pos  = y.sum()
        g      = groups_arr
        n_genomes = len(set(g[y == 1]))

        try:
            std_probs = standard_cv(X, y)
            std_auc   = roc_auc_score(y, std_probs)
        except Exception as e:
            std_auc = float("nan")
            print(f"  {cls}: standard CV error: {e}")

        try:
            lgo_probs = lgo_cv(X, y, g, n_splits=5)
            lgo_auc   = roc_auc_score(y, lgo_probs)
        except Exception as e:
            lgo_auc = float("nan")
            print(f"  {cls}: LGO CV error: {e}")

        drop = std_auc - lgo_auc
        rows.append({"class": cls, "n_pos": n_pos, "n_genomes": n_genomes,
                     "std_auc": std_auc, "lgo_auc": lgo_auc, "drop": drop})

        # Save checkpoint after each class
        pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)

        flag = " <-- LARGE DROP" if drop > 0.05 else ""
        print(f"  {cls:<45} {n_pos:>5}  {std_auc:>13.4f}  {lgo_auc:>13.4f}  "
              f"{drop:>+6.3f}  {n_genomes:>9}{flag}")

    df = pd.DataFrame(rows).sort_values("drop", ascending=False)
    df.to_csv("results/homology_leakage_check.csv", index=False)
    print(f"\nFull results → results/homology_leakage_check.csv")

    print(f"\n── Top {args.top_n} classes by AUC drop (std − LGO) ──")
    print(df.head(args.top_n)[["class","n_pos","n_genomes","std_auc","lgo_auc","drop"]
                               ].to_string(index=False))

    mean_std = df["std_auc"].mean()
    mean_lgo = df["lgo_auc"].mean()
    print(f"\nMean std-CV AUC : {mean_std:.4f}")
    print(f"Mean LGO-CV AUC : {mean_lgo:.4f}")
    print(f"Mean drop       : {mean_std - mean_lgo:+.4f}")


if __name__ == "__main__":
    main()
