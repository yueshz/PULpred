"""
predict_esm2mean_multitask.py

One-vs-rest multi-class classifier covering all dbCAN-seq substrates with
>= MIN_SAMPLES annotated CGCs.  Outputs per-CGC probability vectors for 126k
unannotated CGCs; flags CGCs where all probabilities are below threshold as
"putative novel".

Models:
  linear (default) — LinearSVC + Platt calibration. ~2 min/class, 31 classes
                      complete in ~1 hour. Recommended.
  rbf              — RBF SVM. ~50 min/class, too slow for 31 classes (27 hrs).
  mlp              — sklearn MLPClassifier, ~3 min/class.

Usage:
  cd /work3/zhayu/PULpred
  conda run -p /work3/zhayu/envs/pulpred \\
      python multitaskSVM/predict_esm2mean_multitask.py                  # linear SVM
      python multitaskSVM/predict_esm2mean_multitask.py --model mlp
      python multitaskSVM/predict_esm2mean_multitask.py --min_samples 200
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

# ── Config ────────────────────────────────────────────────────────────────────

TRAIN_NPZ = Path("ablation/esm2mean_embeddings.npz")
SCORE_NPZ = Path("data/dbCAN_seq/embeddings/dbcanseq_unannotated_esm2mean_embeddings.npz")
OUT_DIR   = Path("results/multitask_esm2mean")

# These meta-labels overlap with other substrates — exclude from classes
EXCLUDE_SUBSTRATES = {"host glycan"}

MIN_SAMPLES       = 100
SVM_WEIGHT        = 10      # class_weight for minority class
NOVELTY_THRESHOLD = 0.3     # CGC is "putative novel" if max(prob) < this
TOP_N_NOVEL       = 200     # top CGCs flagged as novel in output


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_best_threshold(y_true, y_prob):
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.001, 0.99, 500):
        f = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f > best_f1: best_f1, best_t = f, t
    return best_t


def make_linear_cv():
    """Plain LinearSVC pipeline for cross_val_predict (decision_function, no nested CV)."""
    return Pipeline([
        ("sc",  StandardScaler()),
        ("clf", LinearSVC(class_weight={1: SVM_WEIGHT, 0: 1},
                          max_iter=2000, random_state=42)),
    ])


def make_linear_final():
    """LinearSVC + Platt calibration for the saved model (predict_proba support)."""
    linear = LinearSVC(class_weight={1: SVM_WEIGHT, 0: 1},
                       max_iter=2000, random_state=42)
    return Pipeline([
        ("sc",  StandardScaler()),
        ("clf", CalibratedClassifierCV(linear, cv=5, method="sigmoid")),
    ])


def make_rbf():
    """RBF SVM. Accurate but O(n²) — too slow for 31 classes on 40k samples."""
    return Pipeline([
        ("sc",  StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True,
                    class_weight={1: SVM_WEIGHT, 0: 1},
                    random_state=42)),
    ])


def make_mlp():
    return Pipeline([
        ("sc",  StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(512, 256),
                              activation="relu",
                              max_iter=300,
                              early_stopping=True,
                              validation_fraction=0.1,
                              random_state=42)),
    ])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",       default="linear", choices=["linear", "rbf", "mlp"])
    ap.add_argument("--min_samples", type=int, default=MIN_SAMPLES)
    ap.add_argument("--retrain",     action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_dir = OUT_DIR / args.model
    model_dir.mkdir(exist_ok=True)

    # ── Load training data ────────────────────────────────────────────────────
    print("[1/6] Loading training embeddings…")
    npz  = np.load(TRAIN_NPZ, allow_pickle=True)
    ids  = npz["cgc_id"].tolist()
    subs = npz["substrate"].tolist()
    X    = npz["embedding"].astype(np.float32)
    print(f"  {len(ids):,} annotated CGCs, {X.shape[1]}-dim ESM2mean\n")

    # ── Select classes ────────────────────────────────────────────────────────
    from collections import Counter
    counts  = Counter(subs)
    classes = sorted(
        s for s, n in counts.items()
        if n >= args.min_samples and s not in EXCLUDE_SUBSTRATES
    )
    print(f"[2/6] Classes with >= {args.min_samples} samples "
          f"(excl. {EXCLUDE_SUBSTRATES}):")
    for cls in classes:
        print(f"  {cls:<45} {counts[cls]:>5}")
    print(f"  Total: {len(classes)} classes\n")

    labels_path = model_dir / "classes.json"
    labels_path.write_text(json.dumps(classes, indent=2))

    # ── Train one model per class ─────────────────────────────────────────────
    print(f"[3/6] Training {args.model.upper()} classifiers (one-vs-rest)…")
    subs_arr = np.array(subs)
    cv_results = []

    for cls in classes:
        clf_path = model_dir / f"clf_{cls.replace(' ','_').replace('/','_')}.joblib"
        thr_path = model_dir / f"thr_{cls.replace(' ','_').replace('/','_')}.npy"

        if clf_path.exists() and thr_path.exists() and not args.retrain:
            print(f"  {cls}: cached")
            cv_results.append({"class": cls, "status": "cached"})
            continue

        y = (subs_arr == cls).astype(int)
        print(f"  {cls}: {y.sum():>4} pos / {(y==0).sum():>6} neg … ", end="", flush=True)

        if args.model == "linear":
            # CV scoring: plain LinearSVC (5 fits, no nested calibration)
            cv_scores = cross_val_predict(
                make_linear_cv(), X, y,
                cv=StratifiedKFold(5, shuffle=True, random_state=42),
                method="decision_function")
            # Convert decision scores to [0,1] with sigmoid for threshold/AUC
            from scipy.special import expit
            cv_probs = expit(cv_scores)
            # Final model: calibrated (6 fits total)
            clf = make_linear_final()
        elif args.model == "rbf":
            cv_probs = cross_val_predict(
                make_rbf(), X, y,
                cv=StratifiedKFold(5, shuffle=True, random_state=42),
                method="predict_proba")[:, 1]
            clf = make_rbf()
        else:
            cv_probs = cross_val_predict(
                make_mlp(), X, y,
                cv=StratifiedKFold(5, shuffle=True, random_state=42),
                method="predict_proba")[:, 1]
            clf = make_mlp()

        auc  = roc_auc_score(y, cv_probs)
        ap_  = average_precision_score(y, cv_probs)
        thr  = find_best_threshold(y, cv_probs)
        f1   = f1_score(y, (cv_probs >= thr).astype(int))
        print(f"AUC={auc:.4f}  AP={ap_:.4f}  F1={f1:.4f}  thr={thr:.3f}")

        clf.fit(X, y)
        joblib.dump(clf, clf_path)
        np.save(thr_path, thr)
        cv_results.append({"class": cls, "n_pos": int(y.sum()),
                            "AUC": auc, "AP": ap_, "F1": f1, "threshold": thr})

    cv_df = pd.DataFrame([r for r in cv_results if "AUC" in r])
    cv_csv = model_dir / "cv_scores.csv"
    cv_df.to_csv(cv_csv, index=False)
    print(f"\nCV scores → {cv_csv}\n")
    if len(cv_df):
        print(cv_df[["class","n_pos","AUC","AP","F1","threshold"]].to_string(index=False))
    print()

    # ── Load classifiers and thresholds ───────────────────────────────────────
    print("[4/6] Loading all classifiers…")
    clfs, thrs = {}, {}
    for cls in classes:
        clf_path = model_dir / f"clf_{cls.replace(' ','_').replace('/','_')}.joblib"
        thr_path = model_dir / f"thr_{cls.replace(' ','_').replace('/','_')}.npy"
        clfs[cls] = joblib.load(clf_path)
        thrs[cls] = float(np.load(thr_path))
    print(f"  Loaded {len(clfs)} classifiers\n")

    # ── Score unannotated CGCs ────────────────────────────────────────────────
    print("[5/6] Scoring unannotated CGCs…")
    npz_u   = np.load(SCORE_NPZ, allow_pickle=True)
    cgc_ids = npz_u["cgc_id"].tolist()
    envs    = npz_u["environment"].tolist()
    X_u     = npz_u["embedding"].astype(np.float32)
    print(f"  {len(cgc_ids):,} unannotated CGCs\n")

    prob_matrix = np.zeros((len(cgc_ids), len(classes)), dtype=np.float32)
    for j, cls in enumerate(classes):
        print(f"  Scoring {cls}…", flush=True)
        prob_matrix[:, j] = clfs[cls].predict_proba(X_u)[:, 1]

    # ── Build output ──────────────────────────────────────────────────────────
    print("\n[6/6] Building output tables…")

    # Full probability matrix
    prob_df = pd.DataFrame(
        prob_matrix,
        columns=[f"p_{c.replace(' ','_')}" for c in classes]
    )
    prob_df.insert(0, "cgc_id",      cgc_ids)
    prob_df.insert(1, "environment", envs)

    max_prob  = prob_matrix.max(axis=1)
    best_idx  = prob_matrix.argmax(axis=1)
    best_cls  = [classes[i] for i in best_idx]

    # Check if above per-class threshold
    above_any = np.array([
        prob_matrix[i, j] >= thrs[classes[j]]
        for i in range(len(cgc_ids))
        for j in range(len(classes))
    ]).reshape(len(cgc_ids), len(classes)).any(axis=1)

    prob_df["top_class"]      = best_cls
    prob_df["top_prob"]       = max_prob.round(4)
    prob_df["above_threshold"]= above_any
    prob_df["putative_novel"] = max_prob < NOVELTY_THRESHOLD

    prob_csv = OUT_DIR / "all_cgc_probabilities.csv"
    prob_df.to_csv(prob_csv, index=False)
    print(f"  Full probability matrix → {prob_csv}")

    # Summary: top assignment per CGC (above any threshold)
    summary = prob_df[above_any].copy()
    summary = summary.sort_values("top_prob", ascending=False)
    summary_csv = OUT_DIR / "cgc_assignments.csv"
    summary[["cgc_id","environment","top_class","top_prob","above_threshold"]
            ].to_csv(summary_csv, index=False)
    print(f"  CGC assignments (above threshold): {len(summary):,} → {summary_csv}")

    # Novelty: all probabilities below NOVELTY_THRESHOLD
    novel_mask = prob_matrix.max(axis=1) < NOVELTY_THRESHOLD
    n_novel    = novel_mask.sum()
    print(f"\n  Putative novel CGCs (max_prob < {NOVELTY_THRESHOLD}): {n_novel:,}")

    novel_df = pd.DataFrame({
        "cgc_id":      [cgc_ids[i] for i in range(len(cgc_ids)) if novel_mask[i]],
        "environment": [envs[i]    for i in range(len(cgc_ids)) if novel_mask[i]],
        "max_prob":    max_prob[novel_mask].round(4),
    }).sort_values("max_prob")

    novel_csv = OUT_DIR / "putative_novel_cgcs.csv"
    novel_df.head(TOP_N_NOVEL).to_csv(novel_csv, index=False)
    print(f"  Top {TOP_N_NOVEL} most-novel → {novel_csv}")

    # Ambiguous: multiple classes above threshold
    above_count = (prob_matrix >= np.array([thrs[c] for c in classes])).sum(axis=1)
    ambiguous   = (above_count > 1).sum()
    print(f"  Ambiguous (>1 class above threshold): {ambiguous:,}")

    # Environment breakdown of novel candidates
    from collections import Counter
    env_counts = Counter(novel_df["environment"])
    print("\n  Environment breakdown of putative novel CGCs:")
    for env, n in sorted(env_counts.items(), key=lambda x: -x[1]):
        print(f"    {env:<20} {n:>5}")

    # ── Summary stats ─────────────────────────────────────────────────────────
    print("\n── Assignment summary ─────────────────────────────────────────────")
    for cls in classes:
        j = classes.index(cls)
        n_above = (prob_matrix[:, j] >= thrs[cls]).sum()
        print(f"  {cls:<45} {n_above:>6} assigned  (thr={thrs[cls]:.3f})")

    print(f"\nOutputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
