#!/usr/bin/env python3
"""
Discriminative classifier on frozen pre-trained PULTransformer CLS embeddings.

Models evaluated:
  - Logistic Regression (L2)
  - Linear SVM
  - RBF SVM

Evaluation: stratified 5-fold CV → macro F1, weighted F1, accuracy
Final models are trained on all data and saved to checkpoints/classify/.

Usage:
    python classify.py \
        --npz            data/embeddings/labelled_pul_embeddings.npz \
        --min_samples    5 \
        --n_folds        5 \
        --out_dir        checkpoints/classify
"""

import argparse
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import LinearSVC, SVC


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--npz",          default="data/embeddings/labelled_pul_embeddings.npz")
    p.add_argument("--min_samples",  type=int, default=5,
                   help="Drop substrate classes with fewer samples than this (default 5)")
    p.add_argument("--n_folds",      type=int, default=5)
    p.add_argument("--out_dir",      default="checkpoints/classify")
    p.add_argument("--report_dir",   default="results")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_data(npz_path: str, min_samples: int):
    data = np.load(npz_path, allow_pickle=True)
    embeddings = data["embedding"].astype(np.float32)   # [N, 512]
    substrates = data["substrate"].tolist()

    counts = Counter(substrates)
    keep = np.array([counts[s] >= min_samples for s in substrates])
    embeddings = embeddings[keep]
    substrates = [s for s, k in zip(substrates, keep) if k]

    print(f"Loaded {len(substrates):,} PULs after dropping classes < {min_samples} samples")
    print(f"Unique substrates: {len(set(substrates))}")
    print()

    le = LabelEncoder()
    labels = le.fit_transform(substrates)
    return embeddings, labels, le


def cv_eval(X, y, model_name: str, pipeline: Pipeline, n_folds: int):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    macro_f1s, weighted_f1s, accs = [], [], []

    for fold, (tr, val) in enumerate(skf.split(X, y)):
        pipeline.fit(X[tr], y[tr])
        preds = pipeline.predict(X[val])
        macro_f1s.append(f1_score(y[val], preds, average="macro",    zero_division=0))
        weighted_f1s.append(f1_score(y[val], preds, average="weighted", zero_division=0))
        accs.append(accuracy_score(y[val], preds))

    print(f"  {model_name:<20s}  "
          f"macro-F1 {np.mean(macro_f1s):.3f}±{np.std(macro_f1s):.3f}  "
          f"weighted-F1 {np.mean(weighted_f1s):.3f}±{np.std(weighted_f1s):.3f}  "
          f"acc {np.mean(accs):.3f}±{np.std(accs):.3f}")

    return np.mean(macro_f1s)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    out_dir    = Path(args.out_dir);    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path(args.report_dir); report_dir.mkdir(parents=True, exist_ok=True)

    X, y, le = load_data(args.npz, args.min_samples)

    # Print class distribution summary
    counts = Counter(y)
    print(f"  Samples/class — min:{min(counts.values())}  "
          f"median:{int(np.median(list(counts.values())))}  "
          f"max:{max(counts.values())}\n")

    # ── Models (L2-normalise embeddings first; helps cosine-ish SVM) ─────────
    models = {
        "LogisticReg": Pipeline([
            ("norm", Normalizer()),
            ("clf",  LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                                        multi_class="multinomial", random_state=42)),
        ]),
        "LinearSVM": Pipeline([
            ("norm", Normalizer()),
            ("clf",  LinearSVC(max_iter=5000, C=0.5, random_state=42)),
        ]),
        "RBF-SVM": Pipeline([
            ("norm", Normalizer()),
            ("clf",  SVC(kernel="rbf", C=10, gamma="scale", random_state=42)),
        ]),
    }

    print(f"── {args.n_folds}-fold cross-validation ─────────────────────────────")
    best_name, best_score, best_pipeline = None, -1, None
    for name, pipeline in models.items():
        score = cv_eval(X, y, name, pipeline, args.n_folds)
        if score > best_score:
            best_score, best_name, best_pipeline = score, name, pipeline

    print(f"\nBest model: {best_name}  (macro-F1 {best_score:.3f})")

    # ── Train final models on full data ──────────────────────────────────────
    print("\n── Training final models on all data ───────────────────────────────")
    for name, pipeline in models.items():
        pipeline.fit(X, y)
        path = out_dir / f"{name.lower().replace('-','_').replace(' ','_')}.pkl"
        with open(path, "wb") as f:
            pickle.dump(pipeline, f)
        print(f"  Saved → {path}")

    with open(out_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    print(f"  Saved → {out_dir}/label_encoder.pkl")

    # ── Classification report for best model on full data ────────────────────
    preds_all = best_pipeline.predict(X)
    report = classification_report(
        y, preds_all,
        target_names=le.classes_,
        zero_division=0,
    )
    report_path = report_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Best model: {best_name}\n")
        f.write(f"CV macro-F1: {best_score:.4f}\n\n")
        f.write("Full-data classification report (train=test — for class breakdown only):\n\n")
        f.write(report)

    print(f"\nReport saved → {report_path}")
    print("\nFull-data class breakdown (train=test, optimistic):\n")
    print(report)


if __name__ == "__main__":
    main()
