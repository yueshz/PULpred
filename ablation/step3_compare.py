"""
step3_compare.py  [CPU]

4-way ablation comparison for GAG (and optionally alginate) classification.

Methods compared:
  1. BoF-SVM        — protein-cluster binary vector (no sequence LLM)
  2. ESM2mean-SVM   — average ESM2 embedding per CGC (no Transformer)
  3. Random-CLS-SVM — CLS from randomly-initialised PULTransformer
  4. CLS-SVM        — CLS from pretrained PULTransformer  ← ours

Evaluation: 5-fold stratified CV → AUC, F1, Precision, Recall, Accuracy

Usage:
  cd /work3/zhayu/PULpred
  conda run -p /work3/zhayu/envs/pulpred python ablation/step3_compare.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score, accuracy_score
)

# ── Config ────────────────────────────────────────────────────────────────────

CLS_NPZ     = Path("data/dbCAN_seq/embeddings/dbcanseq_cls_embeddings.npz")
ESM2_NPZ    = Path("ablation/esm2mean_embeddings.npz")
RCLS_NPZ    = Path("ablation/random_cls_embeddings.npz")
BOF_NPZ     = Path("ablation/bof_embeddings.npz")
DBCAN_DIR   = Path("data/dbCAN_seq")
OUT_DIR     = Path("ablation")

GAG_FAMILIES = {
    "PL8", "PL12", "PL13", "PL15", "PL21", "PL23",
    "PL29", "PL30", "PL33", "PL35", "GH88",
}
EXCLUDE_SUBSTRATES = {"glycosaminoglycan", "host glycan"}

SVM_WEIGHT = 10
CV_FOLDS   = 5
RANDOM_SEED = 42

TASKS = {
    "GAG":      "glycosaminoglycan",
    "Alginate": "alginate",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def base_family(s: str) -> str:
    return re.sub(r"_\d+$", "", s.strip())


def parse_dbcanseq_contents(data_dir: Path) -> dict[str, str]:
    """Return {cgc_id: contents_string} from all substrate files."""
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
                    cgc2contents[parts[1].strip()] = parts[2].strip()
    return cgc2contents


def contents_has_gag_family(contents: str) -> bool:
    for tok in re.split(r"[,|]", contents):
        if base_family(tok) in GAG_FAMILIES:
            return True
    return False


def build_labels(cgc_ids, substrates, task_substrate, cgc2contents) -> tuple:
    """
    Returns (keep_idx, labels) for the given task.
    GAG: positive = substrate==task AND has GAG marker family
         negative = substrate not in EXCLUDE_SUBSTRATES
    Alginate: positive = substrate==task
              negative = all others
    """
    keep, labels = [], []
    for i, (cid, sub) in enumerate(zip(cgc_ids, substrates)):
        if task_substrate == "glycosaminoglycan":
            if sub == task_substrate:
                if contents_has_gag_family(cgc2contents.get(cid, "")):
                    labels.append(1); keep.append(i)
            elif sub not in EXCLUDE_SUBSTRATES:
                labels.append(0); keep.append(i)
        else:
            labels.append(1 if sub == task_substrate else 0)
            keep.append(i)
    return np.array(keep), np.array(labels)


def find_best_threshold(y_true, y_prob):
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.001, 0.99, 500):
        f = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t


def evaluate(name: str, X: np.ndarray, y: np.ndarray) -> dict:
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(kernel="rbf", class_weight={1: SVM_WEIGHT, 0: 1},
                       probability=True, random_state=RANDOM_SEED)),
    ])
    cv       = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_probs = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    threshold = find_best_threshold(y, cv_probs)
    y_pred    = (cv_probs >= threshold).astype(int)

    return {
        "Method":    name,
        "Threshold": round(threshold, 4),
        "AUC":       round(roc_auc_score(y, cv_probs), 4),
        "F1":        round(f1_score(y, y_pred, zero_division=0), 4),
        "Precision": round(precision_score(y, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y, y_pred, zero_division=0), 4),
        "Accuracy":  round(accuracy_score(y, y_pred), 4),
        "n_pos":     int(y.sum()),
        "n_neg":     int((y == 0).sum()),
    }


def load_npz(path: Path):
    npz = np.load(path, allow_pickle=True)
    return (npz["cgc_id"].tolist(),
            npz["substrate"].tolist(),
            npz["embedding"].astype(np.float32))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # check which files exist
    available = {}
    for tag, path in [("CLS", CLS_NPZ), ("ESM2mean", ESM2_NPZ),
                      ("Random-CLS", RCLS_NPZ), ("BoF", BOF_NPZ)]:
        if path.exists():
            available[tag] = path
        else:
            print(f"  WARNING: {path} not found — skipping {tag}")

    if not available:
        raise RuntimeError("No embedding files found. Run step1 and step2 first.")

    # load contents for GAG label parsing
    print("Parsing substrate contents for GAG labels…")
    cgc2contents = parse_dbcanseq_contents(DBCAN_DIR)
    print(f"  {len(cgc2contents):,} CGC content entries\n")

    all_results = []

    for task_name, task_substrate in TASKS.items():
        print(f"{'='*60}")
        print(f"Task: {task_name}  (substrate='{task_substrate}')")
        print(f"{'='*60}")

        task_results = []
        for method_name, path in available.items():
            cgc_ids, substrates, X = load_npz(path)
            keep_idx, labels = build_labels(cgc_ids, substrates,
                                            task_substrate, cgc2contents)
            X_task = X[keep_idx]

            print(f"\n[{method_name}]  dim={X_task.shape[1]}  "
                  f"pos={labels.sum()}  neg={(labels==0).sum()}")
            result = evaluate(method_name, X_task, labels)
            result["Task"] = task_name
            task_results.append(result)
            print(f"  AUC={result['AUC']:.4f}  F1={result['F1']:.4f}  "
                  f"P={result['Precision']:.4f}  R={result['Recall']:.4f}")

        all_results.extend(task_results)

        # per-task summary table
        df_task = pd.DataFrame(task_results)[
            ["Method", "AUC", "F1", "Precision", "Recall", "Accuracy", "Threshold"]
        ]
        print(f"\n{task_name} Summary:\n{df_task.to_string(index=False)}\n")

    # save full results
    df_all = pd.DataFrame(all_results)
    out_csv = OUT_DIR / "ablation_results.csv"
    df_all.to_csv(out_csv, index=False)
    print(f"\nFull results saved → {out_csv}")

    # final comparison table
    print("\n" + "="*60)
    print("ABLATION SUMMARY")
    print("="*60)
    for task_name in TASKS:
        df_t = df_all[df_all["Task"] == task_name][
            ["Method", "AUC", "F1", "Precision", "Recall"]
        ].reset_index(drop=True)
        print(f"\n{task_name}:\n{df_t.to_string(index=False)}")


if __name__ == "__main__":
    main()
