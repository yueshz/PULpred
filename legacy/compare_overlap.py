"""
compare_overlap.py

Compares ESM2mean vs CLS novel candidates (min_proteins=5) for GAG and Alginate.
Prints overlap statistics and saves combined tables.
"""

from pathlib import Path
import pandas as pd

MIN_PROT = 5

PAIRS = {
    "GAG": {
        "esm2mean": Path(f"PULpredSVM/results/gag_esm2mean_minprot{MIN_PROT}/novel_gag_candidates.csv"),
        "cls":      Path(f"PULpredSVM/results/gag_cls_minprot{MIN_PROT}/novel_gag_candidates.csv"),
    },
    "Alginate": {
        "esm2mean": Path(f"PULpredSVM/results/alginate_esm2mean_minprot{MIN_PROT}/novel_alginate_candidates.csv"),
        "cls":      Path(f"PULpredSVM/results/alginate_cls_minprot{MIN_PROT}/novel_alginate_candidates.csv"),
    },
}

for task, paths in PAIRS.items():
    print(f"\n{'='*60}")
    print(f"  {task}  (min_proteins={MIN_PROT})")
    print(f"{'='*60}")

    df_esm = pd.read_csv(paths["esm2mean"])
    df_cls = pd.read_csv(paths["cls"])

    ids_esm = set(df_esm.cgc_id)
    ids_cls = set(df_cls.cgc_id)
    overlap = ids_esm & ids_cls
    esm_only = ids_esm - ids_cls
    cls_only = ids_cls - ids_esm

    print(f"\n  ESM2mean candidates : {len(ids_esm)}")
    print(f"  CLS candidates      : {len(ids_cls)}")
    print(f"  Overlap (both)      : {len(overlap)}  "
          f"({100*len(overlap)/max(len(ids_esm),1):.0f}% of ESM2mean, "
          f"{100*len(overlap)/max(len(ids_cls),1):.0f}% of CLS)")
    print(f"  ESM2mean only       : {len(esm_only)}")
    print(f"  CLS only            : {len(cls_only)}")

    if overlap:
        print(f"\n  Overlapping CGCs (high-confidence — found by both models):")
        # Show with both models' scores
        df_both = df_esm[df_esm.cgc_id.isin(overlap)][
            ["cgc_id","environment","n_proteins","svm_score","GH_families","PL_families"]
        ].rename(columns={"svm_score": "score_esm2mean"})
        cls_scores = df_cls.set_index("cgc_id")["svm_score"].rename("score_cls")
        df_both = df_both.join(cls_scores, on="cgc_id")
        df_both = df_both.sort_values("score_esm2mean", ascending=False)
        print(df_both.to_string(index=False))

        # Save high-confidence set
        out = Path(f"PULpredSVM/results/{task.lower()}_highconf_minprot{MIN_PROT}.csv")
        df_both.to_csv(out, index=False)
        print(f"\n  High-confidence set saved → {out}")

    print(f"\n  ESM2mean-only top candidates:")
    print(df_esm[df_esm.cgc_id.isin(esm_only)].head(10)[
        ["rank","cgc_id","environment","n_proteins","svm_score","GH_families"]
    ].to_string(index=False))

    print(f"\n  CLS-only top candidates:")
    print(df_cls[df_cls.cgc_id.isin(cls_only)].head(10)[
        ["rank","cgc_id","environment","n_proteins","svm_score","GH_families"]
    ].to_string(index=False))
