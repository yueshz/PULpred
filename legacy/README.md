# Legacy code

First-generation approach, superseded by the ESM2mean pipeline documented in
the top-level [README](../README.md). Kept for provenance, not maintained,
not part of any currently-runnable pipeline.

## What's here

**PULTransformer / PULDB track** — pretrain a custom Set Transformer on
PULDB, fine-tune it for substrate classification, then classify on its CLS
embeddings:

- `pretrain.py` — masked protein modeling pretraining on PULDB → `checkpoints/pretrain/`
- `finetune.py` — supervised fine-tuning on CANPUL_substrates.xlsx → `checkpoints/finetune/`
- `classify.py` — sklearn heads on frozen CLS embeddings → `checkpoints/classify/`
- `predict_substrate.py` — inference with the fine-tuned model

Nothing in the active pipeline calls any of these four scripts, and
`checkpoints/finetune/` and `checkpoints/classify/` have no downstream
readers anywhere in the codebase.

**dbCAN-seq CLS embedding + comparison track** — scaled the PULTransformer
CLS approach up to dbCAN-seq and compared it against ESM2mean:

- `embed_dbcanseq.py` — CLS embeddings for annotated CGCs (uses `checkpoints/pretrain/` via `src/model.py`)
- `compare_overlap.py` — diffed ESM2mean vs. CLS candidate sets (originally lived in `PULpredSVM/`)
- `run_taskspecific_compare.sh` — driver for the above; **currently broken** —
  it calls `PULpredSVM/01_score_svm.py --embed_model cls`, a flag that
  script no longer accepts after being simplified to ESM2mean-only. Left
  as-is rather than patched, since fixing it would mean re-adding a code
  path the project deliberately dropped.

The unannotated-CGC counterpart of `embed_dbcanseq.py`,
`embed_dbcanseq_unannotated.py`, is **not** here — it's dual-purpose and
still active in `--mean_only` mode (see the top-level file's docstring).

The formal, working version of "why ESM2mean over CLS" is `ablation/` at
the repo root, not this comparison track — that's what the README's
ablation table is drawn from.

## Why these are archived

The ablation in `ablation/` showed ESM2mean embeddings (mean-pooled ESM2 per
CGC, no learned encoder) outperform the pretrained PULTransformer's CLS
token on both downstream tasks. Once that was established, every active
pipeline (`PULpredSVM/`, `predict_esm2mean_multitask.py`) moved to ESM2mean-only
and `src/model.py`'s PULTransformer stopped being needed for production
inference — it only still gets imported here, for regenerating the
now-archived CLS comparison.
