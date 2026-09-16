# PULpred

PULpred predicts the substrate specificity of Polysaccharide Utilization Loci
(PULs) in gut and marine bacteria, and discovers novel PULs in unannotated
metagenome-assembled genomes.

**Representation.** Each CGC (CAZyme Gene Cluster) is represented as the mean
of per-protein ESM2 embeddings (`esm2_t30_150M_UR50D`, 640-dim) across all
proteins in the locus — **ESM2mean**.

**Classification.** RBF SVM (`probability=True`, `class_weight={1:10}`)
trained on ESM2mean embeddings. Separate models for task-specific discovery
(GAG, alginate) and a one-vs-rest multi-task classifier covering 31
substrate classes.

An earlier version of this project pretrained a custom Set Transformer
(PULTransformer) and classified on its CLS token instead. A controlled
ablation showed plain ESM2mean beats the pretrained transformer on both
downstream tasks, so the transformer track was dropped — see
[Ablation](#ablation) and [`legacy/`](legacy/README.md).

---

## Results

### Task-specific SVMs (ESM2mean, RBF, 5-fold CV)

| Task | AUC | F1 | CV threshold |
|------|-----|----|-------------|
| GAG (glycosaminoglycan) | 0.9942 | 0.829 | 0.568 |
| Alginate | 0.9904 | 0.930 | 0.851 |

Threshold = CV F1-optimal probability, used as a high-confidence flag in
output; candidates are always ranked by raw SVM score.

### Ablation

ESM2mean vs. the pretrained PULTransformer's CLS token vs. an untrained
("Random-CLS") version of the same transformer:

| Method | GAG AUC | GAG F1 | Alginate AUC | Alginate F1 |
|--------|---------|--------|--------------|-------------|
| CLS (pretrained PULTransformer) | 0.9768 | 0.709 | 0.9892 | 0.773 |
| **ESM2mean** | **0.9942** | **0.829** | **0.9904** | **0.930** |
| Random-CLS (no pretraining) | 0.9875 | 0.761 | 0.9946 | 0.874 |

### Sequence homology leakage check

CV scores are not inflated by sequence-level redundancy between train/test
splits:
- LGO (Leave-Genome-Out) CV: GAG AUC drop = **+0.0003** (negligible)
- SeqID-80% CAZyme GroupKFold: GAG AUC drop = **+0.011** (negligible)

---

## Repository layout

```
PULpred/
├── PULpredSVM/                     # core: substrate-specific discovery pipeline (run in order)
│   ├── 01_score_svm.py         #   SVM scoring on ESM2mean embeddings — python PULpredSVM/01_score_svm.py --task gag|alginate
│   ├── 02_filter_diamond.py    #   DIAMOND-vs-CAZyDB exclusion filter
│   ├── 03_fold_structure.py    #   ESMFold + FoldSeek structural annotation
│   ├── 04_check_active_site.py #   active-site validation of top hits
│   └── 05_blast_consensus.py   #   BLAST consensus check
│
├── predict_esm2mean_multitask.py  # core: one-vs-rest multi-task classifier (31 substrate classes)
├── embed_dbcanseq_unannotated.py  # core (--mean_only): ESM2mean embeddings for the 126k unannotated CGCs
├── check_seqid_leakage.py         # validation: MMseqs2 + GroupKFold leakage check
├── check_homology_leakage.py      # validation: Leave-Genome-Out leakage check
│
├── ablation/                    # provenance: the ESM2mean-vs-CLS comparison above (completed, not re-run)
│   ├── step1_embed_variants.py
│   └── step3_compare.py
│
├── src/                         # PULTransformer (Set Transformer, CLS token) model + dataset code
│                                 #   only still used by ablation/ and legacy/
│
├── legacy/                      # archived first-gen PULTransformer/PULDB track — see legacy/README.md
│
├── run_*.sh, setup_foldseek_db.sh   # LSF (bsub) job scripts for the DTU HPC cluster
│
├── data/          # (gitignored) raw + intermediate data — see "Data & model availability"
├── checkpoints/   # (gitignored) model weights
└── results/       # (gitignored) run outputs
```

`PULpredSVM/01_score_svm.py` is the deliverable, but it isn't
self-contained: it reads training embeddings from `ablation/` and scoring
embeddings from `data/dbCAN_seq/embeddings/`, both produced by scripts
outside `PULpredSVM/`. `predict_esm2mean_multitask.py` is a sibling pipeline at the
same "core" status, not a dependency of `PULpredSVM/`.

### Data & model availability

`data/`, `checkpoints/`, and `results/` are gitignored — they're 19GB+ of
downloaded reference databases (CAZy, UniProt, PDB, AlphaFold/Swiss-Prot via
FoldSeek), computed ESM2 embeddings, and trained model weights, none of
which belong in git. Reference databases are re-downloadable with
`setup_foldseek_db.sh` and the DIAMOND/CAZy setup described below; the
computed embeddings and trained models represent real compute time and
aren't currently published anywhere — **contact the author to obtain them**
rather than regenerating from scratch.

---

## Setup

```bash
conda env create -f environment.yml
conda activate pulpred
```

`environment.yml` installs the conda-only CLI tools (MMseqs2, FoldSeek,
TM-align, DIAMOND) plus everything in `requirements.txt` via pip. The
`torch` pin targets CUDA 11.8; adjust the `--extra-index-url` in
`requirements.txt` for your platform (e.g. drop it entirely for CPU-only).

### External data setup

- **FoldSeek databases** (PDB + AlphaFold/Swiss-Prot, ~10GB): `bash setup_foldseek_db.sh` (or `bsub < setup_foldseek_db.sh` on an LSF cluster)
- **CAZy DIAMOND database**: see the header of `PULpredSVM/02_filter_diamond.py` for the download + `diamond makedb` command
- **dbCAN-seq CGC data**: per-genome CGC FASTAs, expected under `data/dbCAN_seq/{HUMAN_GUT,COW_RUMEN,HUMAN_ORAL,MARINE}/`

---

## Usage

```bash
# Discovery pipeline (SVM discovery, ~10 min first run to build a protein-count cache)
python PULpredSVM/01_score_svm.py --task gag
python PULpredSVM/01_score_svm.py --task alginate

# FoldSeek structural annotation of top candidates (GPU)
bash run_foldseek_annotate.sh

# Multi-task classifier (31 substrate classes)
bash run_multitask.sh

# Leakage validation
bash run_seqid_leakage.sh
bash run_leakage_check.sh
```

SVM checkpoints in `PULpredSVM/results/{task}_svm/` are reused across runs; pass
`--retrain` to `01_score_svm.py` to force retraining.

### CGC ID format

`GENOME_X|CGCN`, e.g. `MGYG000000949_143|CGC1`. FASTA path reconstruction:

```python
stem   = cgc_id.replace("|", "#")             # MGYG000000949_143#CGC1
genome = stem.split("#")[0].rsplit("_", 1)[0]  # MGYG000000949
path   = f"data/dbCAN_seq/{ENV}/{genome}/dbcan_out/CGC_fasta/{stem}.fasta"
```

---

## Label definitions

**GAG positives**: substrate == `glycosaminoglycan` AND CGC contains one of
`{PL8, PL12, PL13, PL15, PL21, PL23, PL29, PL30, PL33, PL35, GH88}`

**GAG negatives**: all substrates except `glycosaminoglycan` and `host glycan`

**Alginate positives**: substrate == `alginate`

**DIAMOND exclusion families** (used to filter out CGCs with already-known
CAZyme families before flagging a candidate as novel):
- GAG: `{PL8, PL12, PL13, PL15, PL21, PL23, PL29, PL30, PL33, PL35}` (GH88 passes through)
- Alginate: `{PL5, PL6, PL7, PL14, PL15, PL17, PL18, PL31, PL32, PL34, PL36, PL38, PL39}`

---

## Compute notes

Developed on DTU HPC (LSF). Shell scripts under `run_*.sh` are `bsub` job
files with hardcoded absolute paths (`/work3/...`, `/zhome/...`) — adjust
these for your own environment; they aren't parameterized for portability.
