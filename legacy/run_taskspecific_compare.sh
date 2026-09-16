#!/bin/bash
# STALE — predates the ESM2mean-only refactor of PULpredSVM/01_score_svm.py.
# --embed_model is no longer a valid flag on that script (it now hardcodes
# ESM2mean), so this will fail with an argparse error as-is. Kept for
# reference on how the CLS-vs-ESM2mean comparison used to be run; see
# ablation/ for the current, working version of that comparison.
#BSUB -J taskspecific_compare
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=48GB]"
#BSUB -W 4:00
#BSUB -o /work3/zhayu/PULpred/taskspecific_compare.log
#BSUB -e /work3/zhayu/PULpred/taskspecific_compare.log

set -e
cd /work3/zhayu/PULpred

PY="/zhome/68/5/210030/anaconda3/bin/conda run -p /work3/zhayu/envs/pulpred python -u"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "=== [$(date)] Task-specific predictions with min_proteins=5 ==="

echo ""
echo "--- GAG: ESM2mean (SVM checkpoint cached, skip training) ---"
$PY PULpredSVM/01_score_svm.py --task gag --embed_model esm2mean --min_proteins 5

echo ""
echo "--- GAG: CLS ---"
$PY PULpredSVM/01_score_svm.py --task gag --embed_model cls --min_proteins 5

echo ""
echo "--- Alginate: ESM2mean (SVM checkpoint cached, skip training) ---"
$PY PULpredSVM/01_score_svm.py --task alginate --embed_model esm2mean --min_proteins 5

echo ""
echo "--- Alginate: CLS ---"
$PY PULpredSVM/01_score_svm.py --task alginate --embed_model cls --min_proteins 5

echo ""
echo "--- Overlap analysis ---"
$PY legacy/compare_overlap.py

echo ""
echo "=== [$(date)] Done ==="
