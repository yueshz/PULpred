#!/bin/bash
#BSUB -J foldseek_annotate
#BSUB -q gpua100
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 2:00
#BSUB -o /work3/zhayu/PULpred/foldseek_annotate.log
#BSUB -e /work3/zhayu/PULpred/foldseek_annotate.log

set -e
cd /work3/zhayu/PULpred

# Redirect HuggingFace cache to work3 (home quota is too small for ESMFold ~2.5GB)
export HF_HOME=/work3/zhayu/hf_cache
export TRANSFORMERS_CACHE=/work3/zhayu/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /work3/zhayu/hf_cache

PY="/zhome/68/5/210030/anaconda3/bin/conda run -p /work3/zhayu/envs/pulpred python -u"

echo "=== [$(date)] FoldSeek annotation: GAG ==="
$PY PULpredSVM/03_fold_structure.py --task gag --threads 8

echo "=== [$(date)] FoldSeek annotation: Alginate ==="
$PY PULpredSVM/03_fold_structure.py --task alginate --threads 8

echo "=== [$(date)] Done ==="
