#!/bin/bash
#BSUB -J multitask_linear
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -W 3:00
#BSUB -o /work3/zhayu/PULpred/multitask_linear.log
#BSUB -e /work3/zhayu/PULpred/multitask_linear.log

set -e
cd /work3/zhayu/PULpred

PY="/zhome/68/5/210030/anaconda3/bin/conda run -p /work3/zhayu/envs/pulpred python -u"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "=== [$(date)] Multi-task LinearSVC (31 classes) ==="
$PY multitaskSVM/predict_esm2mean_multitask.py --model linear
echo "=== [$(date)] Done ==="
