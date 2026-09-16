#!/bin/bash
#BSUB -J leakage_check
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 4:00
#BSUB -o /work3/zhayu/PULpred/leakage_check.log
#BSUB -e /work3/zhayu/PULpred/leakage_check.log

set -e
cd /work3/zhayu/PULpred

PY="/zhome/68/5/210030/anaconda3/bin/conda run -p /work3/zhayu/envs/pulpred python -u"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "=== [$(date)] Homology leakage check (31 classes, Std-5fold vs LGO) ==="
$PY check_homology_leakage.py
echo "=== [$(date)] Done ==="
