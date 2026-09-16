#!/bin/bash
#BSUB -J seqid_leakage
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -W 4:00
#BSUB -o /work3/zhayu/PULpred/seqid_leakage.log
#BSUB -e /work3/zhayu/PULpred/seqid_leakage.log

set -e
cd /work3/zhayu/PULpred

PY="/zhome/68/5/210030/anaconda3/bin/conda run -p /work3/zhayu/envs/pulpred python -u"
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

echo "=== [$(date)] SeqID leakage check (MMseqs2 30%/80% + GroupKFold) ==="
$PY check_seqid_leakage.py --threads 16
echo "=== [$(date)] Done ==="
