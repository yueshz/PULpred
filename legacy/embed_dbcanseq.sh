#!/bin/bash
#BSUB -J embed_dbcanseq
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -W 4:00
#BSUB -o /tmp/embed_dbcanseq_lsf.log
#BSUB -e /tmp/embed_dbcanseq_lsf.log

cd /work3/zhayu/PULpred
conda run -p /work3/zhayu/envs/pulpred python -u legacy/embed_dbcanseq.py --device cuda
