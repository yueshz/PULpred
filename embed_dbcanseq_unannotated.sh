#!/bin/bash
#BSUB -J embed_dbcanseq_unannotated
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=64GB]"
#BSUB -W 12:00
#BSUB -o /work3/zhayu/PULpred/embed_dbcanseq_unannotated.log
#BSUB -e /work3/zhayu/PULpred/embed_dbcanseq_unannotated.log

cd /work3/zhayu/PULpred
/zhome/68/5/210030/anaconda3/bin/conda run -p /work3/zhayu/envs/pulpred python -u \
    embed_dbcanseq_unannotated.py --device cuda --esm_batch 32 --mean_only
