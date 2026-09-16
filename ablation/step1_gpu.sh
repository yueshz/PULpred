#!/bin/bash
#BSUB -J ablation_embed
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=64GB]"
#BSUB -W 4:00
#BSUB -o /work3/zhayu/PULpred/ablation/step1.log
#BSUB -e /work3/zhayu/PULpred/ablation/step1.log

cd /work3/zhayu/PULpred
/zhome/68/5/210030/anaconda3/bin/conda run -p /work3/zhayu/envs/pulpred python -u \
    ablation/step1_embed_variants.py --device cuda --esm_batch 32 --cls_batch 256
