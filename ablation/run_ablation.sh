#!/bin/bash
# Waits for step1 GPU job output, then runs step3_compare.py

LOG=/work3/zhayu/PULpred/ablation/ablation.log
PY="/zhome/68/5/210030/anaconda3/bin/conda run -p /work3/zhayu/envs/pulpred python"

cd /work3/zhayu/PULpred

echo "[$(date)] Waiting for step1 GPU job output..." >> $LOG

while [ ! -f ablation/esm2mean_embeddings.npz ] || [ ! -f ablation/random_cls_embeddings.npz ]; do
    sleep 60
done

echo "[$(date)] Step1 outputs detected. Running ablation comparison..." >> $LOG
$PY ablation/step3_compare.py >> $LOG 2>&1
echo "[$(date)] Done. Results in ablation/ablation_results.csv" >> $LOG
