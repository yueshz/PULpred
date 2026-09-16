#!/bin/bash
#BSUB -J foldseek_db
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 3:00
#BSUB -o /work3/zhayu/PULpred/foldseek_db.log
#BSUB -e /work3/zhayu/PULpred/foldseek_db.log

set -e
cd /work3/zhayu/PULpred

FS="/work3/zhayu/envs/pulpred/bin/foldseek"
DB="data/foldseek_db"
mkdir -p $DB/tmp

echo "=== [$(date)] Downloading FoldSeek PDB database (~8 GB) ==="
$FS databases PDB $DB/pdb $DB/tmp --threads 4

echo "=== [$(date)] Downloading AlphaFold/Swiss-Prot (~1.5 GB) ==="
$FS databases Alphafold/Swiss-Prot $DB/swissprot $DB/tmp --threads 4

echo "=== [$(date)] Done. Disk usage: ==="
du -sh $DB/
