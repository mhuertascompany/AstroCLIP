#!/bin/bash
#SBATCH --job-name=cweb_eval
#SBATCH --output=/n03data/huertas/COSMOS-Web/cosmosweb_clip/eval_alignment.out
#SBATCH --error=/n03data/huertas/COSMOS-Web/cosmosweb_clip/eval_alignment.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=n03
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

# ── environment ───────────────────────────────────────────────────────────────
source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/

REPO_DIR=/n03data/huertas/python/AstroCLIP
OUTPUT_DIR=/n03data/huertas/COSMOS-Web/cosmosweb_clip

cd ${REPO_DIR}

# ── run ───────────────────────────────────────────────────────────────────────
# --n_eval 0      → all galaxies for retrieval metrics (chunked, low memory)
# --n_probe 30000 → subsample for linear/MLP probes (sklearn CV is slow at 137k)
# --chunk 500     → ~500*N*4 bytes per similarity pass; lower if OOM
python -m cosmosweb.evaluate_alignment \
    --npz      ${OUTPUT_DIR}/cosmosweb_umap_zoobot_v1.npz \
    --h5       ${OUTPUT_DIR}/cosmosweb_dataset_v2.h5 \
    --output   ${OUTPUT_DIR}/alignment_eval_zoobot_v1.pdf \
    --n_eval   0 \
    --n_probe  30000 \
    --k        10 \
    --chunk    500 \
    --batch_eval 128 \
    --n_batches  500
