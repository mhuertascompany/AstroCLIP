#!/bin/bash
#SBATCH --job-name=tfg_stamps
#SBATCH --output=/n03data/huertas/COSMOS-Web/tfg_laura/collect.out
#SBATCH --error=/n03data/huertas/COSMOS-Web/tfg_laura/collect.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=n03
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --chdir=/n03data/huertas/python/AstroCLIP

# ── environment ───────────────────────────────────────────────────────────────
source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/

REPO_DIR=/n03data/huertas/python/AstroCLIP
OUTPUT_DIR=/n03data/huertas/COSMOS-Web/tfg_laura

mkdir -p ${OUTPUT_DIR}
cd ${REPO_DIR}

# ── run ───────────────────────────────────────────────────────────────────────
python -m cosmosweb.collect_tfg_stamps \
    --csv_dir  ${REPO_DIR}/cosmosweb/galaxias_select \
    --output   ${OUTPUT_DIR} \
    --arcsec   5.0
