#!/bin/bash
#SBATCH --job-name=astroclip_subset
#SBATCH --output=/n03data/huertas/astroinfo/logs/subset.out
#SBATCH --error=/n03data/huertas/astroinfo/logs/subset.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --partition=cpu
#SBATCH --chdir=/n03data/huertas/astroinfo/AstroCLIP

module purge
module load cuda/11.8

source /home/huertas/.bashrc
source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate astroclip-teaching

export PYTHONPATH=/n03data/huertas/astroinfo/AstroCLIP:$PYTHONPATH
export ASTROCLIP_ROOT=/n03data/huertas/astroinfo/astroclip_data
export HF_DATASETS_CACHE=/n03data/huertas/astroinfo/hf_cache

python teaching_scripts/prepare_dataset.py \
    --dataset EiffL/AstroCLIP \
    --output-dir ${ASTROCLIP_ROOT}/teaching_demo/astroclip_subset_tiny \
    --train-size 1024 \
    --test-size 256 \
    --streaming \
    --overwrite
