#!/bin/bash
#SBATCH --job-name=astroclip_morph
#SBATCH --output=/n03data/huertas/astroinfo/logs/morph.out
#SBATCH --error=/n03data/huertas/astroinfo/logs/morph.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --chdir=/n03data/huertas/astroinfo/AstroCLIP

module purge
module load cuda/11.8

source /home/huertas/.bashrc
source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate astroclip-teaching

export PYTHONPATH=/n03data/huertas/astroinfo/AstroCLIP:$PYTHONPATH
export ASTROCLIP_ROOT=/n03data/huertas/astroinfo/astroclip_data

GZ_DIR=${ASTROCLIP_ROOT}/datasets/galaxy_zoo
CROSSMATCH=${GZ_DIR}/gz5_decals_crossmatched.hdf5
EMB_OUTPUT=${GZ_DIR}/gz5_decals_embeddings.hdf5
IMAGE_CKPT=${ASTROCLIP_ROOT}/teaching_demo/image_autoencoder.ckpt

mkdir -p ${GZ_DIR}

if [ ! -f "${CROSSMATCH}" ]; then
    python downstream_tasks/morphology_classification/morphology_utils/cross_match.py \
        --root_dir ${ASTROCLIP_ROOT}/datasets/decals \
        --save_path ${CROSSMATCH} \
        --survey_path ${GZ_DIR}/gz_decals_volunteers_5.csv
fi

python teaching_scripts/create_morphology_embeddings.py \
    --crossmatch-file ${CROSSMATCH} \
    --image-ckpt ${IMAGE_CKPT} \
    --output ${EMB_OUTPUT} \
    --batch-size 256 \
    --num-workers 4
