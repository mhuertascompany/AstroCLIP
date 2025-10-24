#!/bin/bash
#SBATCH --job-name=astroclip_astroinfo
#SBATCH --output=/n03data/huertas/astroinfo/logs/astroclip.out
#SBATCH --error=/n03data/huertas/astroinfo/logs/astroclip.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=n03
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/n03data/huertas/astroinfo/AstroCLIP

module load cuda/11.8
# Initialize Conda in this shell
source /home/huertas/.bashrc
source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate astroclip-teaching

# Set CUDA_VISIBLE_DEVICES to use GPU device 0
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PATH=$CONDA_PREFIX/bin:$PATH
export CUDA_HOME=$CONDA_PREFIX
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/n03data/huertas/astroinfo/AstroCLIP:$PYTHONPATH

export ASTROCLIP_ROOT=/n03data/huertas/astroinfo/astroclip_data

#python teaching_scripts/train_image_encoder.py  --dataset $ASTROCLIP_ROOT/teaching_demo/astroclip_dataset  --output $ASTROCLIP_ROOT/teaching_demo/image_autoencoder.ckpt --max-epochs 10
python teaching_scripts/train_spectrum_encoder.py  --dataset $ASTROCLIP_ROOT/teaching_demo/astroclip_dataset  --output $ASTROCLIP_ROOT/teaching_demo/spectrum_autoencoder.ckpt  --max-epochs 15