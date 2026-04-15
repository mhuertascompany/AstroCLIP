#!/bin/bash
#SBATCH --job-name=cosmos_finetune
#SBATCH --output=/n03data/huertas/COSMOS-Web/ilbert_finetune/zoobot.out
#SBATCH --error=/n03data/huertas/COSMOS-Web/ilbert_finetune/zoobot.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=n03
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1


source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual/

cd /n03data/huertas/python/cosmosweb_lowQ


python /n03data/huertas/python/cosmosweb_lowQ/moprhology/zoobot/apply_models_to_master.py --catalog /n07data/ilbert/COSMOS-Web/photoz_MASTER_v3.1.0/MORPHO/COSMOSWeb_mastercatalog_v1.fits --stamp-root /n03data/huertas/COSMOS-Web/zoobot/stamps_ilbert --ckpt-regular /n03data/huertas/COSMOS-Web/zoobot/models/ilbert_finetune/checkpoints/regular_6.ckpt --ckpt-family /n03data/huertas/COSMOS-Web/zoobot/models/ilbert_finetune/checkpoints/family_2.ckpt --ckpt-binary /n03data/huertas/COSMOS-Web/zoobot/models/ilbert_finetune/checkpoints/binary_3-v1.ckpt --output /n03data/huertas/COSMOS-Web/ilbert_finetune/ilbert_visual_zoobot_morphology.fits --accelerator gpu --devices 1
