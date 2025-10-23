# Teaching Environment Setup

The steps below create an isolated conda environment with GPU-ready PyTorch and all
dependencies needed to run the teaching utilities (dataset preparation, verification,
and the upcoming notebooks).

## 1. Load system modules (if applicable)

```bash
# Example – adjust to your cluster
module purge
module load cuda/11.8 gcc/12.3
```

If CUDA/CuDNN are already available on your server, this step can be skipped.

## 2. Create and activate a conda environment

```bash
conda create -n astroclip-teaching python=3.10 -y
conda activate astroclip-teaching

# Ensure packages install into this environment, not the user-site directory
export PYTHONNOUSERSITE=1
```

## 3. Install GPU-enabled PyTorch

```bash
pip install --upgrade pip
pip install --extra-index-url https://download.pytorch.org/whl/cu117 \
    torch==2.0.0 torchvision==0.15.0 torchmetrics==0.10.3
```

If your server uses a different CUDA version, pick the matching command from
https://pytorch.org/get-started/locally/.

## 4. Install project dependencies

To avoid pip’s resolver conflicts between the CUDA-labelled PyTorch wheels and the
strict requirements declared by some git dependencies, install the packages in two
groups: core PyPI wheels followed by the git-based repos without pulling their deps.

```bash
# Core wheels
python -m pip install astropy datasets==4.2.0 huggingface_hub jaxtyping wandb \
    python-dotenv scikit-image scikit-learn plotly pyro-ppl lightning[extra]==2.3.3

# Dependencies required by provabgs
python -m pip install "h5py>=3.0" emcee speclite zeus-mcmc

# Git dependencies (deps already satisfied above)
python -m pip install --no-deps git+https://github.com/facebookresearch/dinov2.git@2302b6bf46953431b969155307b9bed152754069
python -m pip install --no-deps git+https://github.com/changhoonhahn/provabgs.git

# Extras for notebooks, plotting, and evaluation
python -m pip install jupyterlab ipywidgets umap-learn seaborn
```

## 5. Install AstroCLIP in editable mode

```bash
pip install -e . --no-deps
```

This exposes the console scripts (`astroclip_trainer`, `spectrum_trainer`,
`image_trainer`) and ensures the teaching scripts can import the package.

## 6. (Optional) Verify the setup

```bash
python teaching_scripts/prepare_dataset.py --streaming
python teaching_scripts/verify_dataset.py
```

These commands download a small subset of the paired dataset and confirm that a GPU
machine can load a batch successfully.

The environment is now ready for running the full training pipeline and the upcoming
instructional notebooks. Remember to activate the environment (`conda activate
astroclip-teaching`) in each new shell before launching scripts.
