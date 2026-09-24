# Current Euclid architecture

`euclid_current_architecture.pdf` and `.svg` are vector figures; `.png` is the raster preview. Regenerate with `python euclid/figures/draw_architecture.py` from the repository root (requires Matplotlib).

The figure describes `slurm_train_zoobot_clip_bright_frozen_mlp.sh`: the imported ZooBot Euclid backbone and the pretrained SFH transformer remain frozen, while the image MLP, residual SFH MLP, and contrastive temperature are trained on paired images and median SFHs. SFH pretraining uses a masked posterior draw to reconstruct the median SFH with a transformer decoder; the decoder is not used in this alignment configuration. Conditional pixel diffusion is a separate downstream experiment.

Implementation references: `euclid/slurm_pretrain_sfh_autoencoder.sh`, `euclid/pretrain_sfh_autoencoder.py`, `cosmosweb/sfh_autoencoder.py`, `cosmosweb/sfh_transformer.py`, `cosmosweb/zoobot_encoder.py`, and `cosmosweb/model_zoobot.py`. The paired sample count refers to the reported bright-sample run, not the number of objects used for SFH pretraining.
