"""Train image--SFH alignment on Euclid VIS JPEG stamps.

The SFH dimension is read from the preprocessed HDF5 file. During training the
data loader can draw a valid posterior SFH realization for each galaxy; the
validation set always uses the posterior-median SFH.
"""

import argparse
from pathlib import Path

import h5py
import lightning as L
import numpy as np
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger

from cosmosweb.sfh_autoencoder import load_sfh_autoencoder_checkpoint

from .dataset_zoobot import EuclidZooBotDataModule
from .model_zoobot import EuclidZooBotCLIP
from .training_index import inspect_sfh_file


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train an image--SFH contrastive model on Euclid data.',
    )
    data = parser.add_argument_group('data')
    data.add_argument('--dataset', type=Path, required=True,
                      help='Preprocessed Euclid SFH HDF5 file.')
    data.add_argument('--stamp-root', type=Path,
                      help='Directory containing BAND/BAND_object_id.jpg for ZooBot.')
    data.add_argument('--fits-root', type=Path,
                      help='Native cutout run containing cutouts/VIS/object_id.fits.')
    data.add_argument(
        '--eligibility-stamp-root', type=Path,
        help=(
            'Optional BAND/BAND_object_id.jpg root used only to restrict the '
            'training population while images are read from --fits-root.'
        ),
    )
    data.add_argument('--image-stats', type=Path,
                      help='GalaxyTikTok-style global VIS percentile JSON.')
    data.add_argument('--asinh-scale', type=float, default=20.0)
    data.add_argument('--band', default='VIS')
    data.add_argument('--image-size', type=int, default=224)
    data.add_argument('--val-fraction', type=float, default=0.1)
    data.add_argument('--seed', type=int, default=42)
    data.add_argument('--max-pairs', type=int,
                      help='Limit matched pairs for a smoke test.')
    data.add_argument(
        '--max-vis-mag', type=float,
        help='Keep VIS-detected objects at or brighter than this AB magnitude.',
    )
    data.add_argument(
        '--vis-flux-column', default='flux_detection_total',
        help='HDF5 or selection-catalog microJy flux used for the VIS cut.',
    )
    data.add_argument('--vis-detection-column', default='vis_det')
    data.add_argument(
        '--selection-catalog', type=Path,
        help='Optional FITS photometry catalog joined to HDF5 rows by object ID.',
    )
    data.add_argument('--selection-id-column', default='object_id')
    data.add_argument(
        '--exclude-edge-on-axis-ratio-below', type=float,
        help='Exclude only when Sérsic b/a is strictly below this value.',
    )
    data.add_argument(
        '--exclude-edge-on-probability-above', type=float,
        help='Exclude only when conditional ZooBot P(edge-on) is above this value.',
    )
    data.add_argument(
        '--require-vis-detection', action=argparse.BooleanOptionalAction,
        default=True,
        help='Require VIS_DET=1 when applying a VIS magnitude cut.',
    )
    data.add_argument(
        '--sample-posterior',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Draw a valid SFH posterior realization in training (default: on).',
    )
    data.add_argument('--batch-size', type=int, default=128)
    data.add_argument('--num-workers', type=int, default=8)

    model = parser.add_argument_group('model')
    encoder_source = model.add_mutually_exclusive_group(required=True)
    encoder_source.add_argument(
        '--zoobot-model-name',
        help='Timm/Hugging Face encoder name, e.g. hf_hub:mwalmsley/zoobot-encoder-euclid.',
    )
    encoder_source.add_argument(
        '--galactiktok-checkpoint',
        type=Path,
        help='GalaxyTikTok save_pretrained directory (config.json + model.safetensors).',
    )
    model.add_argument('--galactiktok-band', default='euclid-vis')
    model.add_argument('--token-pool-hidden-dim', type=int, default=256)
    model.add_argument('--token-pool-heads', type=int, default=4)
    model.add_argument('--token-pool-layers', type=int, default=2)
    encoder_source.add_argument(
        '--zoobot-ckpt',
        type=Path,
        help='Local FinetuneableZoobotClassifier checkpoint.',
    )
    model.add_argument('--embed-dim', type=int, default=256)
    model.add_argument('--temperature', type=float, default=0.07)
    model.add_argument('--queue-size', type=int, default=4096)
    model.add_argument('--momentum', type=float, default=0.995)
    model.add_argument(
        '--alignment-objective',
        choices=('clip', 'pcmepp', 'cwcl', 'cyclip'), default='clip',
        help=(
            'Cross-modal objective: exact-pair CLIP, probabilistic PCME++, '
            'SFH-AE continuously weighted CWCL, or cyclic-consistency CyCLIP.'
        ),
    )
    model.add_argument(
        '--pcme-pseudo-positive-weight', type=float, default=0.1,
        help='PCME++ pseudo-positive BCE weight (paper default: 0.1).',
    )
    model.add_argument(
        '--pcme-vib-weight', type=float, default=1e-4,
        help='PCME++ variational information-bottleneck weight.',
    )
    model.add_argument(
        '--pcme-initial-scale', type=float, default=5.0,
        help='Initial positive scale a in PCME++ match logit -a*d+b.',
    )
    model.add_argument(
        '--pcme-initial-bias', type=float, default=5.0,
        help='Initial bias b in PCME++ match logit -a*d+b.',
    )
    model.add_argument(
        '--pcme-initial-uncertainty', type=float, default=0.01,
        help='Initial summed diagonal variance for each modality.',
    )
    model.add_argument(
        '--cwcl-similarity-temperature', type=float, default=0.1,
        help='Temperature of the continuous SFH-AE cosine kernel.',
    )
    model.add_argument(
        '--cwcl-reverse-exact-weight', type=float, default=1.0,
        help='Weight of the exact paired SFH-to-image term in CWCL.',
    )
    model.add_argument(
        '--cyclip-clip-weight', type=float, default=0.25,
        help='Weight of the paired symmetric CLIP anchor in CyCLIP.',
    )
    model.add_argument(
        '--cyclip-inmodal-weight', type=float, default=1.0,
        help='Weight matching image-image and SFH-SFH cosine geometry.',
    )
    model.add_argument(
        '--cyclip-crossmodal-weight', type=float, default=0.25,
        help='Weight enforcing symmetry of the cross-modal cosine matrix.',
    )
    model.add_argument('--unfreeze-blocks', type=int, default=0)
    model.add_argument('--backbone-lr-scale', type=float, default=0.1)
    model.add_argument(
        '--image-projection', choices=('legacy', 'mlp'), default='legacy',
        help='Image adapter architecture; mlp enables an unrestricted deep adapter.',
    )
    model.add_argument('--image-projection-hidden-dim', type=int, default=512)
    model.add_argument('--image-projection-hidden-layers', type=int, default=2)
    model.add_argument(
        '--sfh-encoder', choices=('mlp', 'transformer'), default='transformer',
    )
    model.add_argument('--sfh-d-model', type=int, default=128)
    model.add_argument('--sfh-n-heads', type=int, default=4)
    model.add_argument('--sfh-n-layers', type=int, default=4)
    model.add_argument('--sfh-lr-scale', type=float, default=1.0)
    model.add_argument(
        '--sfh-pretrained-checkpoint', type=Path,
        help='SFH encoder-decoder checkpoint used to initialize a new CLIP run.',
    )
    model.add_argument(
        '--sfh-reconstruction-weight', type=float, default=0.0,
        help='Auxiliary SFH reconstruction weight during contrastive training.',
    )
    model.add_argument('--sfh-reconstruction-w1-weight', type=float, default=0.5)
    model.add_argument('--sfh-decoder-layers', type=int, default=2)
    model.add_argument(
        '--sfh-projection',
        choices=('identity', 'linear', 'residual_mlp', 'mlp'), default='identity',
        help='Map from the SFH encoder latent into CLIP space (default: identity).',
    )
    model.add_argument(
        '--sfh-projection-hidden-dim', type=int, default=512,
        help='Hidden width of the MLP SFH projection.',
    )
    model.add_argument('--sfh-projection-hidden-layers', type=int, default=2)
    model.add_argument(
        '--sfh-projection-residual-scale', type=float, default=0.1,
        help='Fixed residual-branch scale for the residual MLP projection.',
    )
    model.add_argument(
        '--freeze-sfh-encoder', action='store_true',
        help='Keep the SFH encoder fixed while training the CLIP projection.',
    )
    model.add_argument(
        '--freeze-sfh-decoder', action='store_true',
        help='Keep the pretrained reconstruction decoder fixed.',
    )
    model.add_argument(
        '--soft-positive-weight', type=float, default=0.0,
        help='Weight assigned to W1-neighbour SFHs; zero preserves exact InfoNCE.',
    )
    model.add_argument(
        '--soft-positive-k', type=int, default=8,
        help='Number of within-batch W1-nearest SFHs receiving soft target mass.',
    )
    model.add_argument(
        '--ae-false-negative-k', type=int, default=0,
        help=(
            'Exclude this many nearest off-diagonal SFHs per anchor from the '
            'InfoNCE denominator, using cosine distance in the fixed pretrained '
            'SFH-autoencoder latent space (default: disabled).'
        ),
    )
    model.add_argument(
        '--ae-false-negative-max-distance', type=float,
        help=(
            'Optional maximum AE cosine distance for excluded negatives. With '
            '--ae-false-negative-k 0, exclude every pair within this distance.'
        ),
    )
    model.add_argument(
        '--ae-adjacency-weight', type=float, default=0.0,
        help=(
            'Weight of the parameter-free FNAC-style L1 consistency between '
            'cross-modal probabilities and fixed AE latent adjacency.'
        ),
    )
    model.add_argument(
        '--ae-adjacency-warmup-epochs', type=int, default=3,
        help='Exact-InfoNCE warm-up before enabling AE adjacency regularization.',
    )
    model.add_argument(
        '--ae-adjacency-temperature', type=float, default=0.07,
        help='Fixed temperature for AE and cross-modal adjacency distributions.',
    )

    optimization = parser.add_argument_group('optimization')
    optimization.add_argument('--lr', type=float, default=1e-4)
    optimization.add_argument('--weight-decay', type=float, default=0.05)
    optimization.add_argument('--warmup-epochs', type=int, default=5)
    optimization.add_argument('--max-epochs', type=int, default=100)
    optimization.add_argument('--patience', type=int, default=20)

    runtime = parser.add_argument_group('runtime')
    runtime.add_argument('--accelerator', default='auto')
    runtime.add_argument('--devices', default='auto')
    runtime.add_argument('--precision', default='16-mixed')
    runtime.add_argument('--output-dir', type=Path, required=True)
    runtime.add_argument('--run-name', default='euclid_zoobot_clip')
    runtime.add_argument('--resume-from', type=Path)
    return parser.parse_args()


def validate_args(args):
    if not args.dataset.is_file():
        raise FileNotFoundError(f'SFH dataset not found: {args.dataset}')
    if args.galactiktok_checkpoint is None:
        if args.stamp_root is None:
            raise ValueError('ZooBot training requires --stamp-root.')
        stamp_dir = args.stamp_root / args.band
        if not stamp_dir.is_dir():
            raise FileNotFoundError(f'JPEG directory not found: {stamp_dir}')
        if args.fits_root is not None or args.image_stats is not None:
            raise ValueError('FITS inputs are only used with GalaxyTikTok.')
        if args.eligibility_stamp_root is not None:
            raise ValueError(
                '--eligibility-stamp-root is only needed with GalaxyTikTok.'
            )
    else:
        if args.fits_root is None or args.image_stats is None:
            raise ValueError(
                'GalaxyTikTok training requires --fits-root and --image-stats.'
            )
        from .vis_fits import load_image_stats, resolve_fits_directory
        resolve_fits_directory(args.fits_root, args.band)
        if not args.image_stats.is_file():
            raise FileNotFoundError(f'Image statistics not found: {args.image_stats}')
        load_image_stats(args.image_stats, args.galactiktok_band)
        if args.stamp_root is not None:
            raise ValueError('Use --fits-root, not --stamp-root, with GalaxyTikTok.')
    if args.zoobot_ckpt is not None and not args.zoobot_ckpt.is_file():
        raise FileNotFoundError(f'ZooBot checkpoint not found: {args.zoobot_ckpt}')
    if (args.galactiktok_checkpoint is not None
            and not args.galactiktok_checkpoint.is_dir()):
        raise FileNotFoundError(
            f'GalaxyTikTok checkpoint not found: {args.galactiktok_checkpoint}'
        )
    if args.galactiktok_checkpoint is not None:
        required = ('config.json', 'model.safetensors')
        missing = [name for name in required
                   if not (args.galactiktok_checkpoint / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f'GalaxyTikTok checkpoint is missing {missing}: '
                f'{args.galactiktok_checkpoint}'
            )
        if args.unfreeze_blocks:
            raise ValueError('--unfreeze-blocks is only supported for ZooBot.')
    if min(args.token_pool_hidden_dim, args.token_pool_heads,
           args.token_pool_layers) < 1:
        raise ValueError('Token-pooler dimensions must be positive.')
    if args.token_pool_hidden_dim % args.token_pool_heads:
        raise ValueError('--token-pool-hidden-dim must divide by --token-pool-heads.')
    if args.resume_from is not None and not args.resume_from.is_file():
        raise FileNotFoundError(f'Resume checkpoint not found: {args.resume_from}')
    if (args.sfh_pretrained_checkpoint is not None
            and not args.sfh_pretrained_checkpoint.is_file()):
        raise FileNotFoundError(
            f'SFH pretraining checkpoint not found: {args.sfh_pretrained_checkpoint}'
        )
    if args.batch_size < 2:
        raise ValueError('--batch-size must be at least 2.')
    if not np.isfinite(args.asinh_scale) or args.asinh_scale <= 0:
        raise ValueError('--asinh-scale must be finite and positive.')
    if args.max_vis_mag is not None and not np.isfinite(args.max_vis_mag):
        raise ValueError('--max-vis-mag must be finite.')
    edge_thresholds = (
        args.exclude_edge_on_axis_ratio_below,
        args.exclude_edge_on_probability_above,
    )
    if (edge_thresholds[0] is None) != (edge_thresholds[1] is None):
        raise ValueError('Both edge-on exclusion thresholds must be provided.')
    if edge_thresholds[0] is not None and not (
        0 < edge_thresholds[0] <= 1 and 0 <= edge_thresholds[1] <= 1
    ):
        raise ValueError('Edge-on exclusion thresholds are outside [0, 1].')
    if args.queue_size < 0:
        raise ValueError('--queue-size cannot be negative.')
    if args.queue_size and args.queue_size < args.batch_size:
        raise ValueError('--queue-size must be zero or at least --batch-size.')
    if args.queue_size and args.queue_size % args.batch_size:
        raise ValueError('--queue-size must be divisible by --batch-size.')
    if args.pcme_pseudo_positive_weight < 0:
        raise ValueError('--pcme-pseudo-positive-weight cannot be negative.')
    if args.pcme_vib_weight < 0:
        raise ValueError('--pcme-vib-weight cannot be negative.')
    if args.pcme_initial_scale <= 0:
        raise ValueError('--pcme-initial-scale must be positive.')
    if args.pcme_initial_uncertainty <= 0:
        raise ValueError('--pcme-initial-uncertainty must be positive.')
    if args.cwcl_similarity_temperature <= 0:
        raise ValueError('--cwcl-similarity-temperature must be positive.')
    if args.cwcl_reverse_exact_weight < 0:
        raise ValueError('--cwcl-reverse-exact-weight cannot be negative.')
    if min(
        args.cyclip_clip_weight,
        args.cyclip_inmodal_weight,
        args.cyclip_crossmodal_weight,
    ) < 0:
        raise ValueError('CyCLIP weights cannot be negative.')
    if (
        args.cyclip_clip_weight
        + args.cyclip_inmodal_weight
        + args.cyclip_crossmodal_weight
    ) == 0:
        raise ValueError('At least one CyCLIP weight must be positive.')
    if not 0 <= args.soft_positive_weight <= 1:
        raise ValueError('--soft-positive-weight must lie in [0, 1].')
    if args.soft_positive_k < 1:
        raise ValueError('--soft-positive-k must be positive.')
    if args.soft_positive_weight and args.queue_size:
        raise ValueError('--soft-positive-weight requires --queue-size 0.')
    ae_filtering = (
        args.ae_false_negative_k > 0
        or args.ae_false_negative_max_distance is not None
    )
    if args.ae_false_negative_k < 0:
        raise ValueError('--ae-false-negative-k cannot be negative.')
    if args.ae_false_negative_max_distance is not None and not (
        0 <= args.ae_false_negative_max_distance <= 2
    ):
        raise ValueError('--ae-false-negative-max-distance must lie in [0, 2].')
    if ae_filtering and args.queue_size:
        raise ValueError('AE false-negative filtering requires --queue-size 0.')
    if ae_filtering and args.soft_positive_weight:
        raise ValueError(
            'Choose either AE false-negative filtering or soft positives.'
        )
    if ae_filtering and not args.freeze_sfh_encoder:
        raise ValueError(
            'AE false-negative filtering requires --freeze-sfh-encoder.'
        )
    if args.ae_adjacency_weight < 0:
        raise ValueError('--ae-adjacency-weight cannot be negative.')
    if args.ae_adjacency_warmup_epochs < 0:
        raise ValueError('--ae-adjacency-warmup-epochs cannot be negative.')
    if args.ae_adjacency_temperature <= 0:
        raise ValueError('--ae-adjacency-temperature must be positive.')
    if args.ae_adjacency_weight:
        if args.queue_size:
            raise ValueError('AE adjacency regularization requires --queue-size 0.')
        if args.soft_positive_weight:
            raise ValueError(
                'Choose either AE adjacency regularization or soft positives.'
            )
        if ae_filtering:
            raise ValueError(
                'Choose either continuous AE adjacency regularization or hard '
                'AE false-negative filtering.'
            )
        if not args.freeze_sfh_encoder:
            raise ValueError(
                'AE adjacency regularization requires --freeze-sfh-encoder.'
            )
    if args.alignment_objective in {'pcmepp', 'cwcl', 'cyclip'}:
        incompatible = []
        if args.queue_size:
            incompatible.append('--queue-size')
        if args.soft_positive_weight:
            incompatible.append('--soft-positive-weight')
        if ae_filtering:
            incompatible.append('--ae-false-negative-*')
        if args.ae_adjacency_weight:
            incompatible.append('--ae-adjacency-weight')
        if incompatible:
            raise ValueError(
                f'{args.alignment_objective} uses its own in-batch objective; disable '
                + ', '.join(incompatible) + '.'
            )
    if args.alignment_objective == 'cwcl' and not args.freeze_sfh_encoder:
        raise ValueError(
            'CWCL requires --freeze-sfh-encoder so its SFH-AE targets are fixed.'
        )
    if args.sfh_encoder == 'transformer':
        if min(args.sfh_d_model, args.sfh_n_heads, args.sfh_n_layers) < 1:
            raise ValueError('SFH transformer dimensions must be positive.')
        if args.sfh_d_model % args.sfh_n_heads:
            raise ValueError('--sfh-d-model must be divisible by --sfh-n-heads.')
    if args.sfh_lr_scale <= 0:
        raise ValueError('--sfh-lr-scale must be positive.')
    if args.sfh_reconstruction_weight < 0:
        raise ValueError('--sfh-reconstruction-weight cannot be negative.')
    if not 0 <= args.sfh_reconstruction_w1_weight <= 1:
        raise ValueError('--sfh-reconstruction-w1-weight must lie in [0, 1].')
    if args.sfh_decoder_layers < 1:
        raise ValueError('--sfh-decoder-layers must be positive.')
    if args.sfh_reconstruction_weight and args.sfh_encoder != 'transformer':
        raise ValueError('SFH reconstruction requires --sfh-encoder transformer.')
    if ((args.freeze_sfh_encoder or args.freeze_sfh_decoder)
            and args.sfh_pretrained_checkpoint is None
            and args.resume_from is None):
        raise ValueError(
            'Freezing the SFH autoencoder requires --sfh-pretrained-checkpoint '
            'for a new run, or --resume-from.'
        )
    if args.freeze_sfh_decoder and args.sfh_reconstruction_weight <= 0:
        raise ValueError(
            '--freeze-sfh-decoder requires --sfh-reconstruction-weight > 0.'
        )
    if args.sfh_projection_hidden_dim <= 0:
        raise ValueError('--sfh-projection-hidden-dim must be positive.')
    if args.image_projection_hidden_dim <= 0:
        raise ValueError('--image-projection-hidden-dim must be positive.')
    if min(args.image_projection_hidden_layers,
           args.sfh_projection_hidden_layers) <= 0:
        raise ValueError('Projection hidden-layer counts must be positive.')
    if args.sfh_projection_residual_scale <= 0:
        raise ValueError('--sfh-projection-residual-scale must be positive.')
    if args.unfreeze_blocks < 0:
        raise ValueError('--unfreeze-blocks cannot be negative.')
    if args.backbone_lr_scale <= 0:
        raise ValueError('--backbone-lr-scale must be positive.')


def main():
    args = parse_args()
    validate_args(args)
    _, n_bins, n_realizations = inspect_sfh_file(args.dataset)
    with h5py.File(args.dataset, 'r') as source:
        sfh_log_epsilon = float(source.attrs.get('sfh_log_epsilon', 1e-10))
    L.seed_everything(args.seed, workers=True)

    datamodule = EuclidZooBotDataModule(
        sfh_path=args.dataset,
        stamp_root=(
            args.fits_root
            if args.galactiktok_checkpoint is not None else args.stamp_root
        ),
        band=args.band,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
        sample_posterior=args.sample_posterior,
        max_pairs=args.max_pairs,
        max_vis_mag=args.max_vis_mag,
        vis_flux_column=args.vis_flux_column,
        vis_detection_column=args.vis_detection_column,
        require_vis_detection=args.require_vis_detection,
        selection_catalog=args.selection_catalog,
        selection_id_column=args.selection_id_column,
        image_format=(
            'fits' if args.galactiktok_checkpoint is not None else 'jpg'
        ),
        image_stats=args.image_stats,
        asinh_scale=args.asinh_scale,
        eligibility_stamp_root=args.eligibility_stamp_root,
        exclude_edge_on_axis_ratio_below=(
            args.exclude_edge_on_axis_ratio_below
        ),
        exclude_edge_on_probability_above=(
            args.exclude_edge_on_probability_above
        ),
    )
    datamodule.setup('fit')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair_index = datamodule.pair_index
    split_data = dict(
        train_rows=pair_index.train_rows,
        train_ids=pair_index.train_ids,
        val_rows=pair_index.val_rows,
        val_ids=pair_index.val_ids,
    )
    if args.max_vis_mag is not None:
        split_data.update(
            max_vis_mag=np.float32(args.max_vis_mag),
            vis_flux_column=np.asarray(args.vis_flux_column),
            vis_detection_column=np.asarray(args.vis_detection_column),
            require_vis_detection=np.asarray(args.require_vis_detection),
            selection_catalog=np.asarray(
                str(args.selection_catalog) if args.selection_catalog else ''
            ),
            selection_id_column=np.asarray(args.selection_id_column),
        )
    if args.exclude_edge_on_axis_ratio_below is not None:
        split_data.update(
            exclude_edge_on_axis_ratio_below=np.float32(
                args.exclude_edge_on_axis_ratio_below
            ),
            exclude_edge_on_probability_above=np.float32(
                args.exclude_edge_on_probability_above
            ),
            n_edge_on_excluded=np.int64(pair_index.n_edge_on_excluded),
        )
    np.savez_compressed(args.output_dir / 'pair_split.npz', **split_data)
    model = EuclidZooBotCLIP(
        zoobot_ckpt=str(args.zoobot_ckpt) if args.zoobot_ckpt else None,
        zoobot_model_name=args.zoobot_model_name,
        image_encoder_type=(
            'galactiktok' if args.galactiktok_checkpoint is not None else 'zoobot'
        ),
        galactiktok_checkpoint=(
            str(args.galactiktok_checkpoint)
            if args.galactiktok_checkpoint is not None else None
        ),
        galactiktok_band=args.galactiktok_band,
        token_pool_hidden_dim=args.token_pool_hidden_dim,
        token_pool_heads=args.token_pool_heads,
        token_pool_layers=args.token_pool_layers,
        embed_dim=args.embed_dim,
        sfh_input_dim=n_bins,
        temperature=args.temperature,
        queue_size=args.queue_size,
        momentum=args.momentum,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.max_epochs,
        warmup_epochs=args.warmup_epochs,
        unfreeze_blocks=args.unfreeze_blocks,
        backbone_lr_scale=args.backbone_lr_scale,
        image_projection_type=args.image_projection,
        image_projection_hidden_dim=args.image_projection_hidden_dim,
        image_projection_hidden_layers=args.image_projection_hidden_layers,
        sfh_encoder_type=args.sfh_encoder,
        sfh_d_model=args.sfh_d_model,
        sfh_n_heads=args.sfh_n_heads,
        sfh_n_layers=args.sfh_n_layers,
        sfh_lr_scale=args.sfh_lr_scale,
        soft_positive_weight=args.soft_positive_weight,
        soft_positive_k=args.soft_positive_k,
        ae_false_negative_k=args.ae_false_negative_k,
        ae_false_negative_max_distance=args.ae_false_negative_max_distance,
        ae_adjacency_weight=args.ae_adjacency_weight,
        ae_adjacency_warmup_epochs=args.ae_adjacency_warmup_epochs,
        ae_adjacency_temperature=args.ae_adjacency_temperature,
        sfh_log_epsilon=sfh_log_epsilon,
        sfh_reconstruction_weight=args.sfh_reconstruction_weight,
        sfh_reconstruction_w1_weight=args.sfh_reconstruction_w1_weight,
        sfh_decoder_layers=args.sfh_decoder_layers,
        sfh_projection_type=args.sfh_projection,
        sfh_projection_hidden_dim=args.sfh_projection_hidden_dim,
        sfh_projection_residual_scale=args.sfh_projection_residual_scale,
        sfh_projection_hidden_layers=args.sfh_projection_hidden_layers,
        freeze_sfh_encoder=args.freeze_sfh_encoder,
        freeze_sfh_decoder=args.freeze_sfh_decoder,
        alignment_objective=args.alignment_objective,
        pcme_pseudo_positive_weight=args.pcme_pseudo_positive_weight,
        pcme_vib_weight=args.pcme_vib_weight,
        pcme_initial_scale=args.pcme_initial_scale,
        pcme_initial_bias=args.pcme_initial_bias,
        pcme_initial_uncertainty=args.pcme_initial_uncertainty,
        cwcl_similarity_temperature=args.cwcl_similarity_temperature,
        cwcl_reverse_exact_weight=args.cwcl_reverse_exact_weight,
        cyclip_clip_weight=args.cyclip_clip_weight,
        cyclip_inmodal_weight=args.cyclip_inmodal_weight,
        cyclip_crossmodal_weight=args.cyclip_crossmodal_weight,
    )
    if args.sfh_pretrained_checkpoint is not None and args.resume_from is None:
        load_sfh_autoencoder_checkpoint(
            model.sfh_encoder,
            args.sfh_pretrained_checkpoint,
            decoder=model.sfh_decoder,
        )
        model.sfh_encoder_m.load_state_dict(model.sfh_encoder.state_dict())
        model.sfh_projection_m.load_state_dict(model.sfh_projection.state_dict())
        print(
            f'Initialized SFH encoder'
            f'{" and decoder" if model.sfh_decoder is not None else ""} from '
            f'{args.sfh_pretrained_checkpoint}',
            flush=True,
        )
    elif args.sfh_pretrained_checkpoint is not None:
        print(
            'Ignoring --sfh-pretrained-checkpoint because --resume-from restores '
            'the complete CLIP state.',
            flush=True,
        )

    checkpoint_dir = args.output_dir / 'checkpoints'
    log_dir = args.output_dir / 'logs'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=args.run_name + '-{epoch:03d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
    )
    callbacks = [
        checkpoint_callback,
        EarlyStopping(
            monitor='val_loss',
            patience=args.patience,
            mode='min',
            verbose=True,
        ),
        LearningRateMonitor(logging_interval='epoch'),
    ]
    logger = CSVLogger(save_dir=str(log_dir), name=args.run_name)

    sfh_projection_details = (
        f'hidden={args.sfh_projection_hidden_dim} x '
        f'{args.sfh_projection_hidden_layers}'
    )
    if args.sfh_projection == 'residual_mlp':
        sfh_projection_details += (
            f', residual scale={args.sfh_projection_residual_scale:g}'
        )
    image_encoder_description = (
        f'GalaxyTikTok {args.galactiktok_checkpoint}; attention pooler '
        f'{args.token_pool_hidden_dim}d x {args.token_pool_layers}'
        if args.galactiktok_checkpoint is not None else
        f'ZooBot {args.zoobot_model_name or args.zoobot_ckpt}; image projection '
        f'{args.image_projection}, hidden={args.image_projection_hidden_dim} x '
        f'{args.image_projection_hidden_layers}'
    )
    print(
        f'Training with {n_bins} SFH bins and {n_realizations} posterior '
        f'realizations; posterior sampling={args.sample_posterior}; '
        f'alignment objective={args.alignment_objective}; '
        f'image encoder={image_encoder_description}; '
        f'SFH encoder={args.sfh_encoder}; '
        f'SFH encoder frozen={args.freeze_sfh_encoder}; '
        f'SFH projection={args.sfh_projection}; '
        f'SFH projection {sfh_projection_details}; '
        f'SFH reconstruction weight={args.sfh_reconstruction_weight:g}; '
        f'SFH soft-positive weight={args.soft_positive_weight:g}, '
        f'k={args.soft_positive_k}; '
        f'AE false-negative k={args.ae_false_negative_k}, '
        f'max distance={args.ae_false_negative_max_distance}; '
        f'AE adjacency weight={args.ae_adjacency_weight:g}, '
        f'warmup={args.ae_adjacency_warmup_epochs}, '
        f'temperature={args.ae_adjacency_temperature:g}; '
        f'PCME++ pseudo-positive weight={args.pcme_pseudo_positive_weight:g}, '
        f'VIB weight={args.pcme_vib_weight:g}, '
        f'initial scale={args.pcme_initial_scale:g}, '
        f'initial bias={args.pcme_initial_bias:g}, '
        f'initial uncertainty={args.pcme_initial_uncertainty:g}; '
        f'CWCL similarity temperature={args.cwcl_similarity_temperature:g}, '
        f'reverse exact weight={args.cwcl_reverse_exact_weight:g}; '
        f'CyCLIP weights clip/in-modal/cross-modal='
        f'{args.cyclip_clip_weight:g}/{args.cyclip_inmodal_weight:g}/'
        f'{args.cyclip_crossmodal_weight:g}',
        flush=True,
    )
    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=20,
        gradient_clip_val=1.0,
    )
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=str(args.resume_from) if args.resume_from else None,
    )
    print(f'Best checkpoint: {checkpoint_callback.best_model_path}', flush=True)
    print(f'Last checkpoint: {checkpoint_callback.last_model_path}', flush=True)
    print(f'Logs: {logger.log_dir}', flush=True)


if __name__ == '__main__':
    main()
