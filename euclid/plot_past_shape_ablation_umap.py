"""Joint UMAP of real and independent-past SFHs in the frozen AE space."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap

from .export_sfh_autoencoder_embeddings import encode_sfhs
from .plot_null_sfh_resampling_examples import linear_sfh
from .pretrain_sfh_autoencoder import EuclidSFHAutoencoder
from .sfh_surrogates import recent_preserved_independent_past


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-objects", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--recent-fraction", type=float, default=.1)
    parser.add_argument("--transition-bins", type=int, default=8)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=.1)
    parser.add_argument("--seed", type=int, default=173)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    with h5py.File(args.h5, "r") as source:
        n_total = len(source["sfh"])
        rows = np.sort(rng.choice(
            n_total, min(args.max_objects, n_total), replace=False,
        ))
        ids = np.asarray(source["galaxy_id"][rows], dtype=np.int64)
        time = np.asarray(source["sfh_time_grid"], dtype=np.float64)
        epsilon = float(source.attrs.get("sfh_log_epsilon", 1e-10))
        real = linear_sfh(np.asarray(source["sfh"][rows]), epsilon)

    donor_order = rng.permutation(len(real))
    if np.any(donor_order == np.arange(len(real))):
        donor_order = np.roll(donor_order, 1)
    generated = np.empty_like(real)
    accepted = np.empty(len(real), dtype=np.int64)
    for row in range(len(real)):
        donor = int(donor_order[row])
        for _ in range(len(real)):
            try:
                generated[row], _, _ = recent_preserved_independent_past(
                    real[row], real[donor], time,
                    recent_fraction=args.recent_fraction,
                    transition_bins=args.transition_bins,
                )
                break
            except ValueError:
                donor = (donor + 1) % len(real)
        else:
            raise RuntimeError("no valid donor found")
        accepted[row] = donor

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = EuclidSFHAutoencoder.load_from_checkpoint(
        str(args.checkpoint), map_location="cpu", weights_only=False,
    ).to(device)
    real_embedding, _ = encode_sfhs(
        model, np.log10(real + epsilon).astype(np.float32),
        device, args.batch_size,
    )
    generated_embedding, _ = encode_sfhs(
        model, np.log10(generated + epsilon).astype(np.float32),
        device, args.batch_size,
    )
    combined = np.concatenate((real_embedding, generated_embedding), axis=0)
    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors, min_dist=args.min_dist,
        metric="cosine", random_state=args.seed, n_jobs=1,
    )
    xy = reducer.fit_transform(combined).astype(np.float32)
    xy_real, xy_generated = np.split(xy, 2)

    recent = time <= args.recent_fraction
    recent_mass = real[:, recent].sum(axis=1)
    log_recent = np.log10(np.maximum(recent_mass, 1e-5))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output.with_suffix(".npz"), rows=rows, galaxy_id=ids,
        donor_rows=rows[accepted], donor_id=ids[accepted], time=time,
        recent_mass=recent_mass.astype(np.float32),
        xy_real=xy_real, xy_generated=xy_generated,
        real_embedding=real_embedding.astype(np.float32),
        generated_embedding=generated_embedding.astype(np.float32),
    )

    figure, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    axis = axes[0, 0]
    axis.scatter(xy_real[:, 0], xy_real[:, 1], s=3, alpha=.18,
                 color="#3b3b3b", rasterized=True, label="real SFH")
    axis.scatter(xy_generated[:, 0], xy_generated[:, 1], s=3, alpha=.18,
                 color="#1b9e77", rasterized=True, label="generated donor-past SFH")
    vector_rows = rng.choice(len(real), min(250, len(real)), replace=False)
    delta = xy_generated[vector_rows] - xy_real[vector_rows]
    axis.quiver(
        xy_real[vector_rows, 0], xy_real[vector_rows, 1],
        delta[:, 0], delta[:, 1], angles="xy", scale_units="xy", scale=1,
        width=.0015, alpha=.18, color="#d95f02",
    )
    axis.legend(frameon=False, markerscale=3)
    axis.set_title("Common UMAP with paired displacements")

    axis = axes[0, 1]
    origin = np.r_[np.zeros(len(real)), np.ones(len(real))]
    image = axis.hexbin(
        xy[:, 0], xy[:, 1], C=origin, reduce_C_function=np.mean,
        gridsize=55, mincnt=4, cmap="PiYG", vmin=0, vmax=1,
    )
    figure.colorbar(image, ax=axis, label="Local generated fraction")
    axis.set_title("Local separation of real and generated SFHs")

    vmin, vmax = np.quantile(log_recent, [.02, .98])
    for axis, coordinates, title in (
        (axes[1, 0], xy_real, "Real SFHs"),
        (axes[1, 1], xy_generated, "Generated SFHs: real recent + donor past"),
    ):
        image = axis.scatter(
            coordinates[:, 0], coordinates[:, 1], c=log_recent,
            s=5, alpha=.55, cmap="viridis", vmin=vmin, vmax=vmax,
            linewidths=0, rasterized=True,
        )
        axis.set_title(title)
        figure.colorbar(image, ax=axis, label="log10 recent formed-mass fraction")
    for axis in axes.ravel():
        axis.set_xlabel("UMAP 1")
        axis.set_ylabel("UMAP 2")
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(
        "Frozen SFH-AE geometry: real versus independent-past ablation",
        fontsize=15,
    )
    figure.savefig(args.output.with_suffix(".png"), dpi=180)
    figure.savefig(args.output.with_suffix(".pdf"))
    print(f"Saved {args.output.with_suffix('.png')}")
    print(f"Saved {args.output.with_suffix('.pdf')}")
    print(f"Saved {args.output.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
