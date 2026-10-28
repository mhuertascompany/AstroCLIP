"""Measure whether bridged independent-past SFHs are OOD for the fixed AE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
import torch

from .export_sfh_autoencoder_embeddings import encode_sfhs
from .plot_null_sfh_resampling_examples import linear_sfh
from .pretrain_sfh_autoencoder import EuclidSFHAutoencoder
from .sfh_surrogates import recent_preserved_independent_past


def reconstruction_metrics(reconstruction, target):
    return {
        "mae": np.abs(reconstruction - target).mean(axis=1),
        "w1": np.abs(
            np.cumsum(reconstruction, axis=1) - np.cumsum(target, axis=1)
        ).mean(axis=1),
    }


def paired_auc(real, altered, seed, splits=5):
    features = np.concatenate((real, altered), axis=0)
    labels = np.r_[np.zeros(len(real)), np.ones(len(altered))]
    groups = np.r_[np.arange(len(real)), np.arange(len(real))]
    values = []
    splitter = GroupShuffleSplit(n_splits=splits, test_size=.3, random_state=seed)
    for train, test in splitter.split(features, labels, groups):
        model = LogisticRegression(max_iter=1000, C=1.0)
        model.fit(features[train], labels[train])
        values.append(roc_auc_score(labels[test], model.predict_proba(features[test])[:, 1]))
    return np.asarray(values)


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(values)),
        "p16": float(np.quantile(values, .16)),
        "p84": float(np.quantile(values, .84)),
        "mean": float(np.mean(values)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recent-fraction", type=float, default=.1)
    parser.add_argument("--transition-bins", type=int, default=8)
    parser.add_argument("--max-objects", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=173)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    with h5py.File(args.h5, "r") as source:
        time = np.asarray(source["sfh_time_grid"], dtype=np.float64)
        epsilon = float(source.attrs.get("sfh_log_epsilon", 1e-10))
        n_total = len(source["sfh"])
        rows = np.sort(rng.choice(
            n_total, min(args.max_objects, n_total), replace=False,
        ))
        weights = linear_sfh(np.asarray(source["sfh"][rows]), epsilon)

    order = rng.permutation(len(weights))
    if np.any(order == np.arange(len(order))):
        order = np.roll(order, 1)
    smooth = np.empty_like(weights)
    raw = np.empty_like(weights)
    self_control = np.empty_like(weights)
    accepted_donors = np.empty(len(weights), dtype=np.int64)
    for row in range(len(weights)):
        donor = int(order[row])
        for _ in range(len(weights)):
            try:
                smooth[row], raw[row], _ = recent_preserved_independent_past(
                    weights[row], weights[donor], time,
                    recent_fraction=args.recent_fraction,
                    transition_bins=args.transition_bins,
                )
                break
            except ValueError:
                donor = (donor + 1) % len(weights)
        else:
            raise RuntimeError("no valid independent donor found")
        accepted_donors[row] = donor
        self_control[row], _, _ = recent_preserved_independent_past(
            weights[row], weights[row], time,
            recent_fraction=args.recent_fraction,
            transition_bins=args.transition_bins,
        )

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = EuclidSFHAutoencoder.load_from_checkpoint(
        str(args.checkpoint), map_location="cpu", weights_only=False,
    ).to(device)
    names = ("real", "self_bridge", "raw_splice", "smooth_splice")
    curves = (weights, self_control, raw, smooth)
    embeddings = {}
    reconstructions = {}
    errors = {}
    for name, values in zip(names, curves):
        log_values = np.log10(values + epsilon).astype(np.float32)
        embedding, reconstruction = encode_sfhs(
            model, log_values, device, args.batch_size,
        )
        embeddings[name] = embedding
        reconstructions[name] = reconstruction
        errors[name] = reconstruction_metrics(reconstruction, values)

    nn = NearestNeighbors(n_neighbors=2, metric="cosine").fit(embeddings["real"])
    real_distance = nn.kneighbors(embeddings["real"], return_distance=True)[0][:, 1]
    latent_distance = {"real": real_distance}
    for name in names[1:]:
        latent_distance[name] = nn.kneighbors(
            embeddings[name], n_neighbors=1, return_distance=True,
        )[0][:, 0]

    auc = {
        name: paired_auc(embeddings["real"], embeddings[name], args.seed)
        for name in names[1:]
    }
    split = int(np.searchsorted(time, args.recent_fraction, side="right"))
    curvature = {
        name: np.abs(np.diff(values, n=2, axis=1)[:, max(split - 3, 0):split + 3]).max(axis=1)
        for name, values in zip(names, curves)
    }

    report = {
        "n_objects": len(weights), "device": str(device),
        "recent_fraction": args.recent_fraction,
        "transition_bins": args.transition_bins,
        "reconstruction_mae": {name: summarize(errors[name]["mae"]) for name in names},
        "reconstruction_w1": {name: summarize(errors[name]["w1"]) for name in names},
        "nearest_real_latent_cosine_distance": {
            name: summarize(latent_distance[name]) for name in names
        },
        "real_vs_altered_latent_auc": {
            name: {**summarize(values), "folds": values.tolist()}
            for name, values in auc.items()
        },
        "boundary_second_difference": {
            name: summarize(curvature[name]) for name in names
        },
        "max_integral_error": {
            name: float(np.max(np.abs(values.sum(axis=1) - 1.0)))
            for name, values in zip(names, curves)
        },
        "max_recent_error": float(np.max(np.abs(
            smooth[:, :split] - weights[:, :split]
        ))),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    colors = dict(real="black", self_bridge="#7570b3", raw_splice="#d95f02", smooth_splice="#1b9e77")
    labels = dict(real="real", self_bridge="self-donor bridge", raw_splice="raw donor splice", smooth_splice="smoothed donor past")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    positive_latent = np.concatenate([
        values[values > 0] for values in latent_distance.values()
    ])
    latent_floor = max(float(np.quantile(positive_latent, .001)), 1e-8)
    latent_bins = np.geomspace(
        latent_floor,
        max(float(positive_latent.max()), latent_floor * 10),
        60,
    )
    for name in names:
        axes[0, 0].hist(errors[name]["w1"], bins=60, density=True,
                        histtype="step", lw=1.8, color=colors[name], label=labels[name])
        axes[0, 1].hist(np.maximum(latent_distance[name], latent_floor),
                        bins=latent_bins, density=True,
                        histtype="step", lw=1.8, color=colors[name], label=labels[name])
        axes[1, 0].hist(curvature[name], bins=np.geomspace(1e-8, max(1e-2, curvature[name].max()), 60),
                        density=True, histtype="step", lw=1.8,
                        color=colors[name], label=labels[name])
    axes[0, 0].set(xlabel="AE reconstruction W1", ylabel="Density", title="Fixed-AE reconstruction")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set(xlabel="Cosine distance to nearest real AE latent", ylabel="Density", title="Distance from the real latent manifold")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set(xlabel="Maximum local |second difference|", ylabel="Density", title="Boundary roughness")
    auc_names = list(auc)
    auc_median = [np.median(auc[name]) for name in auc_names]
    auc_lo = [np.quantile(auc[name], .16) for name in auc_names]
    auc_hi = [np.quantile(auc[name], .84) for name in auc_names]
    x = np.arange(len(auc_names))
    axes[1, 1].errorbar(x, auc_median,
                        yerr=[np.subtract(auc_median, auc_lo), np.subtract(auc_hi, auc_median)],
                        fmt="o", capsize=4, color="#333333")
    axes[1, 1].axhline(.5, color="0.4", ls="--")
    axes[1, 1].set_xticks(x, [labels[name] for name in auc_names], rotation=15, ha="right")
    axes[1, 1].set_ylim(.45, 1.01)
    axes[1, 1].set(ylabel="Held-out ROC AUC", title="Can the AE latent identify the alteration?")
    for axis in axes.ravel():
        axis.grid(alpha=.18)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("OOD diagnostics for independent-past SFH ablation", fontsize=14)
    fig.savefig(args.output.with_suffix(".png"), dpi=180)
    fig.savefig(args.output.with_suffix(".pdf"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
