"""Remove the smooth edge-on trend from an aligned image embedding.

This is a diagnostic, not a replacement for nuisance-aware training.  A spline
ridge model predicts every image-embedding coordinate from the continuous
ZooBot edge-on probability.  The predicted trend is centred and subtracted,
then the image and joint embeddings are renormalized and their UMAPs refitted.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import matplotlib
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "astroclip_numba"))
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .compare_alignment_geometry import effective_rank, normalize
from .umap_zoobot_clip import fit_umap


MORPHOLOGY = (
    ("zoobot_edge_on_probability", "P(edge-on)", "magma"),
    ("zoobot_smooth_probability", "P(smooth)", "viridis"),
    ("zoobot_spiral_probability", "P(spiral arms)", "viridis"),
    ("zoobot_merger_probability", "P(merger/disturbed)", "viridis"),
)


def _crossvalidated_auc(values: np.ndarray, target: np.ndarray, seed: int) -> float:
    valid = np.isfinite(target) & np.isfinite(values).all(axis=1)
    binary = target[valid] >= 0.8
    train, test = train_test_split(
        np.arange(binary.size), test_size=0.3, random_state=seed, stratify=binary,
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1),
    )
    model.fit(values[valid][train], binary[train])
    return float(roc_auc_score(binary[test], model.predict_proba(values[valid][test])[:, 1]))


def _sampled_geometry(image: np.ndarray, sfh: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed)
    first = rng.integers(0, len(image), 250_000)
    second = rng.integers(0, len(image), 250_000)
    keep = first != second
    first, second = first[keep], second[keep]
    image_similarity = (image[first] * image[second]).sum(axis=1)
    sfh_similarity = (sfh[first] * sfh[second]).sum(axis=1)
    return float(spearmanr(image_similarity, sfh_similarity).statistic)


def residualize(image: np.ndarray, edge_on: np.ndarray, knots: int):
    valid = np.isfinite(edge_on)
    if not valid.all():
        raise ValueError("Edge-on probability must be finite for every archived object")
    nuisance = edge_on[:, None].astype(np.float64)
    model = make_pipeline(
        SplineTransformer(n_knots=knots, degree=3, include_bias=False),
        RidgeCV(alphas=np.logspace(-5, 1, 13)),
    )
    model.fit(nuisance, image)
    prediction = model.predict(nuisance)
    effect = prediction - prediction.mean(axis=0, keepdims=True)
    corrected = normalize(image - effect)
    ridge = model[-1]
    return corrected.astype(np.float32), prediction.astype(np.float32), effect, float(ridge.alpha_)


def _scatter(ax, xy, values, title, cmap):
    finite = np.isfinite(values)
    low, high = np.nanpercentile(values[finite], [1, 99])
    artist = ax.scatter(
        xy[finite, 0], xy[finite, 1], c=values[finite], s=2.0,
        cmap=cmap, vmin=low, vmax=high, linewidths=0, rasterized=True,
    )
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return artist


def make_report(original, corrected, metrics, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        for key, label, cmap in MORPHOLOGY:
            if key not in original:
                continue
            values = original[key]
            fig, axes = plt.subplots(2, 2, figsize=(12, 10), layout="constrained")
            for ax, xy_key, title in (
                (axes[0, 0], "xy_image", "Original aligned image"),
                (axes[0, 1], "xy_image", "Edge-on residualized image"),
                (axes[1, 0], "xy_joint", "Original joint average"),
                (axes[1, 1], "xy_joint", "Residualized joint average"),
            ):
                source = original if "Original" in title else corrected
                artist = _scatter(ax, source[xy_key], values, title, cmap)
                fig.colorbar(artist, ax=ax, label=label, shrink=0.82)
            fig.suptitle(f"Inclination residualization diagnostic: {label}", fontsize=14)
            pdf.savefig(fig, dpi=180)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
        names = list(metrics["original"])
        x = np.arange(len(names))
        width = 0.36
        ax.bar(x - width / 2, [metrics["original"][k] for k in names], width,
               label="original")
        ax.bar(x + width / 2, [metrics["residualized"][k] for k in names], width,
               label="residualized")
        ax.set_xticks(x, [name.replace("_", "\n") for name in names])
        ax.set_ylabel("Metric value")
        ax.set_title("Information removed and retained")
        ax.grid(axis="y", alpha=0.2)
        ax.legend()
        pdf.savefig(fig, dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-archive", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--knots", type=int, default=6)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with np.load(args.archive) as source:
        original = {key: np.asarray(source[key]) for key in source.files}
    required = {"image_embedding", "sfh_embedding", "zoobot_edge_on_probability"}
    missing = required.difference(original)
    if missing:
        raise KeyError(f"Archive lacks required arrays: {sorted(missing)}")

    image = normalize(original["image_embedding"])
    sfh = normalize(original["sfh_embedding"])
    edge_on = np.asarray(original["zoobot_edge_on_probability"], dtype=np.float32)
    corrected_image, prediction, effect, alpha = residualize(image, edge_on, args.knots)
    corrected_joint = normalize(corrected_image + sfh)

    print("Fitting residualized image UMAP", flush=True)
    xy_image = fit_umap(corrected_image, args.n_neighbors, args.min_dist, args.seed)
    print("Fitting residualized joint UMAP", flush=True)
    xy_joint = fit_umap(corrected_joint, args.n_neighbors, args.min_dist, args.seed)

    corrected = dict(original)
    corrected.update({
        "image_embedding": corrected_image,
        "joint_embedding": corrected_joint.astype(np.float32),
        "xy_image": xy_image,
        "xy_joint": xy_joint,
        "edge_on_residualization_prediction": prediction,
        "edge_on_residualization_effect_norm": np.linalg.norm(effect, axis=1).astype(np.float32),
    })
    args.output_archive.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_archive, **corrected)

    metrics = {
        "original": {
            "edge_on_AUC": _crossvalidated_auc(image, edge_on, args.seed),
            "image_effective_rank": effective_rank(image),
            "joint_effective_rank": effective_rank(normalize(image + sfh)),
            "image_SFH_geometry": _sampled_geometry(image, sfh, args.seed),
            "paired_cosine": float(np.mean(np.sum(image * sfh, axis=1))),
        },
        "residualized": {
            "edge_on_AUC": _crossvalidated_auc(corrected_image, edge_on, args.seed),
            "image_effective_rank": effective_rank(corrected_image),
            "joint_effective_rank": effective_rank(corrected_joint),
            "image_SFH_geometry": _sampled_geometry(corrected_image, sfh, args.seed),
            "paired_cosine": float(np.mean(np.sum(corrected_image * sfh, axis=1))),
        },
        "configuration": {
            "source": str(args.archive),
            "spline_knots": args.knots,
            "ridge_alpha": alpha,
            "n_objects": len(image),
            "mean_removed_effect_norm": float(np.mean(np.linalg.norm(effect, axis=1))),
            "p95_removed_effect_norm": float(np.percentile(np.linalg.norm(effect, axis=1), 95)),
        },
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n")
    make_report(original, corrected, metrics, args.output_pdf)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Saved archive: {args.output_archive}", flush=True)
    print(f"Saved report: {args.output_pdf}", flush=True)


if __name__ == "__main__":
    main()
