"""Compare progenitor-analogue tracks across aligned and unaligned UMAPs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


SPACE_KEYS = {
    "Joint average": "xy_joint",
    "Aligned SFH": "xy_sfh",
    "Unaligned SFH latent": "xy_sfh_preprojection",
    "Unaligned ZooBot morphology": "xy_image",
}


def _load_coordinates(bundle: Path, archive_path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    import h5py

    with h5py.File(bundle / "euclid_explorer.h5", "r") as source:
        ids = source["galaxy_id"][:].astype(np.int64)
    with np.load(archive_path) as archive:
        missing = [key for key in SPACE_KEYS.values() if key not in archive]
        if missing:
            raise ValueError(f"UMAP archive is missing coordinate arrays: {missing}")
        archive_ids = archive["galaxy_id"].astype(np.int64)
        lookup = {int(g): i for i, g in enumerate(archive_ids)}
        positions = np.array([lookup.get(int(g), -1) for g in ids], dtype=int)
        if np.any(positions < 0):
            raise ValueError("UMAP archive does not contain every explorer object")
        coordinates = {
            label: np.asarray(archive[key][positions], dtype=float)
            for label, key in SPACE_KEYS.items()
        }
    return ids, coordinates


def _robust_normalize(xy: np.ndarray) -> np.ndarray:
    centre = np.nanmedian(xy, axis=0)
    scale = np.nanpercentile(xy, 95, axis=0) - np.nanpercentile(xy, 5, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    return (xy - centre) / scale


def _track_metrics(points: np.ndarray) -> tuple[float, float, float]:
    points = np.asarray(points, dtype=float)
    steps = np.diff(points, axis=0)
    lengths = np.linalg.norm(steps, axis=1)
    path = float(np.sum(lengths))
    endpoint = float(np.linalg.norm(points[-1] - points[0]))
    tortuosity = path / max(endpoint, 1e-8)
    usable = lengths > 1e-10
    angles = []
    for i in range(len(steps) - 1):
        if not (usable[i] and usable[i + 1]):
            continue
        cosine = np.dot(steps[i], steps[i + 1]) / (lengths[i] * lengths[i + 1])
        angles.append(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    mean_angle = float(np.mean(angles)) if angles else np.nan
    return path, tortuosity, mean_angle


def build_tracks(
    ids: np.ndarray,
    coordinates: dict[str, np.ndarray],
    descendants: pd.DataFrame,
    candidates: pd.DataFrame,
) -> tuple[dict[int, dict[str, np.ndarray]], pd.DataFrame]:
    tracks: dict[int, dict[str, np.ndarray]] = {}
    metrics = []
    for row in descendants.itertuples(index=False):
        descendant_id = int(row.galaxy_id)
        descendant_index = int(row.bundle_index)
        selected = candidates[candidates.descendant_id == descendant_id]
        if selected.empty:
            continue
        stage_rows = []
        for stage, group in selected.groupby("stage"):
            stage_rows.append(
                (
                    float(group.formed_mass_fraction.iloc[0]),
                    float(group.state_lookback_gyr.iloc[0]),
                    group.bundle_index.to_numpy(dtype=int),
                )
            )
        stage_rows.sort(key=lambda item: item[0])
        fractions = np.array([item[0] for item in stage_rows] + [1.0])
        lookback = np.array([item[1] for item in stage_rows] + [0.0])
        track = {"formed_mass_fraction": fractions, "lookback_gyr": lookback}
        for label, xy in coordinates.items():
            points = np.vstack(
                [np.nanmedian(xy[item[2]], axis=0) for item in stage_rows]
                + [xy[descendant_index]]
            )
            track[label] = points
            normalized = _robust_normalize(xy)
            normalized_points = np.vstack(
                [np.nanmedian(normalized[item[2]], axis=0) for item in stage_rows]
                + [normalized[descendant_index]]
            )
            path, tortuosity, angle = _track_metrics(normalized_points)
            metrics.append(
                {
                    "galaxy_id": descendant_id,
                    "space": label,
                    "n_points": len(points),
                    "normalized_path_length": path,
                    "tortuosity": tortuosity,
                    "mean_turning_angle_deg": angle,
                    "phz_delta_ms": float(row.phz_delta_ms),
                    "log_stellar_mass": float(row.log_stellar_mass),
                }
            )
        tracks[descendant_id] = track
    return tracks, pd.DataFrame(metrics)


def _background(ax: plt.Axes, xy: np.ndarray) -> None:
    ax.scatter(
        xy[:, 0], xy[:, 1], s=1.4, color="0.60", alpha=0.16,
        linewidths=0, rasterized=True,
    )


def _plot_all_tracks(
    pdf: PdfPages,
    coordinates: dict[str, np.ndarray],
    descendants: pd.DataFrame,
    tracks: dict[int, dict[str, np.ndarray]],
) -> None:
    delta_lookup = descendants.set_index("galaxy_id").phz_delta_ms
    finite_delta = delta_lookup[np.isfinite(delta_lookup)].to_numpy(float)
    norm = Normalize(
        vmin=float(np.percentile(finite_delta, 2)),
        vmax=float(np.percentile(finite_delta, 98)),
    )
    cmap = plt.get_cmap("coolwarm")
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.8), layout="constrained")
    for ax, (label, xy) in zip(axes, coordinates.items()):
        _background(ax, xy)
        for galaxy_id, track in tracks.items():
            color = cmap(norm(float(delta_lookup.loc[galaxy_id])))
            points = track[label]
            ax.plot(points[:, 0], points[:, 1], color=color, alpha=0.12, lw=0.65)
        ax.set(xlabel="UMAP 1", ylabel="UMAP 2", title=label)
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(scalar, ax=axes, label="Descendant PHZ deltaMS [dex]", shrink=0.82)
    fig.suptitle(
        f"Same SFH-selected analogue tracks in four embedding spaces ({len(tracks):,} descendants)\n"
        "Each checkpoint is the median position of the same five selected analogues",
        fontsize=14,
    )
    pdf.savefig(fig, dpi=190)
    plt.close(fig)


def _plot_metrics(pdf: PdfPages, metrics: pd.DataFrame) -> dict[str, float]:
    spaces = list(SPACE_KEYS)
    colors = ["#7b3294", "#355f8d", "#2a9d8f", "#d97706"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.3), layout="constrained")
    summary: dict[str, float] = {}
    for ax, column, ylabel in [
        (axes[0], "tortuosity", "Tortuosity (path / endpoint)"),
        (axes[1], "mean_turning_angle_deg", "Mean turning angle [deg]"),
    ]:
        values = [metrics.loc[metrics.space == space, column].replace([np.inf, -np.inf], np.nan).dropna() for space in spaces]
        parts = ax.violinplot(values, showmedians=True, showextrema=False)
        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color); body.set_alpha(0.55)
        parts["cmedians"].set_color("black")
        ax.set_xticks(
            range(1, len(spaces) + 1),
            ["Joint\naverage", "Aligned\nSFH", "Unaligned\nSFH", "ZooBot\nmorphology"],
        )
        ax.set_ylabel(ylabel)
        if column == "tortuosity":
            ax.set_yscale("log")
            ax.set_ylabel("Tortuosity (path / endpoint; log scale)")
        ax.grid(axis="y", alpha=0.18)
        for space, series in zip(spaces, values):
            summary[f"median_{column}_{SPACE_KEYS[space]}"] = float(np.median(series))

    pivot = metrics.pivot(index="galaxy_id", columns="space", values="tortuosity").dropna()
    joint = pivot["Joint average"]
    for space, color in zip(spaces[1:], colors[1:]):
        difference = pivot[space] - joint
        axes[2].hist(difference.clip(-5, 5), bins=45, alpha=0.55, color=color, label=space)
        summary[f"fraction_joint_lower_tortuosity_than_{SPACE_KEYS[space]}"] = float(np.mean(joint < pivot[space]))
    aligned = pivot["Aligned SFH"]
    for space in ("Unaligned SFH latent", "Unaligned ZooBot morphology"):
        summary[f"fraction_aligned_lower_tortuosity_than_{SPACE_KEYS[space]}"] = float(
            np.mean(aligned < pivot[space])
        )
    axes[2].axvline(0, color="black", ls="--", lw=1)
    axes[2].set(
        xlabel="Other-space tortuosity - joint-average tortuosity",
        ylabel="Descendants",
        title="Positive values favor smoother joint tracks",
    )
    axes[2].legend(fontsize=8)
    fig.suptitle(
        "Track smoothness after robustly normalizing each UMAP axis\n"
        "UMAP geometry is descriptive; these statistics are not physical distances",
        fontsize=14,
    )
    pdf.savefig(fig, dpi=190)
    plt.close(fig)
    return summary


def _plot_detailed(
    pdf: PdfPages,
    coordinates: dict[str, np.ndarray],
    descendants: pd.DataFrame,
    tracks: dict[int, dict[str, np.ndarray]],
    metrics: pd.DataFrame,
    detailed_ids: list[int],
) -> None:
    descendant_lookup = descendants.set_index("galaxy_id")
    for galaxy_id in detailed_ids:
        if galaxy_id not in tracks or galaxy_id not in descendant_lookup.index:
            continue
        track = tracks[galaxy_id]
        row = descendant_lookup.loc[galaxy_id]
        norm = Normalize(vmin=float(np.min(track["lookback_gyr"])), vmax=float(np.max(track["lookback_gyr"])))
        cmap = plt.get_cmap("viridis")
        fig, axes_grid = plt.subplots(2, 2, figsize=(12.5, 10.5), layout="constrained")
        axes = axes_grid.ravel()
        for ax, (label, xy) in zip(axes, coordinates.items()):
            _background(ax, xy)
            points = track[label]
            segments = np.stack([points[:-1], points[1:]], axis=1)
            collection = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2.0)
            collection.set_array(track["lookback_gyr"][:-1])
            ax.add_collection(collection)
            ax.scatter(
                points[:-1, 0], points[:-1, 1], c=track["lookback_gyr"][:-1],
                cmap=cmap, norm=norm, s=30, edgecolor="white", linewidth=0.35, zorder=3,
            )
            ax.scatter(points[-1, 0], points[-1, 1], marker="*", s=120,
                       color="#d73027", edgecolor="white", linewidth=0.6, zorder=4)
            for number, point in enumerate(points[:-1], start=1):
                ax.text(point[0], point[1], str(number), fontsize=6, ha="left", va="bottom")
            metric = metrics[(metrics.galaxy_id == galaxy_id) & (metrics.space == label)].iloc[0]
            ax.set(
                xlabel="UMAP 1", ylabel="UMAP 2",
                title=f"{label}\nT={metric.tortuosity:.2f}; turn={metric.mean_turning_angle_deg:.1f} deg",
            )
        scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        fig.colorbar(scalar, ax=axes, label="Descendant checkpoint lookback [Gyr]", shrink=0.82)
        fig.suptitle(
            f"Descendant {galaxy_id}: log M*={row.log_stellar_mass:.2f}, "
            f"z={row.redshift:.3f}, PHZ deltaMS={row.phz_delta_ms:+.2f}\n"
            "Numbers follow earliest progenitor to descendant (red star)",
            fontsize=14,
        )
        pdf.savefig(fig, dpi=190)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--detailed-selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    args = parser.parse_args()

    descendants = pd.read_csv(args.population / "descendants.csv")
    candidates = pd.read_csv(args.population / "analogue_candidates.csv")
    ids, coordinates = _load_coordinates(args.bundle, args.archive)
    tracks, metrics = build_tracks(ids, coordinates, descendants, candidates)
    args.output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output / "track_smoothness.csv", index=False)

    detailed_ids: list[int] = []
    if args.detailed_selection is not None:
        detailed = pd.read_csv(args.detailed_selection)
        detailed_ids = detailed.galaxy_id.astype(np.int64).tolist()

    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.pdf) as pdf:
        _plot_all_tracks(pdf, coordinates, descendants, tracks)
        summary = _plot_metrics(pdf, metrics)
        _plot_detailed(pdf, coordinates, descendants, tracks, metrics, detailed_ids)

    manifest = {
        "n_descendants": len(descendants),
        "n_tracks": len(tracks),
        "spaces": SPACE_KEYS,
        "detailed_descendant_ids": detailed_ids,
        "metric_normalization": "Each UMAP axis centered by its median and divided by its 5th-95th percentile span",
        "summary": summary,
        "pdf": str(args.pdf),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
