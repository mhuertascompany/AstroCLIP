"""Find SFH-selected progenitor analogues and visualize their morphologies.

The analogue search deliberately does not use redshift, morphology, image
embedding, or UMAP position.  For each earlier state of a massive descendant,
it matches the predicted stellar mass and the *renormalized cumulative SFH*
before that state.  Images and SFH-UMAP positions are shown only after matching.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image
from astropy.table import Table

from .cosmic_sfh import COSMOLOGY
from .main_sequence_sfh import main_sequence_along_sfh
from .recent_ms import recent_offset

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D


DEFAULT_BUNDLE = Path(
    "/Users/marchuertascompany/Documents/data/EUCLID/DR1/explorer_bundle"
)
DEFAULT_ARCHIVE = DEFAULT_BUNDLE / (
    "00_training_bright_frozen_mlp_euclid_clip_umap_diagnostics.npz"
)
DEFAULT_OUTPUT = Path("euclid/diagnostics/progenitor_analogues")
DEFAULT_PDF = Path("output/pdf/euclid_progenitor_analogue_pilot.pdf")
DEFAULT_CATALOG = Path(
    "/Users/marchuertascompany/Documents/data/EUCLID/DR1/"
    "phz_sfr_mass_matched_bright.fits"
)
DEFAULT_CLUSTER_MEMBERSHIP = Path(
    "euclid/diagnostics/cluster_recent_ms_100myr/cluster_membership.csv"
)
CLUSTER_COLORS = {1: "#236C9C", 2: "#B35D21", 3: "#228461"}
DEFAULT_FORMED_FRACTIONS = [
    0.99, 0.98, 0.97, 0.96, 0.95, 0.90, 0.85, 0.80,
    0.60, 0.40, 0.20, 0.10, 0.03, 0.01,
]


def bin_edges_from_centres(centres: np.ndarray) -> np.ndarray:
    """Return edges on [0, 1] for the common fractional-lookback grid."""
    centres = np.asarray(centres, dtype=float)
    if centres.ndim != 1 or len(centres) < 2 or np.any(np.diff(centres) <= 0):
        raise ValueError("SFH time centres must be a strictly increasing 1-D array")
    edges = np.empty(len(centres) + 1, dtype=float)
    edges[1:-1] = 0.5 * (centres[:-1] + centres[1:])
    edges[0], edges[-1] = 0.0, 1.0
    return edges


def normalize_sfh(weights: np.ndarray) -> np.ndarray:
    weights = np.maximum(np.asarray(weights, dtype=float), 0.0)
    total = weights.sum(axis=-1, keepdims=True)
    if np.any(~np.isfinite(total)) or np.any(total <= 0):
        raise ValueError("Every SFH must have finite, positive total weight")
    return weights / total


def older_fraction(
    weights: np.ndarray,
    edges: np.ndarray,
    time_norm_gyr: float,
    lookback_gyr: np.ndarray | float,
) -> np.ndarray:
    """Fraction of final formed mass already present at an earlier epoch.

    At lookback ``L``, this is the integral of the SFH from ``L`` to the oldest
    bin.  Piecewise-constant SFR inside each bin makes the cumulative curve
    linear between bin edges.
    """
    weights = normalize_sfh(weights)
    cumulative_young = np.concatenate(([0.0], np.cumsum(weights)))
    u = np.asarray(lookback_gyr, dtype=float) / float(time_norm_gyr)
    younger = np.interp(np.clip(u, 0.0, 1.0), edges, cumulative_young)
    result = 1.0 - younger
    result = np.where(u <= 0.0, 1.0, result)
    result = np.where(u >= 1.0, 0.0, result)
    return result


def lookback_at_formed_fraction(
    weights: np.ndarray, edges: np.ndarray, time_norm_gyr: float, fraction: float
) -> float:
    """Invert ``older_fraction`` to the lookback time for a target fraction."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("Formed fraction must lie strictly between zero and one")
    weights = normalize_sfh(weights)
    cumulative_young = np.concatenate(([0.0], np.cumsum(weights)))
    u = np.interp(1.0 - fraction, cumulative_young, edges)
    return float(u * time_norm_gyr)


def descendant_state_curve(
    weights: np.ndarray,
    edges: np.ndarray,
    time_norm_gyr: float,
    state_lookback_gyr: float,
    state_fraction: float,
    tau_gyr: np.ndarray,
) -> np.ndarray:
    """Cumulative assembly before a state, renormalized to one at that state."""
    return older_fraction(
        weights, edges, time_norm_gyr, state_lookback_gyr + tau_gyr
    ) / state_fraction


def candidate_curve(
    weights: np.ndarray,
    edges: np.ndarray,
    time_norm_gyr: float,
    tau_gyr: np.ndarray,
) -> np.ndarray:
    """Candidate cumulative assembly, normalized at its observation epoch."""
    return older_fraction(weights, edges, time_norm_gyr, tau_gyr)


def candidate_curves(
    weights: np.ndarray,
    edges: np.ndarray,
    time_norm_gyr: np.ndarray,
    tau_gyr: np.ndarray,
) -> np.ndarray:
    """Vectorized candidate cumulative curves for a shared physical-time grid."""
    weights = normalize_sfh(weights)
    norms = np.asarray(time_norm_gyr, dtype=float)
    tau = np.asarray(tau_gyr, dtype=float)
    if weights.ndim != 2 or norms.shape != (len(weights),):
        raise ValueError("Expected SFHs shaped (objects, bins) and one time norm per object")
    cumulative = np.concatenate(
        [np.zeros((len(weights), 1)), np.cumsum(weights, axis=1)], axis=1
    )
    u = tau[None, :] / norms[:, None]
    indices = np.searchsorted(edges, u, side="right") - 1
    indices = np.clip(indices, 0, weights.shape[1] - 1)
    left = edges[indices]
    width = edges[indices + 1] - left
    fraction = np.clip((u - left) / width, 0.0, 1.0)
    younger = np.take_along_axis(cumulative, indices, axis=1)
    younger += fraction * np.take_along_axis(weights, indices, axis=1)
    result = 1.0 - younger
    result[u <= 0.0] = 1.0
    result[u >= 1.0] = 0.0
    return result


def curve_distance(target: np.ndarray, candidate: np.ndarray) -> float:
    """Mean absolute cumulative-SFH difference over the comparison interval."""
    return float(np.mean(np.abs(np.asarray(target) - np.asarray(candidate))))


def _load_inputs(
    bundle: Path,
    archive_path: Path,
    catalog_path: Path | None = None,
    ms_sfr_offset: float = -0.93,
    cluster_path: Path | None = DEFAULT_CLUSTER_MEMBERSHIP,
) -> dict[str, np.ndarray | float]:
    with h5py.File(bundle / "euclid_explorer.h5", "r") as source:
        ids = source["galaxy_id"][:].astype(np.int64)
        epsilon = float(source.attrs.get("sfh_log_epsilon", 1e-10))
        sfh = np.maximum(10.0 ** source["sfh"][:].astype(float) - epsilon, 0.0)
        sfh_p16 = np.maximum(
            10.0 ** source["sfh_p16"][:].astype(float) - epsilon, 0.0
        )
        sfh_p84 = np.maximum(
            10.0 ** source["sfh_p84"][:].astype(float) - epsilon, 0.0
        )
        time_grid = source["sfh_time_grid"][:].astype(float)
        time_norm_gyr = source["sfh_time_norm"][:].astype(float) / 1000.0
        redshift = source["redshift"][:].astype(float)

    with np.load(archive_path) as archive:
        archive_ids = archive["galaxy_id"].astype(np.int64)
        lookup = {int(g): i for i, g in enumerate(archive_ids)}
        positions = np.array([lookup.get(int(g), -1) for g in ids])
        if np.any(positions < 0):
            raise ValueError("The UMAP archive does not contain every explorer object")
        mass = archive["log_stellar_mass"][positions].astype(float)
        xy_sfh = archive["xy_sfh"][positions].astype(float)
        morphology = {}
        for key in (
            "zoobot_smooth_conditional_fraction",
            "zoobot_spiral_probability",
            "zoobot_merger_probability",
        ):
            morphology[key] = (
                archive[key][positions].astype(float)
                if key in archive
                else np.full(len(ids), np.nan)
            )

    clusters = np.full(len(ids), -1, dtype=int)
    if cluster_path is not None and cluster_path.is_file():
        membership = pd.read_csv(cluster_path, dtype={"galaxy_id": str})
        cluster_lookup = dict(
            zip(membership.galaxy_id, membership.cluster.astype(int), strict=True)
        )
        clusters = np.array([cluster_lookup.get(str(int(g)), -1) for g in ids], dtype=int)

    catalog_log_sfr = np.full(len(ids), np.nan)
    catalog_log_mass = mass.copy()
    physical_parameters_matched = np.zeros(len(ids), dtype=bool)
    if catalog_path is not None:
        table = Table.read(catalog_path)
        names = {name.lower(): name for name in table.colnames}
        catalog_ids = np.asarray(table[names["object_id"]], dtype=np.int64)
        if len(np.unique(catalog_ids)) != len(catalog_ids):
            raise ValueError("The PHZ catalog contains duplicate object IDs")
        catalog_lookup = {int(g): i for i, g in enumerate(catalog_ids)}
        catalog_rows = np.array([catalog_lookup.get(int(g), -1) for g in ids])
        matched = catalog_rows >= 0
        safe = np.maximum(catalog_rows, 0)
        catalog_log_sfr[matched] = np.asarray(
            np.ma.asarray(table[names["phz_pp_median_sfr"]], dtype=float).filled(np.nan)
        )[safe[matched]]
        catalog_log_mass[matched] = np.asarray(
            np.ma.asarray(
                table[names["phz_pp_median_stellarmass"]], dtype=float
            ).filled(np.nan)
        )[safe[matched]]
        physical_parameters_matched[matched] = np.asarray(
            table[names["physical_parameters_matched"]], dtype=bool
        )[safe[matched]]
    catalog_log_ssfr = catalog_log_sfr - catalog_log_mass
    cosmic_age = np.full(len(ids), np.nan)
    valid_ms = (
        physical_parameters_matched
        & np.isfinite(redshift + catalog_log_mass + catalog_log_sfr)
        & (redshift >= 0)
    )
    cosmic_age[valid_ms] = COSMOLOGY.age(redshift[valid_ms]).to_value("Gyr")
    catalog_ms_log_sfr = (
        (0.84 - 0.026 * cosmic_age) * catalog_log_mass
        - (6.51 - 0.11 * cosmic_age)
        + ms_sfr_offset
    )
    catalog_delta_ms = catalog_log_sfr - catalog_ms_log_sfr

    return {
        "ids": ids,
        "sfh": normalize_sfh(sfh),
        "sfh_p16": sfh_p16,
        "sfh_p84": sfh_p84,
        "time_grid": time_grid,
        "time_norm_gyr": time_norm_gyr,
        "redshift": redshift,
        "mass": mass,
        "xy_sfh": xy_sfh,
        "cluster": clusters,
        **morphology,
        "epsilon": epsilon,
        "catalog_log_sfr": catalog_log_sfr,
        "catalog_log_mass": catalog_log_mass,
        "catalog_log_ssfr": catalog_log_ssfr,
        "catalog_ms_log_sfr": catalog_ms_log_sfr,
        "catalog_delta_ms": catalog_delta_ms,
        "physical_parameters_matched": physical_parameters_matched,
        "ms_sfr_offset": ms_sfr_offset,
    }


def _choose_descendant(
    data: dict[str, np.ndarray | float],
    stamps: Path,
    requested_id: int | None,
    minimum_mass: float,
    target_mass: float,
    quenched_log_ssfr_max: float | None = None,
    main_sequence_delta_max: float | None = None,
) -> int:
    ids = np.asarray(data["ids"])
    mass = np.asarray(data["mass"])
    has_stamp = np.array([(stamps / f"VIS_{g}.jpg").is_file() for g in ids])
    eligible = np.isfinite(mass) & (mass >= minimum_mass) & has_stamp
    if quenched_log_ssfr_max is not None:
        log_ssfr = np.asarray(data["catalog_log_ssfr"])
        matched = np.asarray(data["physical_parameters_matched"])
        eligible &= matched & np.isfinite(log_ssfr) & (log_ssfr < quenched_log_ssfr_max)
    if main_sequence_delta_max is not None:
        delta_ms = np.asarray(data["catalog_delta_ms"])
        matched = np.asarray(data["physical_parameters_matched"])
        eligible &= (
            matched
            & np.isfinite(delta_ms)
            & (np.abs(delta_ms) <= main_sequence_delta_max)
        )
    if requested_id is not None:
        matches = np.flatnonzero(ids == requested_id)
        if not len(matches):
            raise ValueError(f"Descendant ID {requested_id} is absent from the bundle")
        index = int(matches[0])
        if not eligible[index]:
            raise ValueError("Requested descendant fails the mass/stamp requirements")
        return index
    available = np.flatnonzero(eligible)
    if not len(available):
        raise ValueError("No descendant satisfies the requested mass and stamp cuts")
    return int(available[np.argmin(np.abs(mass[available] - target_mass))])


def find_analogues(
    data: dict[str, np.ndarray | float],
    descendant_index: int,
    fractions: list[float],
    mass_tolerance: float = 0.15,
    history_gyr: float = 2.0,
    minimum_comparison_gyr: float = 0.5,
    n_analogues: int = 5,
    stamps: Path | None = None,
    stamp_mask: np.ndarray | None = None,
    n_curve_points: int = 256,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, np.ndarray | float]]]:
    """Rank analogue candidates at each descendant mass-fraction checkpoint."""
    ids = np.asarray(data["ids"])
    mass = np.asarray(data["mass"])
    redshift = np.asarray(data["redshift"])
    catalog_delta_ms = np.asarray(data["catalog_delta_ms"])
    catalog_log_ssfr = np.asarray(data["catalog_log_ssfr"])
    sfh = np.asarray(data["sfh"])
    norms = np.asarray(data["time_norm_gyr"])
    edges = bin_edges_from_centres(np.asarray(data["time_grid"]))
    descendant_sfh = sfh[descendant_index]
    descendant_norm = float(norms[descendant_index])
    descendant_mass = float(mass[descendant_index])

    rows: list[dict[str, float | int | bool]] = []
    census: list[dict[str, float | int]] = []
    stages: list[dict[str, np.ndarray | float]] = []
    for stage, fraction in enumerate(fractions, start=1):
        state_lookback = lookback_at_formed_fraction(
            descendant_sfh, edges, descendant_norm, fraction
        )
        predicted_mass = descendant_mass + np.log10(fraction)
        available_history = descendant_norm - state_lookback
        comparison_gyr = min(history_gyr, available_history)
        if comparison_gyr <= 0:
            continue
        tau = np.linspace(0.0, comparison_gyr, n_curve_points)
        target_curve = descendant_state_curve(
            descendant_sfh,
            edges,
            descendant_norm,
            state_lookback,
            fraction,
            tau,
        )

        mass_ok = np.isfinite(mass) & (np.abs(mass - predicted_mass) <= mass_tolerance)
        history_ok = np.isfinite(norms) & (norms >= comparison_gyr)
        if stamp_mask is not None:
            stamp_ok = np.asarray(stamp_mask, dtype=bool)
            if stamp_ok.shape != (len(ids),):
                raise ValueError("stamp_mask must contain one value per object")
        else:
            stamp_ok = (
                np.ones(len(ids), dtype=bool)
                if stamps is None
                else np.array([(stamps / f"VIS_{g}.jpg").is_file() for g in ids])
            )
        pool = np.flatnonzero(mass_ok & history_ok & stamp_ok)
        pool = pool[pool != descendant_index]
        curves = candidate_curves(sfh[pool], edges, norms[pool], tau)
        scores = np.mean(np.abs(curves - target_curve[None, :]), axis=1)
        order = np.argsort(scores)[:n_analogues]
        chosen = pool[order]
        chosen_scores = scores[order]
        for rank, (index, score) in enumerate(zip(chosen, chosen_scores), start=1):
            rows.append(
                {
                    "stage": stage,
                    "formed_mass_fraction": fraction,
                    "state_lookback_gyr": state_lookback,
                    "predicted_log_stellar_mass": predicted_mass,
                    "comparison_history_gyr": comparison_gyr,
                    "shape_comparison_limited": comparison_gyr < minimum_comparison_gyr,
                    "rank": rank,
                    "galaxy_id": int(ids[index]),
                    "bundle_index": int(index),
                    "log_stellar_mass": float(mass[index]),
                    "mass_offset_dex": float(mass[index] - predicted_mass),
                    "redshift": float(redshift[index]),
                    "phz_log_ssfr": float(catalog_log_ssfr[index]),
                    "phz_delta_ms": float(catalog_delta_ms[index]),
                    "cumulative_sfh_distance": float(score),
                    "has_stamp": bool(stamp_ok[index]),
                }
            )
        census.append(
            {
                "stage": stage,
                "formed_mass_fraction": fraction,
                "state_lookback_gyr": state_lookback,
                "predicted_log_stellar_mass": predicted_mass,
                "comparison_history_gyr": comparison_gyr,
                "shape_comparison_limited": comparison_gyr < minimum_comparison_gyr,
                "n_mass_compatible": int(mass_ok.sum()),
                "n_with_history": int((mass_ok & history_ok).sum()),
                "n_with_stamp": int((mass_ok & history_ok & stamp_ok).sum()),
                "n_selected": int(len(chosen)),
                "candidate_redshift_p16": float(np.percentile(redshift[pool], 16))
                if len(pool)
                else np.nan,
                "candidate_redshift_median": float(np.median(redshift[pool]))
                if len(pool)
                else np.nan,
                "candidate_redshift_p84": float(np.percentile(redshift[pool], 84))
                if len(pool)
                else np.nan,
                "selected_delta_ms_median": float(
                    np.nanmedian(catalog_delta_ms[chosen])
                ) if len(chosen) else np.nan,
            }
        )
        stages.append(
            {
                "fraction": fraction,
                "state_lookback_gyr": state_lookback,
                "predicted_mass": predicted_mass,
                "comparison_gyr": comparison_gyr,
                "shape_comparison_limited": comparison_gyr < minimum_comparison_gyr,
                "tau": tau,
                "target_curve": target_curve,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(census), stages


def _plot_sfh(
    ax: plt.Axes,
    data: dict[str, np.ndarray | float],
    index: int,
    color: str,
    uncertainty: bool = True,
) -> None:
    time = np.asarray(data["time_grid"])
    sfh = np.asarray(data["sfh"])[index]
    if uncertainty:
        lo = np.asarray(data["sfh_p16"])[index]
        hi = np.asarray(data["sfh_p84"])[index]
        ax.fill_between(time, lo, hi, color=color, alpha=0.16, linewidth=0)
    ax.plot(time, sfh, color=color, lw=1.45)
    ax.set(xlim=(0, 1), xlabel="Fractional lookback time", ylabel="Normalized SFH weight")
    ax.tick_params(labelsize=7)


def _overlay_candidate_main_sequence(
    ax: plt.Axes,
    data: dict[str, np.ndarray | float],
    index: int,
    show_label: bool = False,
) -> None:
    """Overlay the shifted MS evaluated at this candidate's own z and mass."""
    sfh = np.asarray(data["sfh"])[index]
    epsilon = float(data["epsilon"])
    track = main_sequence_along_sfh(
        np.asarray(data["time_grid"]),
        np.log10(sfh + epsilon),
        float(np.asarray(data["redshift"])[index]),
        float(np.asarray(data["catalog_log_mass"])[index]),
        time_norm_myr=float(np.asarray(data["time_norm_gyr"])[index]) * 1000.0,
        return_fraction=0.0,
        epsilon=epsilon,
        ms_sfr_offset=float(data["ms_sfr_offset"]),
    )
    weights = np.asarray(track["weights"], dtype=float)
    supported = np.asarray(track["supported"], dtype=bool)
    time = np.asarray(data["time_grid"])
    label = "MS at candidate z,M*" if show_label else None
    ax.plot(time, np.where(supported, weights, np.nan), color="#2ca25f", lw=1.5,
            ls="--", label=label)
    ax.plot(time, np.where(~supported, weights, np.nan), color="#2ca25f", lw=1.3,
            ls=":")
    if show_label:
        ax.legend(fontsize=7, loc="best")


def _sfh_delta_ms_100myr(
    data: dict[str, np.ndarray | float], index: int
) -> float:
    """Return the 100 Myr MS offset implied by the reconstructed median SFH."""
    sfh = np.asarray(data["sfh"])[index]
    epsilon = float(data["epsilon"])
    track = main_sequence_along_sfh(
        np.asarray(data["time_grid"]),
        np.log10(sfh + epsilon),
        float(np.asarray(data["redshift"])[index]),
        float(np.asarray(data["catalog_log_mass"])[index]),
        time_norm_myr=float(np.asarray(data["time_norm_gyr"])[index]) * 1000.0,
        return_fraction=0.0,
        epsilon=epsilon,
        ms_sfr_offset=float(data["ms_sfr_offset"]),
    )
    return float(recent_offset(track, 1.0e8, floor=-np.inf))


def _format_delta_ms(value: float) -> str:
    """Format a finite offset or identify an exactly zero recent SFH rate."""
    if np.isneginf(value):
        return "-inf (zero)"
    return f"{value:+.2f}"


def _format_probability(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.2f}"


def _plot_cluster_background(
    ax: plt.Axes, data: dict[str, np.ndarray | float], alpha: float = 0.13
) -> None:
    """Draw the fixed three SFH clusters as a quiet UMAP context layer."""
    xy = np.asarray(data["xy_sfh"])
    labels = np.asarray(data.get("cluster", np.full(len(xy), -1)), dtype=int)
    for cluster, color in CLUSTER_COLORS.items():
        mask = labels == cluster
        if not mask.any():
            continue
        ax.scatter(
            xy[mask, 0], xy[mask, 1], s=2, color=color, alpha=alpha,
            linewidths=0, rasterized=True,
        )
        centre = np.median(xy[mask], axis=0)
        ax.text(
            centre[0], centre[1], f"C{cluster}", color=color, fontsize=10,
            fontweight="bold", ha="center", va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.5),
        )
    unknown = labels < 0
    if unknown.any():
        ax.scatter(
            xy[unknown, 0], xy[unknown, 1], s=2, color="0.82", alpha=0.25,
            linewidths=0, rasterized=True,
        )


def make_report(
    data: dict[str, np.ndarray | float],
    descendant_index: int,
    candidates: pd.DataFrame,
    census: pd.DataFrame,
    stages: list[dict[str, np.ndarray | float]],
    stamps: Path,
    pdf_path: Path,
    output: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    ids = np.asarray(data["ids"])
    mass = np.asarray(data["mass"])
    redshift = np.asarray(data["redshift"])
    xy = np.asarray(data["xy_sfh"])
    lookbacks = np.array([float(stage["state_lookback_gyr"]) for stage in stages])
    time_min = float(np.min(lookbacks))
    time_max = float(np.max(lookbacks))
    if time_max <= time_min:
        time_max = time_min + 1.0
    time_color_norm = matplotlib.colors.Normalize(vmin=time_min, vmax=time_max)
    time_cmap = plt.get_cmap("viridis")
    colors = time_cmap(time_color_norm(lookbacks))
    descendant_id = int(ids[descendant_index])
    descendant_ssfr = float(np.asarray(data["catalog_log_ssfr"])[descendant_index])
    descendant_delta_ms = float(np.asarray(data["catalog_delta_ms"])[descendant_index])
    descendant_sfh_delta_ms = _sfh_delta_ms_100myr(data, descendant_index)

    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(14, 8.5), layout="constrained")
        grid = fig.add_gridspec(2, 3, width_ratios=(1.05, 1.45, 1.45))
        image_ax = fig.add_subplot(grid[0, 0])
        with Image.open(stamps / f"VIS_{descendant_id}.jpg") as image:
            image_ax.imshow(np.asarray(image.convert("L")), cmap="gray", origin="upper")
        image_ax.set_title(
            f"Descendant {descendant_id}\nlog M*={mass[descendant_index]:.2f}, "
            f"z={redshift[descendant_index]:.3f}, log sSFR={descendant_ssfr:.2f}\n"
            f"PHZ deltaMS={descendant_delta_ms:+.2f}\n"
            f"SFH-100Myr deltaMS={_format_delta_ms(descendant_sfh_delta_ms)} dex"
        )
        image_ax.axis("off")

        sfh_ax = fig.add_subplot(grid[1, 0])
        _plot_sfh(sfh_ax, data, descendant_index, "#245c8a")
        for stage, color in zip(stages, colors):
            x = float(stage["state_lookback_gyr"]) / float(
                np.asarray(data["time_norm_gyr"])[descendant_index]
            )
            sfh_ax.axvline(x, color=color, lw=1.4, alpha=0.95)
        sfh_ax.set_title("Descendant SFH and earlier-state checkpoints", fontsize=10)

        umap_ax = fig.add_subplot(grid[:, 1])
        _plot_cluster_background(umap_ax, data)
        centroid_path = []
        for stage_number, (stage, color) in enumerate(zip(stages, colors), start=1):
            selected = candidates[candidates.stage == stage_number]
            if selected.empty:
                continue
            indices = selected.bundle_index.to_numpy(dtype=int)
            umap_ax.scatter(
                xy[indices, 0], xy[indices, 1], s=40, c=[color], edgecolor="white",
                linewidth=0.6, zorder=3,
            )
            centroid_path.append(np.median(xy[indices], axis=0))
        umap_ax.scatter(
            *xy[descendant_index], marker="*", s=180, c="#d73027", edgecolor="white",
            linewidth=0.8, label="descendant", zorder=5,
        )
        if centroid_path:
            path = np.vstack(centroid_path[::-1] + [xy[descendant_index]])
            umap_ax.plot(path[:, 0], path[:, 1], "k--", lw=1.1, alpha=0.75)
        umap_ax.legend(fontsize=8, markerscale=0.9, loc="best")
        fig.colorbar(
            plt.cm.ScalarMappable(norm=time_color_norm, cmap=time_cmap),
            ax=umap_ax, label="Descendant checkpoint lookback [Gyr]",
            fraction=0.045, pad=0.02,
        )
        umap_ax.set(
            xlabel="SFH UMAP 1", ylabel="SFH UMAP 2",
            title="Analogue positions and fixed SFH-cluster regions",
        )

        right = grid[:, 2].subgridspec(4, 1, height_ratios=(1.45, 0.78, 0.78, 0.62))
        table_ax = fig.add_subplot(right[0])
        table_ax.axis("off")
        display = census.copy()
        display["f"] = display.formed_mass_fraction.map(lambda x: f"{x:.2f}")
        display["L"] = display.state_lookback_gyr.map(lambda x: f"{x:.2f}")
        display["logM"] = display.predicted_log_stellar_mass.map(lambda x: f"{x:.2f}")
        display["T"] = display.comparison_history_gyr.map(lambda x: f"{x:.2f}")
        display["N"] = display.n_with_stamp.map(lambda x: f"{x:,}")
        display["zmed"] = display.candidate_redshift_median.map(lambda x: f"{x:.2f}")
        columns = ["f", "L", "logM", "T", "N", "zmed"]
        table = table_ax.table(
            cellText=display[columns].values,
            colLabels=columns,
            cellLoc="center",
            colLoc="center",
            loc="upper center",
            bbox=(0.0, 0.0, 1.0, 1.0),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1, 1.35)
        ms_ax = fig.add_subplot(right[1])
        for stage_number, (stage, color) in enumerate(zip(stages, colors), start=1):
            selected = candidates[candidates.stage == stage_number]
            indices = selected.bundle_index.to_numpy(dtype=int)
            phz_delta = selected.phz_delta_ms.to_numpy(dtype=float)
            sfh_delta = np.array(
                [_sfh_delta_ms_100myr(data, index) for index in indices], dtype=float
            )
            x = np.full(len(phz_delta), float(stage["state_lookback_gyr"]))
            for x_value, phz_value, sfh_value in zip(x, phz_delta, sfh_delta):
                if np.isfinite(phz_value) and np.isfinite(sfh_value):
                    ms_ax.plot(
                        [x_value, x_value], [phz_value, sfh_value],
                        color=color, alpha=0.25, lw=0.7, zorder=1,
                    )
            ms_ax.scatter(
                x, phz_delta, facecolors="none", edgecolors=[color], s=31,
                alpha=0.9, linewidth=0.9, zorder=2,
            )
            ms_ax.scatter(
                x, sfh_delta, color=[color], marker="x", s=29,
                alpha=0.9, linewidth=1.0, zorder=3,
            )
        ms_ax.axhline(0, color="#2ca25f", lw=1.5, ls="--", label="shifted MS")
        ms_ax.axhline(-0.3, color="0.5", lw=0.8, ls=":")
        ms_ax.set(
            xlabel="Descendant checkpoint lookback [Gyr]",
            ylabel="Candidate deltaMS [dex]",
            title="PHZ versus reconstructed-SFH recent activity",
        )
        ms_ax.grid(alpha=0.15)
        ms_ax.legend(
            handles=[
                Line2D([], [], color="#2ca25f", lw=1.5, ls="--", label="shifted MS"),
                Line2D([], [], marker="o", markerfacecolor="none", markeredgecolor="0.25",
                       color="none", label="PHZ 100 Myr"),
                Line2D([], [], marker="x", color="0.25", ls="none",
                       label="SFH 100 Myr"),
            ],
            fontsize=7, loc="best",
        )

        morphology_ax = fig.add_subplot(right[2])
        morphology_fields = [
            ("zoobot_smooth_conditional_fraction", "P(smooth)", "#3B6FB6", "o"),
            ("zoobot_spiral_probability", "P(spiral arms)", "#2A9D5B", "s"),
            ("zoobot_merger_probability", "P(merger/disturbed)", "#C44E52", "^"),
        ]
        for field, label, color, marker in morphology_fields:
            medians = []
            for stage_number in range(1, len(stages) + 1):
                selected = candidates[candidates.stage == stage_number]
                indices = selected.bundle_index.to_numpy(dtype=int)
                values = np.asarray(data[field])[indices]
                medians.append(np.nanmedian(values) if np.isfinite(values).any() else np.nan)
            morphology_ax.plot(
                lookbacks, medians, color=color, marker=marker, ms=3.5, lw=1.2,
                label=label,
            )
        morphology_ax.set(
            ylim=(-0.03, 1.03), xlabel="Descendant checkpoint lookback [Gyr]",
            ylabel="Median probability", title="Morphology of selected analogues",
        )
        morphology_ax.grid(alpha=0.15)
        morphology_ax.legend(fontsize=6.5, ncol=3, loc="best")

        notes_ax = fig.add_subplot(right[3])
        notes_ax.axis("off")
        notes_ax.text(
            0, 1,
            "Selection variables\n"
            "- predicted stellar mass, within +/-0.15 dex\n"
            "- cumulative SFH before each descendant state\n"
            "- 2 Gyr physical-time comparison when available\n\n"
            "Candidate redshift, morphology, deltaMS,\n"
            "image embedding, and UMAP position are\n"
            "outcomes, not selection variables. The\n"
            "dashed path joins ensemble medians, not\n"
            "one galaxy's orbit.",
            va="top", fontsize=8.5, linespacing=1.25,
        )
        if descendant_ssfr < -11.5:
            descendant_kind = "Quenched-descendant"
        elif abs(descendant_delta_ms) <= 0.3:
            descendant_kind = "Main-sequence-descendant"
        else:
            descendant_kind = "SFH-selected"
        fig.suptitle(f"Pilot {descendant_kind} progenitor-analogue sequence", fontsize=16)
        pdf.savefig(fig, dpi=180)
        fig.savefig(output / "overview.png", dpi=170)
        plt.close(fig)

        for stage_number, (stage, color) in enumerate(zip(stages, colors), start=1):
            selected = candidates[candidates.stage == stage_number].sort_values("rank")
            if selected.empty:
                continue
            n = len(selected)
            fig, axes = plt.subplots(
                3, n, figsize=(3.35 * n, 9.3), layout="constrained", squeeze=False
            )
            target_curve = np.asarray(stage["target_curve"])
            tau = np.asarray(stage["tau"])
            for column, row in enumerate(selected.itertuples(index=False)):
                index = int(row.bundle_index)
                with Image.open(stamps / f"VIS_{row.galaxy_id}.jpg") as image:
                    axes[0, column].imshow(np.asarray(image.convert("L")), cmap="gray")
                axes[0, column].axis("off")
                axes[0, column].set_title(
                    f"rank {row.rank}: {row.galaxy_id}\n"
                    f"log M*={row.log_stellar_mass:.2f} ({row.mass_offset_dex:+.2f})  "
                    f"z={row.redshift:.3f}\n"
                    f"PHZ deltaMS={row.phz_delta_ms:+.2f}; "
                    "SFH-100Myr deltaMS="
                    f"{_format_delta_ms(_sfh_delta_ms_100myr(data, index))}\n"
                    f"D={row.cumulative_sfh_distance:.3f}",
                    fontsize=9,
                )
                morphology_text = (
                    f"P(smooth)={_format_probability(np.asarray(data['zoobot_smooth_conditional_fraction'])[index])}  "
                    f"P(spiral)={_format_probability(np.asarray(data['zoobot_spiral_probability'])[index])}\n"
                    f"P(merger)={_format_probability(np.asarray(data['zoobot_merger_probability'])[index])}"
                )
                axes[0, column].text(
                    0.02, 0.02, morphology_text, transform=axes[0, column].transAxes,
                    color="white", fontsize=7, ha="left", va="bottom",
                    bbox=dict(facecolor="black", edgecolor="none", alpha=0.62, pad=2),
                )
                _plot_sfh(axes[1, column], data, index, "#355f8d")
                _overlay_candidate_main_sequence(
                    axes[1, column], data, index, show_label=(column == 0)
                )
                axes[1, column].set_title("Complete candidate SFH", fontsize=9)
                candidate = candidate_curve(
                    np.asarray(data["sfh"])[index],
                    bin_edges_from_centres(np.asarray(data["time_grid"])),
                    float(np.asarray(data["time_norm_gyr"])[index]),
                    tau,
                )
                axes[2, column].plot(
                    tau, target_curve, color="#d73027", lw=2,
                    label="descendant state",
                )
                axes[2, column].plot(
                    tau, candidate, color=color, lw=1.8, ls="--", label="analogue"
                )
                axes[2, column].set(
                    xlim=(0, float(stage["comparison_gyr"])), ylim=(-0.03, 1.03),
                    xlabel="Time before compared epoch [Gyr]",
                    ylabel="Fraction already formed",
                )
                axes[2, column].grid(alpha=0.15)
                axes[2, column].tick_params(labelsize=8)
                if column == 0:
                    axes[2, column].legend(fontsize=8)
            fig.suptitle(
                f"Earlier state f={float(stage['fraction']):.2f}: "
                f"lookback={float(stage['state_lookback_gyr']):.2f} Gyr, "
                f"predicted log M*={float(stage['predicted_mass']):.2f}\n"
                "Candidates ranked by cumulative-SFH distance; redshift and morphology were not matched"
                "\nGreen MS curves use each candidate's own observed z and mass (R=0; shifted MS)"
                + (
                    f"\nCAUTION: only {float(stage['comparison_gyr']):.2f} Gyr of "
                    "pre-state history exists; treat this checkpoint as weakly constrained"
                    if bool(stage["shape_comparison_limited"])
                    else ""
                ),
                fontsize=14,
            )
            pdf.savefig(fig, dpi=180)
            fig.savefig(output / f"stage_{stage_number:02d}.png", dpi=150)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--descendant-id", type=int)
    parser.add_argument("--minimum-descendant-mass", type=float, default=10.5)
    parser.add_argument("--target-descendant-mass", type=float, default=11.0)
    parser.add_argument("--quenched-descendant", action="store_true")
    parser.add_argument("--quenched-log-ssfr-max", type=float, default=-11.5)
    parser.add_argument("--main-sequence-descendant", action="store_true")
    parser.add_argument("--main-sequence-delta-max", type=float, default=0.3)
    parser.add_argument("--ms-sfr-offset", type=float, default=-0.93)
    parser.add_argument("--minimum-progenitor-mass", type=float, default=9.0)
    parser.add_argument("--mass-tolerance", type=float, default=0.15)
    parser.add_argument("--history-gyr", type=float, default=2.0)
    parser.add_argument("--minimum-comparison-gyr", type=float, default=0.5)
    parser.add_argument("--n-analogues", type=int, default=5)
    parser.add_argument(
        "--fractions", type=float, nargs="+",
        default=DEFAULT_FORMED_FRACTIONS,
    )
    args = parser.parse_args()
    if args.quenched_descendant and args.main_sequence_descendant:
        parser.error("Choose either --quenched-descendant or --main-sequence-descendant")

    data = _load_inputs(args.bundle, args.archive, args.catalog, args.ms_sfr_offset)
    stamps = args.bundle / "VIS"
    descendant_index = _choose_descendant(
        data,
        stamps,
        args.descendant_id,
        args.minimum_descendant_mass,
        args.target_descendant_mass,
        args.quenched_log_ssfr_max if args.quenched_descendant else None,
        args.main_sequence_delta_max if args.main_sequence_descendant else None,
    )
    descendant_mass = float(np.asarray(data["mass"])[descendant_index])
    fractions = [
        f for f in args.fractions
        # Treat the configured floor as approximate at the 0.01 dex level;
        # catalog masses and a nominal 10^11 target are not exactly decimal.
        if descendant_mass + np.log10(f) >= args.minimum_progenitor_mass - 0.01
    ]
    candidates, census, stages = find_analogues(
        data,
        descendant_index,
        fractions,
        mass_tolerance=args.mass_tolerance,
        history_gyr=args.history_gyr,
        minimum_comparison_gyr=args.minimum_comparison_gyr,
        n_analogues=args.n_analogues,
        stamps=stamps,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(args.output / "analogue_candidates.csv", index=False)
    census.to_csv(args.output / "checkpoint_census.csv", index=False)
    descendant_id = int(np.asarray(data["ids"])[descendant_index])
    manifest = {
        "descendant_id": descendant_id,
        "descendant_bundle_index": descendant_index,
        "descendant_log_stellar_mass": descendant_mass,
        "descendant_redshift": float(np.asarray(data["redshift"])[descendant_index]),
        "descendant_phz_log_ssfr": float(
            np.asarray(data["catalog_log_ssfr"])[descendant_index]
        ),
        "descendant_phz_delta_ms": float(
            np.asarray(data["catalog_delta_ms"])[descendant_index]
        ),
        "quenched_descendant": args.quenched_descendant,
        "quenched_log_ssfr_max": args.quenched_log_ssfr_max,
        "main_sequence_descendant": args.main_sequence_descendant,
        "main_sequence_delta_max": args.main_sequence_delta_max,
        "minimum_descendant_mass": args.minimum_descendant_mass,
        "minimum_progenitor_mass": args.minimum_progenitor_mass,
        "fractions_used": fractions,
        "mass_tolerance_dex": args.mass_tolerance,
        "history_gyr": args.history_gyr,
        "minimum_comparison_gyr": args.minimum_comparison_gyr,
        "n_analogues_per_stage": args.n_analogues,
        "selection_uses": ["predicted stellar mass", "renormalized cumulative SFH"],
        "selection_does_not_use": ["redshift", "morphology", "image embedding", "UMAP position", "sSFR"],
        "formed_mass_return_fraction": 0.0,
        "ms_sfr_offset_dex": args.ms_sfr_offset,
        "catalog": str(args.catalog),
        "bundle": str(args.bundle),
        "archive": str(args.archive),
        "pdf": str(args.pdf),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    make_report(
        data, descendant_index, candidates, census, stages, stamps, args.pdf, args.output
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
