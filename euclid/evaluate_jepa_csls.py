"""Apply CSLS hubness correction to JEPA SFH-to-VIS retrieval outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .evaluate_multimodal_jepa_retrieval import (
    _load_properties,
    _matched_controls,
    _morphology_figure,
    _show_image,
    _summary,
    _unit_rows,
)
from .sfh_conditional_diffusion import clr_to_weights


def _csls_densities(
    query: np.ndarray, gallery: np.ndarray, ks: tuple[int, ...],
    device: str, gallery_chunk: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Average k-neighbour similarities on both sides of the two domains."""
    query = _unit_rows(query)
    gallery = _unit_rows(gallery)
    maximum_k = max(ks)
    if maximum_k > min(len(query), len(gallery)):
        raise ValueError("CSLS k exceeds the smaller domain")
    query_tensor = torch.from_numpy(query).to(device)
    gallery_density = {
        k: np.empty(len(gallery), dtype=np.float32) for k in ks
    }
    query_top = torch.full(
        (len(query), maximum_k), -torch.inf,
        dtype=query_tensor.dtype, device=device,
    )
    with torch.inference_mode():
        for start in range(0, len(gallery), gallery_chunk):
            stop = min(start + gallery_chunk, len(gallery))
            gallery_tensor = torch.from_numpy(gallery[start:stop]).to(device)
            similarity = query_tensor @ gallery_tensor.T
            column_top = similarity.topk(maximum_k, dim=0).values
            cumulative = column_top.cumsum(0)
            for k in ks:
                gallery_density[k][start:stop] = (
                    cumulative[k - 1].div(k).cpu().numpy()
                )
            query_top = torch.cat((query_top, similarity), dim=1).topk(
                maximum_k, dim=1,
            ).values
    cumulative = query_top.cumsum(1)
    query_density = {
        k: cumulative[:, k - 1].div(k).cpu().numpy().astype(np.float32)
        for k in ks
    }
    return query_density, gallery_density


def _csls_retrieval_many(
    query: np.ndarray, gallery: np.ndarray,
    query_density: dict[int, np.ndarray],
    gallery_density: dict[int, np.ndarray],
    paired_index: np.ndarray, control_index: np.ndarray,
    top_k: int, device: str, query_chunk: int,
) -> dict[int, dict[str, np.ndarray]]:
    """Exact CSLS ranks and neighbours for several local-scaling values."""
    query = _unit_rows(query)
    gallery = _unit_rows(gallery)
    ks = tuple(sorted(query_density))
    if set(ks) != set(gallery_density):
        raise ValueError("Query and gallery density k values differ")
    gallery_tensor = torch.from_numpy(gallery).to(device)
    gallery_local = {
        k: torch.from_numpy(gallery_density[k]).to(device) for k in ks
    }
    collected = {
        k: {name: [] for name in (
            "rank", "control_rank", "paired_similarity",
            "nearest_similarity", "top_index", "top_similarity",
        )} for k in ks
    }
    with torch.inference_mode():
        for start in range(0, len(query), query_chunk):
            stop = min(start + query_chunk, len(query))
            q = torch.from_numpy(query[start:stop]).to(device)
            cosine = q @ gallery_tensor.T
            local = torch.arange(stop - start, device=device)
            paired = torch.as_tensor(paired_index[start:stop], device=device)
            control = torch.as_tensor(control_index[start:stop], device=device)
            for k in ks:
                q_density = torch.from_numpy(
                    query_density[k][start:stop],
                ).to(device)
                score = (
                    2 * cosine - q_density[:, None]
                    - gallery_local[k][None, :]
                )
                paired_score = score[local, paired]
                control_score = score[local, control]
                top_score, top_index = score.topk(top_k, dim=1)
                values = collected[k]
                values["rank"].append(
                    (score > paired_score[:, None]).sum(1).add(1).cpu().numpy()
                )
                values["control_rank"].append(
                    (score > control_score[:, None]).sum(1).add(1).cpu().numpy()
                )
                values["paired_similarity"].append(paired_score.cpu().numpy())
                values["nearest_similarity"].append(top_score[:, 0].cpu().numpy())
                values["top_index"].append(top_index.cpu().numpy())
                values["top_similarity"].append(top_score.cpu().numpy())
    output = {}
    for k, values in collected.items():
        output[k] = {
            name: np.concatenate(parts).astype(
                np.int32 if name in ("rank", "control_rank", "top_index")
                else np.float32
            )
            for name, parts in values.items()
        }
    return output


def _comparison_figure(
    baseline: dict[str, np.ndarray], csls: dict[int, dict[str, np.ndarray]],
    gallery_size: int, output: Path,
):
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    bins = np.linspace(0, 1, 51)
    base_percentile = (baseline["rank"] - 1) / (gallery_size - 1)
    axes[0, 0].hist(
        base_percentile, bins=bins, density=True, histtype="step", lw=2,
        color="0.3", label="cosine",
    )
    for k, result in csls.items():
        percentile = (result["rank"] - 1) / (gallery_size - 1)
        axes[0, 0].hist(
            percentile, bins=bins, density=True, histtype="step", lw=1.6,
            label=f"CSLS k={k}",
        )
    axes[0, 0].set(xlabel="True-pair rank percentile", ylabel="Density",
                   title="Paired retrieval rank")
    axes[0, 0].legend(fontsize=8)

    recall_k = np.array([1, 5, 10, 20, 50, 100])
    axes[0, 1].plot(
        recall_k, [np.mean(baseline["rank"] <= k) for k in recall_k],
        marker="o", color="0.3", label="cosine",
    )
    for k, result in csls.items():
        axes[0, 1].plot(
            recall_k, [np.mean(result["rank"] <= value) for value in recall_k],
            marker="o", label=f"CSLS k={k}",
        )
    axes[0, 1].plot(
        recall_k, recall_k / gallery_size, ls="--", color="0.6", label="chance",
    )
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set(xlabel="Retrieval cutoff", ylabel="Recall",
                   title="True-pair recall")
    axes[0, 1].legend(fontsize=8)

    labels = ["cosine"] + [f"CSLS\n{k}" for k in csls]
    results = [baseline] + list(csls.values())
    maximum_hub = []
    coverage = []
    for result in results:
        count = np.bincount(
            result["top_index"].reshape(-1), minlength=gallery_size,
        )
        maximum_hub.append(count.max() / len(result["rank"]))
        coverage.append(np.mean(count > 0))
    axes[1, 0].bar(labels, maximum_hub, color=["0.45"] + ["#0072B2"] * len(csls))
    axes[1, 0].set(ylabel="Fraction of queries", title="Largest top-100 hub")
    axes[1, 1].bar(labels, coverage, color=["0.45"] + ["#009E73"] * len(csls))
    axes[1, 1].set(ylabel="Fraction of VIS gallery", title="Gallery appearing in top-100")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _example_pdf(
    output: Path, query_indices: np.ndarray, galaxy_ids: np.ndarray,
    h5_rows: np.ndarray, result: dict[str, np.ndarray], packed_vis: Path,
    csls_k: int,
):
    images = np.load(packed_vis / "vis_images.npy", mmap_mode="r")
    clr = np.load(packed_vis / "sfh_clr.npy", mmap_mode="r")
    with PdfPages(output) as pdf:
        for query in query_indices:
            top = result["top_index"][query, :5]
            figure = plt.figure(figsize=(15, 4.8))
            grid = figure.add_gridspec(1, 7, width_ratios=(1.5, 1, 1, 1, 1, 1, 1))
            sfh_axis = figure.add_subplot(grid[0, 0])
            weights = clr_to_weights(np.asarray(clr[h5_rows[query]], dtype=np.float64))
            sfh_axis.plot(np.linspace(0, 1, len(weights)), weights, color="#009E73")
            sfh_axis.set(
                xlabel="Fractional lookback time", ylabel="Normalized SFH weight",
                title=f"Query SFH\nID {galaxy_ids[query]}",
            )
            true_axis = figure.add_subplot(grid[0, 1])
            _show_image(true_axis, images[h5_rows[query]], "True paired VIS")
            for column, gallery_index in enumerate(top, start=2):
                axis = figure.add_subplot(grid[0, column])
                _show_image(
                    axis, images[h5_rows[gallery_index]],
                    f"CSLS #{column - 1}\nID {galaxy_ids[gallery_index]}",
                )
            figure.suptitle(
                f"SFH-to-VIS retrieval - CSLS k={csls_k} - "
                f"paired rank {result['rank'][query]:,}",
                fontsize=13,
            )
            figure.tight_layout(rect=(0, 0, 1, 0.93))
            pdf.savefig(figure)
            plt.close(figure)


def evaluate(args):
    args.output.mkdir(parents=True, exist_ok=True)
    ks = tuple(sorted(set(args.k)))
    with np.load(args.retrieval) as archive:
        galaxy_ids = np.asarray(archive["galaxy_id"], dtype=np.int64)
        h5_rows = np.asarray(archive["h5_row"], dtype=np.int64)
        query = np.asarray(archive["predicted_image_embedding"], dtype=np.float32)
        gallery = np.asarray(archive["teacher_image_embedding"], dtype=np.float32)
        baseline = {
            name: np.asarray(archive[f"pooled_{name}"])
            for name in (
                "rank", "control_rank", "paired_similarity",
                "nearest_similarity", "top_index", "top_similarity",
            )
        }
    paired = np.arange(len(query), dtype=np.int64)
    rng = np.random.default_rng(args.seed)
    shuffled = rng.permutation(len(query))
    if np.any(shuffled == paired):
        shuffled = np.roll(paired, 1)
    query_density, gallery_density = _csls_densities(
        query, gallery, ks, args.device, args.gallery_chunk,
    )
    csls = _csls_retrieval_many(
        query, gallery, query_density, gallery_density,
        paired, shuffled, args.top_k, args.device, args.query_chunk,
    )
    summary = {
        "cosine": _summary(baseline, len(gallery)),
        **{f"csls_k{k}": _summary(result, len(gallery)) for k, result in csls.items()},
    }
    for name, values in summary.items():
        result = baseline if name == "cosine" else csls[int(name.removeprefix("csls_k"))]
        count = np.bincount(result["top_index"].reshape(-1), minlength=len(gallery))
        values["maximum_hub_query_fraction"] = float(count.max() / len(query))
        values["top_k_gallery_coverage"] = float(np.mean(count > 0))
    best_k = min(
        ks,
        key=lambda k: (
            summary[f"csls_k{k}"]["median_rank_percentile"],
            -summary[f"csls_k{k}"]["top_k_gallery_coverage"],
        ),
    )
    summary["selected_k_for_examples"] = best_k
    summary["selection_rule"] = "minimum median true-pair rank percentile"
    (args.output / "csls_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    saved = {
        "galaxy_id": galaxy_ids,
        "h5_row": h5_rows,
        **{f"query_density_k{k}": query_density[k] for k in ks},
        **{f"gallery_density_k{k}": gallery_density[k] for k in ks},
    }
    for k, result in csls.items():
        saved.update({f"csls_k{k}_{name}": value for name, value in result.items()})
    np.savez_compressed(args.output / "csls_results.npz", **saved)
    _comparison_figure(
        baseline, csls, len(gallery), args.output / "csls_retrieval_diagnostics.png",
    )

    properties = _load_properties(args.properties, galaxy_ids)
    eligible = np.ones(len(galaxy_ids), dtype=bool)
    for key in ("log_stellar_mass", "redshift"):
        if key in properties:
            eligible &= np.isfinite(np.asarray(properties[key], dtype=float))
    eligible_indices = np.flatnonzero(eligible)
    morphology_query = rng.choice(
        eligible_indices, min(args.morphology_queries, len(eligible_indices)),
        replace=False,
    )
    morphology_top = csls[best_k]["top_index"][
        morphology_query, :args.morphology_k
    ]
    controls = _matched_controls(
        properties, morphology_query, args.morphology_k, args.seed + 10,
    )
    _morphology_figure(
        properties, morphology_query, morphology_top, controls,
        args.output / f"morphology_csls_k{best_k}.png",
    )
    examples = np.linspace(0, len(galaxy_ids) - 1, args.examples, dtype=int)
    _example_pdf(
        args.output / f"retrieval_examples_csls_k{best_k}.pdf",
        examples, galaxy_ids, h5_rows, csls[best_k], args.packed_vis, best_k,
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved CSLS diagnostics to {args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--packed-vis", type=Path, required=True)
    parser.add_argument("--properties", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, action="append", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gallery-chunk", type=int, default=1024)
    parser.add_argument("--query-chunk", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--morphology-queries", type=int, default=1000)
    parser.add_argument("--morphology-k", type=int, default=20)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Refusing to overwrite non-empty output: {args.output}")
    evaluate(args)


if __name__ == "__main__":
    main()
