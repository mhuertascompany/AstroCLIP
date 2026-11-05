"""Invert an exported JEPA SFH-to-VIS predictor for VIS-to-SFH lookup.

Each observed VIS teacher embedding is used as a query, while the gallery
contains the VIS-target embeddings predicted from every candidate SFH.  This
is an inverse search through the existing SFH-to-VIS conditional predictor;
it is deliberately distinguished from exporting the separately trained
``image_to_sfh`` JEPA branch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .evaluate_jepa_csls import (
    _comparison_figure,
    _csls_densities,
    _csls_retrieval_many,
)
from .evaluate_multimodal_jepa_retrieval import _retrieval, _summary


def evaluate(args):
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.retrieval) as archive:
        galaxy_ids = np.asarray(archive["galaxy_id"], dtype=np.int64)
        h5_rows = np.asarray(archive["h5_row"], dtype=np.int64)
        query = np.asarray(archive["teacher_image_embedding"], dtype=np.float32)
        gallery = np.asarray(archive["predicted_image_embedding"], dtype=np.float32)
    if args.selection is not None:
        with np.load(args.selection) as archive:
            selected_ids = np.asarray(archive["galaxy_id"], dtype=np.int64)
        position = {int(value): index for index, value in enumerate(galaxy_ids)}
        selected = np.array(
            [position.get(int(value), -1) for value in selected_ids], dtype=np.int64,
        )
        if np.any(selected < 0):
            raise ValueError(
                f"Selection contains {np.sum(selected < 0)} IDs absent from retrieval"
            )
        galaxy_ids = galaxy_ids[selected]
        h5_rows = h5_rows[selected]
        query = query[selected]
        gallery = gallery[selected]
    paired = np.arange(len(query), dtype=np.int64)
    rng = np.random.default_rng(args.seed)
    shuffled = rng.permutation(len(query))
    if np.any(shuffled == paired):
        shuffled = np.roll(paired, 1)
    cosine = _retrieval(
        query, gallery, paired, shuffled,
        args.top_k, args.device, args.query_chunk,
    )
    ks = tuple(sorted(set(args.k)))
    query_density, gallery_density = _csls_densities(
        query, gallery, ks, args.device, args.gallery_chunk,
    )
    csls = _csls_retrieval_many(
        query, gallery, query_density, gallery_density,
        paired, shuffled, args.top_k, args.device, args.query_chunk,
    )
    summary = {
        "direction": "vis_to_sfh_by_inverting_sfh_to_vis_predictor",
        "source_retrieval": str(args.retrieval),
        "selection": str(args.selection) if args.selection is not None else None,
        "cosine": _summary(cosine, len(gallery)),
        **{
            f"csls_k{k}": _summary(result, len(gallery))
            for k, result in csls.items()
        },
    }
    saved = {
        "galaxy_id": galaxy_ids,
        "h5_row": h5_rows,
        **{f"cosine_{name}": value for name, value in cosine.items()},
        **{f"query_density_k{k}": query_density[k] for k in ks},
        **{f"gallery_density_k{k}": gallery_density[k] for k in ks},
    }
    for k, result in csls.items():
        saved.update({f"csls_k{k}_{name}": value for name, value in result.items()})
    np.savez_compressed(args.output / "csls_results.npz", **saved)
    (args.output / "csls_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
    )
    _comparison_figure(
        cosine, csls, len(gallery),
        args.output / "csls_retrieval_diagnostics.png",
    )
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, action="append", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gallery-chunk", type=int, default=512)
    parser.add_argument("--query-chunk", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Refusing to overwrite non-empty output: {args.output}")
    evaluate(args)


if __name__ == "__main__":
    main()
