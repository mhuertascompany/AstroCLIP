"""Fit a full-sample UMAP to the exact SFH conditions used by a diffusion run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import umap

from .prepare_diffusion_conditions import sha256


def build(conditions: Path, output: Path, neighbors=30, min_dist=0.1, seed=42):
    if output.exists():
        raise FileExistsError(output)
    with np.load(conditions, allow_pickle=False) as source:
        ids = np.concatenate([source["train_ids"], source["val_ids"]]).astype(np.int64)
        embedding = np.concatenate([
            source["train_condition"], source["val_condition"],
        ]).astype(np.float32)
        condition_metadata = json.loads(str(source["metadata"]))
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Condition cache contains duplicate galaxy IDs.")
    if not np.allclose(np.linalg.norm(embedding, axis=1), 1, atol=1e-4):
        raise ValueError("Condition cache contains non-unit embeddings.")
    reducer = umap.UMAP(
        n_neighbors=neighbors, min_dist=min_dist, metric="cosine",
        random_state=seed, low_memory=True, verbose=True,
    )
    xy = reducer.fit_transform(embedding).astype(np.float32)
    metadata = {
        "condition_cache": str(conditions),
        "condition_cache_sha256": sha256(conditions),
        "condition_metadata": condition_metadata,
        "space": "aligned SFH diffusion-condition embedding",
        "n_neighbors": int(neighbors), "min_dist": float(min_dist),
        "metric": "cosine", "seed": int(seed),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream, galaxy_id=ids, xy=xy, metadata=json.dumps(metadata),
        )
    temporary.replace(output)
    print(json.dumps({**metadata, "n_objects": len(ids), "output": str(output)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build(**vars(args))


if __name__ == "__main__":
    main()
