"""Map full-sample diffusion conditions into an aligned-SFH UMAP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import umap

from .prepare_diffusion_conditions import sha256


def _similarity_alignment(source, target):
    """Return the similarity transform mapping two corresponding 2-D clouds."""
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    centered_source = source - source_mean
    centered_target = target - target_mean
    left, singular_values, right = np.linalg.svd(
        centered_source.T @ centered_target,
    )
    rotation = left @ right
    scale = singular_values.sum() / np.square(centered_source).sum()
    return source_mean, target_mean, rotation, float(scale)


def _apply_similarity(values, transform):
    source_mean, target_mean, rotation, scale = transform
    return (values - source_mean) @ rotation * scale + target_mean


def build(conditions: Path, output: Path, neighbors=15, min_dist=0.1, seed=42,
          reference_archive: Path | None = None):
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
    reference_metadata = None
    if reference_archive is None:
        xy = reducer.fit_transform(embedding).astype(np.float32)
    else:
        with np.load(reference_archive, allow_pickle=False) as reference:
            required = {"galaxy_id", "sfh_embedding", "xy_sfh"}
            missing = sorted(required.difference(reference.files))
            if missing:
                raise ValueError(
                    f"Reference archive is missing arrays: {', '.join(missing)}"
                )
            reference_ids = np.asarray(reference["galaxy_id"], dtype=np.int64)
            reference_embedding = np.asarray(
                reference["sfh_embedding"], dtype=np.float32,
            )
            reference_xy = np.asarray(reference["xy_sfh"], dtype=np.float32)
        if reference_embedding.shape != (len(reference_ids), embedding.shape[1]):
            raise ValueError("Reference SFH embedding has an incompatible shape.")
        if reference_xy.shape != (len(reference_ids), 2):
            raise ValueError("Reference xy_sfh has an incompatible shape.")
        position = {int(galaxy_id): index for index, galaxy_id in enumerate(ids)}
        try:
            reference_position = np.asarray(
                [position[int(galaxy_id)] for galaxy_id in reference_ids],
                dtype=np.int64,
            )
        except KeyError as error:
            raise ValueError(
                f"Reference galaxy {error.args[0]} is absent from the condition cache."
            ) from error
        condition_reference = embedding[reference_position]
        cosine = np.sum(condition_reference * reference_embedding, axis=1) / (
            np.linalg.norm(condition_reference, axis=1)
            * np.linalg.norm(reference_embedding, axis=1)
        )
        if not np.all(np.isfinite(cosine)) or float(cosine.min()) < 0.999:
            raise ValueError(
                "Reference SFH vectors do not match the diffusion conditions: "
                f"minimum cosine={float(np.nanmin(cosine)):.6f}."
            )

        fitted_reference_xy = reducer.fit_transform(reference_embedding)
        transform = _similarity_alignment(fitted_reference_xy, reference_xy)
        aligned_reference_xy = _apply_similarity(fitted_reference_xy, transform)
        alignment_rms = float(np.sqrt(np.mean(np.square(
            aligned_reference_xy - reference_xy,
        ))))

        is_reference = np.zeros(len(ids), dtype=bool)
        is_reference[reference_position] = True
        xy = np.empty((len(ids), 2), dtype=np.float32)
        xy[reference_position] = reference_xy
        if np.any(~is_reference):
            transformed = reducer.transform(embedding[~is_reference])
            xy[~is_reference] = _apply_similarity(
                transformed, transform,
            ).astype(np.float32)
        reference_metadata = {
            "archive": str(reference_archive),
            "archive_sha256": sha256(reference_archive),
            "n_reference": int(len(reference_ids)),
            "minimum_condition_cosine": float(cosine.min()),
            "median_condition_cosine": float(np.median(cosine)),
            "refit_similarity_alignment_rms": alignment_rms,
            "coordinates": "xy_sfh",
        }
    metadata = {
        "condition_cache": str(conditions),
        "condition_cache_sha256": sha256(conditions),
        "condition_metadata": condition_metadata,
        "space": "aligned SFH diffusion-condition embedding",
        "n_neighbors": int(neighbors), "min_dist": float(min_dist),
        "metric": "cosine", "seed": int(seed),
        "reference": reference_metadata,
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
    parser.add_argument("--neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reference-archive", type=Path,
        help=("Anchor the full-sample map to this explorer archive's exact "
              "aligned-SFH coordinates."),
    )
    args = parser.parse_args()
    build(**vars(args))


if __name__ == "__main__":
    main()
