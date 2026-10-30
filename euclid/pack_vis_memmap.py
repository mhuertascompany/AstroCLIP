"""Pack native Euclid VIS FITS cutouts into a row-aligned NumPy memmap.

The output row order is exactly the row order of ``galaxy_id`` in the SFH HDF5
file.  Training can therefore fetch a target and a randomly selected donor with
two direct memmap reads instead of repeatedly opening individual FITS files.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import torch
from numpy.lib.format import open_memmap
from torch.nn import functional as F

from .vis_fits import (
    estimate_image_stats,
    load_image_stats,
    normalize_vis_flux,
    read_vis_fits,
    resolve_fits_directory,
    vis_fits_path,
    write_image_stats,
)
from .sfh_conditional_diffusion import log_sfh_to_weights, weights_to_clr


def _pack_one(arguments):
    row, galaxy_id, directory, band, image_size, p_lo, p_hi, asinh_scale = arguments
    path = vis_fits_path(directory, galaxy_id, band)
    values = normalize_vis_flux(
        read_vis_fits(path), p_lo, p_hi, asinh_scale,
    )
    tensor = torch.from_numpy(values)[None, None]
    resized = F.interpolate(
        tensor, size=(image_size, image_size), mode="bicubic", align_corners=False,
        antialias=True,
    )[0, 0].numpy()
    # p1 maps to -1 and p99 to +1. Preserve bright cores up to three times
    # the display range rather than saturating them at p99.
    resized = np.clip(2.0 * resized - 1.0, -1.0, 3.0)
    return int(row), resized.astype(np.float16), int(galaxy_id)


def pack(args):
    args.output.mkdir(parents=True, exist_ok=True)
    images_path = args.output / "vis_images.npy"
    sfh_path = args.output / "sfh_clr.npy"
    ids_path = args.output / "galaxy_id.npy"
    valid_path = args.output / "valid.npy"
    metadata_path = args.output / "metadata.json"
    if any(path.exists() for path in (images_path, sfh_path, ids_path, valid_path, metadata_path)):
        raise FileExistsError(f"Refusing to overwrite packed VIS data in {args.output}")

    with h5py.File(args.dataset, "r") as source:
        galaxy_ids = np.asarray(source["galaxy_id"], dtype=np.int64)
        n_sfh_bins = int(source["sfh"].shape[1])
    directory = resolve_fits_directory(args.fits_root, args.band)
    stats_path = args.image_stats or args.output / "image_stats.json"
    if stats_path.exists():
        p_lo, p_hi = load_image_stats(stats_path, args.tokenizer_band)
    else:
        p_lo, p_hi, count, pixels = estimate_image_stats(
            directory, galaxy_ids, band=args.band, max_images=args.stats_images,
            max_pixels_per_image=args.stats_pixels_per_image, seed=args.seed,
        )
        write_image_stats(
            stats_path, p_lo, p_hi, band=args.tokenizer_band, n_images=count,
        )
        print(f"Estimated VIS p1/p99 from {count:,} images/{pixels:,} pixels", flush=True)

    images = open_memmap(
        images_path, mode="w+", dtype=np.float16,
        shape=(len(galaxy_ids), args.image_size, args.image_size),
    )
    saved_ids = open_memmap(ids_path, mode="w+", dtype=np.int64, shape=galaxy_ids.shape)
    valid = open_memmap(valid_path, mode="w+", dtype=np.bool_, shape=galaxy_ids.shape)
    sfhs = open_memmap(
        sfh_path, mode="w+", dtype=np.float32,
        shape=(len(galaxy_ids), n_sfh_bins),
    )
    saved_ids[:] = galaxy_ids
    valid[:] = False
    with h5py.File(args.dataset, "r") as source:
        for start in range(0, len(galaxy_ids), 4096):
            stop = min(start + 4096, len(galaxy_ids))
            log_sfh = np.asarray(source["sfh"][start:stop], dtype=np.float32)
            sfhs[start:stop] = weights_to_clr(log_sfh_to_weights(log_sfh))
    sfhs.flush()

    work = (
        (row, int(galaxy_id), str(directory), args.band, args.image_size,
         p_lo, p_hi, args.asinh_scale)
        for row, galaxy_id in enumerate(galaxy_ids)
    )
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for number, (row, image, galaxy_id) in enumerate(
            executor.map(_pack_one, work, chunksize=args.chunk_size), start=1,
        ):
            if galaxy_id != galaxy_ids[row]:
                raise RuntimeError("Packed VIS row/ID order changed unexpectedly")
            images[row] = image
            valid[row] = True
            if number % 1000 == 0 or number == len(galaxy_ids):
                print(f"Packed {number:,}/{len(galaxy_ids):,}", flush=True)
    images.flush()
    saved_ids.flush()
    valid.flush()
    payload = {
        "dataset": str(args.dataset),
        "fits_root": str(args.fits_root),
        "band": args.band,
        "rows": int(len(galaxy_ids)),
        "image_size": int(args.image_size),
        "n_sfh_bins": n_sfh_bins,
        "dtype": "float16",
        "normalization": "clip(2*asinh_global_percentile_scale-1, -1, 3)",
        "p_lo": float(p_lo),
        "p_hi": float(p_hi),
        "asinh_scale": float(args.asinh_scale),
        "valid_rows": int(np.count_nonzero(valid)),
    }
    metadata_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--fits-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-stats", type=Path)
    parser.add_argument("--band", default="VIS")
    parser.add_argument("--tokenizer-band", default="euclid-vis")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--asinh-scale", type=float, default=20.0)
    parser.add_argument("--stats-images", type=int, default=4096)
    parser.add_argument("--stats-pixels-per-image", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.image_size < 16 or args.workers < 1 or args.chunk_size < 1:
        parser.error("image-size, workers, and chunk-size must be positive")
    pack(args)


if __name__ == "__main__":
    main()
