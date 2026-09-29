"""Select star-forming pairs matched in recent SFR, mass, and redshift.

The two members of each pair are deliberately separated in either SFH peak
time or mass-weighted age.  The resulting CSV can be passed directly to
``sample_diffusion_selection`` for a shared-noise morphology comparison.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from astropy.table import Table
from PIL import Image

from .sfh_shape import sfh_recent_activity


def _catalog_values(path, galaxy_ids):
    table = Table.read(path)
    names = {name.lower(): name for name in table.colnames}
    required = ('object_id', 'phz_pp_median_sfr')
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f'Missing PHZ columns: {missing}')
    ids = np.asarray(table[names['object_id']], dtype=np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError('Duplicate object IDs in PHZ catalog.')
    lookup = {int(value): row for row, value in enumerate(ids)}
    rows = np.array([lookup.get(int(value), -1) for value in galaxy_ids])
    sfr = np.full(len(galaxy_ids), np.nan)
    found = rows >= 0
    raw = np.asarray(np.ma.asarray(table[names['phz_pp_median_sfr']], dtype=float).filled(np.nan))
    sfr[found] = raw[rows[found]]
    return sfr, found


def _condition_ids(path):
    if path is None:
        return None
    with np.load(path, allow_pickle=False) as cache:
        return set(map(int, np.r_[cache['train_ids'], cache['val_ids']]))


def _pairs(values, eligible, metric, label, n_pairs, tolerances, min_separation,
           rng, already_used):
    finite = eligible & np.isfinite(metric)
    low_cut, high_cut = np.quantile(metric[finite], [0.25, 0.75])
    low = np.flatnonzero(finite & (metric <= low_cut))
    high = np.flatnonzero(finite & (metric >= high_cut))
    rng.shuffle(low)
    scales = np.asarray(tolerances, dtype=float)
    proposals = []
    for i in low:
        delta = np.abs(values[high] - values[i])
        allowed = np.all(delta <= scales, axis=1) & ((metric[high] - metric[i]) >= min_separation)
        if not allowed.any():
            continue
        candidates = high[allowed]
        distances = np.sum(((values[candidates] - values[i]) / scales) ** 2, axis=1)
        separation = metric[candidates] - metric[i]
        score = distances - 0.05 * separation / min_separation
        j = candidates[np.argmin(score)]
        proposals.append((float(score.min()), -float(metric[j] - metric[i]), int(i), int(j)))
    proposals.sort()
    chosen = []
    used = set(already_used)
    for _, _, i, j in proposals:
        if i in used or j in used:
            continue
        chosen.append((label, i, j))
        used.update((i, j))
        if len(chosen) == n_pairs:
            break
    if len(chosen) < n_pairs:
        raise ValueError(f'Only found {len(chosen)}/{n_pairs} independent {label} pairs.')
    return chosen


def select(args):
    if args.output.exists() or args.output.with_suffix('.json').exists():
        raise FileExistsError(args.output)
    with np.load(args.archive, allow_pickle=False) as archive:
        ids = np.asarray(archive['galaxy_id'], dtype=np.int64)
        mass = np.asarray(archive['log_stellar_mass'], dtype=float)
        redshift = np.asarray(archive['redshift'], dtype=float)
        morphology = {
            name: np.asarray(archive[name], dtype=float) if name in archive else np.full(len(ids), np.nan)
            for name in ('zoobot_smooth_probability', 'zoobot_spiral_probability',
                         'zoobot_merger_probability')
        }
    with h5py.File(args.bundle / 'euclid_explorer.h5', 'r') as source:
        source_ids = np.asarray(source['galaxy_id'][:], dtype=np.int64)
        if np.array_equal(source_ids, ids):
            rows = np.arange(len(ids), dtype=np.int64)
        else:
            lookup = {int(value): row for row, value in enumerate(source_ids)}
            try:
                rows = np.array([lookup[int(value)] for value in ids], dtype=np.int64)
            except KeyError as error:
                raise ValueError(f'Archive ID absent from explorer HDF5: {error.args[0]}') from error
        sfh = np.asarray(source['sfh'][rows], dtype=np.float32)
        time = np.asarray(source['sfh_time_grid'][:], dtype=float)
        age_myr = np.asarray(source['sfh_time_norm'][rows], dtype=float)
        epsilon = float(source.attrs.get('sfh_log_epsilon', 1e-10))
    activity = sfh_recent_activity(sfh, time, epsilon, age_myr)
    recent_log_sfr = mass + activity['sfh_log_sfr_per_stellar_mass_100myr_r0']
    peak_age = np.asarray(activity['sfh_peak_age_gyr'], dtype=float)
    mean_age = np.asarray(activity['sfh_mass_weighted_age_gyr'], dtype=float)
    catalog_log_sfr, catalog_found = _catalog_values(args.catalog, ids)
    catalog_log_ssfr = catalog_log_sfr - mass
    cached = _condition_ids(args.conditions)
    in_cache = np.ones(len(ids), dtype=bool) if cached is None else np.array(
        [int(value) in cached for value in ids], dtype=bool
    )
    eligible = (
        np.isfinite(recent_log_sfr + mass + redshift + catalog_log_ssfr + peak_age + mean_age)
        & catalog_found & in_cache
        & (catalog_log_ssfr >= args.minimum_catalog_log_ssfr)
        & (recent_log_sfr >= args.minimum_sfh_log_sfr)
        & (np.abs(recent_log_sfr - catalog_log_sfr) <= args.maximum_sfr_disagreement)
        & (mass >= args.mass_range[0]) & (mass <= args.mass_range[1])
        & (redshift >= args.redshift_range[0]) & (redshift <= args.redshift_range[1])
    )
    if eligible.sum() < 4 * args.n_pairs:
        raise ValueError(f'Only {eligible.sum()} eligible star-forming galaxies.')
    matching = np.c_[recent_log_sfr, mass, redshift]
    rng = np.random.default_rng(args.seed)
    chosen = _pairs(matching, eligible, peak_age, 'peak', args.n_pairs,
                    args.tolerances, args.minimum_peak_separation, rng, set())
    used = {index for _, i, j in chosen for index in (i, j)}
    chosen += _pairs(matching, eligible, mean_age, 'age', args.n_pairs,
                     args.tolerances, args.minimum_age_separation, rng, used)

    records = []
    pair_counts = {'peak': 0, 'age': 0}
    for mode, i, j in chosen:
        pair_counts[mode] += 1
        pair_number = pair_counts[mode]
        descriptors = ('late_peak', 'early_peak') if mode == 'peak' else ('young', 'old')
        for index, descriptor in zip((i, j), descriptors):
            records.append(dict(
                galaxy_id=int(ids[index]), cluster=f'{mode}_{pair_number:02d}_{descriptor}',
                experiment=mode, pair=pair_number, member=descriptor,
                sfh_log_sfr_100myr_r0=float(recent_log_sfr[index]),
                catalog_log_sfr_100myr=float(catalog_log_sfr[index]),
                catalog_log_ssfr=float(catalog_log_ssfr[index]),
                log_stellar_mass=float(mass[index]), redshift=float(redshift[index]),
                sfh_peak_age_gyr=float(peak_age[index]),
                sfh_mass_weighted_age_gyr=float(mean_age[index]),
                zoobot_p_smooth=float(morphology['zoobot_smooth_probability'][index]),
                zoobot_p_spiral=float(morphology['zoobot_spiral_probability'][index]),
                zoobot_p_merger=float(morphology['zoobot_merger_probability'][index]),
            ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    manifest = dict(
        seed=args.seed, n_pairs_per_experiment=args.n_pairs,
        selection='PHZ star-forming cut; matched SFH SFR100, mass, and redshift; separated history metric',
        minimum_catalog_log_ssfr=args.minimum_catalog_log_ssfr,
        minimum_sfh_log_sfr=args.minimum_sfh_log_sfr,
        maximum_sfh_minus_catalog_sfr_dex=args.maximum_sfr_disagreement,
        mass_range=args.mass_range, redshift_range=args.redshift_range,
        tolerances=dict(log_sfr_100myr=args.tolerances[0], log_mass=args.tolerances[1],
                        redshift=args.tolerances[2]),
        minimum_peak_separation_gyr=args.minimum_peak_separation,
        minimum_age_separation_gyr=args.minimum_age_separation,
        n_eligible=int(eligible.sum()), n_archive=int(len(ids)),
        archive=str(args.archive), catalog=str(args.catalog), conditions=str(args.conditions),
        archive_sha256=hashlib.sha256(args.archive.read_bytes()).hexdigest(), objects=records,
    )
    args.output.with_suffix('.json').write_text(json.dumps(manifest, indent=2) + '\n')

    if args.pdf:
        args.pdf.parent.mkdir(parents=True, exist_ok=True)
        weights = np.maximum(10.0 ** sfh.astype(float) - epsilon, 0.0)
        weights /= weights.sum(axis=1, keepdims=True)
        with PdfPages(args.pdf) as pdf:
            for mode, i, j in chosen:
                fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout='constrained')
                for column, index in enumerate((i, j)):
                    stamp = args.bundle / 'VIS' / f'VIS_{int(ids[index])}.jpg'
                    with Image.open(stamp) as image:
                        axes[0, column].imshow(image.convert('L'), cmap='gray')
                    axes[0, column].axis('off')
                    axes[0, column].set_title(
                        f'{ids[index]}\nlog SFR100={recent_log_sfr[index]:+.2f}, '
                        f'log M★={mass[index]:.2f}, z={redshift[index]:.2f}'
                    )
                    axes[1, column].plot(time, weights[index], color='tab:blue')
                    axes[1, column].axvline(peak_age[index] / (age_myr[index] / 1e3),
                                           color='tab:red', ls='--', label='SFH peak')
                    axes[1, column].set(
                        xlabel='Fractional lookback time', ylabel='Normalized SFH weight',
                        title=f'peak={peak_age[index]:.2f} Gyr; mean age={mean_age[index]:.2f} Gyr'
                    )
                    axes[1, column].legend()
                fig.suptitle(
                    f'Matched recent activity, different {mode}: '
                    f'ΔlogSFR={abs(recent_log_sfr[i]-recent_log_sfr[j]):.3f}, '
                    f'ΔlogM={abs(mass[i]-mass[j]):.3f}, Δz={abs(redshift[i]-redshift[j]):.3f}'
                )
                pdf.savefig(fig); plt.close(fig)
    print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--conditions', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pdf', type=Path)
    parser.add_argument('--n-pairs', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--minimum-catalog-log-ssfr', type=float, default=-11.0)
    parser.add_argument('--minimum-sfh-log-sfr', type=float, default=-1.5)
    parser.add_argument('--maximum-sfr-disagreement', type=float, default=1.0)
    parser.add_argument('--mass-range', type=float, nargs=2, default=[9.5, 11.3])
    parser.add_argument('--redshift-range', type=float, nargs=2, default=[0.2, 1.2])
    parser.add_argument('--tolerances', type=float, nargs=3, default=[0.15, 0.15, 0.10],
                        metavar=('LOGSFR', 'LOGMASS', 'Z'))
    parser.add_argument('--minimum-peak-separation', type=float, default=2.0)
    parser.add_argument('--minimum-age-separation', type=float, default=1.5)
    args = parser.parse_args()
    if args.n_pairs < 1 or any(value <= 0 for value in args.tolerances):
        parser.error('n-pairs and matching tolerances must be positive.')
    select(args)


if __name__ == '__main__':
    main()
