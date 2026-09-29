"""Match cluster-3 main-sequence galaxies to clusters 1/2 in SFR, mass, and z."""
import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

from .cosmic_sfh import COSMOLOGY
from .sfh_shape import sfh_recent_activity


MORPHOLOGY = {
    'Featured / disk probability': 'zoobot_featured_conditional_fraction',
    'Spiral-arm probability': 'zoobot_spiral_probability',
    'Merger / disturbed probability': 'zoobot_merger_probability',
}


def _read_rows(dataset, rows):
    order = np.argsort(rows)
    return np.asarray(dataset[rows[order]])[np.argsort(order)]


def _cluster_labels(embedding, mean_lookback):
    finite = np.isfinite(embedding).all(axis=1) & (np.linalg.norm(embedding, axis=1) > 0)
    raw = np.full(len(embedding), -1, dtype=int)
    raw[finite] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(
        embedding[finite]
    )
    order = sorted(range(3), key=lambda label: np.nanmedian(mean_lookback[raw == label]))
    return np.array([order.index(value) + 1 if value >= 0 else -1 for value in raw])


def _match(target, controls, features, tolerances, neighbours=128):
    scales = np.asarray(tolerances, dtype=float)
    tree = cKDTree(features[controls] / scales)
    k = min(neighbours, len(controls))
    distances, local = tree.query(features[target] / scales, k=k)
    if k == 1:
        distances, local = distances[:, None], local[:, None]
    edges = []
    for target_row in range(len(target)):
        for rank in range(k):
            control_row = int(local[target_row, rank])
            delta = np.abs(features[target[target_row]] - features[controls[control_row]])
            if np.all(delta <= scales):
                edges.append((float(distances[target_row, rank]), target_row, control_row))
    edges.sort()
    assigned_target, assigned_control = {}, set()
    for distance, target_row, control_row in edges:
        if target_row in assigned_target or control_row in assigned_control:
            continue
        assigned_target[target_row] = (control_row, distance, False, True)
        assigned_control.add(control_row)

    # Guarantee one comparison per target. Prefer a within-tolerance reused
    # control; if none exists, record the nearest extrapolative match explicitly.
    for target_row in range(len(target)):
        if target_row in assigned_target:
            continue
        delta = np.abs(features[controls] - features[target[target_row]])
        inside = np.all(delta <= scales, axis=1)
        pool = np.flatnonzero(inside)
        within = len(pool) > 0
        if not within:
            pool = np.arange(len(controls))
        score = np.sum(((features[controls[pool]] - features[target[target_row]]) / scales) ** 2, axis=1)
        control_row = int(pool[np.argmin(score)])
        assigned_target[target_row] = (control_row, float(np.sqrt(score.min())), True, within)
    return assigned_target


def run(args):
    with np.load(args.archive, allow_pickle=False) as archive:
        ids = np.asarray(archive['galaxy_id'], dtype=np.int64)
        embedding = np.asarray(archive['sfh_embedding'], dtype=np.float32)
        mass = np.asarray(archive['log_stellar_mass'], dtype=float)
        redshift = np.asarray(archive['redshift'], dtype=float)
        morphology = {label: np.asarray(archive[key], dtype=float)
                      for label, key in MORPHOLOGY.items()}
    with h5py.File(args.bundle / 'euclid_explorer.h5', 'r') as source:
        source_ids = np.asarray(source['galaxy_id'][:], dtype=np.int64)
        lookup = {int(value): row for row, value in enumerate(source_ids)}
        rows = np.array([lookup[int(value)] for value in ids], dtype=np.int64)
        sfh = _read_rows(source['sfh'], rows).astype(np.float32)
        time = np.asarray(source['sfh_time_grid'][:], dtype=float)
        age_myr = _read_rows(source['sfh_time_norm'], rows).astype(float)
        epsilon = float(source.attrs.get('sfh_log_epsilon', 1e-10))
    with np.errstate(over='ignore', invalid='ignore'):
        weights = np.maximum(10.0 ** sfh.astype(float) - epsilon, 0.0)
        weights /= np.where(weights.sum(axis=1) > 0, weights.sum(axis=1), np.nan)[:, None]
    mean_lookback = weights @ time
    labels = _cluster_labels(embedding, mean_lookback)
    activity = sfh_recent_activity(sfh, time, epsilon, age_myr)
    log_sfr = mass + np.asarray(activity['sfh_log_sfr_per_stellar_mass_100myr_r0'])
    cosmic_age = np.asarray(COSMOLOGY.age(redshift).to_value('Gyr'), dtype=float)
    standard_ms = (.84 - .026 * cosmic_age) * mass - (6.51 - .11 * cosmic_age)
    shifted_ms = standard_ms + args.ms_sfr_offset
    delta_ms = log_sfr - shifted_ms
    finite = np.isfinite(log_sfr + mass + redshift + delta_ms)
    target_mask = (
        finite & (labels == 3)
        & (mass >= args.mass_range[0]) & (mass <= args.mass_range[1])
        & (np.abs(delta_ms) <= args.ms_width)
    )
    control_mask = finite & np.isin(labels, (1, 2))
    target = np.flatnonzero(target_mask)
    controls = np.flatnonzero(control_mask)
    if not len(target) or not len(controls):
        cluster3_mass = finite & (labels == 3) & (mass >= args.mass_range[0]) & (mass <= args.mass_range[1])
        values = delta_ms[cluster3_mass]
        summary = dict(n_controls=int(len(controls)), n_cluster3_mass=int(cluster3_mass.sum()),
                       n_targets=int(len(target)),
                       cluster3_delta_ms_percentiles=(np.percentile(values, [0, 16, 50, 84, 100]).tolist()
                                                      if len(values) else None))
        raise ValueError(f'No eligible targets or controls: {summary}')
    features = np.c_[log_sfr, mass, redshift]
    assignments = _match(target, controls, features, args.tolerances, args.neighbours)
    records = []
    for pair, target_row in enumerate(range(len(target)), start=1):
        control_row, distance, reused, within = assignments[target_row]
        for role, index in (('cluster3_ms', target[target_row]),
                            ('cluster12_match', controls[control_row])):
            record = dict(
                pair=pair, role=role, galaxy_id=str(int(ids[index])), cluster=int(labels[index]),
                log_stellar_mass=float(mass[index]), redshift=float(redshift[index]),
                sfh_log_sfr_100myr_r0=float(log_sfr[index]),
                shifted_ms_log_sfr=float(shifted_ms[index]), delta_ms_100myr=float(delta_ms[index]),
                match_distance=float(distance), control_reused=bool(reused),
                within_requested_tolerances=bool(within),
            )
            for label, values in morphology.items():
                record[label.lower().replace(' / ', '_').replace(' ', '_')] = float(values[index])
            records.append(record)
    matched = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    matched.to_csv(args.output, index=False)

    target_table = matched[matched.role == 'cluster3_ms'].reset_index(drop=True)
    control_table = matched[matched.role == 'cluster12_match'].reset_index(drop=True)
    residuals = {
        'log_sfr': np.abs(target_table.sfh_log_sfr_100myr_r0 - control_table.sfh_log_sfr_100myr_r0),
        'log_mass': np.abs(target_table.log_stellar_mass - control_table.log_stellar_mass),
        'redshift': np.abs(target_table.redshift - control_table.redshift),
    }
    report = dict(
        n_full_sample=int(len(ids)), cluster_counts={str(k): int(np.sum(labels == k)) for k in (1, 2, 3)},
        target_definition=dict(cluster=3, mass_range=args.mass_range,
                               absolute_shifted_delta_ms_max=args.ms_width,
                               window_myr=100, return_fraction=0,
                               ms_sfr_offset_dex=args.ms_sfr_offset),
        n_cluster3_main_sequence=int(len(target)), n_controls=int(len(controls)),
        n_unique_controls=int(control_table.galaxy_id.nunique()),
        n_reused_matches=int(control_table.control_reused.sum()),
        n_outside_tolerances=int((~control_table.within_requested_tolerances).sum()),
        tolerances=dict(log_sfr=args.tolerances[0], log_mass=args.tolerances[1], redshift=args.tolerances[2]),
        match_residuals={name: dict(median=float(value.median()), p90=float(value.quantile(.9)),
                                    maximum=float(value.max())) for name, value in residuals.items()},
    )
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')

    combined_indices = np.r_[target, np.array([controls[assignments[row][0]] for row in range(len(target))])]
    markers = [('Cluster 3 · shifted MS', target, 'o'), ('Matched clusters 1/2',
                np.array([controls[assignments[row][0]] for row in range(len(target))]), '^')]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), layout='constrained', sharex=True, sharey=True)
    median_z = float(np.median(redshift[target]))
    t_median = float(COSMOLOGY.age(median_z).to_value('Gyr'))
    mass_line = np.linspace(args.mass_range[0] - .15, args.mass_range[1] + .15, 100)
    ms_line = (.84 - .026 * t_median) * mass_line - (6.51 - .11 * t_median) + args.ms_sfr_offset
    for ax, (title, values) in zip(axes, morphology.items()):
        valid_values = values[combined_indices][np.isfinite(values[combined_indices])]
        low, high = np.percentile(valid_values, [2, 98]) if len(valid_values) else (0, 1)
        if high <= low:
            high = low + 1
        control_indices = markers[1][1]
        for target_index, control_index in zip(target, control_indices):
            ax.plot([mass[target_index], mass[control_index]],
                    [log_sfr[target_index], log_sfr[control_index]],
                    color='0.55', lw=.7, alpha=.75, zorder=1)
        for label, indices, marker in markers:
            scatter = ax.scatter(mass[indices], log_sfr[indices], c=values[indices],
                                 cmap='viridis', vmin=low, vmax=high, s=52,
                                 marker=marker, alpha=.9, linewidth=.7,
                                 edgecolor='black' if marker == 'o' else 'white',
                                 label=label, rasterized=True, zorder=3)
        for pair_number, index in enumerate(target, start=1):
            ax.annotate(str(pair_number), (mass[index], log_sfr[index]),
                        xytext=(5, 4), textcoords='offset points', fontsize=7,
                        color='0.1', zorder=5)
        ax.plot(mass_line, ms_line, color='black', ls='--', lw=1,
                label=f'shifted MS at median z={median_z:.2f}')
        ax.plot(mass_line, ms_line + args.ms_width, color='0.35', ls=':', lw=.8)
        ax.plot(mass_line, ms_line - args.ms_width, color='0.35', ls=':', lw=.8)
        ax.set(title=title, xlabel='log10(M★ / M☉)', xlim=(9.85, 11.15))
        fig.colorbar(scatter, ax=ax, label=title, shrink=.82)
    axes[0].set_ylabel('SFH log10(SFR100 / M☉ yr⁻¹), R=0')
    axes[0].legend(fontsize=8, loc='best')
    fig.suptitle(
        f'Cluster-3 shifted-MS galaxies and SFR–mass–z matched cluster-1/2 controls\n'
        f'N={len(target):,} pairs; MS shift={args.ms_sfr_offset:+.2f} dex; '
        f'|ΔMS100|≤{args.ms_width:.2f}', fontsize=13,
    )
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=200)
    fig.savefig(args.figure.with_suffix('.png'), dpi=180)
    plt.close(fig)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--figure', type=Path, required=True)
    parser.add_argument('--mass-range', type=float, nargs=2, default=[10., 11.])
    parser.add_argument('--ms-width', type=float, default=.3)
    parser.add_argument('--ms-sfr-offset', type=float, default=-.93)
    parser.add_argument('--tolerances', type=float, nargs=3, default=[.15, .10, .08],
                        metavar=('LOGSFR', 'LOGMASS', 'Z'))
    parser.add_argument('--neighbours', type=int, default=128)
    args = parser.parse_args()
    if args.neighbours < 1 or any(value <= 0 for value in args.tolerances):
        parser.error('neighbours and tolerances must be positive.')
    run(args)


if __name__ == '__main__':
    main()
