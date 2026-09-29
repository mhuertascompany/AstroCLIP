"""Plot paired progenitor-analogue tracks for controlled SFH selections."""
import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from PIL import Image

from .match_cluster3_main_sequence import _cluster_labels
from .progenitor_analogues import _plot_cluster_background


MORPHOLOGY = {
    'P(smooth)': 'zoobot_smooth_probability',
    'P(spiral arms)': 'zoobot_spiral_probability',
    'P(merger/disturbed)': 'zoobot_merger_probability',
}
PAIR_COLORS = ('#2166ac', '#b2182b')


def _normalized_sfh(source, rows):
    epsilon = float(source.attrs.get('sfh_log_epsilon', 1e-10))
    order = np.argsort(rows)
    inverse = np.argsort(order)
    values = np.asarray(source['sfh'][rows[order]], float)[inverse]
    weights = np.maximum(10.0 ** values - epsilon, 0.0)
    weights /= np.where(weights.sum(axis=1) > 0, weights.sum(axis=1), np.nan)[:, None]
    return weights


def run(args):
    selection = pd.read_csv(args.selection, dtype={'galaxy_id': str})
    required = {'galaxy_id', 'experiment', 'pair', 'member'}
    if not required.issubset(selection):
        raise ValueError(f'Selection is missing columns: {sorted(required-set(selection))}')
    checkpoints = pd.read_csv(args.tracks / 'progenitor_checkpoints.csv',
                              dtype={'descendant_id': str})
    candidates = pd.read_csv(args.tracks / 'analogue_candidates.csv',
                             dtype={'descendant_id': str, 'galaxy_id': str})
    descendants = pd.read_csv(args.tracks / 'descendants.csv',
                              dtype={'galaxy_id': str})
    with np.load(args.archive, allow_pickle=False) as archive:
        archive_ids = np.asarray(archive['galaxy_id'], dtype=np.int64)
        xy = np.asarray(archive['xy_joint'], dtype=float)
        cluster = _cluster_labels(
            np.asarray(archive['sfh_embedding'], dtype=float),
            np.asarray(archive['sfh_mean_lookback'], dtype=float),
        )
        morphology = {label: np.asarray(archive[key], dtype=float)
                      for label, key in MORPHOLOGY.items()}
    archive_lookup = {str(int(value)): row for row, value in enumerate(archive_ids)}
    with h5py.File(args.bundle / 'euclid_explorer.h5', 'r') as source:
        h5_ids = np.asarray(source['galaxy_id'][:], dtype=np.int64)
        h5_lookup = {str(int(value)): row for row, value in enumerate(h5_ids)}
        rows = np.array([h5_lookup[value] for value in selection.galaxy_id], dtype=int)
        sfh = _normalized_sfh(source, rows)
        time = np.asarray(source['sfh_time_grid'][:], dtype=float)
        order = np.argsort(rows)
        time_norm_gyr = (
            np.asarray(source['sfh_time_norm'][rows[order]], dtype=float)[np.argsort(order)]
            / 1000.0
        )
    sfh_by_id = {value: sfh[row] for row, value in enumerate(selection.galaxy_id)}
    time_norm_by_id = {
        value: time_norm_gyr[row] for row, value in enumerate(selection.galaxy_id)
    }
    cluster_by_id = {
        value: int(cluster[archive_lookup[value]]) for value in selection.galaxy_id
    }
    cluster_background = {'xy_joint': xy, 'cluster': cluster}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    groups = list(selection.groupby(['experiment', 'pair'], sort=False))
    with PdfPages(args.output) as pdf:
        for (experiment, pair), members in groups:
            if len(members) != 2:
                raise ValueError(f'{experiment} pair {pair} has {len(members)} members.')
            members = members.reset_index(drop=True)
            fig = plt.figure(figsize=(16, 10), layout='constrained')
            grid = fig.add_gridspec(3, 4, height_ratios=(1.0, 1.05, .85),
                                    width_ratios=(1, 1.45, 1.45, 1))
            umap_ax = fig.add_subplot(grid[:2, 1:3])
            _plot_cluster_background(umap_ax, cluster_background, alpha=0.11)
            all_lookback = []
            member_tracks = []
            for member_index, row in members.iterrows():
                gid = row.galaxy_id
                track = checkpoints[checkpoints.descendant_id == gid].sort_values('stage')
                if track.empty:
                    raise ValueError(f'No checkpoints for {gid}.')
                member_tracks.append(track)
                all_lookback.extend(track.state_lookback_gyr)
            norm = Normalize(min(all_lookback), max(all_lookback))
            cmap = plt.get_cmap('viridis')
            for member_index, row in members.iterrows():
                gid = row.galaxy_id
                track = member_tracks[member_index]
                x = track.analogue_centroid_x.to_numpy()
                y = track.analogue_centroid_y.to_numpy()
                color = PAIR_COLORS[member_index]
                marker = 'o' if member_index == 0 else 's'
                umap_ax.plot(x, y, color=color, lw=1.6, alpha=.85)
                umap_ax.scatter(x, y, c=track.state_lookback_gyr, cmap=cmap, norm=norm,
                                s=46, marker=marker, edgecolor=color, linewidth=.9, zorder=4)
                descendant = descendants[descendants.galaxy_id == gid].iloc[0]
                umap_ax.scatter(descendant.joint_umap_x, descendant.joint_umap_y,
                                marker='*', s=230, color=color, edgecolor='white',
                                linewidth=.8, zorder=5,
                                label=f"{row.member}: {gid}")
                for _, point in track.iloc[::max(1, len(track)//5)].iterrows():
                    umap_ax.annotate(f"{point.formed_mass_fraction:.2f}",
                                     (point.analogue_centroid_x, point.analogue_centroid_y),
                                     xytext=(4, 3), textcoords='offset points', fontsize=6,
                                     color=color)
            umap_ax.set(xlabel='Joint-average UMAP 1', ylabel='Joint-average UMAP 2',
                        title='SFH-selected progenitor-analogue tracks')
            umap_ax.legend(fontsize=7, loc='best')
            fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=umap_ax,
                         label='Descendant-frame lookback time [Gyr]', shrink=.65)

            morphology_grid = grid[2, :].subgridspec(1, 3)
            morphology_axes = [fig.add_subplot(morphology_grid[0, column]) for column in range(3)]
            for member_index, member in members.iterrows():
                gid = member.galaxy_id
                track = member_tracks[member_index]
                selected = candidates[candidates.descendant_id == gid]
                color = PAIR_COLORS[member_index]
                for ax, (label, values) in zip(morphology_axes, morphology.items()):
                    medians = []
                    xvalues = []
                    for _, checkpoint in track.iterrows():
                        stage_rows = selected[selected.stage == checkpoint.stage]
                        indices = stage_rows.bundle_index.to_numpy(dtype=int)
                        if len(indices):
                            medians.append(float(np.nanmedian(values[indices])))
                            xvalues.append(float(checkpoint.state_lookback_gyr))
                    ax.plot(xvalues, medians, '-o', color=color, ms=3,
                            label=member.member)
                    descendant_index = archive_lookup[gid]
                    ax.scatter([0], [values[descendant_index]], marker='*', s=80,
                               color=color, edgecolor='white', linewidth=.5, zorder=5)
                    ax.set(ylabel=label, ylim=(-.03, 1.03))
            morphology_axes[0].legend(fontsize=7)
            morphology_axes[-1].set_xlabel('Descendant-frame lookback time [Gyr]')

            for member_index, member in members.iterrows():
                gid = member.galaxy_id
                stamp_ax = fig.add_subplot(grid[0, 0 if member_index == 0 else 3])
                with Image.open(args.bundle / 'VIS' / f'VIS_{gid}.jpg') as image:
                    stamp_ax.imshow(image.convert('L'), cmap='gray')
                stamp_ax.axis('off')
                stamp_ax.set_title(
                    f"{member.member} · C{cluster_by_id[gid]}\n{gid}\n"
                    f"SFH log SFR100={member.sfh_log_sfr_100myr_r0:+.2f}; "
                    f"PHZ={member.catalog_log_sfr_100myr:+.2f}\n"
                    f"log M★={member.log_stellar_mass:.2f}, z={member.redshift:.2f}", fontsize=8,
                )
                sfh_ax = fig.add_subplot(grid[1, 0 if member_index == 0 else 3])
                sfh_ax.plot(time, sfh_by_id[gid], color=PAIR_COLORS[member_index], lw=1.2)
                peak_fraction = time[np.argmax(sfh_by_id[gid])]
                sfh_ax.axvline(peak_fraction, color='tab:red', ls='--', lw=1)
                recent_fraction = min(0.1 / time_norm_by_id[gid], 1.0)
                sfh_ax.axvline(
                    recent_fraction, color='0.15', ls=':', lw=1.4,
                    label='100 Myr before observation',
                )
                sfh_ax.set(xlabel='Fractional lookback time', ylabel='Normalized SFH weight',
                           title=f"peak={member.sfh_peak_age_gyr:.2f} Gyr; "
                                 f"mean age={member.sfh_mass_weighted_age_gyr:.2f} Gyr")
                sfh_ax.text(
                    0.98, 0.96,
                    f"log SFR100\nSFH {member.sfh_log_sfr_100myr_r0:+.2f}\n"
                    f"PHZ {member.catalog_log_sfr_100myr:+.2f}\n"
                    f"Δ {member.sfh_log_sfr_100myr_r0-member.catalog_log_sfr_100myr:+.2f} dex",
                    transform=sfh_ax.transAxes, ha='right', va='top', fontsize=7.2,
                    bbox=dict(facecolor='white', edgecolor='0.75', alpha=.86, pad=2),
                )
                sfh_ax.legend(fontsize=6.5, loc='upper left')

            a, b = members.iloc[0], members.iloc[1]
            fig.suptitle(
                f'{experiment} pair {int(pair)}: matched recent activity, different history\n'
                f'Δlog SFR100={abs(a.sfh_log_sfr_100myr_r0-b.sfh_log_sfr_100myr_r0):.3f}, '
                f'Δlog M★={abs(a.log_stellar_mass-b.log_stellar_mass):.3f}, '
                f'Δz={abs(a.redshift-b.redshift):.3f}; '
                f'Δpeak={abs(a.sfh_peak_age_gyr-b.sfh_peak_age_gyr):.2f} Gyr, '
                f'Δmean age={abs(a.sfh_mass_weighted_age_gyr-b.sfh_mass_weighted_age_gyr):.2f} Gyr',
                fontsize=13,
            )
            pdf.savefig(fig, dpi=170)
            plt.close(fig)
    print(f'Wrote {len(groups)} paired track pages to {args.output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--tracks', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
