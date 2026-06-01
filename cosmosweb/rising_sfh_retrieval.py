"""
Rising SFH cross-modal retrieval.

For each randomly sampled rising-SFH query galaxy (QI < threshold):
  1. Score every galaxy by cosine similarity to the query in joint embedding space
  2. Bin galaxies into score tiers (high → low similarity)
  3. Per page:
       - query SFH (9 CIGALE bins) + query image
       - morphology property distributions across score tiers
       - representative image strips at each score tier

Usage:
    python -m cosmosweb.rising_sfh_retrieval \\
        --npz   /path/to/cosmosweb_umap_zoobot_v8.npz \\
        --h5    /path/to/cosmosweb_dataset_v6.h5 \\
        --output rising_sfh_retrieval.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

SFH_EPS = 1e-10

# Score tiers: (label, lower_percentile, upper_percentile, colour)
TIERS = [
    ('top 5%',    95, 100, '#d62728'),
    ('5–10%',     90,  95, '#ff7f0e'),
    ('10–20%',    80,  90, '#9467bd'),
    ('20–50%',    50,  80, '#2ca02c'),
    ('bottom 50%', 0,  50, '#1f77b4'),
]
# Only these tiers get image+SFH strips; the rest appear only in histograms
STRIP_TIERS = {'top 5%', '5–10%'}

N_STRIP = 5   # images per tier in the strip


# ── SFH helpers ───────────────────────────────────────────────────────────────

def compute_sfh_stats(h5_path: Path, h5_indices: np.ndarray) -> dict:
    """
    Returns dict with SFH scalar statistics:
      qi   = f(5-10%) - f(0-5%)           — negative = rising SFH
      f5   = f(0-5%)                       — fraction of SF in last 5% of Hubble time
      iX   = (w[0,X/2] - w[X/2,X]) / w[0,X]  — increase index (positive = rising)
      dX   = -iX                           — decline index (positive = declining)
    for X in (10, 20, 30, 50).
    """
    with h5py.File(h5_path, 'r') as f:
        sfh_log = f['sfh'][list(h5_indices)].astype(np.float64)
        t_frac  = (f['sfh_time_grid'][:].astype(np.float64)
                   if 'sfh_time_grid' in f
                   else np.linspace(0, 1, sfh_log.shape[1]))

    sfr = np.maximum(10.0 ** sfh_log - SFH_EPS, 0.0)
    sfr /= sfr.sum(axis=1, keepdims=True).clip(min=1e-10)

    f5       = sfr[:, t_frac <= 0.05].sum(axis=1)
    cumul_10 = sfr[:, t_frac <= 0.10].sum(axis=1)
    qi       = (cumul_10 - f5) - f5

    stats = dict(qi=qi, f5=f5)
    # cumulative recent fractions f(X%) = total SF in last X% of Hubble time
    for pct in (10, 20, 30, 50):
        stats[f'f{pct}'] = sfr[:, t_frac <= pct/100.0].sum(axis=1)
    for pct in (5, 10, 20, 30, 50):
        half    = pct / 200.0   # X/2 as fraction
        full    = pct / 100.0
        w_rec   = sfr[:, t_frac <= half].sum(axis=1)
        w_old   = sfr[:, (t_frac > half) & (t_frac <= full)].sum(axis=1)
        w_tot   = sfr[:, t_frac <= full].sum(axis=1)
        d       = np.full(len(sfr), np.nan)
        nz      = w_tot > 0
        d[nz]   = (w_old[nz] - w_rec[nz]) / w_tot[nz]
        stats[f'd{pct}'] =  d
        stats[f'i{pct}'] = -d
    return stats


def farthest_point_sample(emb: np.ndarray, n: int,
                          rng: np.random.Generator) -> np.ndarray:
    """
    Greedy farthest-point sampling in embedding space (cosine distance).
    Returns indices into emb (local, not global).
    """
    N = len(emb)
    first     = int(rng.integers(N))
    selected  = [first]
    min_dists = 1.0 - (emb @ emb[first])   # cosine distance to first point
    for _ in range(n - 1):
        idx = int(np.argmax(min_dists))
        selected.append(idx)
        dists     = 1.0 - (emb @ emb[idx])
        min_dists = np.minimum(min_dists, dists)
    return np.array(selected)


def load_batch(h5_path: Path, h5_idx_list: list[int]) -> dict:
    """Batch-load images, SFH bins, redshifts (h5py requires sorted indices)."""
    orig  = np.array(h5_idx_list, dtype=int)
    si    = np.argsort(orig)
    ri    = np.argsort(si)
    sidx  = orig[si].tolist()

    with h5py.File(h5_path, 'r') as f:
        imgs      = f['images'][sidx, 1].astype(np.float32)
        bins_log  = f['sfh_bins_log'][sidx].astype(np.float64)
        times_myr = f['sfh_times_myr'][sidx].astype(np.float64)
        t_norms   = f['sfh_time_norm'][sidx].astype(np.float64)
        zs        = f['redshift'][sidx].astype(np.float32)

    return dict(imgs=imgs[ri], bins_log=bins_log[ri], times_myr=times_myr[ri],
                t_norms=t_norms[ri], zs=zs[ri])


# ── plotting helpers ──────────────────────────────────────────────────────────

def plot_sfh_bars(ax, bins_log, times_myr, t_norm, qi, z):
    w = np.maximum(10.0 ** bins_log - SFH_EPS, SFH_EPS)
    w = w / w.sum()
    tc = times_myr / t_norm
    edges = np.empty(len(tc) + 1)
    edges[0]    = 0.0
    edges[1:-1] = 0.5 * (tc[:-1] + tc[1:])
    edges[-1]   = 1.0
    ax.bar(edges[:-1], w, width=np.diff(edges), align='edge',
           color='steelblue', alpha=0.8, edgecolor='steelblue', linewidth=0.5)
    ax.set_yscale('log')
    ax.set_xlim(0, 1)
    ax.tick_params(labelsize=5)
    ax.set_xlabel('Fractional lookback time', fontsize=5)
    ax.set_ylabel('SFH weight', fontsize=5)
    ax.set_title(f'z={z:.2f}  QI={qi:.3f}', fontsize=6, pad=2)


def plot_stamp(ax, img, title='', color=None):
    vmin, vmax = np.percentile(img, [0.5, 99.5])
    ax.imshow(img, origin='lower', cmap='gray',
              vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=4, pad=1,
                     color=color if color else 'black')


def plot_morph_distributions(axes_row, tier_masks, prop_arrays):
    """
    prop_arrays: list of (label, values_array, log_x) tuples.
    tier_masks:  list of boolean arrays, one per TIER.
    """
    for ax, (label, vals, log_x) in zip(axes_row, prop_arrays):
        finite = np.isfinite(vals)
        vmin = np.nanpercentile(vals[finite], 1) if finite.any() else 0
        vmax = np.nanpercentile(vals[finite], 99) if finite.any() else 1
        for (tlabel, _, _, color), mask in zip(TIERS, tier_masks):
            m = mask & finite
            if m.sum() < 5:
                continue
            if log_x:
                bins = np.geomspace(max(vmin, 1e-2), max(vmax, 1e-1), 21)
            else:
                bins = np.linspace(vmin, vmax, 21)
            ax.hist(vals[m], bins=bins, density=True, histtype='step',
                    color=color, linewidth=1.2, label=tlabel)
        if log_x:
            ax.set_xscale('log')
        ax.set_xlabel(label, fontsize=5)
        ax.tick_params(labelsize=4)
        ax.set_yticks([])
        if ax is axes_row[0]:
            ax.legend(fontsize=4, loc='upper right',
                      framealpha=0.5, handlelength=1)


# ── KS-based cluster merging ─────────────────────────────────────────────────

KS_PROPS = [
    ('log M★',        'log_mass',         False),
    ('Redshift z',    'redshifts',        False),
    ('P(Early)',      None,               False),   # computed below
    ('P(Late)',       None,               False),   # computed below
    ('P(Disturbed)',  'binary_disturbed', False),
    ('Axis ratio',    'axratio_sersic',   False),
    ('Sersic n',      'sersic',           True),    # log-transform before KS
]


def build_ks_props(npz) -> list[tuple[str, np.ndarray]]:
    """Return (label, values) for the 7 properties used in KS merging."""
    def get(key):
        return npz[key].astype(float) if key in npz else None

    out = []
    for label, key, log_x in KS_PROPS:
        if key is not None:
            v = get(key)
        elif label == 'P(Early)':
            ell, s0 = get('family_elliptical'), get('family_s0')
            v = np.clip(ell + s0, 0, 1) if (ell is not None and s0 is not None) else None
        elif label == 'P(Late)':
            ed, ld = get('family_early_disk'), get('family_late_disk')
            v = np.clip(ed + ld, 0, 1) if (ed is not None and ld is not None) else None
        else:
            v = None
        if v is not None:
            if log_x:
                v = np.log10(np.clip(v, 1e-2, None))
            out.append((label, v))
    return out


def ks_separability_matrix(cluster_labels: np.ndarray,
                            ks_props: list[tuple[str, np.ndarray]],
                            rising_idx: np.ndarray
                            ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Compute pairwise max-KS matrix across all properties for the rising sample.
    Returns:
      ks_max  : (n_clusters, n_clusters) max KS stat across properties
      ks_all  : (n_clusters, n_clusters, n_props) per-property KS stats
      prop_labels : list of property names
    """
    from scipy.stats import ks_2samp

    unique = sorted(c for c in set(cluster_labels) if c >= 0)
    n = len(unique)
    n_props = len(ks_props)
    ks_all = np.zeros((n, n, n_props))

    for pi, (_, vals) in enumerate(ks_props):
        pv = vals[rising_idx]   # values for rising-SFH galaxies
        for i, ci in enumerate(unique):
            for j, cj in enumerate(unique):
                if i >= j:
                    continue
                a = pv[cluster_labels == ci]
                b = pv[cluster_labels == cj]
                a = a[np.isfinite(a)]
                b = b[np.isfinite(b)]
                if len(a) >= 5 and len(b) >= 5:
                    stat, _ = ks_2samp(a, b)
                else:
                    stat = 0.0
                ks_all[i, j, pi] = stat
                ks_all[j, i, pi] = stat

    ks_max  = ks_all.max(axis=2)
    ks_mean = ks_all.mean(axis=2)
    np.fill_diagonal(ks_max,  1.0)
    np.fill_diagonal(ks_mean, 1.0)
    return ks_max, ks_mean, ks_all, [p[0] for p in ks_props]


def merge_by_ks(cluster_labels: np.ndarray,
                ks_max: np.ndarray,
                threshold: float) -> np.ndarray:
    """
    Greedily merge cluster pairs whose max-KS < threshold.
    Uses union-find so transitive merges propagate correctly.
    Returns new cluster_labels array (re-indexed from 0, noise = -1 preserved).
    """
    unique = sorted(c for c in set(cluster_labels) if c >= 0)
    n = len(unique)

    # union-find
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        parent[find(x)] = find(y)

    for i in range(n):
        for j in range(i + 1, n):
            if ks_max[i, j] < threshold:
                union(i, j)
                print(f'  merging cluster {unique[i]+1} + {unique[j]+1}  '
                      f'(max KS={ks_max[i,j]:.3f} < {threshold})')

    # remap labels
    roots = [find(i) for i in range(n)]
    unique_roots = sorted(set(roots))
    root_to_new = {r: new for new, r in enumerate(unique_roots)}

    new_labels = np.full_like(cluster_labels, -1)
    for i, c in enumerate(unique):
        new_labels[cluster_labels == c] = root_to_new[find(i)]
    return new_labels


def plot_ks_heatmap(ks_max, ks_mean, ks_all, prop_labels, unique_clusters,
                    threshold, merge_threshold):
    """Return a figure with the KS separability heatmap + per-property breakdown."""
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    n = len(unique_clusters)
    n_props = len(prop_labels)
    # 2 summary panels (max + mean) + per-property panels
    n_cols = 2 + n_props
    fig, axes = plt.subplots(1, n_cols,
                              figsize=(2.2 * n_cols, 2.2 + 0.4 * n),
                              constrained_layout=True)

    clabels = [f'C{c+1}' for c in unique_clusters]
    norm = Normalize(vmin=0, vmax=1)
    cmap = plt.cm.RdYlGn

    def _fill(ax, mat, title, star_mask=None):
        ax.imshow(mat, cmap=cmap, norm=norm, aspect='auto')
        ax.set_xticks(range(n)); ax.set_xticklabels(clabels, fontsize=7)
        ax.set_yticks(range(n)); ax.set_yticklabels(clabels, fontsize=7)
        ax.set_title(title, fontsize=8)
        for i in range(n):
            for j in range(n):
                v = mat[i, j]
                txt_col = 'white' if v < 0.4 or v > 0.85 else 'black'
                marker = '★' if (star_mask is not None and star_mask[i, j]) else ''
                ax.text(j, i, f'{v:.2f}{marker}', ha='center', va='center',
                        fontsize=6, color=txt_col)
        plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.6)

    star_mask = (np.arange(n)[:, None] != np.arange(n)[None, :]) & (ks_mean < merge_threshold)
    _fill(axes[0], ks_max,  'max KS\n(all props)',  star_mask=star_mask)
    _fill(axes[1], ks_mean, 'mean KS\n(all props)', star_mask=star_mask)

    for pi, (pname, ax) in enumerate(zip(prop_labels, axes[2:])):
        mat = ks_all[:, :, pi].copy()
        np.fill_diagonal(mat, 1.0)
        _fill(ax, mat, pname)

    fig.suptitle(
        f'KS separability between clusters  '
        f'(★ = max KS < merge threshold {merge_threshold})',
        fontsize=9,
    )
    return fig


# ── main ──────────────────────────────────────────────────────────────────────

def build_prop_arrays(npz) -> list[tuple[str, np.ndarray, bool]]:
    """
    Returns list of (label, values, log_x) for histogram panels.
    Computes composite morphology quantities from npz arrays.
    """
    def get(key):
        return npz[key].astype(float) if key in npz else None

    props = []

    # P(Disturbed)
    v = get('binary_disturbed')
    if v is not None:
        props.append(('P(Disturbed)', v, False))

    # P(Early) = Ell + S0
    ell = get('family_elliptical')
    s0  = get('family_s0')
    if ell is not None and s0 is not None:
        props.append(('P(Early: Ell+S0)', np.clip(ell + s0, 0, 1), False))

    # P(Late) = early disk + late disk
    edsk = get('family_early_disk')
    ldsk = get('family_late_disk')
    if edsk is not None and ldsk is not None:
        props.append(('P(Late: disks)', np.clip(edsk + ldsk, 0, 1), False))

    # Redshift
    v = get('redshifts')
    if v is not None:
        props.append(('Redshift z', v, False))

    # Stellar mass
    v = get('log_mass')
    if v is not None:
        props.append(('log M★', v, False))

    # Sersic n (log x-axis)
    v = get('sersic')
    if v is not None:
        props.append(('Sersic n', v, True))

    # Axis ratio
    v = get('axratio_sersic')
    if v is not None:
        props.append(('Axis ratio b/a', v, False))

    return props


def mean_intra_cosine(emb: np.ndarray) -> float:
    """
    Mean pairwise cosine similarity within a set of L2-normalised vectors.
    Uses the identity: mean_pairwise = (n * ||mean_vec||^2 - 1) / (n - 1)
    O(d) computation — no pairwise matrix needed.
    """
    n = len(emb)
    if n < 2:
        return np.nan
    mean_vec  = emb.mean(axis=0)
    mean_norm2 = float(mean_vec @ mean_vec)
    return (n * mean_norm2 - 1.0) / (n - 1.0)


def cluster_embedding_cohesion(cluster_labels: np.ndarray,
                                rising_idx: np.ndarray,
                                embeddings: dict[str, np.ndarray],
                                rng: np.random.Generator,
                                n_random: int = 10) -> plt.Figure:
    """
    For each cluster compute mean intra-cluster cosine similarity in each
    embedding space and compare to a random-subsample baseline.

    embeddings: dict of {label: L2-normalised (N, D) array}
    Returns a figure with one grouped bar chart per embedding space.
    """
    unique_cls   = sorted(c for c in set(cluster_labels) if c >= 0)
    n_cls        = len(unique_cls)
    emb_names    = list(embeddings.keys())
    n_emb        = len(emb_names)
    cmap_cls     = plt.cm.get_cmap('tab10', max(n_cls, 1))

    # per-cluster cohesion
    cohesion = {name: [] for name in emb_names}
    sizes    = []
    for c in unique_cls:
        m   = cluster_labels == c
        idx = rising_idx[m]
        sizes.append(int(m.sum()))
        for name, emb in embeddings.items():
            cohesion[name].append(mean_intra_cosine(emb[idx]))

    # random baseline: n_random draws of size=median_cluster_size from rising_idx
    median_size = int(np.median(sizes))
    baseline = {name: [] for name in emb_names}
    for _ in range(n_random):
        rand_idx = rng.choice(rising_idx, size=median_size, replace=False)
        for name, emb in embeddings.items():
            baseline[name].append(mean_intra_cosine(emb[rand_idx]))
    baseline_mean = {name: float(np.mean(baseline[name])) for name in emb_names}
    baseline_std  = {name: float(np.std(baseline[name]))  for name in emb_names}

    # print summary
    print('\nIntra-cluster cosine similarity (higher = more cohesive):')
    header = f"  {'cluster':>9}  {'n':>6}  " + '  '.join(f'{n:>10}' for n in emb_names)
    print(header)
    for i, c in enumerate(unique_cls):
        row = f"  cluster {c+1:>2}  {sizes[i]:>6}  "
        row += '  '.join(f'{cohesion[n][i]:>10.4f}' for n in emb_names)
        print(row)
    print(f"  {'random':>9}  {median_size:>6}  "
          + '  '.join(f'{baseline_mean[n]:>10.4f}±{baseline_std[n]:.4f}'
                      for n in emb_names))

    # figure
    fig, axes = plt.subplots(1, n_emb, figsize=(3.5 * n_emb, 4),
                              constrained_layout=True)
    if n_emb == 1:
        axes = [axes]

    x = np.arange(n_cls)
    for ax, name in zip(axes, emb_names):
        vals = np.array(cohesion[name])
        bars = ax.bar(x, vals,
                      color=[cmap_cls(i) for i in range(n_cls)],
                      alpha=0.85, edgecolor='k', linewidth=0.5)
        # random baseline band
        bm, bs = baseline_mean[name], baseline_std[name]
        ax.axhline(bm,  color='black', lw=1.5, ls='--', label='random baseline')
        ax.axhspan(bm - bs, bm + bs, alpha=0.12, color='black')
        ax.set_xticks(x)
        ax.set_xticklabels([f'C{c+1}\n(n={sizes[i]})' for i, c in enumerate(unique_cls)],
                           fontsize=7)
        ax.set_ylabel('Mean intra-cluster cosine sim.', fontsize=8)
        ax.set_title(f'{name} space', fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)

    fig.suptitle('Cluster cohesion in embedding spaces\n'
                 '(dashed = random subsample baseline ± 1σ)', fontsize=9)
    return fig


def plot_main_sequence(npz, cluster_labels, rising_idx,
                       cmap_cls, unique_cls,
                       z_bins=None, sel_label=''):
    """
    Multi-panel M★ vs SFR plot in redshift bins.
    Full sample in light gray; cluster members coloured by cluster.
    """
    log_mass  = npz['log_mass'].astype(float)  if 'log_mass'  in npz else None
    log_sfr   = npz['log_sfr_100myr'].astype(float) if 'log_sfr_100myr' in npz else \
                npz['log_sfr'].astype(float)         if 'log_sfr'       in npz else None
    redshifts = npz['redshifts'].astype(float) if 'redshifts' in npz else None

    if log_mass is None or log_sfr is None or redshifts is None:
        print('Skipping main-sequence plot: log_mass / log_sfr / redshifts not in npz.')
        return None

    if z_bins is None:
        z_bins = [0.0, 0.5, 1.0, 1.5, 2.5, 4.0, 8.0]

    # drop bins with fewer than 10 rising galaxies
    active_bins = []
    for zlo, zhi in zip(z_bins[:-1], z_bins[1:]):
        z_r = redshifts[rising_idx]
        if ((z_r >= zlo) & (z_r < zhi)).sum() >= 10:
            active_bins.append((zlo, zhi))
    if not active_bins:
        return None

    n_bins = len(active_bins)
    ncols  = min(n_bins, 4)
    nrows  = (n_bins + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.0 * ncols, 3.8 * nrows),
                              constrained_layout=True, squeeze=False)
    axes = axes.flatten()

    # global axis limits from full sample (finite values only)
    fin = np.isfinite(log_mass) & np.isfinite(log_sfr)
    xlim = (np.percentile(log_mass[fin], 1),  np.percentile(log_mass[fin], 99))
    ylim = (np.percentile(log_sfr[fin],  1),  np.percentile(log_sfr[fin],  99))

    for bi, (zlo, zhi) in enumerate(active_bins):
        ax = axes[bi]

        # full sample background
        z_mask_all = ((redshifts >= zlo) & (redshifts < zhi) &
                      np.isfinite(log_mass) & np.isfinite(log_sfr))
        ax.scatter(log_mass[z_mask_all], log_sfr[z_mask_all],
                   c='#cccccc', s=1, alpha=0.4, linewidths=0,
                   rasterized=True, zorder=1)

        # cluster members
        for i, c in enumerate(unique_cls):
            m   = cluster_labels == c
            idx = rising_idx[m]
            z_r = redshifts[idx]
            c_mask = ((z_r >= zlo) & (z_r < zhi) &
                      np.isfinite(log_mass[idx]) & np.isfinite(log_sfr[idx]))
            if c_mask.sum() < 3:
                continue
            ax.scatter(log_mass[idx[c_mask]], log_sfr[idx[c_mask]],
                       c=[cmap_cls(i)], s=10, alpha=0.85, linewidths=0,
                       label=f'C{c+1} (n={int(c_mask.sum())})',
                       rasterized=True, zorder=2 + i)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('log M★ [M☉]', fontsize=8)
        ax.set_ylabel('log SFR$_{100}$ [M☉ yr⁻¹]', fontsize=8)
        ax.set_title(f'{zlo:.1f} ≤ z < {zhi:.1f}', fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=5, loc='upper left', framealpha=0.6,
                  markerscale=2, handlelength=1)

    for bi in range(n_bins, len(axes)):
        axes[bi].set_visible(False)

    fig.suptitle(f'Star-forming main sequence by redshift bin\n'
                 f'(gray = full sample   |   coloured = clusters   |   {sel_label})',
                 fontsize=9)
    return fig


def arcsec_to_kpc(radius_arcsec: np.ndarray, redshifts: np.ndarray) -> np.ndarray:
    """Convert Sérsic radius from arcsec to kpc using astropy cosmology."""
    try:
        from astropy.cosmology import FlatLambdaCDM
        import astropy.units as u
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        z_vals  = np.clip(redshifts, 1e-3, None)
        da_mpc  = cosmo.angular_diameter_distance(z_vals).to(u.Mpc).value  # Mpc
        kpc_per_arcsec = da_mpc * 1e3 / 206.265   # kpc/arcsec  (1 rad = 206265 arcsec)
        return radius_arcsec * kpc_per_arcsec
    except Exception as e:
        print(f'  arcsec→kpc conversion failed ({e}); using arcsec')
        return radius_arcsec


def plot_size_mass(npz, cluster_labels, rising_idx,
                   cmap_cls, unique_cls,
                   z_bins=None, sel_label=''):
    """
    Multi-panel log Re (kpc) vs log M★ plot in redshift bins.
    Full sample in light gray; cluster members coloured by cluster.
    """
    log_mass  = npz['log_mass'].astype(float)     if 'log_mass'     in npz else None
    re_arcsec = npz['radius_sersic'].astype(float) if 'radius_sersic' in npz else None
    redshifts = npz['redshifts'].astype(float)    if 'redshifts'    in npz else None

    if log_mass is None or re_arcsec is None or redshifts is None:
        print('Skipping size-mass plot: log_mass / radius_sersic / redshifts not in npz.')
        return None

    re_kpc   = arcsec_to_kpc(re_arcsec, redshifts)
    log_re   = np.where(re_kpc > 0, np.log10(re_kpc), np.nan)

    if z_bins is None:
        z_bins = [0.0, 0.5, 1.0, 1.5, 2.5, 4.0, 8.0]

    active_bins = []
    for zlo, zhi in zip(z_bins[:-1], z_bins[1:]):
        z_r = redshifts[rising_idx]
        if ((z_r >= zlo) & (z_r < zhi)).sum() >= 10:
            active_bins.append((zlo, zhi))
    if not active_bins:
        return None

    n_bins = len(active_bins)
    ncols  = min(n_bins, 4)
    nrows  = (n_bins + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.0 * ncols, 3.8 * nrows),
                              constrained_layout=True, squeeze=False)
    axes = axes.flatten()

    fin  = np.isfinite(log_mass) & np.isfinite(log_re)
    xlim = (np.percentile(log_mass[fin], 1),  np.percentile(log_mass[fin], 99))
    ylim = (np.percentile(log_re[fin],   1),  np.percentile(log_re[fin],   99))

    for bi, (zlo, zhi) in enumerate(active_bins):
        ax = axes[bi]

        z_mask_all = ((redshifts >= zlo) & (redshifts < zhi) &
                      np.isfinite(log_mass) & np.isfinite(log_re))
        ax.scatter(log_mass[z_mask_all], log_re[z_mask_all],
                   c='#cccccc', s=1, alpha=0.4, linewidths=0,
                   rasterized=True, zorder=1)

        for i, c in enumerate(unique_cls):
            m   = cluster_labels == c
            idx = rising_idx[m]
            z_r = redshifts[idx]
            c_mask = ((z_r >= zlo) & (z_r < zhi) &
                      np.isfinite(log_mass[idx]) & np.isfinite(log_re[idx]))
            if c_mask.sum() < 3:
                continue
            ax.scatter(log_mass[idx[c_mask]], log_re[idx[c_mask]],
                       c=[cmap_cls(i)], s=10, alpha=0.85, linewidths=0,
                       label=f'C{c+1} (n={int(c_mask.sum())})',
                       rasterized=True, zorder=2 + i)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('log M★ [M☉]', fontsize=8)
        ax.set_ylabel('log R$_e$ [kpc]', fontsize=8)
        ax.set_title(f'{zlo:.1f} ≤ z < {zhi:.1f}', fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=5, loc='upper left', framealpha=0.6,
                  markerscale=2, handlelength=1)

    for bi in range(n_bins, len(axes)):
        axes[bi].set_visible(False)

    fig.suptitle(f'Size–mass relation by redshift bin\n'
                 f'(gray = full sample   |   coloured = clusters   |   {sel_label})',
                 fontsize=9)
    return fig


def cluster_stability(emb: np.ndarray, k_range: range,
                      n_runs: int = 10) -> list[float]:
    """
    For each k, run k-means n_runs times and return the mean pairwise ARI
    across runs.  High ARI = stable clusters at this k.
    emb should be L2-normalised (cosine ≈ euclidean on unit sphere).
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    stabilities = []
    for k in k_range:
        print(f'  k={k} …', end=' ', flush=True)
        labels_runs = []
        for run in range(n_runs):
            km = KMeans(n_clusters=k, random_state=run, n_init=3, max_iter=100)
            labels_runs.append(km.fit_predict(emb))
        aris = [adjusted_rand_score(labels_runs[i], labels_runs[j])
                for i in range(n_runs) for j in range(i + 1, n_runs)]
        mean_ari = float(np.mean(aris))
        print(f'ARI={mean_ari:.3f}')
        stabilities.append(mean_ari)
    return stabilities


def sample_cluster_representatives(emb: np.ndarray, k: int,
                                   random_state: int = 42
                                   ) -> tuple[np.ndarray, np.ndarray]:
    """
    Run k-means with n_init=20 for a stable solution.
    Returns (rep_local_indices, labels) where rep_local_indices[c] is the
    index (into emb) of the galaxy closest to cluster c's centroid.
    """
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, random_state=random_state, n_init=20)
    labels    = km.fit_predict(emb)
    centroids = km.cluster_centers_
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True).clip(1e-10)

    reps = []
    for c in range(k):
        idx_in_cluster = np.where(labels == c)[0]
        sims = emb[idx_in_cluster] @ centroids[c]
        reps.append(idx_in_cluster[np.argmax(sims)])
    return np.array(reps), labels


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--npz',          type=Path, required=True)
    p.add_argument('--h5',           type=Path, required=True)
    p.add_argument('--output',       type=Path,
                   default=Path('rising_sfh_retrieval.pdf'))
    p.add_argument('--n_queries',    type=int,   default=20)
    p.add_argument('--qi_threshold', type=float, default=None,
                   help='Quenching index upper bound (QI < value). '
                        'Default -0.4 if --qi_min not set, else no upper bound.')
    p.add_argument('--qi_min',       type=float, default=None,
                   help='Quenching index lower bound (QI >= value). '
                        'E.g. 0.1 to select quenching galaxies.')
    p.add_argument('--f5_min',       type=float, default=0.0,
                   help='Minimum f(0-5%%) fraction; set to 0.5 to exclude '
                        'rejuvenated galaxies with low absolute recent SF')
    # iX / dX selection criteria
    for _pct in (5, 10, 20, 30, 50):
        p.add_argument(f'--i{_pct}_min', type=float, default=None,
                       help=f'Minimum i{_pct} (increase index over last {_pct}%%)')
        p.add_argument(f'--d{_pct}_max', type=float, default=None,
                       help=f'Maximum d{_pct} (decline index over last {_pct}%%)')
    for _pct in (10, 20, 30, 50):
        p.add_argument(f'--f{_pct}_min', type=float, default=None,
                       help=f'Minimum f({_pct}%%) = cumulative SF fraction in last {_pct}%% of Hubble time')
    p.add_argument('--restrict',      action='store_true',
                   help='Restrict neighbour pool to the selected galaxies only')
    p.add_argument('--retrieval',    choices=['joint', 'cross_modal'],
                   default='joint',
                   help='joint: score by joint_emb similarity (default); '
                        'cross_modal: score all img_emb by sfh_emb[query] similarity')
    p.add_argument('--emb_space',   choices=['joint', 'sfh', 'img'], default='joint',
                   help='Embedding space used for clustering: '
                        'joint (default) = img+sfh; sfh = SFH encoder only; '
                        'img = image encoder only')
    # ── query sampling strategy ───────────────────────────────────────────────
    p.add_argument('--sampling',     choices=['fps', 'kmeans', 'hdbscan'], default='fps',
                   help='fps: farthest-point sampling; kmeans: one rep per cluster; '
                        'hdbscan: density clusters, one rep per cluster')
    p.add_argument('--cluster_k',    type=int,   default=0,
                   help='k-means k (kmeans mode). 0 = auto-detect via stability')
    p.add_argument('--k_min',        type=int,   default=2)
    p.add_argument('--k_max',        type=int,   default=15)
    p.add_argument('--n_stability_runs', type=int, default=10,
                   help='k-means runs per k for stability estimation')
    p.add_argument('--min_cluster_size', type=int, default=0,
                   help='HDBSCAN min_cluster_size (0 = 5%% of rising sample)')
    p.add_argument('--min_samples',      type=int, default=5,
                   help='HDBSCAN min_samples (controls noise sensitivity)')
    p.add_argument('--pca_components',  type=int, default=0,
                   help='PCA dimensionality reduction before clustering '
                        '(0 = disabled, e.g. 50)')
    p.add_argument('--merge_threshold', type=float, default=0.0,
                   help='Merge cluster pairs whose mean KS stat < this value '
                        '(0 = disabled). E.g. 0.2 merges indistinguishable clusters.')
    p.add_argument('--n_per_cluster', type=int,   default=1,
                   help='Queries per cluster (kmeans mode): 1 = centroid rep only, '
                        '>1 = centroid rep + random extras from the same cluster')
    p.add_argument('--n_strip',      type=int,   default=N_STRIP,
                   help='Images per score tier in the strip')
    p.add_argument('--seed',         type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng  = np.random.default_rng(args.seed)

    # ── load embeddings ───────────────────────────────────────────────────────
    npz       = np.load(args.npz, allow_pickle=True)
    img_emb   = npz['img_emb'].astype(np.float32)
    sfh_emb   = npz['sfh_emb'].astype(np.float32)
    img_emb   = img_emb / np.linalg.norm(img_emb, axis=1, keepdims=True).clip(1e-10)
    sfh_emb   = sfh_emb / np.linalg.norm(sfh_emb, axis=1, keepdims=True).clip(1e-10)
    joint_emb = img_emb + sfh_emb
    joint_emb = joint_emb / np.linalg.norm(joint_emb, axis=1, keepdims=True).clip(1e-10)
    h5_indices = npz['h5_indices'].astype(int)
    galaxy_ids = npz['galaxy_ids']
    N          = len(h5_indices)

    # embedding space for clustering
    if args.emb_space == 'sfh':
        clust_emb_full = sfh_emb
    elif args.emb_space == 'img':
        clust_emb_full = img_emb
    else:
        clust_emb_full = joint_emb
    print(f'Clustering embedding space: {args.emb_space}')

    # ── SFH stats & selection ─────────────────────────────────────────────────
    print(f'Computing SFH statistics for {N} galaxies…')
    sfh_stats = compute_sfh_stats(args.h5, h5_indices)
    qi = sfh_stats['qi']
    f5 = sfh_stats['f5']

    # resolve defaults: if no selection given, use rising-SFH QI default
    qi_max = args.qi_threshold
    qi_min = args.qi_min
    has_ix_dx = any(
        getattr(args, f'i{p}_min', None) is not None or
        getattr(args, f'd{p}_max', None) is not None
        for p in (5, 10, 20, 30, 50)
    ) or any(
        getattr(args, f'f{p}_min', None) is not None
        for p in (10, 20, 30, 50)
    )
    if qi_max is None and qi_min is None and not has_ix_dx:
        qi_max = -0.4

    sel_mask = np.ones(N, dtype=bool)
    parts    = []
    if qi_min is not None:
        sel_mask &= qi >= qi_min
        parts.append(f'QI ≥ {qi_min}')
    if qi_max is not None:
        sel_mask &= qi < qi_max
        parts.append(f'QI < {qi_max}')
    if args.f5_min > 0:
        sel_mask &= f5 >= args.f5_min
        parts.append(f'f(5%) ≥ {args.f5_min}')
    for pct in (5, 10, 20, 30, 50):
        imin = getattr(args, f'i{pct}_min', None)
        dmax = getattr(args, f'd{pct}_max', None)
        if imin is not None:
            v = sfh_stats[f'i{pct}']
            sel_mask &= np.where(np.isfinite(v), v >= imin, False)
            parts.append(f'i{pct} ≥ {imin}')
        if dmax is not None:
            v = sfh_stats[f'd{pct}']
            sel_mask &= np.where(np.isfinite(v), v <= dmax, False)
            parts.append(f'd{pct} ≤ {dmax}')
    for pct in (10, 20, 30, 50):
        fmin = getattr(args, f'f{pct}_min', None)
        if fmin is not None:
            v = sfh_stats[f'f{pct}']
            sel_mask &= np.where(np.isfinite(v), v >= fmin, False)
            parts.append(f'f({pct}%) ≥ {fmin}')
    sel_label = '  &  '.join(parts) if parts else 'all'

    # keep legacy name "rising_idx" / "rising_mask" internally
    rising_mask = sel_mask
    rising_idx  = np.where(rising_mask)[0]
    print(f'Selection ({sel_label}): {len(rising_idx)} / {N}')

    if len(rising_idx) == 0:
        raise RuntimeError('No galaxies match selection; adjust thresholds.')

    rising_emb       = joint_emb[rising_idx]            # always used for retrieval scoring
    rising_emb_clust_base = clust_emb_full[rising_idx]  # used for clustering

    pca_model = None
    if args.pca_components > 0:
        from sklearn.decomposition import PCA
        n_comp = min(args.pca_components, rising_emb_clust_base.shape[1],
                     rising_emb_clust_base.shape[0])
        pca_model = PCA(n_components=n_comp, random_state=args.seed)
        rising_emb_clust = pca_model.fit_transform(rising_emb_clust_base)
        var_explained = pca_model.explained_variance_ratio_.sum()
        print(f'PCA: {rising_emb_clust_base.shape[1]}D → {n_comp}D  '
              f'(cumulative variance explained: {var_explained:.1%})')
    else:
        rising_emb_clust = rising_emb_clust_base

    if args.sampling == 'fps':
        n_queries = min(args.n_queries, len(rising_idx))
        fps_local = farthest_point_sample(rising_emb, n_queries, rng)
        query_idx      = rising_idx[fps_local]
        cluster_labels = None
        query_cluster_ids = list(range(n_queries))
        print(f'Selected {n_queries} queries via farthest-point sampling')

    elif args.sampling == 'hdbscan':
        from hdbscan import HDBSCAN

        min_cs = args.min_cluster_size if args.min_cluster_size > 0 \
                 else max(5, int(0.05 * len(rising_idx)))
        print(f'HDBSCAN clustering  min_cluster_size={min_cs}  '
              f'min_samples={args.min_samples}…')
        hdb = HDBSCAN(min_cluster_size=min_cs, min_samples=args.min_samples,
                      metric='euclidean')
        hdb_labels = hdb.fit_predict(rising_emb_clust)

        unique_clusters = sorted(c for c in set(hdb_labels) if c >= 0)
        n_noise         = int((hdb_labels == -1).sum())
        best_k          = len(unique_clusters)
        print(f'Found {best_k} clusters, {n_noise} noise points')
        cluster_labels  = hdb_labels   # (M,) — noise = -1
        stabilities    = None
        _stability_fig = None
        # query list built in the shared KS block below

    else:   # kmeans
        k_range = range(args.k_min, args.k_max + 1)

        if args.cluster_k > 0:
            best_k = args.cluster_k
            stabilities = None
            print(f'Using k={best_k} (manually set)')
        else:
            print(f'Stability analysis k={args.k_min}…{args.k_max} '
                  f'({args.n_stability_runs} runs each)…')
            stabilities = cluster_stability(rising_emb_clust, k_range,
                                            n_runs=args.n_stability_runs)
            # choose largest k before ARI first drops below 0.8 × its max
            ari_arr   = np.array(stabilities)
            threshold = 0.8 * ari_arr.max()
            first_unstable = next(
                (k for k, a in zip(k_range, stabilities) if a < threshold),
                None,
            )
            if first_unstable is not None:
                best_k = first_unstable - 1
            else:
                best_k = max(k_range)
            print(f'Auto-selected k={best_k}  '
                  f'(ARI threshold={threshold:.3f}, '
                  f'first unstable k={first_unstable})')

        print(f'Final k-means clustering with k={best_k}…')
        rep_local, cluster_labels_local = sample_cluster_representatives(
            rising_emb_clust, best_k, random_state=args.seed)
        cluster_labels = cluster_labels_local   # (M,) labels for rising galaxies
        # query list built in the shared KS block below

        # ── stability page (kmeans only) ──────────────────────────────────────
        if stabilities is not None:
            fig_stab, ax_stab = plt.subplots(figsize=(6, 3.5))
            ax_stab.plot(list(k_range), stabilities, 'o-', color='steelblue')
            ax_stab.axvline(best_k, color='firebrick', lw=1.5, ls='--',
                            label=f'chosen k={best_k}')
            ax_stab.set_xlabel('Number of clusters k', fontsize=9)
            ax_stab.set_ylabel('Mean pairwise ARI', fontsize=9)
            pca_str = f' (PCA {pca_model.n_components_}D)' if pca_model else ' (256D)'
            ax_stab.set_title(f'k-means stability across runs{pca_str}', fontsize=10)
            ax_stab.legend(fontsize=8)
            ax_stab.set_ylim(0, 1)
            fig_stab.tight_layout()
            _stability_fig = fig_stab   # save for first PDF page
        else:
            _stability_fig = None

    # ── KS separability + optional merging (kmeans & hdbscan) ────────────────
    ks_fig = None
    if args.sampling in ('kmeans', 'hdbscan') and cluster_labels is not None:
        ks_props  = build_ks_props(npz)
        unique_cls_pre = sorted(c for c in set(cluster_labels) if c >= 0)
        ks_max, ks_mean, ks_all, prop_labels = ks_separability_matrix(
            cluster_labels, ks_props, rising_idx)

        ks_fig = plot_ks_heatmap(ks_max, ks_mean, ks_all, prop_labels, unique_cls_pre,
                                  threshold=None,
                                  merge_threshold=args.merge_threshold)

        if args.merge_threshold > 0:
            print(f'Merging clusters with mean KS < {args.merge_threshold}…')
            cluster_labels = merge_by_ks(cluster_labels, ks_mean,
                                          args.merge_threshold)
            best_k = len(set(cluster_labels) - {-1})
            print(f'Clusters after merging: {best_k}')

        # rebuild query list from (possibly merged) cluster_labels
        unique_cls = sorted(c for c in set(cluster_labels) if c >= 0)
        query_local_list  = []
        query_cluster_ids = []
        for c in unique_cls:
            members  = np.where(cluster_labels == c)[0]
            centroid = rising_emb_clust[members].mean(axis=0)
            centroid /= (np.linalg.norm(centroid) + 1e-10)
            dists    = np.linalg.norm(rising_emb_clust[members] - centroid, axis=1)
            rep      = members[np.argmin(dists)]
            extras   = [m for m in members if m != rep]
            n_extra  = min(args.n_per_cluster - 1, len(extras))
            chosen   = rng.choice(extras, size=n_extra, replace=False) if n_extra > 0 else []
            for idx in [rep] + list(chosen):
                query_local_list.append(idx)
                query_cluster_ids.append(c)

        query_idx = rising_idx[np.array(query_local_list)]
        n_queries = len(query_idx)
        print(f'Final queries: {n_queries} '
              f'({best_k} clusters × up to {args.n_per_cluster} each)')

    # precompute property arrays once
    prop_arrays = build_prop_arrays(npz)
    n_props     = len(prop_arrays)
    n_tiers     = len(TIERS)
    n_strip     = args.n_strip
    strip_tier_labels = [t[0] for t in TIERS if t[0] in STRIP_TIERS]
    n_strip_tiers = len(strip_tier_labels)
    # total rows: 1 (query) + 1 (histograms) + n_strip_tiers * 2 (sfh+img)
    n_rows = 2 + n_strip_tiers * 2
    ncols  = max(n_props, n_strip)

    print(f'Writing {n_queries} pages to {args.output} …')
    with PdfPages(args.output) as pdf:
        if args.sampling == 'kmeans' and '_stability_fig' in dir() and _stability_fig is not None:
            pdf.savefig(_stability_fig, dpi=120, bbox_inches='tight')
            plt.close(_stability_fig)

        if args.sampling in ('kmeans', 'hdbscan') and cluster_labels is not None:
            pca_str  = f', PCA {pca_model.n_components_}D' if pca_model else ', 256D'
            algo_str = 'HDBSCAN' if args.sampling == 'hdbscan' else f'k-means k={best_k}'
            unique_cls = sorted(c for c in set(cluster_labels) if c >= 0)
            cmap_cls   = plt.cm.get_cmap('tab10', max(len(unique_cls), 1))

            # diagnostic: print UMAP centroid per cluster to verify index mapping
            if 'xy_joint' in npz:
                xy_diag = npz['xy_joint']
                print('Cluster UMAP-joint centroids (index sanity check):')
                for c in unique_cls:
                    m   = cluster_labels == c
                    pos = xy_diag[rising_idx[m]]
                    print(f'  cluster {c+1}: n={m.sum():5d}  '
                          f'mean=({pos[:,0].mean():.2f}, {pos[:,1].mean():.2f})  '
                          f'std=({pos[:,0].std():.2f}, {pos[:,1].std():.2f})')
                all_pos = xy_diag[rising_idx]
                print(f'  all selected: mean=({all_pos[:,0].mean():.2f}, {all_pos[:,1].mean():.2f})  '
                      f'std=({all_pos[:,0].std():.2f}, {all_pos[:,1].std():.2f})')

            def _umap_page(xy, title):
                fig_u, ax_u = plt.subplots(figsize=(7, 6))
                ax_u.scatter(xy[:, 0], xy[:, 1], c='#cccccc', s=1,
                             linewidths=0, alpha=0.4, rasterized=True, label='all')
                if args.sampling == 'hdbscan':
                    noise_m = cluster_labels == -1
                    if noise_m.any():
                        ax_u.scatter(xy[rising_idx[noise_m], 0],
                                     xy[rising_idx[noise_m], 1],
                                     c='#555555', s=4, linewidths=0, alpha=0.5,
                                     label='noise', rasterized=True)
                for i, c in enumerate(unique_cls):
                    m = cluster_labels == c
                    ax_u.scatter(xy[rising_idx[m], 0], xy[rising_idx[m], 1],
                                 c=[cmap_cls(i)], s=6, linewidths=0, alpha=0.7,
                                 label=f'cluster {c+1}', rasterized=True)
                ax_u.scatter(xy[query_idx, 0], xy[query_idx, 1],
                             c='black', s=40, marker='*', zorder=5,
                             label='representative')
                ax_u.set_xlabel('UMAP-1', fontsize=9)
                ax_u.set_ylabel('UMAP-2', fontsize=9)
                ax_u.set_title(title, fontsize=9)
                ax_u.legend(fontsize=6, markerscale=1.5, ncol=2,
                            loc='upper right', framealpha=0.7)
                fig_u.tight_layout()
                return fig_u

            # always show joint UMAP first (reference / comparison)
            if 'xy_joint' in npz:
                fig_uj = _umap_page(
                    npz['xy_joint'],
                    f'Joint-embedding UMAP — coloured by {args.emb_space.upper()}-space clusters '
                    f'({algo_str}{pca_str})\n{sel_label}',
                )
                pdf.savefig(fig_uj, dpi=150, bbox_inches='tight')
                plt.close(fig_uj)

            # clustering-space UMAP (skip if same as joint, already shown above)
            _clust_umap_key = {'sfh': 'xy_sfh', 'img': 'xy_img', 'joint': 'xy_joint'}
            _clust_xy_key   = _clust_umap_key.get(args.emb_space, 'xy_joint')
            if _clust_xy_key != 'xy_joint' and _clust_xy_key in npz:
                fig_uc = _umap_page(
                    npz[_clust_xy_key],
                    f'{args.emb_space.upper()}-embedding UMAP — clusters '
                    f'({algo_str}{pca_str})\n{sel_label}',
                )
                pdf.savefig(fig_uc, dpi=150, bbox_inches='tight')
                plt.close(fig_uc)

            # image UMAP (skip if clustering was in image space)
            if args.emb_space != 'img' and 'xy_img' in npz:
                fig_ui = _umap_page(
                    npz['xy_img'],
                    f'Image-embedding UMAP — coloured by {args.emb_space.upper()}-space clusters '
                    f'({algo_str}{pca_str})\n{sel_label}',
                )
                pdf.savefig(fig_ui, dpi=150, bbox_inches='tight')
                plt.close(fig_ui)

            # SFH UMAP (skip if clustering was in SFH space)
            if args.emb_space != 'sfh' and 'xy_sfh' in npz:
                fig_us = _umap_page(
                    npz['xy_sfh'],
                    f'SFH-embedding UMAP — coloured by {args.emb_space.upper()}-space clusters '
                    f'({algo_str}{pca_str})\n{sel_label}',
                )
                pdf.savefig(fig_us, dpi=150, bbox_inches='tight')
                plt.close(fig_us)

        if ks_fig is not None:
            pdf.savefig(ks_fig, dpi=120, bbox_inches='tight')
            plt.close(ks_fig)

        if args.sampling in ('kmeans', 'hdbscan') and cluster_labels is not None:
            # ── embedding cohesion page ───────────────────────────────────────
            fig_coh = cluster_embedding_cohesion(
                cluster_labels, rising_idx,
                embeddings={'img_emb':   img_emb,
                            'sfh_emb':   sfh_emb,
                            'joint_emb': joint_emb},
                rng=rng,
            )
            pdf.savefig(fig_coh, dpi=120, bbox_inches='tight')
            plt.close(fig_coh)

        if args.sampling in ('kmeans', 'hdbscan') and cluster_labels is not None:
            ks_props     = build_ks_props(npz)   # (label, values) for rising sample
            unique_cls   = sorted(c for c in set(cluster_labels) if c >= 0)
            n_cls        = len(unique_cls)
            n_props_ks   = len(ks_props)
            cmap_cls     = plt.cm.get_cmap('tab10', max(n_cls, 1))

            fig_ph, axes_ph = plt.subplots(
                1, n_props_ks,
                figsize=(2.5 * n_props_ks, 3.5),
                constrained_layout=True,
            )
            if n_props_ks == 1:
                axes_ph = [axes_ph]

            for pi, (plabel, pvals) in enumerate(ks_props):
                ax = axes_ph[pi]
                pv = pvals[rising_idx]
                finite = np.isfinite(pv)
                vmin = np.nanpercentile(pv[finite], 1)  if finite.any() else 0
                vmax = np.nanpercentile(pv[finite], 99) if finite.any() else 1
                bins = np.linspace(vmin, vmax, 25)
                for i, c in enumerate(unique_cls):
                    m = (cluster_labels == c) & finite
                    if m.sum() < 3:
                        continue
                    ax.hist(pv[m], bins=bins, density=True, histtype='step',
                            color=cmap_cls(i), linewidth=1.4,
                            label=f'C{c+1} (n={m.sum()})')
                ax.set_xlabel(plabel, fontsize=7)
                ax.tick_params(labelsize=6)
                ax.set_yticks([])
                if pi == 0:
                    ax.legend(fontsize=6, loc='upper right',
                              framealpha=0.6, handlelength=1)

            fig_ph.suptitle(
                f'Property distributions per cluster  '
                f'({algo_str}{pca_str}, {args.emb_space}-emb)\n'
                f'{sel_label}',
                fontsize=9,
            )
            pdf.savefig(fig_ph, dpi=120, bbox_inches='tight')
            plt.close(fig_ph)

            # ── mean properties per cluster ───────────────────────────────────
            fig_mp, axes_mp = plt.subplots(
                1, n_props_ks,
                figsize=(2.5 * n_props_ks, 3.5),
                constrained_layout=True,
            )
            if n_props_ks == 1:
                axes_mp = [axes_mp]

            x = np.arange(n_cls)
            for pi, (plabel, pvals) in enumerate(ks_props):
                ax    = axes_mp[pi]
                pv    = pvals[rising_idx]
                medians = []
                p16s    = []
                p84s    = []
                for c in unique_cls:
                    m  = (cluster_labels == c) & np.isfinite(pv)
                    v  = pv[m]
                    medians.append(np.median(v) if len(v) else np.nan)
                    p16s.append(np.percentile(v, 16) if len(v) else np.nan)
                    p84s.append(np.percentile(v, 84) if len(v) else np.nan)
                medians = np.array(medians)
                err_lo  = medians - np.array(p16s)
                err_hi  = np.array(p84s) - medians
                bars = ax.bar(x, medians, color=[cmap_cls(i) for i in range(n_cls)],
                              alpha=0.8, edgecolor='k', linewidth=0.5)
                ax.errorbar(x, medians, yerr=[err_lo, err_hi],
                            fmt='none', color='black', capsize=3, linewidth=1)
                ax.set_xticks(x)
                ax.set_xticklabels([f'C{c+1}' for c in unique_cls], fontsize=7)
                ax.set_ylabel(plabel, fontsize=7)
                ax.tick_params(labelsize=6)

            fig_mp.suptitle(
                f'Median properties per cluster (error bars: 16th–84th pct)  '
                f'({algo_str}{pca_str}, {args.emb_space}-emb)\n{sel_label}',
                fontsize=9,
            )
            pdf.savefig(fig_mp, dpi=120, bbox_inches='tight')
            plt.close(fig_mp)

            # ── main sequence by redshift bin ─────────────────────────────────
            fig_ms = plot_main_sequence(
                npz, cluster_labels, rising_idx,
                cmap_cls=cmap_cls, unique_cls=unique_cls,
                sel_label=sel_label,
            )
            if fig_ms is not None:
                pdf.savefig(fig_ms, dpi=120, bbox_inches='tight')
                plt.close(fig_ms)

            # ── size–mass by redshift bin ─────────────────────────────────────
            fig_sm = plot_size_mass(
                npz, cluster_labels, rising_idx,
                cmap_cls=cmap_cls, unique_cls=unique_cls,
                sel_label=sel_label,
            )
            if fig_sm is not None:
                pdf.savefig(fig_sm, dpi=120, bbox_inches='tight')
                plt.close(fig_sm)

        for q_pos, emb_idx in enumerate(query_idx):
            h5_idx   = int(h5_indices[emb_idx])
            gid      = int(galaxy_ids[emb_idx])
            query_qi = float(qi[emb_idx])

            # ── score galaxies ────────────────────────────────────────────────
            if args.retrieval == 'cross_modal':
                scores      = img_emb @ sfh_emb[emb_idx]       # img similarity to query SFH
            else:
                scores      = joint_emb @ joint_emb[emb_idx]   # joint similarity
            scores[emb_idx] = -np.inf                           # exclude self
            if args.restrict or args.emb_space != 'joint':
                # mask out galaxies outside the selection
                scores[~rising_mask] = -np.inf
            pcts        = np.percentile(scores, [t[1] for t in TIERS] +
                                                [t[2] for t in TIERS])

            valid_mask   = np.isfinite(scores)
            valid_scores = scores[valid_mask]
            tier_masks = []
            for i, (_, lo, hi, _) in enumerate(TIERS):
                lo_val = np.percentile(valid_scores, lo)
                hi_val = np.percentile(valid_scores, hi)
                m = valid_mask & (scores >= lo_val) & (scores <= hi_val)
                if i == 0:
                    m = valid_mask & (scores >= lo_val)
                tier_masks.append(m)

            # ── load query data ───────────────────────────────────────────────
            qdata = load_batch(args.h5, [h5_idx])

            # ── load strip images for each tier ──────────────────────────────
            strip_h5 = []
            strip_qi = []
            for mask in tier_masks:
                candidates = np.where(mask)[0]
                chosen     = rng.choice(candidates,
                                        size=min(n_strip, len(candidates)),
                                        replace=False)
                strip_h5.append([int(h5_indices[i]) for i in chosen])
                strip_qi.append(qi[chosen])

            all_strip_h5 = [idx for tier in strip_h5 for idx in tier]
            strip_data   = load_batch(args.h5, all_strip_h5)

            # ── figure ────────────────────────────────────────────────────────
            row_heights = [2.5, 2.0] + [1.3, 1.6] * n_strip_tiers
            fig, axes = plt.subplots(
                n_rows, ncols,
                figsize=(ncols * 1.8, sum(row_heights)),
                gridspec_kw={'height_ratios': row_heights},
                constrained_layout=True,
            )
            restr_str   = '  [restricted]' if args.restrict else ''
            ret_str     = '  [cross-modal]' if args.retrieval == 'cross_modal' else ''
            if args.sampling in ('kmeans', 'hdbscan'):
                cid       = query_cluster_ids[q_pos]
                clust_str = f'  cluster {cid+1}/{best_k}  (query {q_pos+1}/{n_queries})'
            else:
                clust_str = f'  query {q_pos+1}/{n_queries}'
            fig.suptitle(
                f'Galaxy {gid}{clust_str}  '
                f'QI={query_qi:.3f}  f(5%)={f5[emb_idx]:.2f}  '
                f'z={float(qdata["zs"][0]):.2f}\n{sel_label}{restr_str}{ret_str}',
                fontsize=8,
            )

            # row 0: query SFH + query image; remaining cols blank
            for ax in axes[0]:
                ax.axis('off')
            axes[0, 0].axis('on')
            plot_sfh_bars(axes[0, 0], qdata['bins_log'][0], qdata['times_myr'][0],
                          float(qdata['t_norms'][0]), query_qi,
                          float(qdata['zs'][0]))
            axes[0, 0].set_title('QUERY SFH', fontsize=6, color='firebrick',
                                  fontweight='bold', pad=2)
            axes[0, 1].axis('on')
            plot_stamp(axes[0, 1], qdata['imgs'][0],
                       title='QUERY image', color='firebrick')

            # row 1: morphology histograms
            for ax in axes[1]:
                ax.axis('off')
            for ax, prop in zip(axes[1, :n_props], prop_arrays):
                ax.axis('on')
            plot_morph_distributions(axes[1, :n_props], tier_masks, prop_arrays)

            # rows 2+: strip tiers (top 5% and 5-10% only)
            strip_row = 2
            strip_ptr = 0
            for t_i, (tlabel, _, _, tcolor) in enumerate(TIERS):
                if tlabel not in STRIP_TIERS:
                    strip_ptr += n_strip
                    continue

                n_avail   = min(n_strip, strip_data['imgs'].shape[0] - strip_ptr)
                tier_imgs = strip_data['imgs'][strip_ptr:strip_ptr + n_avail]
                tier_bins = strip_data['bins_log'][strip_ptr:strip_ptr + n_avail]
                tier_tms  = strip_data['times_myr'][strip_ptr:strip_ptr + n_avail]
                tier_tns  = strip_data['t_norms'][strip_ptr:strip_ptr + n_avail]
                tier_zs   = strip_data['zs'][strip_ptr:strip_ptr + n_avail]
                tier_qis  = strip_qi[t_i]

                for ax in axes[strip_row]:
                    ax.axis('off')
                for ax in axes[strip_row + 1]:
                    ax.axis('off')

                for s_i in range(n_avail):
                    qi_val = float(tier_qis[s_i]) if s_i < len(tier_qis) else 0.0
                    ax_s = axes[strip_row, s_i]
                    ax_s.axis('on')
                    plot_sfh_bars(ax_s, tier_bins[s_i], tier_tms[s_i],
                                  float(tier_tns[s_i]), qi_val,
                                  float(tier_zs[s_i]))
                    if s_i == 0:
                        ax_s.set_title(tlabel, fontsize=6, color=tcolor,
                                       fontweight='bold', pad=2)
                    ax_i = axes[strip_row + 1, s_i]
                    ax_i.axis('on')
                    plot_stamp(ax_i, tier_imgs[s_i],
                               title=f'z={tier_zs[s_i]:.2f} QI={qi_val:.2f}',
                               color=tcolor)

                strip_ptr += n_strip
                strip_row += 2

            pdf.savefig(fig, dpi=120)
            plt.close(fig)

            if (q_pos + 1) % 5 == 0:
                print(f'  {q_pos + 1} / {n_queries}')

    print(f'Done → {args.output}')


if __name__ == '__main__':
    main()
