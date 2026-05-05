"""
Interactive COSMOS-Web CLIP explorer.

Lasso- or box-select galaxies in the UMAP embedding space; the bottom
panels instantly show a random sample of their F150W stamps and CIGALE
SFHs from the selected region.

Files needed (download from the server):
    cosmosweb_dataset.h5     – images + SFHs
    cosmosweb_umap.npz       – UMAP coordinates (produced by umap_embeddings.py)

Installation (once, on your laptop):
    pip install panel bokeh h5py numpy matplotlib

Run:
    panel serve cosmosweb/explore.py --show \\
        --args --h5 /path/to/cosmosweb_dataset.h5 \\
               --umap /path/to/cosmosweb_umap.npz

    # Or just launch directly (same result):
    python cosmosweb/explore.py \\
        --h5   /path/to/cosmosweb_dataset.h5 \\
        --umap /path/to/cosmosweb_umap.npz
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import panel as pn
from bokeh.models import BasicTicker, BooleanFilter, CDSView, ColorBar, ColumnDataSource, LinearColorMapper
from bokeh.palettes import Inferno256, Plasma256, Viridis256
from bokeh.plotting import figure as bk_figure

pn.extension(sizing_mode='stretch_width')

# ── constants ─────────────────────────────────────────────────────────────────
SFH_EPS    = 1e-10
SFH_N_BINS = 50
N_DISPLAY  = 16    # max stamps / SFHs shown at once
NCOLS      = 4     # columns in each gallery grid

# Fractional lookback-time grid — read from HDF5 at runtime (grid-agnostic).
# Fallback to linspace for datasets that pre-date the sfh_time_grid dataset.
_T_FRAC = np.linspace(0, 1, SFH_N_BINS)   # (50,) fallback

# Fractions of cosmic time for which we compute cumulative SFR fractions
SFH_TIME_FRACS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

_PALETTES = {
    'plasma':  Plasma256,
    'viridis': Viridis256,
    'inferno': Inferno256,
}


# ── argument parsing ──────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='CosmosWebCLIP interactive explorer')
    p.add_argument('--h5',   type=Path, required=True,
                   help='cosmosweb_dataset.h5')
    p.add_argument('--umap', type=Path, required=True,
                   help='cosmosweb_umap.npz from umap_embeddings.py')
    p.add_argument('--port', type=int, default=5007)
    # parse_known_args so panel serve's own flags are ignored
    args, _ = p.parse_known_args()
    return args


# ── SFH shape parameters ─────────────────────────────────────────────────────

def _sfh_properties(h5_path: Path, h5_indices: np.ndarray) -> dict[str, np.ndarray]:
    """
    Derive scalar SFH shape descriptors from the log10 SFH vectors in the HDF5.

    The SFH array has shape (N, 50) in log10 of normalised fractions.
    The time grid is read from sfh_time_grid in the HDF5 (grid-agnostic).
    """
    with h5py.File(h5_path, 'r') as f:
        sfh_log = f['sfh'][list(h5_indices)].astype(np.float64)   # (N, 50)
        t_frac  = (f['sfh_time_grid'][:].astype(np.float64)
                   if 'sfh_time_grid' in f else _T_FRAC)           # (50,)

    # Convert to linear fractions; re-normalise to correct for log rounding
    sfr = np.maximum(10.0 ** sfh_log - SFH_EPS, 0.0)
    sfr_sum = sfr.sum(axis=1, keepdims=True)
    sfr_sum = np.where(sfr_sum > 0, sfr_sum, 1.0)
    sfr = sfr / sfr_sum                                            # (N, 50), sums to 1

    t = t_frac[np.newaxis, :]                                      # (1, 50)

    # ── 1. log ratio: old SFR / recent SFR ───────────────────────────────────
    log_old_recent = sfh_log[:, -1] - sfh_log[:, 0]

    # ── 2. Mass-weighted mean formation epoch (fractional lookback time) ──────
    mean_t = (sfr * t).sum(axis=1)

    # ── 3. Cumulative + differential SFR fractions ────────────────────────────
    # Cumulative recent: t_frac ≤ threshold
    # Cumulative old:    t_frac ≥ 1 - threshold
    # Differential:      fraction in each interval between thresholds
    sfr_fracs = {}
    cumul_recent = {}   # cache for computing deltas
    cumul_old    = {}
    for threshold in SFH_TIME_FRACS:
        pct = int(round(threshold * 100))
        mask_recent = t_frac <= threshold
        mask_old    = t_frac >= (1.0 - threshold)
        cumul_recent[pct] = sfr[:, mask_recent].sum(axis=1)
        cumul_old[pct]    = sfr[:, mask_old].sum(axis=1)
        sfr_fracs[f'SFH: f(recent {pct}%)'] = cumul_recent[pct]
        sfr_fracs[f'SFH: f(old {pct}%)']    = cumul_old[pct]

    # Differential fractions: SFR in each interval (recent end)
    pcts = [int(round(t * 100)) for t in SFH_TIME_FRACS]
    for i, pct in enumerate(pcts):
        prev_pct = pcts[i - 1] if i > 0 else 0
        prev_val = cumul_recent[prev_pct] if i > 0 else np.zeros(sfr.shape[0])
        sfr_fracs[f'SFH: Δf({prev_pct}–{pct}% recent)'] = cumul_recent[pct] - prev_val

    # Differential fractions: SFR in each interval (old end)
    for i, pct in enumerate(pcts):
        prev_pct = pcts[i - 1] if i > 0 else 0
        prev_val = cumul_old[prev_pct] if i > 0 else np.zeros(sfr.shape[0])
        sfr_fracs[f'SFH: Δf({prev_pct}–{pct}% old)'] = cumul_old[pct] - prev_val

    # ── 4. Fractional lookback time of the SFH peak ───────────────────────────
    peak_bin = np.argmax(sfr, axis=1)
    peak_t   = t_frac[peak_bin]

    # ── 5. Quenching / rising SFR index ──────────────────────────────────────
    # Difference between the 5-10% interval and 0-5% interval.
    # Positive → SFR was higher 5-10% ago than now → currently quenching
    # Negative → SFR is rising → rejuvenation or ongoing starburst
    # Requires both 5% and 10% thresholds to be in SFH_TIME_FRACS.
    quench_index = np.full(sfr.shape[0], np.nan)
    if 5 in pcts and 10 in pcts:
        delta_0_5  = cumul_recent[5]
        delta_5_10 = cumul_recent[10] - cumul_recent[5]
        quench_index = delta_5_10 - delta_0_5

    return {
        'SFH: log(old/recent)':      log_old_recent,
        'SFH: mean formation epoch':  mean_t,
        **sfr_fracs,
        'SFH: peak lookback t_frac':  peak_t,
        'SFH: quenching index':       quench_index,
    }


# ── data loading ──────────────────────────────────────────────────────────────

def _load(h5_path: Path, umap_path: Path) -> dict:
    npz = np.load(umap_path, allow_pickle=True)

    # UMAP coordinate sets
    xy = {
        'Joint (img + SFH)': npz['xy_joint'].astype(float),
        'Image encoder':     npz.get('xy_img',   npz['xy_joint']).astype(float),
        'SFH encoder':       npz.get('xy_sfh',   npz['xy_joint']).astype(float),
    }

    # Physical properties available for coloring
    color_props: dict[str, np.ndarray] = {}
    _prop_map = {
        'redshifts':          'Redshift z',
        'zfinal':             'Redshift z (LePhare)',
        'radius_sersic':      'Sersic radius',
        'sersic':             'Sersic index n',
        'axratio_sersic':     'Axis ratio b/a',
        'log_mass':           'log M★',
        'log_sfr':            'log SFR',
        'log_ssfr':           'log sSFR',
        # CIGALE SFH properties
        'age_form':           'Formation age [Myr]',
        'log_sfr_inst':       'log SFR_inst',
        'log_sfr_100myr':     'log SFR_100Myr',
        'sfr_mass_vector_dir':  'SFR–M★ vector dir.',
        'sfr_mass_vector_norm': 'SFR–M★ vector norm',
    }
    for npz_key, label in _prop_map.items():
        if npz_key in npz:
            color_props[label] = npz[npz_key].astype(float)

    # Family morphology columns
    for key in sorted(npz.files):
        if 'family' in key:
            label = key.replace('family_', 'P(').upper() + ')'
            color_props[label] = npz[key].astype(float)

    # Composite morphology: P_early = P(Ell) + P(S0), P_late = P(early disk) + P(late disk)
    _ell  = npz['family_elliptical'].astype(float) if 'family_elliptical' in npz else None
    _s0   = npz['family_s0'].astype(float)         if 'family_s0'         in npz else None
    _edsk = npz['family_early_disk'].astype(float) if 'family_early_disk' in npz else None
    _ldsk = npz['family_late_disk'].astype(float)  if 'family_late_disk'  in npz else None
    if _ell is not None and _s0 is not None:
        color_props['P_early (Ell + S0)'] = np.clip(_ell + _s0, 0.0, 1.0)
    if _edsk is not None and _ldsk is not None:
        color_props['P_late (early + late disk)'] = np.clip(_edsk + _ldsk, 0.0, 1.0)
    if 'binary_disturbed' in npz:
        color_props['P(Disturbed)'] = npz['binary_disturbed'].astype(float)

    # B+D colour gradients (optional — produced by compute_color_gradients.py)
    _bd_map = {
        'delta_col_150_444': 'ΔF150W−F444W B+D (bulge − disk)',
        'col_bulge_150_444': 'F150W−F444W B+D bulge',
        'col_disk_150_444':  'F150W−F444W B+D disk',
        'delta_col_115_277': 'ΔF115W−F277W B+D (bulge − disk)',
        'BT_f115w':          'B/T (F115W)',
        'BT_f150w':          'B/T (F150W)',
        'BT_f277w':          'B/T (F277W)',
        'BT_f444w':          'B/T (F444W)',
    }
    for npz_key, label in _bd_map.items():
        if npz_key in npz:
            arr = npz[npz_key].astype(float)
            color_props[label] = np.where(np.isfinite(arr), arr, np.nan)

    # Aperture colour gradients (optional — produced by compute_aperture_gradients.py)
    _APER_DIAMS = ['0.2"', '0.3"', '0.5"', '0.75"', '1.0"']
    _aper_map = {}
    for i, diam in enumerate(_APER_DIAMS):
        _aper_map[f'col_aper{i}_150_444'] = f'F150W−F444W aper {diam}'
        _aper_map[f'col_aper{i}_115_277'] = f'F115W−F277W aper {diam}'
    _aper_map['col_auto_150_444']    = 'F150W−F444W Kron (auto)'
    _aper_map['grad_150_444_1v4']    = 'ΔF150W−F444W 0.3"−1.0" (gradient)'
    _aper_map['grad_150_444_2v4']    = 'ΔF150W−F444W 0.5"−1.0" (gradient)'
    _aper_map['grad_115_277_1v4']    = 'ΔF115W−F277W 0.3"−1.0" (gradient)'
    _ANNULUS_LABELS = ['0–0.1"', '0.1–0.15"', '0.15–0.25"', '0.25–0.375"', '0.375–0.5"']
    for i, ann in enumerate(_ANNULUS_LABELS):
        _aper_map[f'col_ann{i}_150_444'] = f'F150W−F444W annulus {ann}'
        _aper_map[f'col_ann{i}_115_277'] = f'F115W−F277W annulus {ann}'
    _aper_map['grad_ann_150_444_0v4'] = 'ΔF150W−F444W core−outermost annulus'
    _aper_map['grad_ann_150_444_0v3'] = 'ΔF150W−F444W core−3rd annulus'
    _aper_map['grad_ann_115_277_0v4'] = 'ΔF115W−F277W core−outermost annulus'
    for npz_key, label in _aper_map.items():
        if npz_key in npz:
            arr = npz[npz_key].astype(float)
            color_props[label] = np.where(np.isfinite(arr), arr, np.nan)

    # Rest-frame Balmer-break colour gradients (produced by compute_restframe_gradient.py)
    # Filter pair is chosen per galaxy so that the Balmer break (3646 Å) falls between the
    # two filters: F814W/F115W (z 1.2-2.2), F115W/F150W (z 2.2-3.1), F150W/F277W (z 3.1-6.6)
    _rf_map = {}
    _APER_DIAMS_RF = ['0.2"', '0.3"', '0.5"', '0.75"', '1.0"']
    for i, diam in enumerate(_APER_DIAMS_RF):
        _rf_map[f'col_rf_aper{i}'] = f'Balmer break colour aper {diam} (rest-frame)'
    for i, ann in enumerate(_ANNULUS_LABELS):
        _rf_map[f'col_rf_ann{i}'] = f'Balmer break colour annulus {ann} (rest-frame)'
    _rf_map['grad_rf_1v4']     = 'ΔBalmer break 0.3"−1.0" (rest-frame gradient)'
    _rf_map['grad_rf_2v4']     = 'ΔBalmer break 0.5"−1.0" (rest-frame gradient)'
    _rf_map['grad_rf_ann_0v4'] = 'ΔBalmer break core−outermost annulus (rest-frame)'
    _rf_map['grad_rf_ann_0v3'] = 'ΔBalmer break core−3rd annulus (rest-frame)'
    _rf_map['rf_pair']         = 'Rest-frame pair index (0=F814/115, 1=F115/150, 2=F150/277)'
    for npz_key, label in _rf_map.items():
        if npz_key in npz:
            arr = npz[npz_key].astype(float)
            color_props[label] = np.where(np.isfinite(arr), arr, np.nan)

    # SFH shape descriptors computed directly from the HDF5
    h5_indices = npz['h5_indices'].astype(int)
    sfh_props  = _sfh_properties(h5_path, h5_indices)
    color_props.update(sfh_props)

    # HDBSCAN cluster labels (optional — produced by umap_embeddings_zoobot.py)
    if 'hdbscan_labels' in npz:
        from bokeh.palettes import Turbo256
        labels     = npz['hdbscan_labels'].astype(int)
        n_clusters = int(labels.max()) + 1 if labels.max() >= 0 else 0
        step       = max(1, 256 // max(n_clusters, 1))
        palette    = [Turbo256[min(i * step, 255)] for i in range(n_clusters)]
        cluster_hex = [
            '#aaaaaa' if lbl < 0 else palette[lbl % len(palette)]
            for lbl in labels
        ]
        color_props['Cluster (HDBSCAN)'] = labels.astype(float)
    else:
        labels      = None
        n_clusters  = 0
        cluster_hex = ['#888888'] * len(h5_indices)

    # Default color: first available
    default_color = next(iter(color_props), None)

    redshifts = npz['redshifts'].astype(float) if 'redshifts' in npz else np.full(len(h5_indices), np.nan)

    return dict(
        xy            = xy,
        color_props   = color_props,
        default_color = default_color,
        h5_indices    = h5_indices,
        galaxy_ids    = npz['galaxy_ids'],
        cluster_labels = labels,
        cluster_hex    = cluster_hex,
        n_clusters     = n_clusters,
        redshifts      = redshifts,
    )


# ── rendering helpers ─────────────────────────────────────────────────────────

def _render_stamp(ax, img: np.ndarray):
    ch = img[1]   # F277W channel, raw arcsinh flux
    vmin, vmax = np.percentile(ch, [0.5, 99.5])
    ax.imshow(ch, origin='lower', cmap='gray',
              vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])


def _render_sfh(ax, sfh_log: np.ndarray):
    w = np.maximum(10.0 ** sfh_log - SFH_EPS, SFH_EPS)
    t = np.linspace(0, 1, len(sfh_log))
    ax.step(t[1:], w[1:], where='post', color='steelblue', lw=0.9)
    ax.fill_between(t[1:], w[1:], step='post', alpha=0.2, color='steelblue')
    ax.set_yscale('log')
    ax.set_xlim(0, 1)
    ax.tick_params(labelsize=4)
    ax.set_xlabel('Fractional lookback time', fontsize=4)
    ax.set_ylabel('SFH weight (norm.)', fontsize=4)


_BLANK_HTML = (
    '<div style="height:220px;display:flex;align-items:center;'
    'justify-content:center;color:#aaaaaa;font-size:13px">'
    'Draw a selection on the UMAP.'
    '</div>'
)


def _fig_to_html(fig: plt.Figure) -> str:
    """Render *fig* to a base64 PNG img tag, then close the figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return (
        f'<img src="data:image/png;base64,{b64}" '
        'style="width:100%;max-width:600px"/>'
    )


def _make_gallery(h5_path: Path, h5_rows: np.ndarray, d: dict,
                  rng: np.random.Generator) -> tuple[plt.Figure, plt.Figure]:
    n = min(N_DISPLAY, len(h5_rows))
    sample = np.sort(rng.choice(len(h5_rows), n, replace=False))
    rows   = h5_rows[sample]

    with h5py.File(h5_path, 'r') as f:
        imgs = f['images'][list(rows)]    # (n, 3, 64, 64)
        sfhs = f['sfh'][list(rows)]       # (n, N_BINS)
        zs   = f['redshift'][list(rows)]  # (n,)

    nrows = max(1, (n + NCOLS - 1) // NCOLS)
    kw    = dict(figsize=(NCOLS * 1.7, nrows * 1.7), squeeze=False)

    fig_img, axs_img = plt.subplots(nrows, NCOLS, **kw)
    fig_sfh, axs_sfh = plt.subplots(nrows, NCOLS, **kw)

    for axs in (axs_img.flatten(), axs_sfh.flatten()):
        for ax in axs:
            ax.set_visible(False)

    for i in range(n):
        axs_img.flatten()[i].set_visible(True)
        _render_stamp(axs_img.flatten()[i], imgs[i])
        axs_img.flatten()[i].set_title(f'z={zs[i]:.2f}', fontsize=5, pad=1)

        axs_sfh.flatten()[i].set_visible(True)
        _render_sfh(axs_sfh.flatten()[i], sfhs[i])
        axs_sfh.flatten()[i].set_title(f'z={zs[i]:.2f}', fontsize=5, pad=1)

    for fig in (fig_img, fig_sfh):
        fig.tight_layout(pad=0.4)

    return fig_img, fig_sfh


def _mean_sfh_fig(h5_path: Path, h5_rows: np.ndarray) -> plt.Figure:
    """Mean ± std SFH across all h5_rows (not just the display subsample)."""
    with h5py.File(h5_path, 'r') as f:
        sfhs = f['sfh'][sorted(h5_rows.tolist())].astype(np.float64)   # (N, 50)
    sfr = np.maximum(10.0 ** sfhs - SFH_EPS, 0.0)
    sfr /= sfr.sum(axis=1, keepdims=True).clip(min=1e-10)
    mean = sfr.mean(axis=0)
    std  = sfr.std(axis=0)
    t    = _T_FRAC
    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.fill_between(t[1:], (mean - std)[1:], (mean + std)[1:],
                    step='post', alpha=0.25, color='steelblue')
    ax.step(t[1:], mean[1:], where='post', color='steelblue', lw=1.5,
            label=f'mean ± std  (N={len(h5_rows):,})')
    ax.set_yscale('log')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Fractional lookback time', fontsize=8)
    ax.set_ylabel('SFH weight (norm.)', fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.6)
    return fig


# ── Panel application ─────────────────────────────────────────────────────────

def build_app(h5_path: Path, d: dict) -> pn.viewable.Viewable:
    rng  = np.random.default_rng(42)
    N    = len(d['h5_indices'])
    xy0  = d['xy']['Joint (img + SFH)']
    keys = list(d['color_props'].keys())

    # ── shared data source ────────────────────────────────────────────────
    # c1 / c2 are the colour fields for the left and right plots respectively.
    def _safe(key):
        """Impute NaN → median; used only for filter/range-slider bounds."""
        vals = d['color_props'].get(key, np.zeros(N))
        med  = np.nanmedian(vals[np.isfinite(vals)]) if np.any(np.isfinite(vals)) else 0.0
        return np.where(np.isfinite(vals), vals, med)

    def _raw(key):
        """Return colour values with NaN preserved; NaN points are hidden by nan_color."""
        vals = d['color_props'].get(key, np.full(N, np.nan))
        return np.where(np.isfinite(vals), vals, np.nan).astype(np.float64)

    # Default: first key left, second key right (usually a morphology vs SFH split)
    key1 = keys[0] if keys else ''
    key2 = keys[1] if len(keys) > 1 else key1
    c1_s = _safe(key1)
    c2_s = _safe(key2)

    _DYN_LABEL = 'Dynamic k-means'

    src = ColumnDataSource(dict(
        x           = xy0[:, 0].tolist(),
        y           = xy0[:, 1].tolist(),
        c1          = _raw(key1).tolist(),
        c2          = _raw(key2).tolist(),
        cluster_hex = d['cluster_hex'],
        dyn_hex     = ['#cccccc'] * N,   # filled after on-the-fly k-means
    ))

    # ── combined visibility mask ──────────────────────────────────────────
    # z_mask:   driven by the redshift slider
    # prop_mask: driven by the property filter (all True until clustering runs)
    # z_view holds the AND of both; hidden points are not rendered → not selectable
    z_all     = d['redshifts']
    z_safe    = np.where(np.isfinite(z_all), z_all, -1.0)
    z_mask    = np.ones(N, dtype=bool)
    prop_mask = np.ones(N, dtype=bool)   # updated when "Cluster filtered" fires
    z_view    = CDSView(filter=BooleanFilter(booleans=[True] * N))

    def _update_view():
        # Replace the filter object (not mutate in place) so Bokeh reliably
        # detects the property change and pushes it to the browser.
        z_view.filter = BooleanFilter(booleans=(z_mask & prop_mask).tolist())

    # ── helper: build one Bokeh plot panel ────────────────────────────────
    def _make_plot(color_field: str, init_safe: np.ndarray, title: str):
        lo = float(np.nanpercentile(init_safe, 1))
        hi = float(np.nanpercentile(init_safe, 99))
        mapper = LinearColorMapper(palette=Plasma256, low=lo, high=hi,
                                   nan_color=(0, 0, 0, 0))
        cbar   = ColorBar(color_mapper=mapper, ticker=BasicTicker(),
                          label_standoff=8, width=12, location=(0, 0))
        plot = bk_figure(
            width=500, height=480,
            tools='lasso_select,box_select,wheel_zoom,pan,reset',
            active_drag='lasso_select',
            title=title,
            output_backend='webgl',
        )
        r_cont = plot.scatter(
            'x', 'y', source=src, view=z_view,
            color=dict(field=color_field, transform=mapper),
            size=2.5, alpha=0.7, line_width=0,
            selection_color='white', selection_alpha=1.0,
            nonselection_alpha=0.12,
        )
        r_clust = plot.scatter(
            'x', 'y', source=src, view=z_view,
            fill_color='cluster_hex', line_width=0,
            size=2.5, alpha=0.7,
            selection_fill_color='white', selection_alpha=1.0,
            nonselection_alpha=0.12, visible=False,
        )
        r_dyn = plot.scatter(
            'x', 'y', source=src, view=z_view,
            fill_color='dyn_hex', line_width=0,
            size=2.5, alpha=0.8,
            selection_fill_color='white', selection_alpha=1.0,
            nonselection_alpha=0.15, visible=False,
        )
        plot.add_layout(cbar, 'right')
        plot.xaxis.axis_label = 'UMAP 1'
        plot.yaxis.axis_label = 'UMAP 2'
        return plot, mapper, r_cont, r_clust, r_dyn

    plot1, mapper1, r_cont1, r_clust1, r_dyn1 = _make_plot('c1', c1_s, 'Left UMAP — draw to select')
    plot2, mapper2, r_cont2, r_clust2, r_dyn2 = _make_plot('c2', c2_s, 'Right UMAP')

    # ── per-plot colour widgets ───────────────────────────────────────────
    keys_plus = keys + [_DYN_LABEL]   # includes dynamic option once computed

    def _make_color_widgets(init_key, init_safe, label_prefix):
        color_w = pn.widgets.Select(
            name=f'{label_prefix}: colour by',
            options=keys_plus, value=init_key, width=220,
        )
        lo = float(np.nanpercentile(init_safe, 1))
        hi = float(np.nanpercentile(init_safe, 99))
        cbar_w = pn.widgets.RangeSlider(
            name=f'{label_prefix}: colour range',
            start=float(np.nanmin(init_safe)), end=float(np.nanmax(init_safe)),
            value=(lo, hi),
            step=max(float((np.nanmax(init_safe) - np.nanmin(init_safe)) / 200), 1e-6),
            width=220,
        )
        return color_w, cbar_w

    color_w1, cbar_w1 = _make_color_widgets(key1, c1_s, 'Left')
    color_w2, cbar_w2 = _make_color_widgets(key2, c2_s, 'Right')

    # ── shared widgets ────────────────────────────────────────────────────
    embed_w = pn.widgets.Select(
        name='Embedding space',
        options=list(d['xy'].keys()),
        value='Joint (img + SFH)', width=200,
    )
    info_md      = pn.pane.Markdown('_Draw a selection on the UMAP._', width=210)
    resample_btn = pn.widgets.Button(name='New random sample',
                                     button_type='primary', width=200)

    img_pane      = pn.pane.HTML(_BLANK_HTML, width=550)
    sfh_pane      = pn.pane.HTML(_BLANK_HTML, width=550)
    mean_sfh_pane = pn.pane.HTML(_BLANK_HTML, width=400)

    # ── redshift slider ───────────────────────────────────────────────────
    z_finite = z_safe[np.isfinite(z_all)]
    z_lo = float(np.floor(z_finite.min() * 10) / 10) if len(z_finite) else 0.0
    z_hi = float(np.ceil( z_finite.max() * 10) / 10) if len(z_finite) else 6.0
    z_slider = pn.widgets.RangeSlider(
        name='Redshift range', start=z_lo, end=z_hi,
        value=(z_lo, z_hi), step=0.1, width=200,
    )

    # ── cluster selector ──────────────────────────────────────────────────
    n_clusters = d['n_clusters']
    if n_clusters > 0 and d['cluster_labels'] is not None:
        labels    = d['cluster_labels']
        has_noise = bool((labels == -1).any())
        cl_opts   = (['— all —'] +
                     (['noise'] if has_noise else []) +
                     [f'cluster {i}' for i in range(n_clusters)])
        cluster_w = pn.widgets.Select(
            name='Jump to cluster', options=cl_opts, value='— all —', width=200)
        cluster_widget_row = pn.Column(pn.layout.Divider(), cluster_w)
    else:
        cluster_w          = None
        cluster_widget_row = pn.pane.Markdown('')

    # ── callbacks ─────────────────────────────────────────────────────────
    def _update_embed(event):
        xy = d['xy'][embed_w.value]
        src.data['x'] = xy[:, 0].tolist()
        src.data['y'] = xy[:, 1].tolist()
        src.selected.indices = []
        plot1.title.text = f'Left UMAP ({embed_w.value}) — draw to select'
        plot2.title.text = f'Right UMAP ({embed_w.value})'

    def _set_renderers(r_cont, r_clust, r_dyn, mode):
        """mode: 'cont' | 'clust' | 'dyn'"""
        r_cont.visible  = (mode == 'cont')
        r_clust.visible = (mode == 'clust')
        r_dyn.visible   = (mode == 'dyn')

    def _make_color_cb(color_field, mapper, r_cont, r_clust, r_dyn, color_w, cbar_w):
        def _cb(event):
            val = color_w.value
            if val == 'Cluster (HDBSCAN)':
                _set_renderers(r_cont, r_clust, r_dyn, 'clust')
                return
            if val == _DYN_LABEL:
                _set_renderers(r_cont, r_clust, r_dyn, 'dyn')
                return
            _set_renderers(r_cont, r_clust, r_dyn, 'cont')
            raw  = _raw(val)
            safe = _safe(val)   # for range-slider bounds only
            src.data[color_field] = raw.tolist()
            lo = float(np.nanpercentile(safe, 1))
            hi = float(np.nanpercentile(safe, 99))
            mapper.low  = lo
            mapper.high = hi
            cbar_w.start = float(np.nanmin(safe))
            cbar_w.end   = float(np.nanmax(safe))
            cbar_w.value = (lo, hi)
        return _cb

    def _make_cbar_cb(mapper, color_w):
        def _cb(event):
            if color_w.value not in ('Cluster (HDBSCAN)', _DYN_LABEL):
                mapper.low  = float(event.new[0])
                mapper.high = float(event.new[1])
        return _cb

    # ── on-the-fly property filter + k-means ─────────────────────────────
    filter_prop_w = pn.widgets.Select(
        name='Filter property', options=keys, value=keys[0] if keys else '', width=200,
    )
    _fp0  = _safe(keys[0]) if keys else np.zeros(N)
    filter_range_w = pn.widgets.RangeSlider(
        name='Filter range',
        start=float(np.nanmin(_fp0)), end=float(np.nanmax(_fp0)),
        value=(float(np.nanpercentile(_fp0, 1)), float(np.nanpercentile(_fp0, 99))),
        step=max(float((np.nanmax(_fp0) - np.nanmin(_fp0)) / 200), 1e-6),
        width=200,
    )
    k_input = pn.widgets.IntInput(name='k (clusters)', value=6, start=2, end=50, width=80)
    cluster_btn = pn.widgets.Button(
        name='Cluster filtered subset', button_type='success', width=200,
    )
    reset_btn = pn.widgets.Button(
        name='Show all', button_type='light', width=95,
    )
    kmeans_info = pn.pane.Markdown('', width=210, styles={'font-size': '11px'})

    def _on_filter_prop(event):
        safe = _safe(filter_prop_w.value)
        lo, hi = float(np.nanmin(safe)), float(np.nanmax(safe))
        filter_range_w.start = lo
        filter_range_w.end   = hi
        filter_range_w.value = (float(np.nanpercentile(safe, 1)),
                                 float(np.nanpercentile(safe, 99)))
        filter_range_w.step  = max((hi - lo) / 200, 1e-6)

    def _on_filter_range(event):
        """Live update: filter_range_w moved → update visibility immediately."""
        nonlocal prop_mask
        prop_vals = _safe(filter_prop_w.value)
        flo, fhi  = filter_range_w.value
        prop_mask = (prop_vals >= flo) & (prop_vals <= fhi)
        _update_view()

    def _on_cluster_filtered(event):
        nonlocal prop_mask, z_mask
        from sklearn.cluster import KMeans
        from bokeh.palettes import Turbo256

        prop_vals  = _safe(filter_prop_w.value)
        flo, fhi   = filter_range_w.value
        z_lo, z_hi = z_slider.value

        prop_mask = (prop_vals >= flo) & (prop_vals <= fhi)
        z_mask    = (z_safe >= z_lo) & (z_safe <= z_hi)
        idx       = np.where(prop_mask & z_mask)[0]
        k         = k_input.value

        if len(idx) < k:
            kmeans_info.object = (f'⚠ Only **{len(idx)}** galaxies pass the '
                                  f'filter — need ≥ k={k}.')
            return

        # Update combined visibility — only filtered points shown
        _update_view()
        src.selected.indices = []
        _refresh([])

        xy     = np.column_stack([src.data['x'], src.data['y']])
        km     = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = km.fit_predict(xy[idx])

        step = max(1, 256 // k)
        pal  = [Turbo256[min(i * step, 255)] for i in range(k)]
        dyn  = ['#cccccc'] * N
        for i, gi in enumerate(idx):
            dyn[gi] = pal[labels[i] % len(pal)]
        src.data['dyn_hex'] = dyn

        # Switch left plot to dynamic view
        color_w1.value = _DYN_LABEL

        sizes = [int((labels == c).sum()) for c in range(k)]
        kmeans_info.object = (
            f'**{len(idx):,}** galaxies → **{k}** clusters  \n'
            + '  \n'.join(f'cluster {c}: {sizes[c]:,}' for c in range(k))
        )

    def _on_reset(event):
        nonlocal prop_mask, z_mask
        prop_mask = np.ones(N, dtype=bool)
        _update_view()
        src.selected.indices = []
        _refresh([])
        kmeans_info.object = ''

    def _on_z_filter(event):
        nonlocal z_mask
        lo, hi = z_slider.value
        z_mask = (z_safe >= lo) & (z_safe <= hi)
        _update_view()
        src.selected.indices = []
        _refresh([])

    def _on_cluster_select(event):
        if cluster_w is None:
            return
        val = cluster_w.value
        if val == '— all —':
            src.selected.indices = []
            return
        elif val == 'noise':
            idx = np.where(labels == -1)[0].tolist()
        else:
            cid = int(val.split()[-1])
            idx = np.where(labels == cid)[0].tolist()
        src.selected.indices = idx
        _refresh(idx)

    def _refresh(sel):
        sel = list(sel)
        if not sel:
            info_md.object       = '_No galaxies selected._'
            img_pane.object      = _BLANK_HTML
            sfh_pane.object      = _BLANK_HTML
            mean_sfh_pane.object = _BLANK_HTML
            return
        h5_rows = d['h5_indices'][np.array(sel, dtype=int)]
        n_show  = min(N_DISPLAY, len(h5_rows))
        info_md.object = (f'**{len(sel):,}** selected '
                          f'— showing {n_show} random examples')
        fig_img, fig_sfh = _make_gallery(h5_path, h5_rows, d, rng)
        img_pane.object = _fig_to_html(fig_img)
        sfh_pane.object = _fig_to_html(fig_sfh)
        mean_sfh_pane.object = (
            _fig_to_html(_mean_sfh_fig(h5_path, h5_rows))
            if len(h5_rows) >= 20 else _BLANK_HTML
        )

    def _on_resample(event):
        _refresh(src.selected.indices)

    _prev_key: list[tuple] = [()]

    def _poll():
        key = tuple(sorted(src.selected.indices))
        if key == _prev_key[0]:
            return
        _prev_key[0] = key
        _refresh(list(src.selected.indices))

    # wire up
    embed_w.param.watch(_update_embed, 'value')
    color_w1.param.watch(_make_color_cb('c1', mapper1, r_cont1, r_clust1, r_dyn1, color_w1, cbar_w1), 'value')
    color_w2.param.watch(_make_color_cb('c2', mapper2, r_cont2, r_clust2, r_dyn2, color_w2, cbar_w2), 'value')
    cbar_w1.param.watch(_make_cbar_cb(mapper1, color_w1), 'value')
    cbar_w2.param.watch(_make_cbar_cb(mapper2, color_w2), 'value')
    z_slider.param.watch(_on_z_filter, 'value')
    if cluster_w is not None:
        cluster_w.param.watch(_on_cluster_select, 'value')
    filter_prop_w.param.watch(_on_filter_prop, 'value')
    filter_range_w.param.watch(_on_filter_range, 'value')
    cluster_btn.on_click(_on_cluster_filtered)
    reset_btn.on_click(_on_reset)
    resample_btn.on_click(_on_resample)
    pn.state.add_periodic_callback(_poll, period=350)

    # ── layout ────────────────────────────────────────────────────────────
    sidebar = pn.Column(
        pn.pane.Markdown('## COSMOS-Web CLIP\n### Explorer'),
        pn.layout.Divider(),
        embed_w,
        cluster_widget_row,
        pn.layout.Divider(),
        z_slider,
        pn.layout.Divider(),
        pn.pane.Markdown('**Property filter → k-means**',
                         styles={'font-size': '12px'}),
        filter_prop_w,
        filter_range_w,
        pn.Row(k_input, cluster_btn),
        pn.Row(reset_btn),
        kmeans_info,
        pn.layout.Divider(),
        info_md,
        resample_btn,
        pn.pane.Markdown(
            '_Lasso or box-select on either UMAP.  '
            f'Up to {N_DISPLAY} random examples are shown._',
            styles={'font-size': '11px', 'color': '#888888'},
        ),
        width=230,
    )

    app = pn.Column(
        pn.Row(
            pn.Column(color_w1, cbar_w1, pn.pane.Bokeh(plot1)),
            pn.Column(color_w2, cbar_w2, pn.pane.Bokeh(plot2)),
            sidebar,
        ),
        pn.layout.Divider(),
        pn.Row(
            pn.Column(pn.pane.Markdown('#### F277W stamps'),           img_pane),
            pn.Column(pn.pane.Markdown('#### CIGALE SFHs'),            sfh_pane),
            pn.Column(pn.pane.Markdown('#### Mean SFH (all selected)'), mean_sfh_pane),
        ),
    )
    return app


# ── entry point ───────────────────────────────────────────────────────────────

_args = _parse_args()
_data = _load(_args.h5, _args.umap)
app   = build_app(_args.h5, _data)
app.servable()   # picked up by `panel serve`

if __name__ == '__main__':
    app.show(port=_args.port, open=True)
