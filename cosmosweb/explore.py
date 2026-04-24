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

# Fractional lookback-time grid matching prepare_dataset.py
# bin 0 = observation epoch (t_frac=0.0), bin 49 = Big Bang (t_frac=1.0)
_T_FRAC = np.linspace(0, 1, SFH_N_BINS)   # (50,)  matches SFH_T_FRAC in prepare_dataset.py

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
    Bin 0 = most recent lookback time (t_frac=0.02).
    Bin 49 = oldest lookback time   (t_frac=1.0).
    """
    with h5py.File(h5_path, 'r') as f:
        sfh_log = f['sfh'][list(h5_indices)].astype(np.float64)  # (N, 50)

    # Convert to linear fractions; re-normalise to correct for log rounding
    sfr = np.maximum(10.0 ** sfh_log - SFH_EPS, 0.0)
    sfr_sum = sfr.sum(axis=1, keepdims=True)
    sfr_sum = np.where(sfr_sum > 0, sfr_sum, 1.0)
    sfr = sfr / sfr_sum                                  # (N, 50), sums to 1

    t = _T_FRAC[np.newaxis, :]                           # (1, 50)

    # ── 1. log ratio: old SFR / recent SFR ───────────────────────────────────
    # Positive → dominated by old stars (quiescent / early-type)
    # Negative → dominated by recent SF (star-forming / late-type)
    log_old_recent = sfh_log[:, -1] - sfh_log[:, 0]

    # ── 2. Mass-weighted mean formation epoch (fractional lookback time) ──────
    # High (→1) = formed mostly at early times = old stellar population
    # Low  (→0) = formed mostly recently       = young stellar population
    mean_t = (sfr * t).sum(axis=1)

    # ── 3. Fraction of SFH in most recent 20% of lookback time (bins 0–9) ────
    n_recent = max(1, SFH_N_BINS // 5)
    f_recent = sfr[:, :n_recent].sum(axis=1)

    # ── 4. Fractional lookback time of the SFH peak ───────────────────────────
    peak_bin = np.argmax(sfr, axis=1)
    peak_t   = _T_FRAC[peak_bin]

    return {
        'SFH: log(old/recent)':     log_old_recent,
        'SFH: mean formation epoch': mean_t,
        'SFH: f(recent 20%)':        f_recent,
        'SFH: peak lookback t_frac': peak_t,
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
    rng = np.random.default_rng(42)

    # ── Bokeh UMAP scatter ────────────────────────────────────────────────
    xy0    = d['xy']['Joint (img + SFH)']
    c0_key = d['default_color']
    c0     = d['color_props'].get(c0_key, np.zeros(len(xy0)))
    c0_s   = np.where(np.isfinite(c0), c0, np.nanmedian(c0[np.isfinite(c0)]))

    src = ColumnDataSource(dict(
        x           = xy0[:, 0].tolist(),
        y           = xy0[:, 1].tolist(),
        c           = c0_s.tolist(),
        cluster_hex = d['cluster_hex'],
    ))

    mapper = LinearColorMapper(
        palette = Plasma256,
        low     = float(np.nanpercentile(c0_s, 1)),
        high    = float(np.nanpercentile(c0_s, 99)),
    )
    cbar = ColorBar(color_mapper=mapper, ticker=BasicTicker(),
                    label_standoff=8, width=12, location=(0, 0))

    plot = bk_figure(
        width=580, height=520,
        tools='lasso_select,box_select,wheel_zoom,pan,reset',
        active_drag='lasso_select',
        title='Draw a lasso or box to select galaxies',
        output_backend='webgl',
    )
    # Redshift filter view — updated by the z RangeSlider
    z_all  = d['redshifts']
    z_safe = np.where(np.isfinite(z_all), z_all, -1.0)
    z_view = CDSView(filter=BooleanFilter(booleans=[True] * len(z_safe)))

    r_cont = plot.scatter(
        'x', 'y', source=src, view=z_view,
        color=dict(field='c', transform=mapper),
        size=2.5, alpha=0.7, line_width=0,
        selection_color='white', selection_alpha=1.0,
        nonselection_alpha=0.12,
    )
    r_clust = plot.scatter(
        'x', 'y', source=src, view=z_view,
        fill_color='cluster_hex', line_width=0,
        size=2.5, alpha=0.7,
        selection_fill_color='white', selection_alpha=1.0,
        nonselection_alpha=0.12,
        visible=False,
    )
    plot.add_layout(cbar, 'right')
    plot.xaxis.axis_label = 'UMAP 1'
    plot.yaxis.axis_label = 'UMAP 2'

    # ── widgets ───────────────────────────────────────────────────────────
    embed_w = pn.widgets.Select(
        name='Embedding space',
        options=list(d['xy'].keys()),
        value='Joint (img + SFH)',
        width=200,
    )
    color_w = pn.widgets.Select(
        name='Color by',
        options=list(d['color_props'].keys()),
        value=c0_key or '',
        width=200,
    )
    info_md = pn.pane.Markdown(
        '_Draw a selection on the UMAP._', width=210)
    resample_btn = pn.widgets.Button(
        name='New random sample', button_type='primary', width=200)

    # HTML panes are used instead of Matplotlib panes so that every update
    # is a fresh object (new bytes) — Panel detects the change reliably.
    img_pane      = pn.pane.HTML(_BLANK_HTML, width=550)
    sfh_pane      = pn.pane.HTML(_BLANK_HTML, width=550)
    mean_sfh_pane = pn.pane.HTML(_BLANK_HTML, width=400)

    # Colour range slider (synced to current property; user can drag to override)
    _c0_lo = float(np.nanpercentile(c0_s, 1))
    _c0_hi = float(np.nanpercentile(c0_s, 99))
    cbar_slider = pn.widgets.RangeSlider(
        name='Colour range',
        start=float(np.nanmin(c0_s)), end=float(np.nanmax(c0_s)),
        value=(_c0_lo, _c0_hi),
        step=float((np.nanmax(c0_s) - np.nanmin(c0_s)) / 200),
        width=200,
    )

    def _on_cbar_range(event):
        if color_w.value == 'Cluster (HDBSCAN)':
            return
        mapper.low, mapper.high = float(cbar_slider.value[0]), float(cbar_slider.value[1])

    cbar_slider.param.watch(_on_cbar_range, 'value')

    # Redshift range slider
    z_finite = z_safe[np.isfinite(z_all)]
    z_lo = float(np.floor(z_finite.min() * 10) / 10) if len(z_finite) else 0.0
    z_hi = float(np.ceil( z_finite.max() * 10) / 10) if len(z_finite) else 6.0
    z_slider = pn.widgets.RangeSlider(
        name='Redshift range', start=z_lo, end=z_hi,
        value=(z_lo, z_hi), step=0.1, width=200,
    )

    def _on_z_filter(event):
        lo, hi = z_slider.value
        z_view.filter.booleans = [bool(lo <= z <= hi) for z in z_safe]
        src.selected.indices = []
        _refresh([])

    z_slider.param.watch(_on_z_filter, 'value')

    # Cluster selector widget (only shown when clusters are available)
    n_clusters = d['n_clusters']
    if n_clusters > 0 and d['cluster_labels'] is not None:
        labels = d['cluster_labels']
        has_noise = bool((labels == -1).any())
        cl_opts = (['— all —'] +
                   (['noise'] if has_noise else []) +
                   [f'cluster {i}' for i in range(n_clusters)])
        cluster_w = pn.widgets.Select(
            name='Jump to cluster', options=cl_opts, value='— all —', width=200)

        def _on_cluster_select(event):
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

        cluster_w.param.watch(_on_cluster_select, 'value')
        cluster_widget_row = pn.Column(pn.layout.Divider(), cluster_w)
    else:
        cluster_widget_row = pn.pane.Markdown('')

    # ── callbacks ─────────────────────────────────────────────────────────
    def _update_embed(event):
        xy = d['xy'][embed_w.value]
        src.data['x'] = xy[:, 0].tolist()
        src.data['y'] = xy[:, 1].tolist()
        src.selected.indices = []
        plot.title.text = f'UMAP ({embed_w.value}) — draw to select'

    def _update_color(event):
        if color_w.value == 'Cluster (HDBSCAN)':
            r_cont.visible  = False
            r_clust.visible = True
            return
        r_cont.visible  = True
        r_clust.visible = False
        vals = d['color_props'].get(color_w.value, np.zeros(len(xy0)))
        safe = np.where(np.isfinite(vals), vals,
                        np.nanmedian(vals[np.isfinite(vals)]))
        src.data['c'] = safe.tolist()
        lo = float(np.nanpercentile(safe, 1))
        hi = float(np.nanpercentile(safe, 99))
        mapper.low  = lo
        mapper.high = hi
        # sync the range slider to the new property — suppress its callback
        # by updating start/end/value together
        cbar_slider.start = float(np.nanmin(safe))
        cbar_slider.end   = float(np.nanmax(safe))
        cbar_slider.value = (lo, hi)

    def _refresh(sel):
        sel = list(sel)
        if not sel:
            info_md.object = '_No galaxies selected._'
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
        if len(h5_rows) >= 20:
            mean_sfh_pane.object = _fig_to_html(_mean_sfh_fig(h5_path, h5_rows))
        else:
            mean_sfh_pane.object = _BLANK_HTML

    def _on_resample(event):
        _refresh(src.selected.indices)

    # Poll for selection changes every 350 ms.  This is more reliable than
    # src.selected.on_change in Panel's server context because slow HDF5 I/O
    # inside a Bokeh on_change callback can block the Tornado IOLoop and cause
    # subsequent selection events to be silently dropped.
    _prev_key: list[tuple] = [()]

    def _poll():
        key = tuple(sorted(src.selected.indices))
        if key == _prev_key[0]:
            return
        _prev_key[0] = key
        _refresh(list(src.selected.indices))

    embed_w.param.watch(_update_embed, 'value')
    color_w.param.watch(_update_color, 'value')
    resample_btn.on_click(_on_resample)
    pn.state.add_periodic_callback(_poll, period=350)

    # ── layout ────────────────────────────────────────────────────────────
    sidebar = pn.Column(
        pn.pane.Markdown('## COSMOS-Web CLIP\n### Explorer'),
        pn.layout.Divider(),
        embed_w,
        color_w,
        cbar_slider,
        cluster_widget_row,
        pn.layout.Divider(),
        z_slider,
        pn.layout.Divider(),
        info_md,
        resample_btn,
        pn.pane.Markdown(
            '_Lasso or box-select points on the UMAP.  '
            f'Up to {N_DISPLAY} random examples are shown.  '
            'Click **New random sample** for a different draw._',
            styles={'font-size': '11px', 'color': '#888888'},
        ),
        width=220,
    )

    app = pn.Column(
        pn.Row(pn.pane.Bokeh(plot), sidebar),
        pn.layout.Divider(),
        pn.Row(
            pn.Column(pn.pane.Markdown('#### F277W stamps'), img_pane),
            pn.Column(pn.pane.Markdown('#### CIGALE SFHs'),  sfh_pane),
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
