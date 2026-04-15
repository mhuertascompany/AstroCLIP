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
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import panel as pn
from bokeh.models import BasicTicker, ColorBar, ColumnDataSource, LinearColorMapper
from bokeh.palettes import Inferno256, Plasma256, Viridis256
from bokeh.plotting import figure as bk_figure

pn.extension(sizing_mode='stretch_width')

# ── constants ─────────────────────────────────────────────────────────────────
SFH_EPS   = 1e-10
N_DISPLAY = 16    # max stamps / SFHs shown at once
NCOLS     = 4     # columns in each gallery grid

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
        'redshifts':     'Redshift z',
        'zfinal':        'Redshift z (LePhare)',
        'radius_sersic': 'Sersic radius',
        'sersic':        'Sersic index n',
        'axratio_sersic':'Axis ratio b/a',
        'log_mass':      'log M★',
        'log_sfr':       'log SFR',
        'log_ssfr':      'log sSFR',
    }
    for npz_key, label in _prop_map.items():
        if npz_key in npz:
            color_props[label] = npz[npz_key].astype(float)

    # Family morphology columns
    for key in sorted(npz.files):
        if 'family' in key:
            label = key.replace('family_', 'P(').upper() + ')'
            color_props[label] = npz[key].astype(float)

    # Default color: first available
    default_color = next(iter(color_props), None)

    # HDF5 metadata
    with h5py.File(h5_path, 'r') as f:
        img_mean     = f.attrs['img_mean'].astype(np.float32)
        img_std      = f.attrs['img_std'].astype(np.float32)
        time_grid    = f['sfh_time_grid'][:]
        time_is_frac = time_grid.max() <= 1.0
        has_tnorm    = 'sfh_time_norm' in f

    return dict(
        xy           = xy,
        color_props  = color_props,
        default_color= default_color,
        h5_indices   = npz['h5_indices'].astype(int),
        galaxy_ids   = npz['galaxy_ids'],
        img_mean     = img_mean,
        img_std      = img_std,
        time_grid    = time_grid,
        time_is_frac = time_is_frac,
        has_tnorm    = has_tnorm,
    )


# ── rendering helpers ─────────────────────────────────────────────────────────

def _render_stamp(ax, img: np.ndarray, img_mean: np.ndarray, img_std: np.ndarray):
    ch = img[0] * img_std[0] + img_mean[0]          # un-normalise channel 0 (F150W)
    vmin, vmax = np.percentile(ch, [0.5, 99.5])
    ax.imshow(ch, origin='lower', cmap='gray',
              vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])


def _render_sfh(ax, sfh_log: np.ndarray, time_grid: np.ndarray,
                time_is_frac: bool, t_norm: float | None):
    sfr = np.maximum(10.0 ** sfh_log - SFH_EPS, SFH_EPS)
    if time_is_frac and t_norm is not None:
        tx = time_grid * float(t_norm) / 1e3   # fractional → Gyr
    else:
        tx = time_grid / 1e3                   # Myr → Gyr
    t, s = tx[1:], sfr[1:]                     # skip t=0 (log-axis)
    ax.step(t, s, where='post', color='steelblue', lw=0.9)
    ax.fill_between(t, s, step='post', alpha=0.2, color='steelblue')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(t[0] * 0.9, t[-1] * 1.1)
    ax.tick_params(labelsize=4)
    ax.set_xlabel('Lookback time [Gyr]', fontsize=4)
    ax.set_ylabel(r'SFR [M$_\odot$/yr]', fontsize=4)


def _blank(msg: str = '') -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.text(0.5, 0.5, msg, ha='center', va='center',
            transform=ax.transAxes, fontsize=10, color='#999999')
    ax.set_axis_off()
    return fig


def _make_gallery(h5_path: Path, h5_rows: np.ndarray, d: dict,
                  rng: np.random.Generator) -> tuple[plt.Figure, plt.Figure]:
    n = min(N_DISPLAY, len(h5_rows))
    sample = np.sort(rng.choice(len(h5_rows), n, replace=False))
    rows   = h5_rows[sample]

    with h5py.File(h5_path, 'r') as f:
        imgs    = f['images'][list(rows)]           # (n, 3, 64, 64)
        sfhs    = f['sfh'][list(rows)]              # (n, 50)
        zs      = f['redshift'][list(rows)]         # (n,)
        t_norms = (f['sfh_time_norm'][list(rows)]   # (n,) or None
                   if d['has_tnorm'] else [None] * n)

    nrows = max(1, (n + NCOLS - 1) // NCOLS)
    kw    = dict(figsize=(NCOLS * 1.7, nrows * 1.7), squeeze=False)

    fig_img, axs_img = plt.subplots(nrows, NCOLS, **kw)
    fig_sfh, axs_sfh = plt.subplots(nrows, NCOLS, **kw)

    for axs in (axs_img.flatten(), axs_sfh.flatten()):
        for ax in axs:
            ax.set_visible(False)

    for i in range(n):
        axs_img.flatten()[i].set_visible(True)
        _render_stamp(axs_img.flatten()[i], imgs[i],
                      d['img_mean'], d['img_std'])
        axs_img.flatten()[i].set_title(f'z={zs[i]:.2f}', fontsize=5, pad=1)

        axs_sfh.flatten()[i].set_visible(True)
        _render_sfh(axs_sfh.flatten()[i], sfhs[i],
                    d['time_grid'], d['time_is_frac'], t_norms[i])
        axs_sfh.flatten()[i].set_title(f'z={zs[i]:.2f}', fontsize=5, pad=1)

    for fig in (fig_img, fig_sfh):
        fig.tight_layout(pad=0.4)

    return fig_img, fig_sfh


# ── Panel application ─────────────────────────────────────────────────────────

def build_app(h5_path: Path, d: dict) -> pn.viewable.Viewable:
    rng = np.random.default_rng(42)

    # ── Bokeh UMAP scatter ────────────────────────────────────────────────
    xy0    = d['xy']['Joint (img + SFH)']
    c0_key = d['default_color']
    c0     = d['color_props'].get(c0_key, np.zeros(len(xy0)))
    c0_s   = np.where(np.isfinite(c0), c0, np.nanmedian(c0[np.isfinite(c0)]))

    src = ColumnDataSource(dict(
        x  = xy0[:, 0].tolist(),
        y  = xy0[:, 1].tolist(),
        c  = c0_s.tolist(),
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
    plot.scatter(
        'x', 'y', source=src,
        color=dict(field='c', transform=mapper),
        size=2.5, alpha=0.7, line_width=0,
        selection_color='white',  selection_alpha=1.0,
        nonselection_alpha=0.12,
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

    img_pane = pn.pane.Matplotlib(
        _blank('Draw a selection on the UMAP'), tight=True, width=550)
    sfh_pane = pn.pane.Matplotlib(
        _blank('Draw a selection on the UMAP'), tight=True, width=550)

    # ── callbacks ─────────────────────────────────────────────────────────
    def _update_embed(event):
        xy = d['xy'][embed_w.value]
        src.data = dict(src.data,
                        x=xy[:, 0].tolist(),
                        y=xy[:, 1].tolist())
        plot.title.text = f'UMAP ({embed_w.value}) — draw to select'

    def _update_color(event):
        vals = d['color_props'].get(color_w.value, np.zeros(len(xy0)))
        safe = np.where(np.isfinite(vals), vals,
                        np.nanmedian(vals[np.isfinite(vals)]))
        src.data = dict(src.data, c=safe.tolist())
        mapper.low  = float(np.nanpercentile(safe, 1))
        mapper.high = float(np.nanpercentile(safe, 99))

    def _refresh(event=None):
        sel = src.selected.indices
        if not sel:
            info_md.object = '_No galaxies selected._'
            img_pane.object = _blank('No selection')
            sfh_pane.object = _blank('No selection')
            return
        h5_rows = d['h5_indices'][np.array(sel, dtype=int)]
        n_show  = min(N_DISPLAY, len(h5_rows))
        info_md.object = (f'**{len(sel):,}** selected '
                          f'— showing {n_show} random examples')
        fig_img, fig_sfh = _make_gallery(h5_path, h5_rows, d, rng)
        img_pane.object = fig_img
        sfh_pane.object = fig_sfh
        plt.close('all')

    embed_w.param.watch(_update_embed, 'value')
    color_w.param.watch(_update_color, 'value')
    resample_btn.on_click(_refresh)
    src.selected.on_change('indices', lambda attr, old, new: _refresh())

    # ── layout ────────────────────────────────────────────────────────────
    sidebar = pn.Column(
        pn.pane.Markdown('## COSMOS-Web CLIP\n### Explorer'),
        pn.layout.Divider(),
        embed_w,
        color_w,
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
            pn.Column(pn.pane.Markdown('#### F150W stamps'), img_pane),
            pn.Column(pn.pane.Markdown('#### CIGALE SFHs'),  sfh_pane),
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
