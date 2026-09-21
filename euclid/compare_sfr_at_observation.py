"""Compare catalog and 100 Myr SFH rates to the existing Speagle MS, R=0."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from astropy.table import Table
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from .cosmic_sfh import COSMOLOGY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--h5', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Output filename stem')
    args = parser.parse_args()
    table = Table.read(args.catalog)
    names = {n.lower(): n for n in table.colnames}
    ids = np.asarray(table[names['object_id']], dtype=np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError('Duplicate catalog IDs')
    lookup = {int(x): i for i, x in enumerate(ids)}
    with h5py.File(args.h5) as h:
        gid = h['galaxy_id'][:]
        redshift = h['redshift'][:].astype(float)
        duration = h['sfh_time_norm'][:].astype(float)*1e6
        time = h['sfh_time_grid'][:].astype(float)
        weights = np.maximum(10.**h['sfh'][:].astype(float)-1e-10, 0.)
    rows = np.array([lookup[int(x)] for x in gid])
    def column(name):
        return np.asarray(np.ma.asarray(table[names[name]], dtype=float).filled(np.nan))[rows]
    catalog = column('phz_pp_median_sfr')
    mass = column('phz_pp_median_stellarmass')
    total = weights.sum(axis=1)
    weights /= np.where(total > 0, total, np.nan)[:, None]
    edges = np.r_[0., (time[1:]+time[:-1])/2, 1.][None, :]*duration[:, None]
    recent = np.sum(weights*np.maximum(0., np.minimum(edges[:, 1:], 1e8)-edges[:, :-1])
                    /np.diff(edges), axis=1)
    valid = (np.isfinite(catalog+mass+redshift+duration+recent) & (redshift >= 0)
             & (duration >= 1e8) & (total > 0))
    if 'physical_parameters_matched' in names:
        valid &= np.asarray(table[names['physical_parameters_matched']], dtype=bool)[rows]
    cosmic_age = np.full(len(gid), np.nan)
    cosmic_age[valid] = COSMOLOGY.age(redshift[valid]).to_value('Gyr')
    # Same Speagle Eq. 28 and zero IMF mass offset as the explorer overlay.
    ms = (.84-.026*cosmic_age)*mass-(6.51-.11*cosmic_age)
    sfh = np.full(len(gid), np.nan)
    positive = valid & (recent > 0)
    zero = valid & (recent == 0)
    sfh[positive] = np.log10(recent[positive]/1e8)+mass[positive]  # R=0
    dc, ds = catalog-ms, sfh-ms
    assert np.allclose((ds-dc)[positive], (sfh-catalog)[positive])
    def stats(a):
        a = a[np.isfinite(a)]
        return {'n': len(a), 'p16_p50_p84_dex': np.percentile(a, [16, 50, 84]).tolist()} if len(a) else {'n': 0}
    supported = valid & (cosmic_age >= 2.5) & (cosmic_age <= 11.5) & (mass >= 9.7) & (mass <= 11.1)
    near = valid & (np.abs(dc) <= .3)
    report = dict(n_valid=int(valid.sum()), n_positive_sfh=int(positive.sum()), n_zero_sfh=int(zero.sum()),
        R=0, window_myr=100, ms='Speagle 2014 Eq28, observed mass/redshift, zero IMF mass offset',
        catalog_minus_ms_all=stats(dc[valid]), catalog_minus_ms_same_positive_sample=stats(dc[positive]),
        sfh_minus_ms_same_positive_sample=stats(ds[positive]), sfh_minus_catalog=stats((sfh-catalog)[positive]),
        catalog_within_03dex_MS=dict(n=int(near.sum()), zero_sfh=int((near&zero).sum()),
            sfh_minus_ms_positive=stats(ds[near&positive])),
        conservative_display_domain=dict(n=int(supported.sum()), n_positive=int((supported&positive).sum()),
            catalog_minus_ms=stats(dc[supported&positive]), sfh_minus_ms=stats(ds[supported&positive])),
        limitations=['No PHZ quality flags', 'SFH rate from rebinned pointwise median history',
                     'MS is an empirical reference, not ground truth; IMF calibration unverified'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2))
    np.savez(args.output.with_suffix('.npz'), object_id=gid, redshift=redshift,
        log_stellar_mass=mass, catalog_log_sfr=catalog, sfh_log_sfr=sfh, ms_log_sfr=ms,
        valid=valid, zero_sfh=zero, conservative_domain=supported)
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.8), layout='constrained')
    common_norm = LogNorm(vmin=1)
    for ax, values, label in zip(axs[:2], [catalog, sfh], ['Catalog SFR (100 Myr)', 'SFH SFR (100 Myr), R=0']):
        mesh = ax.hexbin(ms[positive], values[positive], gridsize=60, mincnt=1,
            extent=(-2, 4, -2, 4), norm=common_norm, cmap='viridis', linewidths=0)
        ax.plot([-2, 4], [-2, 4], 'r--', lw=1)
        ax.set(xlim=(-2, 4), ylim=(-2, 4), xlabel='MS log SFR at observed mass and redshift',
               ylabel='log SFR [Msun/yr]', title=label, aspect='equal')
    # Share color normalization determined by both panels.
    common_norm.vmax = max(ax.collections[0].get_array().max() for ax in axs[:2])
    fig.colorbar(mesh, ax=list(axs[:2]), label='Galaxies per hexagon', shrink=.65)
    bins = np.linspace(-5, 3, 65)
    ax = axs[2]
    for value, label, color in [(dc, 'Catalog', 'tab:blue'), (ds, 'SFH, R=0', 'tab:orange')]:
        ax.hist(value[positive], bins=bins, weights=np.ones(positive.sum())/positive.sum(),
                histtype='step', lw=1.8, color=color, label=f'{label}: median {np.median(value[positive]):+.2f}')
    ax.axvline(0, color='k', ls='--', lw=1)
    ax.set(xlabel='log(SFR / MS SFR)', ylabel='Fraction of common positive-SFH sample per bin',
           title='Residuals for the same galaxies', xlim=(-5, 3))
    ax.legend(fontsize=9)
    fig.suptitle(f'Observed-epoch comparison: {positive.sum():,} galaxies in all panels; '
                 f'{zero.sum():,} zero-SFH rates omitted', fontsize=14)
    fig.supxlabel('SFH SFR = mass fraction in 0–100 Myr × observed stellar mass / 100 Myr; '
                  'no return correction.\nSpeagle MS uses observed M★ and cosmic age(z), '
                  'without reconstructing past mass. Statistics include values outside display limits.\n'
                  'No PHZ quality cuts; no IMF offset applied. SFH summary is the normalized median history.', fontsize=10)
    for extension in ('.png', '.pdf'):
        fig.savefig(args.output.with_suffix(extension), dpi=180, bbox_inches='tight')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
