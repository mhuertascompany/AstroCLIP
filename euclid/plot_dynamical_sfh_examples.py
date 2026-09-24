"""Compare fractional-time SFHs with a mass-conserving expansion-time grid."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .cosmic_sfh import COSMOLOGY


def rebin_expansion_time(weights, fractional_centers, time_norm_gyr, redshift, u_edges):
    """Piecewise-constant physical-time rate, integrated into common u bins.

    u=ln(a_obs/a_past); bin mass is conserved inside the selected interval.
    No renormalization of truncated mass and no rescaling of the source history.
    """
    weights = np.asarray(weights, dtype=float)
    if np.any(weights < 0) or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError('Expected finite nonnegative mass weights.')
    if np.any(np.diff(u_edges) <= 0) or u_edges[0] != 0:
        raise ValueError('Expected increasing u edges starting at zero.')
    weights = weights / weights.sum()
    edges = np.r_[0., (fractional_centers[:-1]+fractional_centers[1:])/2, 1.]*time_norm_gyr
    age = COSMOLOGY.age(redshift).to_value('Gyr')
    z_edges = (1+redshift)*np.exp(u_edges)-1
    target = age-COSMOLOGY.age(z_edges).to_value('Gyr')
    target[0] = 0.
    cumulative = np.interp(target, edges, np.r_[0., np.cumsum(weights)], left=0., right=1.)
    mass = np.maximum(np.diff(cumulative), 0.)
    return mass, target, edges, weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--h5', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('euclid/diagnostics/dynamical_sfh_examples'))
    parser.add_argument('--u-max', type=float, default=3.)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.u_max <= 8:
        parser.error('--u-max must be in (0,8]')
    with h5py.File(args.h5) as h:
        time = h['sfh_time_grid'][:].astype(float)
        raw = np.maximum(10.**h['sfh'][:].astype(float)-float(h.attrs.get('sfh_log_epsilon', 1e-10)), 0.)
        ids, z = h['galaxy_id'][:], h['redshift'][:]
        scale = h['sfh_time_norm'][:]/1000.
    valid = np.flatnonzero(np.isfinite(raw).all(axis=1) & (raw.sum(axis=1)>0) & np.isfinite(z+scale) & (z>=0) & (scale>0))
    if len(valid) < 6:
        raise ValueError('Need at least six valid SFHs.')
    # Sample across peak-time quantiles to expose different history shapes.
    density = raw[valid]/np.diff(np.r_[0., (time[:-1]+time[1:])/2, 1.])
    ordered = valid[np.argsort(np.argmax(density, axis=1), kind='stable')]
    rng = np.random.default_rng(args.seed)
    chosen = [int(rng.choice(group)) for group in np.array_split(ordered, 6)]
    u_edges = np.linspace(0., args.u_max, 251)
    fig, axes = plt.subplots(6, 3, figsize=(13, 16), layout='constrained')
    saved, report = {}, []
    for row, i in enumerate(chosen):
        mass, target, edges, w = rebin_expansion_time(raw[i], time, scale[i], float(z[i]), u_edges)
        du, dt = np.diff(u_edges), np.diff(target)
        f_edges = edges/scale[i]
        axes[row, 0].stairs(w/np.diff(f_edges), f_edges, color='#236C9C', lw=1.4)
        # Rate per fractional physical time on a new x coordinate only.
        axes[row, 1].stairs(mass/dt*scale[i], u_edges, color='#236C9C', lw=1.4)
        axes[row, 2].stairs(mass/du, u_edges, color='#B55A22', lw=1.4)
        axes[row, 0].set_title(f'{int(ids[i])}   z={z[i]:.2f}', fontsize=10)
        axes[row, 1].set_title('Physical-time rate on the u axis', fontsize=10)
        axes[row, 2].set_title(f'Mass per u · retained {100*mass.sum():.2f}%', fontsize=10)
        for col, ax in enumerate(axes[row]):
            ax.set_xlim(0, 1 if col==0 else args.u_max)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=.18)
            ax.set_ylabel('d(mass fraction)/df' if col<2 else 'd(mass fraction)/du')
            ax.set_xlabel('f = lookback / saved cosmic-age scale' if col==0 else 'u = ln(a_obs/a_past) = ∫ H dt')
        report.append(dict(object_id=str(ids[i]), redshift=float(z[i]), retained_mass_fraction=float(mass.sum())))
        saved[str(ids[i])] = mass
        # Verify retained mass independently by physical overlap with input bins.
        overlap = np.clip(target[-1]-edges[:-1], 0, np.diff(edges))
        np.testing.assert_allclose(mass.sum(), np.sum(w*overlap/np.diff(edges)), atol=1e-12)
    fig.suptitle('SFH shapes on fractional-age and accumulated dynamical-time coordinates\n'
                 'Same six median histories · 250 common u bins · linear vertical scales', fontsize=15)
    fig.supxlabel('Left: original mass density in fractional time. Middle: change the time axis only. '
                   'Right: conserve bin mass, including the 1/H Jacobian.\n'
                   'Each row is selected from a different peak-time quantile; no uncertainty draws. '
                   'Finite u cutoff; retained mass is not renormalized. H₀=70, Ωm=0.3.', fontsize=10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('.pdf', '.png'):
        fig.savefig(args.output.with_suffix(ext), dpi=180)
    np.savez(args.output.with_suffix('.npz'), u_edges=u_edges, **saved)
    args.output.with_suffix('.json').write_text(json.dumps(dict(source=str(args.h5), seed=args.seed,
        u_max=args.u_max, bins=250, examples=report), indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
