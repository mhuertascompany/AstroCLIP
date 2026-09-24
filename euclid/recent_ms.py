"""Recent MS offsets integrated over matched physical-time windows."""
import numpy as np
from .main_sequence_sfh import main_sequence_along_sfh

WINDOWS = {'100 Myr': ('gyr', .1), '0.1 cosmic age': ('fraction', .1),
           '0.2 cosmic age': ('fraction', .2)}


def recent_offset(track, window_years, floor=-4.):
    """Log ratio of integrated rates, using partial-bin overlap.

    No missing-time extrapolation: windows not fully covered return NaN.
    Exact zero numerator is lower-censored at floor for plotting/filtering.
    """
    if 'sfr' not in track or not np.isfinite(window_years) or window_years <= 0:
        return np.nan
    dt = np.asarray(track['duration_years'], float)
    edges = np.r_[0., np.cumsum(dt)]
    if window_years > edges[-1]*(1+1e-10):
        return np.nan
    overlap = np.clip(window_years-edges[:-1], 0., dt)
    included = overlap > 0
    sfr, ms = np.asarray(track['sfr']), np.asarray(track['ms_sfr'])
    if not np.isfinite(sfr[included]+ms[included]).all():
        return np.nan
    numerator = np.sum(sfr[included]*overlap[included])
    denominator = np.sum(ms[included]*overlap[included])
    if numerator < 0 or denominator <= 0:
        return np.nan
    return max(float(np.log10(numerator/denominator)), floor) if numerator > 0 else floor


def catalog_recent_offsets(log_sfhs, time, redshift, masses, time_norm=None, **settings):
    values = np.full((len(log_sfhs), len(WINDOWS)), np.nan)
    for i, sfh in enumerate(log_sfhs):
        track = main_sequence_along_sfh(time, sfh, float(redshift[i]), float(masses[i]),
            time_norm_myr=None if time_norm is None else float(time_norm[i]), **settings)
        for j, (mode, width) in enumerate(WINDOWS.values()):
            years = width*1e9 if mode == 'gyr' else width*track.get('observation_age_years', np.nan)
            values[i,j] = recent_offset(track, years)
    return values


def matched_mask(mass, z, recent, mass_bounds, z_bounds, recent_bounds):
    mask = np.ones(len(mass), dtype=bool)
    for values, (low, high) in zip((mass, z, recent), (mass_bounds, z_bounds, recent_bounds)):
        a = np.asarray(values, float)
        mask &= np.isfinite(a) & (a >= low) & (a <= high)
    return mask
