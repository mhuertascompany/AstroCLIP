"""Separate positive and negative integrated deviations from the MS reference."""
import numpy as np
from .main_sequence_sfh import main_sequence_along_sfh


def integrated_deviations(track):
    """Use bin-integrated rates; undefined pre-formation bins are excluded.

    Weights and reference weights both equal SFR * dt / the same total mass.
    Thus summing them implements physical-time integration, without an extra dt.
    Coverage is the fraction of included physical time in the conservative domain.
    """
    if 'observed_weights' not in track:
        return np.nan, np.nan, np.nan
    blue = np.asarray(track['observed_weights'])
    green = np.asarray(track['weights'])
    dt = np.asarray(track['duration_years'])
    valid = np.isfinite(blue+green+dt) & (dt > 0) & (blue >= 0) & (green >= 0)
    denom = np.sum((blue+green)[valid])
    if not np.any(valid) or denom <= 0:
        return np.nan, np.nan, np.nan
    difference = blue-green
    plus = np.maximum(difference[valid], 0).sum()/denom
    minus = np.maximum(-difference[valid], 0).sum()/denom
    coverage = dt[valid & track['supported']].sum()/dt[valid].sum()
    return float(plus), float(minus), float(coverage)


def catalog_deviations(log_sfhs, time, redshift, masses, time_norm=None, **settings):
    values = np.full((len(log_sfhs), 3), np.nan)
    for i, sfh in enumerate(log_sfhs):
        track = main_sequence_along_sfh(time, sfh, float(redshift[i]), float(masses[i]),
                time_norm_myr=None if time_norm is None else float(time_norm[i]), **settings)
        values[i] = integrated_deviations(track)
    return values
