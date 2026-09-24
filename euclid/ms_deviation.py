"""Separate positive and negative integrated deviations from the MS reference."""
import numpy as np
from functools import lru_cache
from scipy.interpolate import PchipInterpolator
from .cosmic_sfh import COSMOLOGY
from .main_sequence_sfh import main_sequence_along_sfh


@lru_cache(maxsize=1)
def _log_scale_factor_from_age():
    # A shared inverse cosmology avoids thousands of per-bin root solves.
    log_a = np.linspace(-np.log1p(1e5), 0., 8192)
    age = COSMOLOGY.age(np.expm1(-log_a)).to_value('yr')
    return PchipInterpolator(age, log_a, extrapolate=False)


def accumulated_expansion_time(age_years, lookback_years):
    """Integral H dt = ln(a_obs/a_past), not a calibrated halo time count.

    Nonphysical or out-of-table epochs are NaN; no extrapolation near Big Bang.
    """
    inverse = _log_scale_factor_from_age()
    past = age_years - np.asarray(lookback_years, dtype=float)
    result = np.asarray(inverse(age_years) - inverse(past))
    return np.where((past > 0) & (past <= age_years), result, np.nan)


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


def history_summary(track, floor=-3.):
    """Signed extrema and time-weighted mean of lower-floored log(SFR/MS).

    Times are integral H dt from each bin midpoint to observation, equivalent
    to ln(a_obs/a_bin). Ties select the most
    recent bin. An absent positive/negative excursion has amplitude 0 and time
    NaN. Undefined pre-formation bins are excluded; zero SFR is floored.
    """
    if not np.isfinite(floor) or floor >= 0:
        raise ValueError('Delta MS floor must be finite and negative.')
    result = np.full(6, np.nan)
    if 'delta_ms' not in track:
        return result
    delta = np.asarray(track['delta_ms'], dtype=float)
    dt = np.asarray(track['duration_years'], dtype=float)
    valid = ~np.isnan(delta) & ~np.isposinf(delta) & np.isfinite(dt) & (dt > 0)
    if not valid.any():
        return result
    age = float(track['observation_age_years'])
    if not np.isfinite(age) or age <= 0:
        return result
    times = accumulated_expansion_time(age, np.cumsum(dt)-.5*dt)
    clipped = np.maximum(delta[valid], floor)
    selected_times = times[valid]
    high, low = np.argmax(clipped), np.argmin(clipped)
    result[:] = [max(clipped[high], 0.), min(clipped[low], 0.),
                 selected_times[high] if clipped[high] > 0 else np.nan,
                 selected_times[low] if clipped[low] < 0 else np.nan,
                 np.average(clipped, weights=dt[valid]),
                 dt[valid][delta[valid] < floor].sum()/dt[valid].sum()]
    return result


def catalog_deviations(log_sfhs, time, redshift, masses, time_norm=None, include_summary=False, delta_floor=-3., **settings):
    values = np.full((len(log_sfhs), 9 if include_summary else 3), np.nan)
    for i, sfh in enumerate(log_sfhs):
        track = main_sequence_along_sfh(time, sfh, float(redshift[i]), float(masses[i]),
                time_norm_myr=None if time_norm is None else float(time_norm[i]), **settings)
        values[i, :3] = integrated_deviations(track)
        if include_summary:
            values[i, 3:] = history_summary(track, delta_floor)
    return values
