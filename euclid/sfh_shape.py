"""Shape summaries of log-normalized SFH bin weights."""

import numpy as np


def sfh_recent_activity(log_sfh, time, epsilon=1e-10, age_myr=None):
    """Recent birthrate and bounded trend from mass in [0,.1] and [.1,.2].

    Integrate partial bins assuming constant SFR within each bin, using the
    same midpoint edges as preprocessing. Positive trend means rising toward
    observation. Physical rates use formed mass, not surviving stellar mass.
    """
    log_sfh = np.asarray(log_sfh, dtype=float)
    time = np.asarray(time, dtype=float)
    if (log_sfh.ndim != 2 or time.ndim != 1 or len(time) < 2
            or log_sfh.shape[1] != len(time)
            or not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0)
            or time[0] < 0 or time[-1] > 1):
        raise ValueError('Expected SFH rows on an increasing fractional time grid.')
    edges = np.r_[0., (time[:-1] + time[1:]) / 2, 1.]
    with np.errstate(over='ignore', invalid='ignore'):
        weights = np.maximum(10.**log_sfh - epsilon, 0.)
        totals = weights.sum(axis=1)
    valid = np.all(np.isfinite(weights), axis=1) & np.isfinite(totals) & (totals > 0)
    normalized = np.full_like(weights, np.nan)
    normalized[valid] = weights[valid] / totals[valid, None]

    def mass_between(lower, upper):
        overlap = np.maximum(0., np.minimum(edges[1:], upper) - np.maximum(edges[:-1], lower))
        return normalized @ (overlap / np.diff(edges))

    recent = mass_between(0., 0.1)
    previous = mass_between(0.1, 0.2)
    trend = np.full(len(weights), np.nan)
    active = valid & ((recent + previous) > 0)
    trend[active] = (recent[active] - previous[active]) / (recent[active] + previous[active])
    output = {
        'sfh_recent_birthrate': (recent / 0.1).astype(np.float32),
        'sfh_recent_trend': trend.astype(np.float32),
    }
    if age_myr is not None:
        age = np.asarray(age_myr, dtype=float)
        if age.shape != totals.shape:
            raise ValueError('age_myr must contain one value per SFH.')
        rate = np.full(len(weights), np.nan)
        good = valid & np.isfinite(age) & (age > 0)
        rate[good] = recent[good] / (0.1 * age[good] * 1e6)
        output['sfh_recent_sfr_per_formed_mass'] = rate
        # Finite display floor keeps zero-rate galaxies selectable.
        output['sfh_log_recent_sfr_per_formed_mass'] = np.log10(np.maximum(rate, 1e-15))
    return output


def sfh_duration_80(log_sfh, time, epsilon=1e-10):
    """Central 80% mass interval, Q90-Q10, in the supplied time coordinate.

    Quantiles use the first bin center reaching the cumulative fraction;
    precision is limited by the SFH grid. Invalid/zero-mass rows return NaN.
    """
    log_sfh = np.asarray(log_sfh, dtype=float)
    time = np.asarray(time, dtype=float)
    if (log_sfh.ndim != 2 or time.ndim != 1
            or log_sfh.shape[1] != len(time) or len(time) < 2
            or not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0)):
        raise ValueError('Expected SFH rows on a finite, increasing time grid.')
    with np.errstate(over='ignore', invalid='ignore'):
        weights = np.maximum(10.0 ** log_sfh - epsilon, 0)
        totals = weights.sum(axis=1)
    valid = np.all(np.isfinite(weights), axis=1) & np.isfinite(totals) & (totals > 0)
    output = np.full(len(weights), np.nan, dtype=np.float32)
    cumulative = np.cumsum(weights[valid] / totals[valid, None], axis=1)
    q10 = time[np.argmax(cumulative >= 0.1, axis=1)]
    q90 = time[np.argmax(cumulative >= 0.9, axis=1)]
    output[valid] = q90 - q10
    return output
