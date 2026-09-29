"""Shape summaries of log-normalized SFH bin weights."""

import numpy as np
from scipy.ndimage import gaussian_filter1d


def sfh_post_peak_decline(
    log_sfh,
    time,
    age_myr,
    epsilon=1e-10,
    smoothing_myr=300.0,
    minimum_peak_age_myr=300.0,
    minimum_pre_peak_duration_myr=300.0,
    minimum_decline_dex=0.3,
    maximum_secondary_peak_fraction=0.5,
    minimum_monotonic_fraction=0.7,
):
    """Measure the mean logarithmic decline from a clear SFH peak to observation.

    SFHs are smoothed in physical time before classification.  A history is
    retained only when its global maximum is interior, no second local maximum
    exceeds ``maximum_secondary_peak_fraction`` of the primary peak, both
    endpoints lie at least ``minimum_decline_dex`` below the peak, and the
    post-peak branch is predominantly monotonic.  The returned slope is
    positive for a decline toward observation and is measured in dex/Gyr::

        [log10(SFR_peak) - log10(SFR_observation)] / peak_lookback_time

    A second diagnostic measures the mean logarithmic rise from the oldest
    inferred endpoint to the peak, divided by the elapsed physical time.  It
    additionally requires a predominantly monotonic pre-peak branch.  Both
    slopes are positive and rates below 0.1% of the smoothed peak are floored
    for the endpoint ratios. Histories that fail the relevant criteria return
    NaN.
    """
    log_sfh = np.asarray(log_sfh, dtype=float)
    time = np.asarray(time, dtype=float)
    age = np.asarray(age_myr, dtype=float)
    if (log_sfh.ndim != 2 or time.ndim != 1 or len(time) < 3
            or log_sfh.shape[1] != len(time) or age.shape != (len(log_sfh),)
            or not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0)
            or time[0] < 0 or time[-1] > 1):
        raise ValueError('Expected SFH rows, an increasing fractional time grid, and one age.')
    if (smoothing_myr <= 0 or minimum_peak_age_myr < 0
            or minimum_pre_peak_duration_myr < 0
            or minimum_decline_dex < 0
            or not 0 < maximum_secondary_peak_fraction < 1
            or not 0 <= minimum_monotonic_fraction <= 1):
        raise ValueError('Invalid post-peak decline configuration.')

    with np.errstate(over='ignore', invalid='ignore'):
        weights = np.maximum(10.0 ** log_sfh - epsilon, 0.0)
        totals = weights.sum(axis=1)
    valid = (
        np.all(np.isfinite(weights), axis=1) & np.isfinite(totals) & (totals > 0)
        & np.isfinite(age) & (age > 0)
    )
    normalized = np.full_like(weights, np.nan)
    normalized[valid] = weights[valid] / totals[valid, None]

    # The Euclid grid is uniform in fractional lookback time.  Grouping nearby
    # Gaussian widths avoids one scipy call per galaxy while retaining a
    # physical 300 Myr smoothing scale for galaxies with different ages.
    fractional_step = float(np.median(np.diff(time)))
    sigma_bins = smoothing_myr / (age * fractional_step)
    sigma_bins = np.maximum(np.round(sigma_bins * 4.0) / 4.0, 0.5)
    smoothed = np.full_like(normalized, np.nan)
    for sigma in np.unique(sigma_bins[valid]):
        rows = valid & (sigma_bins == sigma)
        smoothed[rows] = gaussian_filter1d(
            normalized[rows], sigma=float(sigma), axis=1, mode='nearest',
        )

    safe = np.where(np.isfinite(smoothed), smoothed, -np.inf)
    peak_index = np.argmax(safe, axis=1)
    row_index = np.arange(len(weights))
    peak = safe[row_index, peak_index]
    interior = valid & (peak_index > 0) & (peak_index < len(time) - 1)

    local_maximum = np.zeros_like(smoothed, dtype=bool)
    local_maximum[:, 1:-1] = (
        (smoothed[:, 1:-1] > smoothed[:, :-2])
        & (smoothed[:, 1:-1] >= smoothed[:, 2:])
    )
    significant_peaks = np.sum(
        local_maximum & (smoothed >= maximum_secondary_peak_fraction * peak[:, None]),
        axis=1,
    )
    single_peak = significant_peaks == 1

    relative_floor = np.maximum(peak * 1e-3, np.finfo(float).tiny)
    observation = np.maximum(smoothed[:, 0], relative_floor)
    oldest = np.maximum(smoothed[:, -1], relative_floor)
    with np.errstate(divide='ignore', invalid='ignore'):
        decline_dex = np.log10(peak / observation)
        older_side_dex = np.log10(peak / oldest)
    peak_age_myr = time[peak_index] * age

    monotonic_fraction = np.full(len(weights), np.nan)
    pre_peak_monotonic_fraction = np.full(len(weights), np.nan)
    for row in np.flatnonzero(interior & single_peak):
        differences = np.diff(smoothed[row, :peak_index[row] + 1])
        tolerance = -0.02 * peak[row]
        monotonic_fraction[row] = np.mean(differences >= tolerance)
        pre_peak_differences = np.diff(smoothed[row, peak_index[row]:])
        pre_peak_monotonic_fraction[row] = np.mean(
            pre_peak_differences <= -tolerance
        )

    selected = (
        interior & single_peak
        & (peak_age_myr >= minimum_peak_age_myr)
        & (decline_dex >= minimum_decline_dex)
        & (older_side_dex >= minimum_decline_dex)
        & (monotonic_fraction >= minimum_monotonic_fraction)
    )
    slope = np.full(len(weights), np.nan, dtype=np.float32)
    slope[selected] = (decline_dex[selected] / (peak_age_myr[selected] / 1e3)).astype(
        np.float32
    )
    pre_peak_duration_myr = (time[-1] - time[peak_index]) * age
    pre_peak_selected = (
        selected
        & (pre_peak_duration_myr >= minimum_pre_peak_duration_myr)
        & (pre_peak_monotonic_fraction >= minimum_monotonic_fraction)
    )
    rise_slope = np.full(len(weights), np.nan, dtype=np.float32)
    rise_slope[pre_peak_selected] = (
        older_side_dex[pre_peak_selected]
        / (pre_peak_duration_myr[pre_peak_selected] / 1e3)
    ).astype(np.float32)
    return {
        'sfh_post_peak_decline_slope_dex_gyr': slope,
        'sfh_pre_peak_rise_slope_dex_gyr': rise_slope,
        'sfh_single_peak_declining': selected.astype(np.float32),
    }


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
        # A fixed 100 Myr window is easier to compare with PHZ SFRs than the
        # fractional 0.1-age window above. Integrate partial edge bins while
        # avoiding an (N_galaxy, N_bin) temporary overlap array.
        upper = np.minimum(1.0e8 / (age * 1.0e6), 1.0)
        recent_100 = np.zeros(len(weights), dtype=float)
        for column, (left, right, width) in enumerate(
            zip(edges[:-1], edges[1:], np.diff(edges))
        ):
            overlap = np.maximum(0.0, np.minimum(right, upper) - left)
            recent_100 += normalized[:, column] * overlap / width
        covered = good & (age >= 100.0)
        recent_100[~covered] = np.nan
        specific_100 = recent_100 / 1.0e8
        output['sfh_recent_mass_fraction_100myr'] = recent_100.astype(np.float32)
        output['sfh_log_sfr_per_stellar_mass_100myr_r0'] = np.log10(
            np.maximum(specific_100, 1e-15)
        ).astype(np.float32)

        peak_index = np.argmax(np.where(np.isfinite(normalized), normalized, -np.inf), axis=1)
        peak_fraction = time[peak_index]
        mean_fraction = normalized @ time
        peak_age = peak_fraction * age / 1.0e3
        mean_age = mean_fraction * age / 1.0e3
        peak_age[~good] = np.nan
        mean_age[~good] = np.nan
        output['sfh_peak_age_gyr'] = peak_age.astype(np.float32)
        output['sfh_mass_weighted_age_gyr'] = mean_age.astype(np.float32)
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
