"""Doubling-number and quenching summaries derived from normalized SFHs."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .cosmic_sfh import COSMOLOGY


def _mass_quantile_lookback(weights, edges, quantile):
    cumulative = np.cumsum(weights, axis=1)
    index = np.argmax(cumulative >= quantile, axis=1)
    rows = np.arange(len(weights))
    previous = np.where(index > 0, cumulative[rows, np.maximum(index - 1, 0)], 0.0)
    mass = weights[rows, index]
    fraction = np.divide(
        quantile - previous, mass,
        out=np.zeros_like(previous), where=mass > 0,
    )
    fraction = np.clip(fraction, 0.0, 1.0)
    return edges[index] + fraction * (edges[index + 1] - edges[index])


@lru_cache(maxsize=1)
def _age_redshift_lookup():
    redshift = np.expm1(np.linspace(0.0, np.log1p(100.0), 50000))
    age = np.asarray(COSMOLOGY.age(redshift).to_value("Gyr"), dtype=float)
    return age[::-1], redshift[::-1]


def _redshift_at_cosmic_age(age_gyr):
    ages, redshifts = _age_redshift_lookup()
    values = np.asarray(age_gyr, dtype=float)
    output = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values) & (values >= ages[0]) & (values <= ages[-1])
    output[valid] = np.interp(values[valid], ages, redshifts)
    return output


def _crossing_position(x0, y0, x1, y1, threshold):
    if not np.isfinite(y0 + y1) or y1 == y0:
        return 0.5 * (x0 + x1)
    fraction = np.clip((threshold - y0) / (y1 - y0), 0.0, 1.0)
    return x0 + fraction * (x1 - x0)


def sfh_quenching_summaries(
    log_sfh,
    time,
    age_myr,
    log_stellar_mass,
    redshift,
    epsilon=1e-10,
    recent_window_myr=100.0,
    smoothing_myr=300.0,
):
    """Compute SF/transition/quiescent and quenching-history diagnostics.

    The normalized formed-mass SFH is multiplied by the catalog stellar mass
    to obtain an ``R=0`` physical SFR, and divided by that same catalog mass to
    obtain sSFR.  The doubling number uses the age of the Universe at the
    observed redshift for the whole reconstructed history::

        D = sSFR * t_H(z_obs)

    ``D >= 1/3`` is star forming, ``D <= 1/20`` is quiescent, and intermediate
    values are transitional.  The observation-state value uses the latest
    100 Myr.  Historical crossings use a 300 Myr Gaussian-smoothed SFH.  When
    several complete SF-to-quiescent passages occur, the most recent completed
    passage is reported.  Histories without a completed passage return NaN for
    its timescale and epoch.
    """
    log_sfh = np.asarray(log_sfh, dtype=float)
    time = np.asarray(time, dtype=float)
    age = np.asarray(age_myr, dtype=float)
    mass_log = np.asarray(log_stellar_mass, dtype=float)
    redshift = np.asarray(redshift, dtype=float)
    n_rows = len(log_sfh)
    if (
        log_sfh.ndim != 2 or time.ndim != 1
        or log_sfh.shape[1] != len(time) or len(time) < 3
        or age.shape != (n_rows,) or mass_log.shape != (n_rows,)
        or redshift.shape != (n_rows,) or np.any(np.diff(time) <= 0)
        or time[0] < 0 or time[-1] > 1 or recent_window_myr <= 0
        or smoothing_myr <= 0
    ):
        raise ValueError("Expected SFH rows, fractional lookback time, ages, mass, and z")

    with np.errstate(over="ignore", invalid="ignore"):
        weights = np.maximum(10.0 ** log_sfh - epsilon, 0.0)
        totals = weights.sum(axis=1)
    valid = (
        np.all(np.isfinite(weights), axis=1) & np.isfinite(totals) & (totals > 0)
        & np.isfinite(age) & (age > recent_window_myr)
        & np.isfinite(mass_log) & (mass_log > 0) & (mass_log < 20)
        & np.isfinite(redshift) & (redshift >= 0)
    )
    normalized = np.full_like(weights, np.nan)
    normalized[valid] = weights[valid] / totals[valid, None]
    edges = np.r_[0.0, (time[:-1] + time[1:]) / 2.0, 1.0]
    fractional_width = np.diff(edges)
    age_years = age * 1e6
    stellar_mass = np.power(10.0, mass_log, where=np.isfinite(mass_log),
                            out=np.full_like(mass_log, np.nan))
    duration_years = age_years[:, None] * fractional_width[None, :]
    sfr = normalized * stellar_mass[:, None] / duration_years
    ssfr = sfr / stellar_mass[:, None]
    doubling_history = ssfr * age_years[:, None]

    # The current value is a partial-bin integral over exactly 100 Myr.
    upper = recent_window_myr / age
    recent_mass_fraction = np.zeros(n_rows, dtype=float)
    for column, (left, right, width) in enumerate(
        zip(edges[:-1], edges[1:], fractional_width)
    ):
        overlap = np.maximum(0.0, np.minimum(right, upper) - left)
        recent_mass_fraction += normalized[:, column] * overlap / width
    recent_sfr = recent_mass_fraction * stellar_mass / (recent_window_myr * 1e6)
    recent_ssfr = recent_sfr / stellar_mass
    doubling_observed = recent_ssfr * age_years
    doubling_observed[~valid] = np.nan
    recent_ssfr[~valid] = np.nan

    activity_class = np.full(n_rows, np.nan, dtype=np.float32)
    activity_class[valid & (doubling_observed <= 1.0 / 20.0)] = 0.0
    activity_class[
        valid & (doubling_observed > 1.0 / 20.0)
        & (doubling_observed < 1.0 / 3.0)
    ] = 1.0
    activity_class[valid & (doubling_observed >= 1.0 / 3.0)] = 2.0

    tau_sf = np.full(n_rows, np.nan, dtype=float)
    if np.any(valid):
        q20 = _mass_quantile_lookback(normalized[valid], edges, 0.2)
        q80 = _mass_quantile_lookback(normalized[valid], edges, 0.8)
        tau_sf[valid] = (q80 - q20) * age[valid] / 1e3

    # Smooth rates in physical time. Grouping equal rounded widths avoids one
    # scipy call per galaxy while preserving a fixed 300 Myr scale.
    step = float(np.median(np.diff(time)))
    sigma_bins = smoothing_myr / (age * step)
    sigma_bins = np.maximum(np.round(sigma_bins * 4.0) / 4.0, 0.5)
    smoothed = np.full_like(doubling_history, np.nan)
    for sigma in np.unique(sigma_bins[valid]):
        rows = valid & (sigma_bins == sigma)
        smoothed[rows] = gaussian_filter1d(
            doubling_history[rows], sigma=float(sigma), axis=1, mode="nearest",
        )

    quenching_time = np.full(n_rows, np.nan, dtype=float)
    quenching_lookback = np.full(n_rows, np.nan, dtype=float)
    quenching_cosmic_age = np.full(n_rows, np.nan, dtype=float)
    forward_fraction = 1.0 - time[::-1]
    upper_threshold = 1.0 / 3.0
    lower_threshold = 1.0 / 20.0
    for row in np.flatnonzero(valid):
        history = smoothed[row, ::-1]
        entries = np.flatnonzero(
            (history[:-1] >= upper_threshold) & (history[1:] < upper_threshold)
        )
        candidates = []
        for entry in entries:
            later = history[entry + 1:]
            rebound = np.flatnonzero(later >= upper_threshold)
            stop = len(history) if not len(rebound) else entry + 1 + int(rebound[0])
            exits = np.flatnonzero(history[entry + 1:stop] <= lower_threshold)
            if not len(exits):
                continue
            exit_index = entry + 1 + int(exits[0])
            entry_fraction = _crossing_position(
                forward_fraction[entry], history[entry],
                forward_fraction[entry + 1], history[entry + 1], upper_threshold,
            )
            exit_fraction = _crossing_position(
                forward_fraction[exit_index - 1], history[exit_index - 1],
                forward_fraction[exit_index], history[exit_index], lower_threshold,
            )
            if exit_fraction >= entry_fraction:
                candidates.append((entry_fraction, exit_fraction))
        if not candidates:
            continue
        entry_fraction, exit_fraction = candidates[-1]
        age_gyr = age[row] / 1e3
        quenching_time[row] = (exit_fraction - entry_fraction) * age_gyr
        midpoint = 0.5 * (entry_fraction + exit_fraction)
        quenching_cosmic_age[row] = midpoint * age_gyr
        quenching_lookback[row] = (1.0 - midpoint) * age_gyr

    quenching_redshift = _redshift_at_cosmic_age(quenching_cosmic_age)
    return {
        "sfh_doubling_number_100myr": doubling_observed.astype(np.float32),
        "sfh_log_ssfr_100myr_r0": np.log10(
            np.maximum(recent_ssfr, 1e-15)
        ).astype(np.float32),
        "sfh_activity_class": activity_class,
        "sfh_tau_sf_20_80_gyr": tau_sf.astype(np.float32),
        "sfh_quenching_timescale_gyr": quenching_time.astype(np.float32),
        "sfh_quenching_epoch_lookback_gyr": quenching_lookback.astype(np.float32),
        "sfh_quenching_epoch_cosmic_age_gyr": quenching_cosmic_age.astype(np.float32),
        "sfh_quenching_epoch_redshift": quenching_redshift.astype(np.float32),
    }
