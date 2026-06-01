"""
Compute SFH scalar statistics for all galaxies in a cosmosweb npz file
and save them back into the same npz.

Statistics added
----------------
qi          Quenching index = f(5–10%) − f(0–5%).
              Negative = rising SFH; positive = declining/quenching.
nqi         Normalised quenching index = qi / (f(5%) + f(5–10%)).
              Bounded [−1, +1]; purely shape-sensitive (= −i10).
f5          Cumulative SF fraction in last  5% of Hubble time.
f10         Cumulative SF fraction in last 10% of Hubble time.
f20         Cumulative SF fraction in last 20% of Hubble time.
f30         Cumulative SF fraction in last 30% of Hubble time.
f50         Cumulative SF fraction in last 50% of Hubble time.
i5          Increase index over last  5%: (w[0–2.5%]  − w[2.5–5%])  / w[0–5%].
i10         Increase index over last 10%: (w[0–5%]    − w[5–10%])   / w[0–10%].
i20         Increase index over last 20%: (w[0–10%]   − w[10–20%])  / w[0–20%].
i30         Increase index over last 30%: (w[0–15%]   − w[15–30%])  / w[0–30%].
i50         Increase index over last 50%: (w[0–25%]   − w[25–50%])  / w[0–50%].
d5  … d50   Decline indices: dX = −iX. Positive = declining over last X%.
sfh_peak_t       Fractional lookback time of SFH peak (0 = now, 1 = Big Bang).
sfh_mean_t       Mass-weighted mean formation epoch (fractional lookback time).

Burstiness statistics (from 9 native CIGALE bins, sfh_bins_log)
----------------------------------------------------------------
Bins are normalised to fractional Hubble time (sfh_times_myr / sfh_time_norm)
and sorted recent-first before computing statistics. Because the normalisation
places all galaxies on a common 0→1 scale, measures are directly comparable
across redshifts.

burst_var        Variance of the 9 normalised SFH weights.
burst_ac1        Lag-1 autocorrelation (≈1 = smooth; low/negative = bursty).
burst_ac2        Lag-2 autocorrelation.
burst_dyn_range  log10(max_weight / min_weight) — dynamic range across all bins.
burst_var_rec    Variance in the 3 most recent bins.
burst_var_old    Variance in the 3 oldest bins.
burst_var_ratio  burst_var_rec / burst_var_old (> 1 = more variable recently).
burst_max_delta  Max absolute difference between consecutive bins (largest jump).

All fractions are computed from the 50-bin interpolated SFH grid stored in the
H5 file (sfh_time_grid dataset, linspace 0→1).  The SFH is first converted from
log10 to linear and re-normalised so that it sums to 1.

Usage
-----
python -m cosmosweb.add_sfh_stats \\
    --npz  /path/to/cosmosweb_umap_zoobot_v8.npz \\
    --h5   /path/to/cosmosweb_dataset_v6.h5

The input npz is overwritten with the new fields appended.
A backup is saved as <npz_path>.bak before writing.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np

SFH_EPS = 1e-10
BATCH   = 10_000   # galaxies per H5 read


def compute_all_sfh_stats(h5_path: Path, h5_indices: np.ndarray) -> dict[str, np.ndarray]:
    N = len(h5_indices)

    # read in batches to avoid loading the full H5 at once
    sfh_all   = np.empty((N, 50), dtype=np.float64)
    t_frac    = None

    si  = np.argsort(h5_indices)
    ri  = np.argsort(si)
    sidx = h5_indices[si]

    with h5py.File(h5_path, 'r') as f:
        t_frac = (f['sfh_time_grid'][:].astype(np.float64)
                  if 'sfh_time_grid' in f
                  else np.linspace(0, 1, 50))
        for start in range(0, N, BATCH):
            end   = min(start + BATCH, N)
            batch = sidx[start:end].tolist()
            sfh_all[start:end] = f['sfh'][batch].astype(np.float64)

    # restore original order
    sfh_all = sfh_all[ri]

    # linear SFR, row-normalised
    sfr = np.maximum(10.0 ** sfh_all - SFH_EPS, 0.0)
    sfr /= sfr.sum(axis=1, keepdims=True).clip(min=1e-10)

    t = t_frac[np.newaxis, :]   # (1, 50)

    stats: dict[str, np.ndarray] = {}

    # ── cumulative recent fractions ───────────────────────────────────────────
    for pct in (5, 10, 20, 30, 50):
        stats[f'f{pct}'] = sfr[:, t_frac <= pct / 100.0].sum(axis=1).astype(np.float32)

    # ── quenching index ───────────────────────────────────────────────────────
    f5    = stats['f5']
    f10   = stats['f10']
    d_5_10 = f10 - f5
    qi    = d_5_10 - f5
    stats['qi'] = qi.astype(np.float32)

    # ── normalised QI ─────────────────────────────────────────────────────────
    denom  = f5 + d_5_10
    nqi    = np.where(denom > 0, qi / denom, np.float32(0))
    stats['nqi'] = nqi.astype(np.float32)

    # ── iX / dX ──────────────────────────────────────────────────────────────
    for pct in (5, 10, 20, 30, 50):
        half   = pct / 200.0
        full   = pct / 100.0
        w_rec  = sfr[:, t_frac <= half].sum(axis=1)
        w_old  = sfr[:, (t_frac > half) & (t_frac <= full)].sum(axis=1)
        w_tot  = sfr[:, t_frac <= full].sum(axis=1)
        d      = np.full(N, np.nan, dtype=np.float32)
        nz     = w_tot > 0
        d[nz]  = ((w_old[nz] - w_rec[nz]) / w_tot[nz]).astype(np.float32)
        stats[f'd{pct}'] =  d
        stats[f'i{pct}'] = -d

    # ── SFH peak and mean formation epoch ─────────────────────────────────────
    stats['sfh_peak_t'] = t_frac[np.argmax(sfr, axis=1)].astype(np.float32)
    stats['sfh_mean_t'] = (sfr * t).sum(axis=1).astype(np.float32)

    return stats


def compute_burstiness(h5_path: Path, h5_indices: np.ndarray) -> dict[str, np.ndarray]:
    """
    Burstiness statistics computed from the 9 native CIGALE bins (sfh_bins_log),
    normalised to fractional Hubble time via sfh_times_myr / sfh_time_norm.

    Because the bins are on a common fractional-time scale, variability measures
    are directly comparable across galaxies at different redshifts.

    Statistics
    ----------
    burst_var        Variance of the 9 normalised SFH weights.
                     High = large spread across bins = variable SFH.
    burst_ac1        Lag-1 autocorrelation of the 9 weights (ordered recent→old).
                     High (≈1) = smooth; low or negative = bursty/oscillating.
    burst_ac2        Lag-2 autocorrelation.
    burst_dyn_range  log10(max_weight / min_weight) across all 9 bins.
                     Large = high dynamic range = bursty episodes.
    burst_var_rec    Variance in the 3 most recent bins (shortest timescales).
    burst_var_old    Variance in the 3 oldest bins (longest timescales).
    burst_var_ratio  burst_var_rec / burst_var_old.
                     > 1 = more variable recently than in the past.
    burst_max_delta  Max absolute difference between consecutive bins (peak jump).
    """
    N    = len(h5_indices)
    si   = np.argsort(h5_indices)
    ri   = np.argsort(si)
    sidx = h5_indices[si]

    with h5py.File(h5_path, 'r') as f:
        bins_log  = np.empty((N, 9), dtype=np.float64)
        times_myr = np.empty((N, 9), dtype=np.float64)
        t_norms   = np.empty(N,      dtype=np.float64)
        for start in range(0, N, BATCH):
            end   = min(start + BATCH, N)
            batch = sidx[start:end].tolist()
            bins_log[start:end]  = f['sfh_bins_log'][batch]
            times_myr[start:end] = f['sfh_times_myr'][batch]
            t_norms[start:end]   = f['sfh_time_norm'][batch]

    # restore original order
    bins_log  = bins_log[ri]
    times_myr = times_myr[ri]
    t_norms   = t_norms[ri]

    # fractional lookback times of bin centres: (N, 9), 0=now, 1=Big Bang
    t_frac_bins = times_myr / t_norms[:, np.newaxis].clip(min=1e-10)

    # linear weights, row-normalised
    w = np.maximum(10.0 ** bins_log - SFH_EPS, 0.0)
    w = w / w.sum(axis=1, keepdims=True).clip(min=1e-10)   # (N, 9)

    # sort bins by increasing fractional lookback time (recent first)
    order = np.argsort(t_frac_bins, axis=1)               # (N, 9)
    w_sorted = np.take_along_axis(w, order, axis=1)        # (N, 9), bin 0 = most recent

    stats: dict[str, np.ndarray] = {}

    # ── overall variance ──────────────────────────────────────────────────────
    stats['burst_var'] = w_sorted.var(axis=1).astype(np.float32)

    # ── lag-1 and lag-2 autocorrelation ───────────────────────────────────────
    def autocorr(x, lag):
        # x: (N, 9); returns (N,) mean lag-k autocorrelation
        n   = x.shape[1] - lag
        mu  = x.mean(axis=1, keepdims=True)
        x0  = x[:, :n]  - mu
        xl  = x[:, lag:] - mu
        var = (x ** 2).mean(axis=1) - mu[:, 0] ** 2
        return np.where(var > 0,
                        (x0 * xl).mean(axis=1) / var.clip(min=1e-10),
                        np.float32(0)).astype(np.float32)

    stats['burst_ac1'] = autocorr(w_sorted, 1)
    stats['burst_ac2'] = autocorr(w_sorted, 2)

    # ── dynamic range ─────────────────────────────────────────────────────────
    wmax = w_sorted.max(axis=1)
    wmin = w_sorted.min(axis=1).clip(min=1e-10)
    stats['burst_dyn_range'] = np.log10(wmax / wmin).astype(np.float32)

    # ── recent vs old variance ────────────────────────────────────────────────
    var_rec = w_sorted[:, :3].var(axis=1)
    var_old = w_sorted[:, 6:].var(axis=1)
    stats['burst_var_rec']   = var_rec.astype(np.float32)
    stats['burst_var_old']   = var_old.astype(np.float32)
    stats['burst_var_ratio'] = np.where(
        var_old > 0, var_rec / var_old.clip(min=1e-10), np.float32(np.nan)
    ).astype(np.float32)

    # ── max consecutive jump ──────────────────────────────────────────────────
    diffs = np.abs(np.diff(w_sorted, axis=1))           # (N, 8)
    stats['burst_max_delta'] = diffs.max(axis=1).astype(np.float32)

    return stats


def main():
    p = argparse.ArgumentParser(description='Add SFH statistics to cosmosweb npz')
    p.add_argument('--npz', type=Path, required=True)
    p.add_argument('--h5',  type=Path, required=True)
    args = p.parse_args()

    print(f'Loading {args.npz} …')
    npz        = np.load(args.npz, allow_pickle=True)
    h5_indices = npz['h5_indices'].astype(int)
    N          = len(h5_indices)

    print(f'Computing SFH statistics for {N} galaxies …')
    sfh_stats  = compute_all_sfh_stats(args.h5, h5_indices)

    print(f'Computing burstiness statistics (9-bin CIGALE) …')
    burst_stats = compute_burstiness(args.h5, h5_indices)

    all_new = {**sfh_stats, **burst_stats}

    # merge with existing arrays
    data = dict(npz)
    overlap = [k for k in all_new if k in data]
    if overlap:
        print(f'Overwriting existing keys: {overlap}')
    data.update(all_new)

    # backup then overwrite
    bak = args.npz.with_suffix('.npz.bak')
    shutil.copy2(args.npz, bak)
    print(f'Backup saved → {bak}')

    np.savez_compressed(args.npz, **data)
    print(f'Saved {len(all_new)} new arrays → {args.npz}')
    print('\nArrays added:')
    for k, v in sorted(all_new.items()):
        print(f'  {k:20s}  {v.shape}  {v.dtype}  '
              f'[{np.nanmin(v):.3f}, {np.nanmax(v):.3f}]')


if __name__ == '__main__':
    main()
