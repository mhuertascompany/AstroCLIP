"""Exploratory Speagle et al. (2014) main-sequence integration tracks."""
import numpy as np
from scipy.integrate import solve_ivp
from .cosmic_sfh import COSMOLOGY


def main_sequence_track(time, redshift, log_mass, time_norm_myr=None,
                        return_fraction=.4, mass_offset=0.):
    """Backward in-situ growth; constant recycling; no mergers or quenching.

    log_mass is surviving stellar mass in log10 solar masses. mass_offset
    converts the input mass to the Kroupa convention of Speagle Eq. 28.
    Stop at 1e6 solar masses or cosmic time zero; do not invent an early SFH.
    Bin weights are normalized over the integrated part of the track only.
    """
    time = np.asarray(time, dtype=float)
    if (time.ndim != 1 or len(time) < 2 or not np.isfinite(time).all()
            or np.any(np.diff(time) <= 0) or time[0] < 0 or time[-1] > 1):
        raise ValueError('Expected increasing fractional time grid in [0,1].')
    if not 0 <= return_fraction < 1:
        raise ValueError('Return fraction must be in [0,1).')
    empty = dict(weights=np.full_like(time, np.nan), supported=np.zeros(len(time), bool))
    mass = log_mass + mass_offset
    if not np.isfinite(mass) or not 6 < mass < 13 or not np.isfinite(redshift) or redshift < 0:
        return empty
    age = float(COSMOLOGY.age(redshift).value)
    scale = age if time_norm_myr is None else float(time_norm_myr)/1000
    if not np.isfinite(scale) or scale <= 0:
        return empty

    def derivative(lookback, y):
        cosmic_time = age-lookback
        log_sfr = (.84-.026*cosmic_time)*y[0] - (6.51-.11*cosmic_time)
        # Gyr derivative of log10 surviving mass; negative backward in time.
        return [-(1-return_fraction)*1e9*10**(log_sfr-y[0])/np.log(10)]

    def seed(lookback, y):
        return y[0]-6.
    seed.terminal = True
    seed.direction = -1
    solution = solve_ivp(derivative, (0, min(age, scale)), [mass], events=seed,
                         dense_output=True, rtol=1e-8, atol=1e-9, max_step=.03)
    if not solution.success:
        raise RuntimeError(solution.message)
    stop = solution.t[-1]
    edges = np.r_[0., (time[:-1]+time[1:])/2, 1.]*scale
    edge_mass = 10**solution.sol(np.minimum(edges, stop))[0]
    formed = np.maximum(-np.diff(edge_mass)/(1-return_fraction), 0.)
    weights = formed/formed.sum()
    weights[edges[:-1] >= stop] = np.nan
    lo_time, hi_time = age-edges[1:], age-edges[:-1]
    # Conservative display guide, not a rectangular completeness claim.
    supported = ((lo_time >= 2.5) & (hi_time <= 11.5)
                 & (edge_mass[1:] >= 10**9.7) & (edge_mass[:-1] <= 10**11.1)
                 & (edges[1:] <= stop))
    return dict(weights=weights, supported=supported, stop_lookback_gyr=stop,
                modeled_formed_mass=formed.sum(), final_mass=10**mass,
                seed_mass=edge_mass[-1])


def main_sequence_along_sfh(time, log_sfh, redshift, log_mass,
                            time_norm_myr=None, return_fraction=.4,
                            mass_offset=0., epsilon=1e-10, ms_sfr_offset=0.):
    """MS reference along the observed SFH's own inferred mass history.

    ms_sfr_offset shifts log10 MS SFR at fixed mass/time (empirical only).
    Constant instantaneous recycling and entirely in-situ growth are assumed.
    At lookback l: Mstar(l)=Mstar_obs * fraction formed at lookbacks >= l.
    MS bin integrals are divided by the SAME formed mass as the observed SFH;
    they are deliberately not normalized to unit sum. Ratios compare bin-
    averaged SFRs, not instantaneous values at bin centers.
    """
    time = np.asarray(time, dtype=float)
    log_sfh = np.asarray(log_sfh, dtype=float)
    if (time.ndim != 1 or len(time) < 2 or log_sfh.shape != time.shape
            or not np.isfinite(time).all() or np.any(np.diff(time) <= 0)
            or time[0] < 0 or time[-1] > 1):
        raise ValueError('Expected matching SFH and increasing fractional grid.')
    if not 0 <= return_fraction < 1:
        raise ValueError('Return fraction must be in [0,1).')
    empty = dict(weights=np.full_like(time, np.nan), supported=np.zeros(len(time), bool))
    mass_log = log_mass + mass_offset
    if not np.isfinite(mass_log) or not 6 < mass_log < 13 or not np.isfinite(redshift) or redshift < 0:
        return empty
    age = float(COSMOLOGY.age(redshift).value)
    scale = age if time_norm_myr is None else float(time_norm_myr)/1000
    if not np.isfinite(scale) or scale <= 0:
        return empty
    observed = np.maximum(10.**log_sfh - epsilon, 0.)
    if not np.isfinite(observed).all() or observed.sum() <= 0:
        return empty
    observed /= observed.sum()
    edges = np.r_[0., (time[:-1]+time[1:])/2, 1.]*scale
    duration_years = np.diff(edges)*1e9
    final_mass = 10.**mass_log
    total_formed_mass = final_mass/(1-return_fraction)
    older = np.cumsum(observed[::-1])[::-1]-observed
    older = np.maximum(older, 0.)
    # Within-bin constant observed SFR implies linearly increasing mass in
    # forward cosmic time. Quadrature averages the varying MS reference.
    nodes, quadrature_weights = np.polynomial.legendre.leggauss(16)
    fraction = (nodes+1)/2
    lookback = edges[:-1, None] + np.diff(edges)[:, None]*fraction
    cosmic_age = np.maximum(age-lookback, 0.)
    mass = final_mass*(older[:, None]+observed[:, None]*(1-fraction))
    with np.errstate(divide='ignore', invalid='ignore'):
        lm = np.log10(mass)
        ms_sfr = 10.**((.84-.026*cosmic_age)*lm-(6.51-.11*cosmic_age))
    ms_sfr[mass <= 0] = 0.
    if not np.isfinite(ms_sfr_offset):
        raise ValueError("MS SFR offset must be finite.")
    reference_sfr = (ms_sfr @ quadrature_weights / 2) * 10.**ms_sfr_offset
    reference = reference_sfr*duration_years/total_formed_mass
    supported = ((age-edges[1:] >= 2.5) & (age-edges[:-1] <= 11.5)
                 & (final_mass*older >= 10**9.7)
                 & (final_mass*(older+observed) <= 10**11.1))
    no_mass = (older+observed) <= 0
    reference[no_mass] = np.nan
    sfr = observed*total_formed_mass/duration_years
    with np.errstate(divide='ignore', invalid='ignore'):
        delta = np.log10(sfr/reference_sfr)
    delta[no_mass] = np.nan
    return dict(weights=reference, supported=supported, observed_weights=observed,
                sfr=sfr, ms_sfr=reference_sfr, delta_ms=delta,
                mass_at_centers=final_mass*(older+.5*observed),
                mass_at_edges=final_mass*np.r_[np.cumsum(observed[::-1])[::-1], 0.],
                total_formed_mass=total_formed_mass, duration_years=duration_years)
