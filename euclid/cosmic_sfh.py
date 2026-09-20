"""Cosmic SFRD as normalized formed-mass bin weights, for shape comparison."""
from functools import lru_cache

import numpy as np
from astropy.cosmology import FlatLambdaCDM

COSMOLOGY = FlatLambdaCDM(H0=70, Om0=0.3)


@lru_cache(maxsize=1)
def _history():
    # Madau & Dickinson 2014, Eq. 15. Smooth extrapolation at early times,
    # not a measurement of the first-star epoch.
    z = np.geomspace(1., 1.e5, 16384) - 1.
    density = .015 * (1 + z)**2.7 / (1 + ((1 + z) / 2.9)**5.6)
    ages = COSMOLOGY.age(z).to_value('Gyr')
    return np.r_[0., ages[::-1]], np.r_[0., density[::-1]]


def cosmic_sfh_weights(time, redshift, time_norm_myr=None):
    """Integrate SFRD before observation in the same bins as Euclid SFHs.

    Return dimensionless bin masses summing to one, not physical SFRD or
    the expected history of an individual galaxy. Uses the saved fractional
    time scale when supplied and evaluates SFRD at cosmic age(z)-lookback.
    """
    time = np.asarray(time, dtype=float)
    if (time.ndim != 1 or len(time) < 2 or not np.isfinite(time).all()
            or np.any(np.diff(time) <= 0) or time[0] < 0 or time[-1] > 1):
        raise ValueError('Expected increasing fractional bin centers in [0,1].')
    if not np.isfinite(redshift) or redshift < 0:
        return np.full_like(time, np.nan)
    observation_age = COSMOLOGY.age(redshift).to_value('Gyr')
    scale = observation_age if time_norm_myr is None else float(time_norm_myr) / 1000
    if not np.isfinite(scale) or scale <= 0:
        return np.full_like(time, np.nan)
    edges = np.r_[0., (time[:-1] + time[1:]) / 2, 1.]
    nodes, weights = np.polynomial.legendre.leggauss(16)
    widths = np.diff(edges) * scale
    lookback = ((edges[:-1] + edges[1:])[:, None] * scale / 2
                + widths[:, None] * nodes / 2)
    ages, density = _history()
    rate = np.interp(observation_age - lookback, ages, density, left=0.)
    mass = (rate @ weights) * widths / 2
    return mass / mass.sum() if mass.sum() > 0 else np.full_like(time, np.nan)
