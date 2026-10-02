"""Published star-forming main-sequence parameterizations used in figures.

References
----------
Popesso et al. 2023, MNRAS, 519, 1526 (arXiv:2203.10487).
JADES SFMS study 2025, MNRAS, 544, 4551 (doi:10.1093/mnras/staf1950).
"""

from __future__ import annotations

import numpy as np

from .cosmic_sfh import COSMOLOGY


POPESSO_2023_REDSHIFT_RANGE = (0.0, 6.0)
POPESSO_2023_LOG_MASS_RANGE = (8.7, 11.3)
JADES_2025_REDSHIFT_RANGE = (3.0, 9.0)
JADES_2025_LOG_MASS_RANGE = (9.0, 10.3)


def popesso_2023_log_sfr(
    log_stellar_mass: np.ndarray | float,
    redshift: float,
    sfr_offset_dex: float = 0.0,
) -> np.ndarray:
    """Popesso et al. (2023) Eq. 15, with cosmic age in Gyr.

    The published relation is calibrated after conversion to a Kroupa IMF.
    ``sfr_offset_dex`` is an optional project-specific vertical recalibration;
    it is not part of the published relation.
    """
    mass = np.asarray(log_stellar_mass, dtype=float)
    if not np.isfinite(redshift) or not POPESSO_2023_REDSHIFT_RANGE[0] <= redshift <= POPESSO_2023_REDSHIFT_RANGE[1]:
        return np.full_like(mass, np.nan, dtype=float)
    cosmic_age_gyr = float(COSMOLOGY.age(redshift).to_value("Gyr"))
    log_sfr_max = 2.71 - 0.186 * cosmic_age_gyr
    log_turnover_mass = 10.86 - 0.0729 * cosmic_age_gyr
    return (
        log_sfr_max
        - np.log10(1.0 + 10.0 ** (log_turnover_mass - mass))
        + float(sfr_offset_dex)
    )


def jades_2025_log_sfr_100myr(
    log_stellar_mass: np.ndarray | float,
    redshift: float,
    sfr_offset_dex: float = 0.0,
) -> np.ndarray:
    """JADES 2025 100-Myr SFMS fit for 3 <= z <= 9.

    The published fit is sSFR/Gyr^-1 = 0.30 *
    (Mstar/1e10 Msun)^0.00 * (1+z)^1.06.  The conversion to SFR
    includes the factor 1e9 yr/Gyr. ``sfr_offset_dex`` is external to
    the published relation.
    """
    mass = np.asarray(log_stellar_mass, dtype=float)
    if not np.isfinite(redshift) or not JADES_2025_REDSHIFT_RANGE[0] <= redshift <= JADES_2025_REDSHIFT_RANGE[1]:
        return np.full_like(mass, np.nan, dtype=float)
    log_ssfr_per_gyr = (
        np.log10(0.30)
        + 0.00 * (mass - 10.0)
        + 1.06 * np.log10(1.0 + redshift)
    )
    return mass - 9.0 + log_ssfr_per_gyr + float(sfr_offset_dex)


def literature_main_sequence(
    log_stellar_mass: np.ndarray | float,
    redshift: float,
    sfr_offset_dex: float = 0.0,
) -> tuple[np.ndarray, str, tuple[float, float]]:
    """Return the preferred relation and its formal mass interval.

    Popesso et al. (2023) is used below z=3, while the explicitly 100-Myr
    JADES 2025 calibration is used from z=3 through z=9. No extrapolation in
    redshift is returned.
    """
    if 0.0 <= redshift < 3.0:
        return (
            popesso_2023_log_sfr(log_stellar_mass, redshift, sfr_offset_dex),
            "Popesso+23",
            POPESSO_2023_LOG_MASS_RANGE,
        )
    if 3.0 <= redshift <= 9.0:
        return (
            jades_2025_log_sfr_100myr(
                log_stellar_mass, redshift, sfr_offset_dex
            ),
            "JADES 2025 (100 Myr)",
            JADES_2025_LOG_MASS_RANGE,
        )
    mass = np.asarray(log_stellar_mass, dtype=float)
    return np.full_like(mass, np.nan, dtype=float), "outside calibration", (np.nan, np.nan)
