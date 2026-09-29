"""Conversions for catalogue angular sizes."""

from __future__ import annotations

import numpy as np

from .cosmic_sfh import COSMOLOGY


def angular_radius_to_proper_kpc(radius_arcsec, redshift, cosmology=COSMOLOGY):
    """Convert an angular radius in arcseconds to a proper radius in kpc.

    Invalid, non-positive radii and non-positive redshifts are returned as
    ``NaN``.  Inputs must have matching shapes.
    """
    radius = np.asarray(radius_arcsec, dtype=float)
    z = np.asarray(redshift, dtype=float)
    if radius.shape != z.shape:
        raise ValueError("Angular radius and redshift must have matching shapes.")

    result = np.full(radius.shape, np.nan, dtype=float)
    valid = np.isfinite(radius) & (radius > 0) & np.isfinite(z) & (z > 0)
    if valid.any():
        kpc_per_arcsec = np.asarray(
            cosmology.kpc_proper_per_arcmin(z[valid]).value,
            dtype=float,
        ) / 60.0
        result[valid] = radius[valid] * kpc_per_arcsec
    return result
