import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM

from euclid.physical_size import angular_radius_to_proper_kpc


COSMOLOGY = FlatLambdaCDM(H0=70, Om0=0.3)


def test_angular_radius_to_proper_kpc_matches_astropy():
    radius = np.array([0.2, 0.7, 1.5])
    redshift = np.array([0.2, 0.7, 1.5])
    expected = radius * COSMOLOGY.kpc_proper_per_arcmin(redshift).value / 60.0

    actual = angular_radius_to_proper_kpc(radius, redshift, COSMOLOGY)

    np.testing.assert_allclose(actual, expected)


def test_angular_radius_to_proper_kpc_masks_invalid_values():
    radius = np.array([1.0, np.nan, 0.0, -1.0, 1.0])
    redshift = np.array([0.5, 0.5, 0.5, 0.5, np.nan])

    actual = angular_radius_to_proper_kpc(radius, redshift, COSMOLOGY)

    assert np.isfinite(actual[0])
    assert np.isnan(actual[1:]).all()


def test_angular_radius_to_proper_kpc_requires_matching_shapes():
    with pytest.raises(ValueError, match="matching shapes"):
        angular_radius_to_proper_kpc(np.ones(2), np.ones(3), COSMOLOGY)
