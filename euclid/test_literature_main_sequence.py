import unittest

import numpy as np

from euclid.literature_main_sequence import (
    jades_2025_log_sfr_100myr,
    literature_main_sequence,
    popesso_2023_log_sfr,
)


class LiteratureMainSequenceTest(unittest.TestCase):
    def test_jades_normalizations_at_published_endpoints(self):
        # At log M=10, log SFR = 1 + log sSFR/Gyr^-1.
        self.assertAlmostEqual(
            float(jades_2025_log_sfr_100myr(10.0, 3.0)),
            1.0 + np.log10(1.30), places=2,
        )
        self.assertAlmostEqual(
            float(jades_2025_log_sfr_100myr(10.0, 9.0)),
            1.0 + np.log10(3.44), places=2,
        )

    def test_offset_is_a_uniform_vertical_shift(self):
        original = popesso_2023_log_sfr(np.array([9.5, 10.5]), 1.0)
        shifted = popesso_2023_log_sfr(
            np.array([9.5, 10.5]), 1.0, sfr_offset_dex=-0.93
        )
        np.testing.assert_allclose(shifted - original, -0.93)

    def test_hybrid_relation_switches_at_z_three(self):
        _, low_name, low_range = literature_main_sequence(10.0, 2.99)
        _, high_name, high_range = literature_main_sequence(10.0, 3.0)
        self.assertEqual(low_name, "Popesso+23")
        self.assertEqual(high_name, "JADES 2025 (100 Myr)")
        self.assertNotEqual(low_range, high_range)

    def test_no_redshift_extrapolation(self):
        values, name, _ = literature_main_sequence(np.array([9.5, 10.0]), 9.1)
        self.assertTrue(np.isnan(values).all())
        self.assertEqual(name, "outside calibration")


if __name__ == "__main__":
    unittest.main()
