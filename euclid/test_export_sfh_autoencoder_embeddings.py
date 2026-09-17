import importlib.util
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


RUNTIME_AVAILABLE = all(
    importlib.util.find_spec(package) is not None
    for package in ('torch', 'lightning', 'sklearn')
)


@unittest.skipUnless(RUNTIME_AVAILABLE, 'autoencoder runtime dependencies are absent')
class ExportSFHAutoencoderEmbeddingsTests(unittest.TestCase):
    def test_validation_rows_are_checked_and_subsampled(self):
        from euclid.export_sfh_autoencoder_embeddings import load_validation_rows

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / 'dataset.h5'
            split = root / 'split.npz'
            with h5py.File(dataset, 'w') as target:
                target['galaxy_id'] = np.arange(100, 110, dtype=np.int64)
            np.savez(
                split,
                val_rows=np.array([1, 3, 5, 7, 9]),
                val_ids=np.array([101, 103, 105, 107, 109]),
            )
            rows, ids = load_validation_rows(dataset, split, max_objects=3, seed=8)
            self.assertEqual(len(rows), 3)
            np.testing.assert_array_equal(ids, np.arange(100, 110)[rows])

    def test_shape_probe_recovers_identical_neighborhoods(self):
        from euclid.export_sfh_autoencoder_embeddings import shape_probe

        rng = np.random.default_rng(5)
        sfhs = rng.dirichlet(np.ones(12), size=80)
        metrics = shape_probe(sfhs.copy(), sfhs, seed=7, k=5)
        self.assertAlmostEqual(metrics['latent_raw_neighbor_overlap_at_k'], 1.0)
        self.assertGreater(
            metrics['latent_neighbor_raw_sfh_cosine_mean'],
            metrics['random_pair_raw_sfh_cosine_mean'],
        )

    def test_compact_h5_preserves_rows_and_reconstructions(self):
        from euclid.export_sfh_autoencoder_embeddings import write_compact_h5

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / 'dataset.h5'
            output = root / 'compact.h5'
            with h5py.File(dataset, 'w') as target:
                target['galaxy_id'] = np.array([10, 11, 12])
                target['sfh'] = np.arange(12, dtype=np.float32).reshape(3, 4)
                target['sfh_p16'] = target['sfh'][:] - 1
                target['sfh_p84'] = target['sfh'][:] + 1
                target['redshift'] = np.array([0.5, 1.0, 1.5])
                target['sfh_time_grid'] = np.linspace(0, 1, 4)
            reconstruction = np.full((2, 4), 0.25, dtype=np.float32)
            write_compact_h5(
                dataset, output, np.array([0, 2]), np.array([10, 12]),
                reconstruction,
            )
            with h5py.File(output, 'r') as compact:
                np.testing.assert_array_equal(compact['galaxy_id'][:], [10, 12])
                np.testing.assert_allclose(
                    compact['sfh_reconstruction'][:], reconstruction,
                )
                np.testing.assert_array_equal(
                    compact['sfh'][:], [[0, 1, 2, 3], [8, 9, 10, 11]],
                )


if __name__ == '__main__':
    unittest.main()
