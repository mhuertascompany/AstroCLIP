import tempfile
import unittest
from pathlib import Path

import numpy as np

from euclid.compare_frozen_image_encoders import (
    compact_archive,
    load_zoobot_archive,
)


class FrozenImageEncoderComparisonTests(unittest.TestCase):
    def test_compact_archive_preserves_ids_coordinates_and_scalar_properties(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / 'source.npz'
            np.savez(
                source_path,
                galaxy_id=np.array([11, 22]),
                h5_row=np.array([3, 7]),
                xy_image=np.array([[0.0, 1.0], [2.0, 3.0]]),
                image_embedding=np.eye(2),
                redshift=np.array([0.4, 0.8]),
                ignored_matrix=np.ones((2, 2)),
            )
            ids, rows, coordinates, embedding, properties = (
                load_zoobot_archive(source_path)
            )
            np.testing.assert_array_equal(ids, [11, 22])
            np.testing.assert_array_equal(rows, [3, 7])
            np.testing.assert_allclose(embedding, np.eye(2))
            self.assertEqual(list(properties), ['redshift'])

            output_path = root / 'compact.npz'
            compact_archive(
                output_path, ids, rows, coordinates, properties,
            )
            with np.load(output_path) as output:
                self.assertEqual(
                    set(output.files),
                    {'galaxy_id', 'h5_row', 'xy_image', 'redshift'},
                )

    def test_duplicate_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'duplicates.npz'
            np.savez(
                path,
                galaxy_id=np.array([11, 11]),
                h5_row=np.array([3, 7]),
                xy_image=np.zeros((2, 2)),
            )
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                load_zoobot_archive(path)


if __name__ == '__main__':
    unittest.main()
