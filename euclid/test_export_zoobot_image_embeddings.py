import importlib.util
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


RUNTIME_AVAILABLE = all(
    importlib.util.find_spec(package) is not None
    for package in ('torch', 'torchvision')
)


@unittest.skipUnless(RUNTIME_AVAILABLE, 'image runtime dependencies are absent')
class ExportZooBotImageEmbeddingsTests(unittest.TestCase):
    def test_stamp_matching_and_seeded_subsampling(self):
        from euclid.export_zoobot_image_embeddings import select_stamp_rows

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / 'dataset.h5'
            stamp_dir = root / 'stamps' / 'VIS'
            stamp_dir.mkdir(parents=True)
            with h5py.File(dataset, 'w') as target:
                target['galaxy_id'] = np.arange(100, 110, dtype=np.int64)
            for galaxy_id in (100, 102, 104, 106, 108):
                (stamp_dir / f'VIS_{galaxy_id}.jpg').write_bytes(b'jpeg')

            actual_dir, rows, galaxy_ids, n_paired, n_h5 = select_stamp_rows(
                dataset, stamp_dir.parent, max_objects=3, seed=9,
            )
            _, rows_again, ids_again, _, _ = select_stamp_rows(
                dataset, stamp_dir.parent, max_objects=3, seed=9,
            )

            self.assertEqual(actual_dir, stamp_dir)
            self.assertEqual(n_paired, 5)
            self.assertEqual(n_h5, 10)
            self.assertEqual(len(rows), 3)
            np.testing.assert_array_equal(galaxy_ids, np.arange(100, 110)[rows])
            np.testing.assert_array_equal(rows_again, rows)
            np.testing.assert_array_equal(ids_again, galaxy_ids)


if __name__ == '__main__':
    unittest.main()
