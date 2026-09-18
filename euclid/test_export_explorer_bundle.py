import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from euclid.export_explorer_bundle import export_bundle


class ExportExplorerBundleTest(unittest.TestCase):
    def test_accepts_image_only_umap_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / 'sfh_clip.h5'
            stamp_dir = root / 'stamps' / 'VIS'
            stamp_dir.mkdir(parents=True)
            with h5py.File(dataset, 'w') as target:
                target['galaxy_id'] = [201, 202]
                target['sfh'] = np.zeros((2, 4), dtype=np.float32)
                target['sfh_p16'] = np.zeros((2, 4), dtype=np.float32)
                target['sfh_p84'] = np.zeros((2, 4), dtype=np.float32)
                target['redshift'] = [0.4, 0.8]
                target['sfh_time_grid'] = np.linspace(0, 1, 4)
            for galaxy_id in (201, 202):
                (stamp_dir / f'VIS_{galaxy_id}.jpg').write_bytes(b'jpeg')
            image_umap = root / 'zoobot_image_umap.npz'
            np.savez_compressed(
                image_umap,
                galaxy_id=np.array([201, 202]),
                h5_row=np.array([0, 1]),
                xy_image=np.zeros((2, 2), dtype=np.float32),
                image_embedding=np.eye(2, dtype=np.float32),
                vis_magnitude=np.array([20.5, 21.0], dtype=np.float32),
            )

            output = root / 'bundle'
            manifest = export_bundle(
                dataset, stamp_dir.parent, [image_umap], output,
            )

            self.assertEqual(manifest['n_galaxies'], 2)
            with np.load(output / manifest['umap_archives'][0]) as archive:
                self.assertIn('xy_image', archive.files)
                self.assertIn('image_embedding', archive.files)
                self.assertNotIn('sfh_embedding', archive.files)

    def test_aligns_runs_and_adds_full_embeddings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / 'sfh_clip.h5'
            stamp_dir = root / 'stamps' / 'VIS'
            stamp_dir.mkdir(parents=True)
            galaxy_ids = np.array([101, 102, 103, 104], dtype=np.int64)
            with h5py.File(dataset, 'w') as target:
                target['galaxy_id'] = galaxy_ids
                target['sfh'] = np.arange(20, dtype=np.float32).reshape(4, 5)
                target['sfh_p16'] = np.zeros((4, 5), dtype=np.float32)
                target['sfh_p84'] = np.ones((4, 5), dtype=np.float32)
                target['redshift'] = np.linspace(0.5, 2.0, 4, dtype=np.float32)
                target['sfh_time_grid'] = np.linspace(0, 1, 5, dtype=np.float32)
                target.attrs['sfh_log_epsilon'] = 1e-10

            run_a = root / 'run_a' / 'evaluation_best'
            run_b = root / 'run_b' / 'evaluation_best'
            run_a.mkdir(parents=True)
            run_b.mkdir(parents=True)
            archive_a = run_a / 'euclid_clip_umap_diagnostics.npz'
            archive_b = run_b / 'euclid_clip_umap_diagnostics.npz'
            self._write_diagnostics(archive_a, [101, 102, 103], [0, 1, 2])
            self._write_diagnostics(archive_b, [103, 102, 104], [2, 1, 3])
            self._write_embeddings(run_a / 'validation_embeddings.npz', [103, 101, 102])
            self._write_embeddings(run_b / 'validation_embeddings.npz', [104, 102, 103])
            for galaxy_id in galaxy_ids:
                (stamp_dir / f'VIS_{galaxy_id}.jpg').write_bytes(b'jpeg')

            output = root / 'bundle'
            manifest = export_bundle(
                dataset, stamp_dir.parent, [archive_a, archive_b], output,
            )

            self.assertEqual(manifest['n_galaxies'], 2)
            self.assertEqual(json.loads((output / 'manifest.json').read_text()), manifest)
            with h5py.File(output / 'euclid_explorer.h5', 'r') as compact:
                np.testing.assert_array_equal(compact['galaxy_id'][:], [102, 103])
                np.testing.assert_array_equal(compact['source_h5_row'][:], [1, 2])
                np.testing.assert_array_equal(compact['sfh'][:], [[5, 6, 7, 8, 9],
                                                                  [10, 11, 12, 13, 14]])
            for name in manifest['umap_archives']:
                with np.load(output / name) as archive:
                    self.assertIn('image_embedding', archive.files)
                    self.assertIn('sfh_embedding', archive.files)
                    self.assertIn('joint_embedding', archive.files)
                    norms = np.linalg.norm(archive['joint_embedding'], axis=1)
                    np.testing.assert_allclose(norms, 1.0, rtol=1e-6)
            with tarfile.open(output / 'VIS_stamps.tar') as stamps:
                self.assertEqual(
                    sorted(stamps.getnames()), ['VIS/VIS_102.jpg', 'VIS/VIS_103.jpg'],
                )

    @staticmethod
    def _write_diagnostics(path, ids, rows):
        ids = np.asarray(ids, dtype=np.int64)
        coordinates = np.column_stack([np.arange(len(ids)), np.arange(len(ids))])
        np.savez_compressed(
            path, galaxy_id=ids, h5_row=np.asarray(rows, dtype=np.int64),
            xy_image=coordinates, xy_sfh=coordinates + 1, xy_joint=coordinates + 2,
            redshift=np.linspace(0.5, 1.5, len(ids)),
        )

    @staticmethod
    def _write_embeddings(path, ids):
        ids = np.asarray(ids, dtype=np.int64)
        image = np.column_stack([np.ones(len(ids)), np.arange(1, len(ids) + 1)])
        sfh = np.column_stack([np.arange(1, len(ids) + 1), np.ones(len(ids))])
        np.savez_compressed(
            path, galaxy_id=ids, image_embedding=image, sfh_embedding=sfh,
        )


if __name__ == '__main__':
    unittest.main()
