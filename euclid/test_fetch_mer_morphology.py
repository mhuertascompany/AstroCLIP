"""Offline tests for the Datalabs MER morphology join."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.table import Table

from euclid.fetch_mer_morphology import fetch_morphology, load_ids, morphology_query


class MorphologyFetchTests(unittest.TestCase):
    def test_batched_exact_join_and_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids = np.array([2688124142666475971 + i for i in range(3)], dtype=np.int64)
            sample = root / 'sample.fits'
            Table({'object_id': ids}).write(sample)
            owner = self

            class Client:
                calls = 0

                def launch_job_async(self, query, **kwargs):
                    self.calls += 1
                    upload = Table.read(kwargs['upload_resource'], format='votable')
                    owner.assertIn('catalogue.mer_catalogue_deep_survey', query)
                    result = Table({
                        'sample_row': upload['sample_row'],
                        'object_id': upload['object_id'],
                        'segmentation_area': np.full(len(upload), 1200.0),
                        'kron_radius': np.full(len(upload), 3.0),
                        'ellipticity': np.full(len(upload), 0.4),
                        'semimajor_axis': np.full(len(upload), 4.0),
                    })

                    class Job:
                        def get_results(self):
                            return result
                    return Job()

            output = root / 'morphology.fits'
            client = Client()
            result = fetch_morphology(sample, output, client, batch_size=2)
            self.assertEqual(client.calls, 2)
            np.testing.assert_array_equal(result['object_id'], ids)
            self.assertEqual(result.colnames[:4], [
                'object_id', 'segmentation_area', 'kron_radius', 'ellipticity'
            ])
            output.unlink()
            cached = fetch_morphology(sample, output, None, batch_size=2)
            np.testing.assert_array_equal(cached['object_id'], ids)

    def test_missing_join_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = root / 'sample.fits'
            Table({'object_id': np.array([101, 202], dtype=np.int64)}).write(sample)

            class Client:
                def launch_job_async(self, query, **kwargs):
                    upload = Table.read(kwargs['upload_resource'], format='votable')[:1]
                    result = Table({
                        'sample_row': upload['sample_row'],
                        'object_id': upload['object_id'],
                        'segmentation_area': [10.0],
                        'kron_radius': [2.0],
                        'ellipticity': [0.2],
                        'semimajor_axis': [1.0],
                    })

                    class Job:
                        def get_results(self):
                            return result
                    return Job()

            with self.assertRaisesRegex(ValueError, '1/2 objects'):
                fetch_morphology(sample, root / 'out.fits', Client())

    def test_query_and_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample.fits'
            ids = np.array([101, 202], dtype=np.int64)
            Table({'OBJECT_ID': ids}).write(path)
            np.testing.assert_array_equal(load_ids(path)['object_id'], ids)
        query = morphology_query()
        self.assertIn('mer.segmentation_area', query)
        self.assertIn('mer.object_id = src.object_id', query)


if __name__ == '__main__':
    unittest.main()
