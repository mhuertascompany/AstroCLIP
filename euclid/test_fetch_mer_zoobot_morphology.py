import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.table import Table

from euclid.fetch_mer_zoobot_morphology import (
    MER_ZOOBOT_COLUMNS,
    fetch_zoobot_morphology,
    zoobot_query,
)


class MerZoobotMorphologyTests(unittest.TestCase):
    def test_exact_join_and_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids = np.array([101, 202, 303], dtype=np.int64)
            sample = root / 'sample.fits'
            Table({'object_id': ids}).write(sample)

            class Client:
                calls = 0

                def launch_job_async(self, query, **kwargs):
                    self.calls += 1
                    uploaded = Table.read(kwargs['upload_resource'], format='votable')
                    data = {
                        'sample_row': uploaded['sample_row'],
                        'object_id': uploaded['object_id'],
                    }
                    data.update({
                        name: np.full(len(uploaded), index + 1.0)
                        for index, name in enumerate(MER_ZOOBOT_COLUMNS)
                    })
                    result = Table(data)

                    class Job:
                        def get_results(self):
                            return result
                    return Job()

            output = root / 'zoobot.fits'
            client = Client()
            table = fetch_zoobot_morphology(sample, output, client, batch_size=2)
            self.assertEqual(client.calls, 2)
            np.testing.assert_array_equal(table['object_id'], ids)
            self.assertIn('smooth_or_featured_smooth', table.colnames)
            output.unlink()
            cached = fetch_zoobot_morphology(sample, output, None, batch_size=2)
            np.testing.assert_array_equal(cached['object_id'], ids)

    def test_query_uses_morphology_table(self):
        query = zoobot_query()
        self.assertIn('catalogue.mer_morphology_deep_survey', query)
        self.assertIn('morph.smooth_or_featured_smooth', query)
        self.assertIn('morph.object_id = src.object_id', query)

    def test_partial_join_is_row_aligned_with_nan_for_missing_objects(self):
        from euclid.fetch_mer_zoobot_morphology import validate_results

        sources = Table({
            'sample_row': np.arange(3, dtype=np.int64),
            'object_id': np.array([101, 202, 303], dtype=np.int64),
        })
        data = {
            'sample_row': np.array([2, 0], dtype=np.int64),
            'object_id': np.array([303, 101], dtype=np.int64),
        }
        data.update({
            name: np.array([index + 3.0, index + 1.0])
            for index, name in enumerate(MER_ZOOBOT_COLUMNS)
        })
        output = validate_results(sources, Table(data))
        np.testing.assert_array_equal(output['object_id'], [101, 202, 303])
        self.assertTrue(np.isnan(output['concentration'][1]))
        np.testing.assert_allclose(output['concentration'][[0, 2]], [1.0, 3.0])


if __name__ == '__main__':
    unittest.main()
