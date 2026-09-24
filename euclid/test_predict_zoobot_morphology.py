import unittest
import tempfile
from pathlib import Path

import numpy as np

from euclid.predict_zoobot_morphology import (
    compatible_checkpoint_hparams,
    model_input_channels,
    normalize_answer_name,
    prediction_table,
    select_stamps_without_h5,
)


class DummySchema:
    label_cols = [
        'smooth-or-featured_smooth',
        'smooth-or-featured_featured-or-disk',
        'smooth-or-featured_problem',
        'has-spiral-arms_yes',
        'has-spiral-arms_no',
        'merging_none',
        'merging_minor-disturbance',
        'merging_major-disturbance',
        'merging_merger',
    ]


class DummyAbstract:
    def __init__(self, name=None, greyscale=False):
        pass


class DummyTree:
    def __init__(self, schema, **super_kwargs):
        pass


class PredictZooBotMorphologyTest(unittest.TestCase):
    def test_selects_stamps_without_h5(self):
        with tempfile.TemporaryDirectory() as directory:
            vis = Path(directory) / 'VIS'
            vis.mkdir()
            for galaxy_id in (30, 10, 20):
                (vis / f'VIS_{galaxy_id}.jpg').touch()
            stamp_dir, rows, ids, n_paired, n_h5 = select_stamps_without_h5(
                directory, max_objects=0,
            )
            self.assertEqual(stamp_dir, vis)
            np.testing.assert_array_equal(ids, [10, 20, 30])
            np.testing.assert_array_equal(rows, [-1, -1, -1])
            self.assertEqual(n_paired, 3)
            self.assertIsNone(n_h5)

    def test_removes_obsolete_checkpoint_hyperparameters(self):
        schema = DummySchema()
        clean, ignored = compatible_checkpoint_hparams(
            {'schema': schema, 'name': 'encoder', 'greyscale': True,
             'n_blocks': 0},
            DummyTree, DummyAbstract,
        )
        self.assertEqual(clean['schema'], schema)
        self.assertEqual(clean['name'], 'encoder')
        self.assertEqual(clean['greyscale'], True)
        self.assertEqual(ignored, ['n_blocks'])

    def test_normalizes_schema_names_and_problem_alias(self):
        self.assertEqual(
            normalize_answer_name('smooth-or-featured_featured-or-disk'),
            'smooth_or_featured_featured_or_disk',
        )
        self.assertEqual(
            normalize_answer_name('smooth-or-featured_problem'),
            'smooth_or_featured_artifact_star_zoom',
        )
        self.assertEqual(
            normalize_answer_name('smooth-or-featured-euclid_smooth'),
            'smooth_or_featured_smooth',
        )
        self.assertEqual(
            normalize_answer_name('smooth-or-featured-euclid_problem'),
            'smooth_or_featured_artifact_star_zoom',
        )
        self.assertEqual(
            normalize_answer_name('has-spiral-arms-euclid_yes'),
            'has_spiral_arms_yes',
        )
        self.assertEqual(
            normalize_answer_name('merging-euclid_major-disturbance'),
            'merging_major_disturbance',
        )

    def test_prediction_table_preserves_exact_ids_and_rows(self):
        values = np.arange(18, dtype=np.float32).reshape(2, 9) + 1
        table, _, columns = prediction_table(
            np.array([2700000000000000001, 2700000000000000002]),
            np.array([7, 9]), values, DummySchema(),
        )
        self.assertEqual(list(table['object_id']), [2700000000000000001, 2700000000000000002])
        self.assertEqual(list(table['h5_row']), [7, 9])
        self.assertIn('has_spiral_arms_yes', columns)
        self.assertIn('merging_merger', table.colnames)

    def test_rejects_nonpositive_concentrations(self):
        values = np.ones((1, 9), dtype=np.float32)
        values[0, 2] = 0
        with self.assertRaisesRegex(ValueError, 'must be positive'):
            prediction_table(np.array([1]), np.array([0]), values, DummySchema())

    def test_reads_model_input_channels(self):
        model = type('Model', (), {})()
        model.encoder = __import__('torch').nn.Sequential(
            __import__('torch').nn.Conv2d(1, 4, 3),
        )
        self.assertEqual(model_input_channels(model), 1)


if __name__ == '__main__':
    unittest.main()
