import unittest

import numpy as np

from euclid.predict_zoobot_morphology import (
    model_input_channels,
    normalize_answer_name,
    prediction_table,
)


class DummySchema:
    label_cols = [
        'smooth-or-featured_smooth',
        'smooth-or-featured_featured-or-disk',
        'smooth-or-featured_problem',
        'has-spiral-arms_yes',
        'has-spiral-arms_no',
        'merging_none',
        'merging_merger',
    ]


class PredictZooBotMorphologyTest(unittest.TestCase):
    def test_normalizes_schema_names_and_problem_alias(self):
        self.assertEqual(
            normalize_answer_name('smooth-or-featured_featured-or-disk'),
            'smooth_or_featured_featured_or_disk',
        )
        self.assertEqual(
            normalize_answer_name('smooth-or-featured_problem'),
            'smooth_or_featured_artifact_star_zoom',
        )

    def test_prediction_table_preserves_exact_ids_and_rows(self):
        values = np.arange(14, dtype=np.float32).reshape(2, 7) + 1
        table, _, columns = prediction_table(
            np.array([2700000000000000001, 2700000000000000002]),
            np.array([7, 9]), values, DummySchema(),
        )
        self.assertEqual(list(table['object_id']), [2700000000000000001, 2700000000000000002])
        self.assertEqual(list(table['h5_row']), [7, 9])
        self.assertIn('has_spiral_arms_yes', columns)
        self.assertIn('merging_merger', table.colnames)

    def test_rejects_nonpositive_concentrations(self):
        values = np.ones((1, 7), dtype=np.float32)
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
