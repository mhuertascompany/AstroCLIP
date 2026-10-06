import numpy as np

from euclid.analyze_conditional_sfh_residuals import cross_validated_r2


def test_residual_features_add_predictive_information():
    rng = np.random.default_rng(12)
    condition = rng.normal(size=(300, 3))
    residual = rng.normal(size=(300, 8))
    target = 0.8 * residual[:, 0] - 0.4 * residual[:, 1] + rng.normal(0, 0.1, 300)
    baseline, count = cross_validated_r2(condition, target, folds=3, seed=4)
    combined, _ = cross_validated_r2(
        np.column_stack([condition, residual]), target, folds=3, seed=4,
    )
    assert count == 300
    assert baseline < 0.1
    assert combined > 0.9
