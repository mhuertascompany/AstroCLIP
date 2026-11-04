import numpy as np

from euclid.evaluate_jepa_csls import _csls_densities, _csls_retrieval_many


def test_csls_identity_retrieval_and_density_shapes():
    gallery = np.eye(7, dtype=np.float32)
    query_density, gallery_density = _csls_densities(
        gallery, gallery, (1, 3), device="cpu", gallery_chunk=3,
    )
    assert query_density[1].shape == (7,)
    assert gallery_density[3].shape == (7,)
    result = _csls_retrieval_many(
        gallery, gallery, query_density, gallery_density,
        np.arange(7), np.roll(np.arange(7), 1), top_k=3,
        device="cpu", query_chunk=2,
    )
    for k in (1, 3):
        np.testing.assert_array_equal(result[k]["rank"], np.ones(7))
        np.testing.assert_array_equal(result[k]["top_index"][:, 0], np.arange(7))
