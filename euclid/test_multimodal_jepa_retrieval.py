import numpy as np

from euclid.evaluate_multimodal_jepa_retrieval import _retrieval, _summary


def test_exact_retrieval_recovers_paired_gallery_and_control_ranks():
    gallery = np.eye(5, dtype=np.float32)
    query = gallery.copy()
    paired = np.arange(5)
    control = np.roll(paired, 1)
    result = _retrieval(
        query, gallery, paired, control, top_k=3,
        device="cpu", chunk_size=2,
    )
    np.testing.assert_array_equal(result["rank"], np.ones(5, dtype=np.int32))
    np.testing.assert_array_equal(result["top_index"][:, 0], paired)
    assert np.all(result["control_rank"] > 1)
    summary = _summary(result, len(gallery))
    assert summary["recall_at_1"] == 1.0
    assert summary["median_rank_percentile"] == 0.0


def test_retrieval_accepts_flattened_patch_tokens():
    rng = np.random.default_rng(3)
    gallery = rng.normal(size=(7, 12)).astype(np.float32)
    result = _retrieval(
        gallery.copy(), gallery, np.arange(7), np.roll(np.arange(7), 1),
        top_k=2, device="cpu", chunk_size=3,
    )
    assert np.all(result["rank"] == 1)
    assert result["top_index"].shape == (7, 2)
