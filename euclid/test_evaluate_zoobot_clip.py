import numpy as np

from euclid.evaluate_zoobot_clip import (
    embedding_diagnostics,
    projection_geometry_diagnostics,
    retrieval_ranks,
    sfh_shape_neighborhood_test,
    summarize_sfh_reconstruction,
    summarize_ranks,
)


def test_retrieval_ranks_find_aligned_pairs():
    embedding = np.eye(4, dtype=np.float32)
    ranks = retrieval_ranks(embedding, embedding, chunk_size=2)
    np.testing.assert_array_equal(ranks, np.ones(4, dtype=np.int32))
    summary = summarize_ranks(ranks)
    assert summary['recall']['R@1'] == 1.0
    assert summary['median_rank'] == 1.0


def test_retrieval_ranks_report_known_order():
    query = np.eye(3, dtype=np.float32)
    gallery = query[[1, 0, 2]]
    ranks = retrieval_ranks(query, gallery, chunk_size=2)
    np.testing.assert_array_equal(ranks, [2, 2, 1])


def test_embedding_diagnostics_distinguish_rank():
    rng = np.random.default_rng(7)
    one_dimensional = np.tile([1.0, 0.0, 0.0], (100, 1)).astype(np.float32)
    diverse = rng.normal(size=(1000, 3)).astype(np.float32)
    diverse /= np.linalg.norm(diverse, axis=1, keepdims=True)
    collapsed = embedding_diagnostics(one_dimensional, rng, n_random_pairs=100)
    healthy = embedding_diagnostics(diverse, rng, n_random_pairs=100)
    assert collapsed['effective_rank'] == 0.0
    assert healthy['effective_rank'] > 2.5


def test_projection_geometry_detects_preserved_neighborhoods():
    rng = np.random.default_rng(11)
    before = rng.normal(size=(100, 12)).astype(np.float32)
    before /= np.linalg.norm(before, axis=1, keepdims=True)
    result = projection_geometry_diagnostics(
        before, before.copy(), rng, subset_size=100, k=5,
        n_random_pairs=500,
    )
    assert result['pre_post_neighbor_overlap_at_k'] == 1.0
    assert result['random_pair_cosine_correlation'] > 0.999
    assert result['random_pair_cosine_mean_absolute_change'] == 0.0


def test_shape_neighborhood_prefers_matching_shapes():
    rng = np.random.default_rng(9)
    angles = np.linspace(0.05, 1.45, 30)
    shape = np.column_stack([np.cos(angles), np.sin(angles)]).astype(np.float32)
    shape /= shape.sum(axis=1, keepdims=True)
    log_shape = np.log10(shape + 1e-10)
    embedding = shape / np.linalg.norm(shape, axis=1, keepdims=True)
    result = sfh_shape_neighborhood_test(
        embedding, embedding, log_shape, rng, subset_size=30, k=3,
    )
    assert result['raw_sfh_neighbor_overlap_at_k'] > 0.9
    assert (
        result['mean_raw_sfh_cosine_of_cross_modal_neighbors'] >
        result['mean_raw_sfh_cosine_of_random_neighbors']
    )


def test_sfh_reconstruction_summary_is_zero_for_exact_shape():
    target = np.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
    result = summarize_sfh_reconstruction(target, np.log10(target + 1e-10))
    assert result['w1_mean'] < 1e-10
    assert result['mae_mean'] < 1e-10
