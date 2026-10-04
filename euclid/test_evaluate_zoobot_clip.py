import numpy as np
import torch
import h5py

from euclid.evaluate_zoobot_clip import (
    embedding_diagnostics,
    extract_embeddings,
    projection_geometry_diagnostics,
    probabilistic_retrieval_ranks,
    retrieval_ranks,
    sfh_shape_neighborhood_test,
    summarize_sfh_reconstruction,
    summarize_ranks,
    validate_saved_split,
)


def test_validate_saved_split_accepts_native_fits(tmp_path):
    dataset = tmp_path / 'sfh.h5'
    ids = np.array([10, 20, 30], dtype=np.int64)
    with h5py.File(dataset, 'w') as target:
        target.attrs['n_galaxies'] = len(ids)
        target['galaxy_id'] = ids
        target['sfh_time_grid'] = np.linspace(0, 1, 4)
        target['sfh'] = np.zeros((3, 4), dtype=np.float32)
        target['sfh_realizations'] = np.zeros((3, 2, 4), dtype=np.float32)
        target['sfh_realization_valid'] = np.ones((3, 2), dtype=bool)
    split = tmp_path / 'split.npz'
    np.savez(
        split,
        train_rows=np.array([0, 1]), train_ids=ids[:2],
        val_rows=np.array([2]), val_ids=ids[2:],
    )
    fits_dir = tmp_path / 'cutouts' / 'VIS'
    fits_dir.mkdir(parents=True)
    (fits_dir / '30.fits').touch()
    result = validate_saved_split(
        dataset, split, tmp_path, image_format='fits',
    )
    np.testing.assert_array_equal(result[2], [2])
    np.testing.assert_array_equal(result[3], [30])


def test_extract_embeddings_can_return_both_preprojection_spaces():
    class FakeModel:
        sfh_decoder = None

        def eval(self):
            return self

        def encode_image_latent(self, image):
            return image.flatten(1)

        def project_image(self, latent):
            return latent[:, :2]

        def encode_sfh_latent(self, sfh):
            return sfh

        def project_sfh(self, latent):
            return latent[:, :2]

    loader = [{
        'image': torch.tensor([
            [[[3., 4.], [0., 0.]]],
            [[[0., 0.], [0., 2.]]],
        ]),
        'sfh': torch.tensor([[1., 2., 3.], [3., 2., 1.]]),
        'galaxy_id': torch.tensor([10, 20]),
    }]
    result = extract_embeddings(
        FakeModel(), loader, torch.device('cpu'),
        include_image_preprojection=True,
    )
    image, sfh, sfh_pre, ids, reconstruction, image_pre = result
    assert image.shape == (2, 2)
    assert sfh.shape == (2, 2)
    assert sfh_pre.shape == (2, 3)
    assert image_pre.shape == (2, 4)
    np.testing.assert_array_equal(ids, [10, 20])
    assert reconstruction is None
    np.testing.assert_allclose(np.linalg.norm(image_pre, axis=1), 1.0)


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


def test_probabilistic_ranks_account_for_gallery_uncertainty():
    query = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
    gallery = query.copy()
    logvar_query = np.full_like(query, np.log(0.01))
    logvar_gallery = np.array([
        [np.log(10.), np.log(10.)],
        [np.log(0.01), np.log(0.01)],
    ], dtype=np.float32)
    ranks = probabilistic_retrieval_ranks(
        query, logvar_query, gallery, logvar_gallery, chunk_size=1,
    )
    np.testing.assert_array_equal(ranks, [2, 1])


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
