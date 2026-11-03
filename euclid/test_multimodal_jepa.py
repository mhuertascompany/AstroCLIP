"""Tests for the paired VIS/SFH masked JEPA."""

import torch

from euclid.multimodal_jepa import MultimodalJEPA, TASKS


def _model():
    return MultimodalJEPA(
        image_size=32, image_patch=8, n_sfh_bins=20, sfh_patch=5,
        width=32, heads=4, layers=2,
        predictor_width=24, predictor_heads=4, predictor_layers=1,
    )


def _inputs(batch=4):
    return torch.randn(batch, 1, 32, 32), torch.randn(batch, 20)


def test_all_masking_tasks_have_visible_context_and_targets():
    model = _model()
    for task in TASKS:
        spec = model.make_mask(task, torch.device("cpu"))
        assert spec.visible[0]
        assert spec.visible[1:].any()
        assert spec.target.any()
        assert not torch.any(spec.visible & spec.target)


def test_all_task_losses_are_finite_and_target_is_frozen():
    model = _model()
    image, sfh = _inputs()
    total = 0
    for task in TASKS:
        loss, metrics = model.loss(image, sfh, task)
        assert torch.isfinite(loss)
        assert all(torch.isfinite(value) for value in metrics.values())
        total = total + loss
    total.backward()
    assert any(parameter.grad is not None for parameter in model.online.parameters())
    assert any(parameter.grad is not None for parameter in model.predictor.parameters())
    assert all(parameter.grad is None for parameter in model.target.parameters())


def test_cross_modal_shuffling_and_error_maps():
    model = _model().eval()
    image, sfh = _inputs()
    for task, expected in (("sfh_to_image", 16), ("image_to_sfh", 4)):
        paired, _ = model.loss(image, sfh, task)
        shuffled, _ = model.loss(image, sfh, task, shuffle_source=True)
        errors = model.cross_modal_errors(image, sfh, task)
        assert torch.isfinite(paired) and torch.isfinite(shuffled)
        assert errors.shape == (len(image), expected)


def test_ema_update_and_embedding_shapes():
    model = _model()
    image, sfh = _inputs()
    before = [parameter.clone() for parameter in model.target.parameters()]
    with torch.no_grad():
        next(model.online.parameters()).add_(1)
    model.update_target(0.5)
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, model.target.parameters(), strict=True)
    )
    for mode in ("joint", "image", "sfh"):
        assert model.embed(image, sfh, mode).shape == (len(image), 32)


def test_predictor_accepts_mixed_precision_context():
    model = _model()
    image, sfh = _inputs()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, _ = model.loss(image, sfh, "sfh_to_image")
    assert torch.isfinite(loss)
