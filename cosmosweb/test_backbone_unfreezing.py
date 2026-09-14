"""Tests for architecture-aware partial backbone unfreezing."""

import unittest

from cosmosweb.backbone_unfreezing import unfreeze_last_feature_blocks


class FakeParameter:
    def __init__(self):
        self.requires_grad = False

    def requires_grad_(self, value):
        self.requires_grad = value


class FakeModule:
    def __init__(self, children=None, n_parameters=1):
        self._children = dict(children or [])
        self._parameters = [FakeParameter() for _ in range(n_parameters)]
        for name, module in self._children.items():
            setattr(self, name, module)

    def named_children(self):
        return self._children.items()

    def parameters(self):
        yield from self._parameters
        for child in self._children.values():
            yield from child.parameters()

    def __getitem__(self, index):
        return list(self._children.values())[index]


class FakeFeatureInfo:
    def module_name(self):
        return [f'stages.{index}' for index in range(4)]


class BackboneUnfreezingTests(unittest.TestCase):
    def make_backbone(self):
        stages = FakeModule(
            [(str(index), FakeModule()) for index in range(4)],
            n_parameters=0,
        )
        backbone = FakeModule([
            ('stem', FakeModule()),
            ('stages', stages),
            ('norm_pre', FakeModule(n_parameters=0)),
            ('head', FakeModule()),
        ], n_parameters=0)
        backbone.feature_info = FakeFeatureInfo()
        return backbone

    def test_convnext_feature_stages_are_selected_instead_of_head(self):
        backbone = self.make_backbone()
        selected = unfreeze_last_feature_blocks(backbone, 2)
        self.assertEqual(selected, ['stages.2', 'stages.3', 'head'])
        self.assertFalse(next(backbone.stages[0].parameters()).requires_grad)
        self.assertFalse(next(backbone.stages[1].parameters()).requires_grad)
        self.assertTrue(next(backbone.stages[2].parameters()).requires_grad)
        self.assertTrue(next(backbone.stages[3].parameters()).requires_grad)
        self.assertTrue(next(backbone.head.parameters()).requires_grad)

    def test_too_many_feature_stages_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'exposes only 4'):
            unfreeze_last_feature_blocks(self.make_backbone(), 5)


if __name__ == '__main__':
    unittest.main()
