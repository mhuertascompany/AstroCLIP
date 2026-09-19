import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from euclid.pixel_diffusion import ConditionalUNet, DiffusionSchedule
from euclid.train_pixel_diffusion import ConditionedStamps, PixelDiffusion, write_samples


class PixelDiffusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_velocity_roundtrip_including_pure_noise(self):
        schedule = DiffusionSchedule(20)
        clean, noise = torch.randn(3, 1, 8, 8), torch.randn(3, 1, 8, 8)
        t = torch.tensor([0, 10, 19])
        noisy, velocity = schedule.noisy_target(clean, noise, t)
        recovered, recovered_noise = schedule.clean_noise(noisy, velocity, t)
        torch.testing.assert_close(recovered, clean)
        torch.testing.assert_close(recovered_noise, noise)
        torch.testing.assert_close(noisy[-1], noise[-1])

    def test_full_resolution_gradients_and_null_condition(self):
        net = ConditionalUNet(16, base=8).eval()
        x, condition = torch.randn(1, 1, 224, 224), torch.randn(1, 16)
        t = torch.tensor([100])
        result = net(x, t, condition)
        self.assertEqual(result.shape, x.shape)
        result.square().mean().backward()
        self.assertGreater(net.condition_mlp[0].weight.grad.abs().sum().item(), 0)
        drop = torch.ones(1, dtype=torch.bool)
        with torch.no_grad():
            torch.testing.assert_close(net(x, t, condition, drop), net(x, t, condition + 10, drop))

    def test_ddim_seed_and_guidance_zero(self):
        model = ConditionalUNet(16, base=8).eval()
        schedule = DiffusionSchedule(10)
        condition = torch.randn(2, 16)
        a = schedule.sample(model, condition, size=32, steps=4, guidance=0, seed=12)
        b = schedule.sample(model, condition + 4, size=32, steps=4, guidance=0, seed=12)
        torch.testing.assert_close(a, b)
        self.assertTrue(torch.isfinite(a).all())
        self.assertLessEqual(a.abs().max().item(), 1)

    def test_data_split_and_lightning_checkpoint(self):
        import lightning as L
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'VIS').mkdir()
            for gid in [1, 2, 3]:
                Image.fromarray(np.full((224, 224), 70, dtype=np.uint8)).save(root / 'VIS' / f'VIS_{gid}.jpg')
            cache = root / 'conditions.npz'
            np.savez(cache, train_ids=[1, 2], val_ids=[3],
                     train_condition=np.eye(16, dtype=np.float32)[:2],
                     val_condition=np.eye(16, dtype=np.float32)[2:3], metadata=json.dumps({}))
            train = ConditionedStamps(cache, root, 'train')
            val = ConditionedStamps(cache, root, 'val')
            self.assertAlmostEqual(train[0][0].mean().item(), 70 / 127.5 - 1, places=5)
            model = PixelDiffusion(16, base=8, steps=10, cache_sha256='test')
            before = model.ema.input.weight.detach().clone()
            trainer = L.Trainer(accelerator='cpu', devices=1, max_epochs=1,
                                limit_val_batches=1, num_sanity_val_steps=0,
                                logger=False, enable_checkpointing=False,
                                enable_progress_bar=False, enable_model_summary=False)
            trainer.fit(model, DataLoader(train, batch_size=2), DataLoader(val))
            self.assertFalse(torch.equal(before, model.ema.input.weight))
            path = root / 'checkpoint.ckpt'
            trainer.save_checkpoint(path)
            restored = PixelDiffusion.load_from_checkpoint(path, map_location='cpu')
            for a, b in zip(model.ema.parameters(), restored.ema.parameters()):
                torch.testing.assert_close(a, b)
            write_samples(restored, val, root / 'samples', steps=2, n=1)
            with Image.open(root / 'samples' / 'conditioning_comparison.png') as panel:
                self.assertEqual(panel.size, (224, 224 * 5))
            np.savez(cache, train_ids=[1], val_ids=[1], train_condition=np.eye(16)[:1], val_condition=np.eye(16)[:1])
            with self.assertRaisesRegex(ValueError, 'overlap'):
                ConditionedStamps(cache, root, 'train')


if __name__ == '__main__':
    unittest.main()
