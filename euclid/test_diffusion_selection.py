import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import torch
from PIL import Image
from euclid.sample_diffusion_selection import selected_conditions, run
from euclid.prepare_diffusion_conditions import sha256


class SelectionTests(unittest.TestCase):
    def test_order_precision_outputs_and_noise(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); cache=root/'cache.npz'; csv=root/'ids.csv'
            ids=[2758212946672935686,2640791304676264916]
            np.savez(cache,train_ids=[ids[0]],val_ids=[ids[1]],train_rows=[0],val_rows=[1],
                     train_condition=[[1.,0.]],val_condition=[[0.,1.]])
            csv.write_text(f'galaxy_id,bundle_h5_row\n{ids[1]},123\n{ids[0]},999\n')
            actual,rows,cond,splits=selected_conditions(csv,cache)
            self.assertEqual(actual.tolist(),ids[::-1]);self.assertEqual(rows.tolist(),[1,0])
            self.assertEqual(splits,['val','train'])
            h5=root/'data.h5'
            with h5py.File(h5,'w') as f:
                f['object_id']=ids;f['sfh']=np.log10(np.full((2,10),.1));f['sfh_time_grid']=np.linspace(0,1,10)
            (root/'VIS').mkdir()
            for gid in ids: Image.new('L',(224,224),80).save(root/'VIS'/f'VIS_{gid}.jpg')
            checkpoint=root/'test.ckpt';checkpoint.write_bytes(b'test snapshot')
            calls=[]
            class Model:
                hparams=SimpleNamespace(cache_sha256=sha256(cache))
                ema=None
                def to(self,*a): return self
                def eval(self): return self
                def requires_grad_(self,*a): return self
            def sample(ema,condition,steps,guidance,seed):
                calls.append((condition.tolist(),guidance,seed))
                return torch.zeros(1,1,224,224)
            model=Model();model.schedule=SimpleNamespace(alpha=np.ones(10),sample=sample)
            args=SimpleNamespace(selection=csv,conditions=cache,dataset=h5,stamps=root,
                                 checkpoint=checkpoint,output=root/'out',device='cpu',steps=2,
                                 seed=42,n_seeds=2,guidance=[1.,2.])
            with patch('euclid.sample_diffusion_selection.PixelDiffusion.load_from_checkpoint',return_value=model):
                run(args)
            self.assertEqual([c[2] for c in calls],[42,43]*4)
            self.assertEqual(calls[0][0],[[0.,1.]])
            self.assertEqual(calls[2][0],[[1.,0.]])
            self.assertEqual(len(list((root/'out').rglob('*.png'))),10)
            csv.write_text('galaxy_id\n123\n')
            with self.assertRaises(ValueError):selected_conditions(csv,cache)
            csv.write_text(f'galaxy_id\n{ids[0]}\n{ids[0]}\n')
            with self.assertRaises(ValueError):selected_conditions(csv,cache)


if __name__=='__main__': unittest.main()
