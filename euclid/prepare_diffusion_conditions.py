"""Cache normalized aligned SFH embeddings for the exact saved CLIP split."""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare(checkpoint, dataset, split, stamps, output, device='cuda', batch_size=256):
    from .evaluate_zoobot_clip import validate_saved_split, _read_rows
    from cosmosweb.model_zoobot import CosmosWebZooBotCLIP
    if Path(output).exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    train_rows, train_ids, val_rows, val_ids, _, _ = validate_saved_split(dataset, split, stamps)
    for ids in (train_ids, val_ids):
        for gid in ids:
            path = Path(stamps) / 'VIS' / f'VIS_{int(gid)}.jpg'
            if not path.is_file():
                raise FileNotFoundError(path)
    model = CosmosWebZooBotCLIP.load_from_checkpoint(str(checkpoint), map_location='cpu')
    model.eval().requires_grad_(False).to(device)
    arrays = {}
    with h5py.File(dataset, 'r') as source, torch.inference_mode():
        for name, rows, ids in [('train', train_rows, train_ids), ('val', val_rows, val_ids)]:
            parts = []
            for start in range(0, len(rows), batch_size):
                sfh = torch.as_tensor(_read_rows(source['sfh'], rows[start:start + batch_size]),
                                      dtype=torch.float32, device=device)
                # Includes the learned SFH adapter, as in CLIP evaluation.
                embedding = F.normalize(model.encode_sfh(sfh).float(), dim=1)
                if not torch.isfinite(embedding).all() or (embedding.norm(dim=1) < .99).any():
                    raise ValueError('Invalid aligned SFH embedding.')
                parts.append(embedding.cpu().numpy())
            arrays.update({f'{name}_ids': ids, f'{name}_rows': rows,
                           f'{name}_condition': np.concatenate(parts)})
            print(f'Encoded {name}: {len(ids):,}', flush=True)
    metadata = {'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint),
                'dataset': str(dataset), 'split': str(split), 'split_sha256': sha256(split),
                'condition': 'L2-normalized CLIP encode_sfh: median SFH, encoder + learned adapter'}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays, metadata=json.dumps(metadata))
    temporary.replace(output)
    print(f'Conditions: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'dataset', 'split', 'stamps', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=256)
    prepare(**vars(parser.parse_args()))
