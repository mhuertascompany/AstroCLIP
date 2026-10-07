"""Cache normalized aligned embeddings for the exact saved CLIP split."""
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


def _validated_embedding(values, modality):
    embedding = F.normalize(values.float(), dim=1)
    if (not torch.isfinite(embedding).all()
            or (embedding.norm(dim=1) < .99).any()):
        raise ValueError(f'Invalid aligned {modality} embedding.')
    return embedding


def prepare(checkpoint, dataset, split, stamps, output, device='cuda',
            batch_size=256, modality='sfh', workers=8):
    from .evaluate_zoobot_clip import (
        _read_rows, extraction_loader, validate_saved_split,
    )
    from .model_zoobot import EuclidZooBotCLIP
    if modality not in {'sfh', 'image'}:
        raise ValueError("modality must be 'sfh' or 'image'.")
    if Path(output).exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    train_rows, train_ids, val_rows, val_ids, _, _ = validate_saved_split(dataset, split, stamps)
    for ids in (train_ids, val_ids):
        for gid in ids:
            path = Path(stamps) / 'VIS' / f'VIS_{int(gid)}.jpg'
            if not path.is_file():
                raise FileNotFoundError(path)
    model = EuclidZooBotCLIP.load_from_checkpoint(
        str(checkpoint), map_location='cpu',
    )
    model.eval().requires_grad_(False).to(device)
    arrays = {}
    with h5py.File(dataset, 'r') as source, torch.inference_mode():
        for name, rows, ids in [('train', train_rows, train_ids), ('val', val_rows, val_ids)]:
            parts = []
            seen_ids = []
            if modality == 'sfh':
                for start in range(0, len(rows), batch_size):
                    sfh = torch.as_tensor(
                        _read_rows(source['sfh'], rows[start:start + batch_size]),
                        dtype=torch.float32, device=device,
                    )
                    # Includes the learned SFH adapter, as in CLIP evaluation.
                    embedding = _validated_embedding(
                        model.encode_sfh(sfh), modality,
                    )
                    parts.append(embedding.cpu().numpy())
            else:
                loader = extraction_loader(
                    dataset, stamps, rows, ids, 'VIS', 224, batch_size,
                    workers, image_format='jpg',
                )
                for batch in loader:
                    image = batch['image'].to(device, non_blocking=True)
                    # Includes the learned image adapter, as in CLIP evaluation.
                    with torch.autocast(
                        device_type=torch.device(device).type,
                        dtype=torch.float16,
                        enabled=torch.device(device).type == 'cuda',
                    ):
                        encoded = model.encode_image(image)
                    embedding = _validated_embedding(encoded, modality)
                    parts.append(embedding.cpu().numpy())
                    seen_ids.append(batch['galaxy_id'].numpy())
                if not np.array_equal(np.concatenate(seen_ids), ids):
                    raise ValueError(
                        f'{name} image loader changed the saved split order.'
                    )
            arrays.update({f'{name}_ids': ids, f'{name}_rows': rows,
                           f'{name}_condition': np.concatenate(parts)})
            print(f'Encoded {name} {modality}: {len(ids):,}', flush=True)
    condition_description = {
        'sfh': ('L2-normalized CLIP encode_sfh: median SFH, encoder + '
                'learned alignment adapter'),
        'image': ('L2-normalized CLIP encode_image: deterministic VIS stamp, '
                  'ZooBot encoder + learned alignment adapter'),
    }[modality]
    metadata = {'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint),
                'dataset': str(dataset), 'split': str(split), 'split_sha256': sha256(split),
                'modality': modality, 'condition': condition_description}
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
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--modality', choices=['sfh', 'image'], default='sfh')
    prepare(**vars(parser.parse_args()))
