"""Select real galaxies near equally spaced locations on a UMAP segment."""
import numpy as np


def sample_segment(coordinates, start, end, count=12, max_distance=np.inf, visible=None):
    xy = np.asarray(coordinates, dtype=float)
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    delta = end - start
    length = np.linalg.norm(delta)
    if not np.isfinite(length) or length == 0:
        raise ValueError('Choose two distinct finite endpoints.')
    if count < 2 or max_distance <= 0:
        raise ValueError('Need at least two samples and a positive distance limit.')
    valid = np.isfinite(xy).all(axis=1)
    if visible is not None:
        valid &= np.asarray(visible, dtype=bool)
    candidates = np.flatnonzero(valid)
    selected = []
    for fraction in np.linspace(0, 1, count):
        if not len(candidates):
            break
        distances = np.linalg.norm(xy[candidates] - (start + fraction * delta), axis=1)
        nearest = int(np.argmin(distances))
        if distances[nearest] <= max_distance:
            selected.append(int(candidates[nearest]))
            candidates = np.delete(candidates, nearest)
    selected = np.asarray(selected, dtype=int)
    position = ((xy[selected] - start) @ delta) / length ** 2
    return selected[np.argsort(position, kind='stable')].tolist()
