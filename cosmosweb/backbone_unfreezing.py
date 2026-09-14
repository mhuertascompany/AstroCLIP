"""Select actual terminal feature stages when partially unfreezing timm models."""

from __future__ import annotations


def _resolve_module(root, path):
    module = root
    for part in path.split('.'):
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def _feature_info_names(backbone):
    info = getattr(backbone, 'feature_info', None)
    if info is None:
        return []
    if hasattr(info, 'module_name'):
        names = info.module_name()
        names = [names] if isinstance(names, str) else list(names)
    elif hasattr(info, 'get_dicts'):
        names = [record['module'] for record in info.get_dicts()]
    else:
        names = [record['module'] for record in info]
    return list(dict.fromkeys(name for name in names if name))


def feature_block_names(backbone):
    """Return feature stages in forward order, excluding pooling/classifier heads."""
    names = _feature_info_names(backbone)
    if names:
        return names
    for container_name in ('stages', 'blocks'):
        container = getattr(backbone, container_name, None)
        if container is not None and hasattr(container, 'named_children'):
            children = list(container.named_children())
            if children:
                return [f'{container_name}.{name}' for name, _ in children]
    excluded = {'head', 'classifier', 'fc', 'global_pool', 'norm_pre'}
    return [
        name for name, module in backbone.named_children()
        if name not in excluded and any(True for _ in module.parameters())
    ]


def unfreeze_last_feature_blocks(backbone, count):
    """Unfreeze the last ``count`` feature stages plus terminal feature norms.

    Returns the module paths that were selected. The classifier/pooling head is
    never counted as a feature block, avoiding the ambiguous top-level-child
    behavior of timm architectures such as ConvNeXt.
    """
    if count < 0:
        raise ValueError('unfreeze_blocks cannot be negative.')
    if count == 0:
        return []
    names = feature_block_names(backbone)
    if count > len(names):
        raise ValueError(
            f'Requested {count} unfrozen feature blocks, but the backbone '
            f'exposes only {len(names)}: {names}'
        )
    selected = names[-count:]
    output_modules = []
    for name in ('norm_pre', 'head'):
        module = getattr(backbone, name, None)
        if module is not None and any(True for _ in module.parameters()):
            output_modules.append(name)
    for name in selected + output_modules:
        for parameter in _resolve_module(backbone, name).parameters():
            parameter.requires_grad_(True)
    return selected + output_modules
