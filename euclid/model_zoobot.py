"""Euclid VIS--SFH alignment model.

The alignment engine was originally developed in the :mod:`cosmosweb`
package, but it is survey-agnostic: it consumes an image tensor and an SFH
tensor and contains no COSMOS-Web catalogue or JWST-filter logic. This module
is the canonical Euclid entry point and gives Euclid tools an accurate model
name while retaining state-dict compatibility with earlier
``CosmosWebZooBotCLIP`` checkpoints.
"""

from cosmosweb.model_zoobot import (
    CosmosWebZooBotCLIP,
    ResidualSFHProjection,
    make_sfh_projection,
)


class EuclidZooBotCLIP(CosmosWebZooBotCLIP):
    """Align Euclid VIS ZooBot features with fixed-grid SFHs.

    Euclid-specific sample construction, VIS loading, posterior SFH sampling,
    and splitting are supplied by
    :class:`euclid.dataset_zoobot.EuclidZooBotDataModule`. This class provides
    the shared neural alignment and loss implementation.
    """


__all__ = [
    'EuclidZooBotCLIP',
    'ResidualSFHProjection',
    'make_sfh_projection',
]
