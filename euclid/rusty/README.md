# Rusty execution wrappers

This directory contains Rusty-specific SLURM wrappers for the Euclid
SFH–morphology experiments.

Shared Python implementations remain in `euclid/` and `cosmosweb/`.
Candide-specific SLURM files remain unchanged.

Large datasets, generated stamps, logs, checkpoints, caches, and virtual
environments are stored outside the Git repository under the Rusty Ceph
filesystem.
