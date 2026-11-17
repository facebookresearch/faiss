# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Explicit NumPy reference and lazily imported tensor E-means."""

from .e_means import NumpyEmeans

__all__ = [
    "Emeans",
    "NumpyEmeans",
    "QuantizedEmeans",
    "DistributedEmeans",
    "DistributedNumpyEmeans",
    "EmeansShardedCentroids",
    "EmeansShardedCentroidsIVF",
]


def __getattr__(name):
    if name == "Emeans":
        from .e_means_torch import Emeans

        return Emeans
    if name == "QuantizedEmeans":
        from .e_means_quantized import QuantizedEmeans

        return QuantizedEmeans
    if name == "DistributedEmeans":
        from .e_means_distributed import DistributedEmeans

        return DistributedEmeans
    if name == "DistributedNumpyEmeans":
        from .e_means_distributed import DistributedNumpyEmeans

        return DistributedNumpyEmeans
    if name == "EmeansShardedCentroids":
        from .sharded_centroids import EmeansShardedCentroids

        return EmeansShardedCentroids
    if name == "EmeansShardedCentroidsIVF":
        from .sharded_centroids_ivf import EmeansShardedCentroidsIVF

        return EmeansShardedCentroidsIVF
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
