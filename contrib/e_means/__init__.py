# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""NumPy E-means and lazily imported Torch implementations."""

from .e_means import Emeans, NumpyEmeans

__all__ = [
    "Emeans",
    "NumpyEmeans",
    "TorchEmeans",
    "QuantizedTorchEmeans",
    "DistributedEmeans",
    "EmeansShardedCentroids",
    "EmeansShardedCentroidsIVF",
]


def __getattr__(name):
    if name == "TorchEmeans":
        from .e_means_torch import TorchEmeans

        return TorchEmeans
    if name == "QuantizedTorchEmeans":
        from .e_means_quantized import QuantizedTorchEmeans

        return QuantizedTorchEmeans
    if name == "DistributedEmeans":
        from .e_means_distributed import DistributedEmeans

        return DistributedEmeans
    if name == "EmeansShardedCentroids":
        from .sharded_centroids import EmeansShardedCentroids

        return EmeansShardedCentroids
    if name == "EmeansShardedCentroidsIVF":
        from .sharded_centroids_ivf import EmeansShardedCentroidsIVF

        return EmeansShardedCentroidsIVF
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
