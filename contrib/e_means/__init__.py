# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .e_means import Emeans
from .e_means_distributed import DistributedEmeans

__all__ = ["Emeans", "DistributedEmeans", "EmeansShardedCentroids"]


def __getattr__(name):
    if name == "EmeansShardedCentroids":
        from .sharded_centroids import EmeansShardedCentroids
        return EmeansShardedCentroids
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
