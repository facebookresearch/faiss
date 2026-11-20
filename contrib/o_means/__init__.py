# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""o-means: GPU e-means with a two-level nearest-centroid assignment.

PyTorch and Triton are imported on first use of Omeans or train_omeans.
"""

__all__ = ["Omeans", "train_omeans"]


def __getattr__(name):
    if name == "Omeans":
        from .o_means import Omeans

        return Omeans
    if name == "train_omeans":
        from .o_means import train_omeans

        return train_omeans
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
