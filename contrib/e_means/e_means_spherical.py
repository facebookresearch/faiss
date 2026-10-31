# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Bounded, zero-preserving centroid projection; no input preprocessing."""


def normalize_into(source, out):
    """Project finite rows, prescaling before the norm and before narrowing.

    Each temporary row matrix is bounded to 32 MiB (one row if wider).
    Zero maps to zero. The source can alias the destination.
    """
    is_torch = hasattr(source, "element_size")
    itemsize = source.element_size() if is_torch else source.itemsize
    rows = max(1, (1 << 25) // (source.shape[1] * itemsize))
    if is_torch:
        import torch

        for begin in range(0, len(source), rows):
            values, target = source[begin:begin + rows], out[begin:begin + rows]
            work = target if values.dtype == target.dtype else torch.empty_like(values)
            maximum = torch.linalg.vector_norm(values, ord=float("inf"), dim=1, keepdim=True)
            work.copy_(values)
            work.div_(torch.where(maximum > 0, maximum, 1))
            norm = torch.linalg.vector_norm(work, dim=1, keepdim=True)
            work.div_(torch.where(norm > 0, norm, 1))
            if work is not target:
                target.copy_(work)
    else:
        import numpy as np

        for begin in range(0, len(source), rows):
            values, target = source[begin:begin + rows], out[begin:begin + rows]
            work = target if values.dtype == target.dtype else np.empty_like(values)
            maximum = np.max(np.abs(values), axis=1, keepdims=True)
            np.divide(values, np.where(maximum > 0, maximum, 1), out=work)
            norm = np.sqrt(np.einsum("ij,ij->i", work, work))[:, None]
            np.divide(work, np.where(norm > 0, norm, 1), out=work)
            if work is not target:
                np.copyto(target, work, casting="unsafe")
    return out
