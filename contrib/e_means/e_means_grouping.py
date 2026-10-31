# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""CCCL-first grouping with a dependency-absent stable Torch fallback.

Tile IDs must be in [0, n_groups); contents are not read on the host. Buffers
must not be resized/rebound while a handle or captured graph uses them.
workspace_bytes counts owned tensor storage, including queried scratch;
retained_bytes also counts borrowed keys' backing storage. Neither includes
opaque compiled state or allocator rounding. Provider selection happens only
when a grouped binding is prepared. CCCL may cache descriptors; only the handle
retains their tensor storage. Torch sort/search temporary allocation is opaque.
"""

from functools import lru_cache
import importlib
import importlib.metadata
from numbers import Integral
from types import SimpleNamespace
from typing import NamedTuple
import warnings


_TORCH_WARNING = (
    "CuTe grouping is using Torch because cuda.compute is not installed; "
    "measured complete-call slowdown was ~0.6–2.4%. Install cuda-cccl for "
    "maximum performance."
)
_provider_warning_emitted = False


class _Grouping(NamedTuple):
    order: object
    offsets: object
    workspace_bytes: int
    scratch_bytes: int
    retained_bytes: int
    arrays: tuple
    tensors: tuple
    sorter: object
    bounder: object
    tile_ids: object
    end_bit: int
    provider: str


class _TorchGrouping(NamedTuple):
    order: object
    offsets: object
    workspace_bytes: int
    scratch_bytes: object
    retained_bytes: int
    tensors: tuple
    pointers: tuple
    sorted_keys: object
    order64: object
    boundaries: object
    offsets64: object
    provider: str


@lru_cache(maxsize=1)
def _grouping_provider():
    try:
        module = importlib.import_module("cuda.compute")
    except ModuleNotFoundError as error:
        if error.name not in ("cuda", "cuda.compute"):
            raise
        try:
            version = importlib.metadata.version("cuda-cccl")
        except importlib.metadata.PackageNotFoundError:
            return "torch"
        raise ImportError(
            "cuda-cccl %s is installed but cuda.compute is unavailable; "
            "install cuda-cccl with cuda.compute support (tested 1.2.0)"
            % version
        ) from error
    required = ("CountingIterator", "SortOrder", "make_lower_bound", "make_radix_sort")
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise ImportError(
            "cuda.compute is missing required grouping API: %s; "
            "install cuda-cccl with cuda.compute support (tested 1.2.0)"
            % ", ".join(missing)
        )
    return "cuda.compute"


def selected_grouping_provider(*, warn=False):
    """Return the stable CuTe grouping provider; optionally warn once."""
    global _provider_warning_emitted
    provider = _grouping_provider()
    if warn and provider == "torch" and not _provider_warning_emitted:
        warnings.warn(_TORCH_WARNING, RuntimeWarning, stacklevel=2)
        _provider_warning_emitted = True
    return provider


def _array(tensor):
    import numpy as np

    shape = tuple(tensor.shape)
    dtype = np.dtype(str(tensor.dtype).removeprefix('torch.'))
    ptr = tensor.data_ptr()
    return SimpleNamespace(shape=shape, dtype=dtype,
                           size=tensor.numel(), nbytes=tensor.numel() * tensor.element_size(),
                           data_ptr=lambda: ptr, __cuda_array_interface__={
                               'shape': shape, 'strides': None, 'typestr': dtype.str,
                               'data': (ptr, False), 'version': 3})


def _storage_bytes(tensors):
    storages = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        storages[(storage.device, storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


def prepare_grouping(keys, n_groups):
    if isinstance(n_groups, bool) or not isinstance(n_groups, Integral):
        raise TypeError('n_groups must be a host integer')
    n_groups = int(n_groups)
    if not 0 < n_groups < 2**31:
        raise ValueError('n_groups must be positive and below 2**31')
    import torch

    if (not isinstance(keys, torch.Tensor) or keys.ndim != 1 or not keys.is_cuda
            or keys.dtype != torch.int32 or not keys.is_contiguous()
            or not 0 < keys.numel() < 2**31):
        raise ValueError('keys must be a nonempty contiguous CUDA INT32 vector with B < 2**31')
    if selected_grouping_provider(warn=True) == "torch":
        with torch.cuda.device(keys.device), torch.no_grad():
            sorted_keys = torch.empty_like(keys)
            order64 = torch.empty(keys.numel(), dtype=torch.int64, device=keys.device)
            order = torch.empty_like(keys)
            boundaries = torch.arange(n_groups + 1, dtype=torch.int32, device=keys.device)
            offsets64 = torch.empty(n_groups + 1, dtype=torch.int64, device=keys.device)
            offsets = offsets64.view(torch.uint64)
            tensors = (keys, sorted_keys, order64, order, offsets, boundaries)
            pointers = tuple((tuple(t.shape), t.dtype, t.device, t.data_ptr()) for t in tensors)
            return _TorchGrouping(order, offsets, _storage_bytes(tensors[1:]), None,
                                  _storage_bytes(tensors), tensors, pointers, sorted_keys, order64,
                                  boundaries, offsets64, "torch")
    from cuda.compute import CountingIterator, SortOrder, make_lower_bound, make_radix_sort
    import numpy as np

    with torch.cuda.device(keys.device), torch.no_grad():
        tensors = (
            keys, torch.empty_like(keys),
            torch.arange(keys.numel(), dtype=torch.int32, device=keys.device),
            torch.empty_like(keys), torch.empty(n_groups + 1, dtype=torch.uint64, device=keys.device))
        k, sorted_keys, rows, order, offsets = map(_array, tensors)
        stream = torch.cuda.current_stream(keys.device)
        sorter = make_radix_sort(d_in_keys=k, d_out_keys=sorted_keys,
                                 d_in_values=rows, d_out_values=order, order=SortOrder.ASCENDING)
        end_bit = max(1, (n_groups - 1).bit_length())
        scratch_bytes = int(sorter(temp_storage=None, d_in_keys=k, d_out_keys=sorted_keys,
                                   d_in_values=rows, d_out_values=order, num_items=k.size,
                                   begin_bit=0, end_bit=end_bit, stream=stream))
        tensors += (torch.empty(scratch_bytes, dtype=torch.uint8, device=keys.device),)
        temp = _array(tensors[-1])
        tile_ids = CountingIterator(np.int32(0))
        bounder = make_lower_bound(d_data=sorted_keys, d_values=tile_ids, d_out=offsets)
        arrays = (k, sorted_keys, rows, order, offsets, temp)
        workspace_bytes = sum(t.untyped_storage().nbytes() for t in tensors[1:])
        return _Grouping(tensors[3], tensors[4], workspace_bytes, scratch_bytes,
                         workspace_bytes + keys.untyped_storage().nbytes(), arrays, tensors, sorter, bounder, tile_ids, end_bit,
                         "cuda.compute")


def launch_grouping(handle):
    import torch

    if handle.provider == "torch":
        if any((tuple(t.shape), t.dtype, t.device, t.data_ptr()) != expected
               for t, expected in zip(handle.tensors, handle.pointers)):
            raise ValueError('grouping buffers must not be resized or rebound')
        keys = handle.tensors[0]
        with torch.cuda.device(keys.device), torch.no_grad():
            torch.sort(keys, stable=True, out=(handle.sorted_keys, handle.order64))
            handle.order.copy_(handle.order64)
            torch.searchsorted(handle.sorted_keys, handle.boundaries, out=handle.offsets64)
        return handle.order, handle.offsets
    k, sorted_keys, rows, order, offsets, temp = handle.arrays
    if any(tuple(t.shape) != a.shape or t.data_ptr() != a.__cuda_array_interface__['data'][0]
           for t, a in zip(handle.tensors, handle.arrays)):
        raise ValueError('grouping buffers must not be resized or rebound')
    with torch.cuda.device(handle.tensors[0].device):
        stream = torch.cuda.current_stream(handle.tensors[0].device)
        handle.sorter(temp_storage=temp, d_in_keys=k, d_out_keys=sorted_keys,
                      d_in_values=rows, d_out_values=order, num_items=k.size,
                      begin_bit=0, end_bit=handle.end_bit, stream=stream)
        handle.bounder(d_data=sorted_keys, num_items=k.size, d_values=handle.tile_ids,
                       num_values=offsets.size, d_out=offsets, comp=None, stream=stream)
    return handle.order, handle.offsets
