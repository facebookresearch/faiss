# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Host adapter for E-means BF16 CuTe assignment on SM90.

Optional CuTe dependencies are resolved when the CuTe implementation is
selected. Assignment selects between the tiled kernel and a resource-legal
low-dimension row-owned kernel; there is no fallback backend.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
import importlib
import importlib.metadata
import importlib.util
import inspect
import weakref


_QUACK_VERSION = "0.6.5"
_CUTLASS_DSL_VERSION = "4.7.1"
_CUDA_PYTHON_MAJOR = 13
_MIN_OPTIN_SMEM = 227 * 1024
_TILE_N = 192
_MAX_BINDINGS = 2


@lru_cache(maxsize=1)
def _stack_status():
    """Return a stable compatibility result without importing on Faiss load."""
    exact_versions = (
        ("quack-kernels", _QUACK_VERSION),
        ("nvidia-cutlass-dsl", _CUTLASS_DSL_VERSION),
    )
    try:
        for distribution, expected in exact_versions:
            if importlib.metadata.version(distribution) != expected:
                return False, "%s version mismatch" % distribution
        cuda_version = importlib.metadata.version("cuda-python")
        if int(cuda_version.split(".", 1)[0]) != _CUDA_PYTHON_MAJOR:
            return False, "cuda-python requires major version 13"
        for module in ("cutlass", "cuda.bindings.driver", "quack"):
            if importlib.util.find_spec(module) is None:
                return False, "%s is unavailable" % module
        cuda_driver = importlib.import_module("cuda.bindings.driver")
        if not callable(getattr(cuda_driver, "CUstream", None)):
            return False, "cuda-python ABI mismatch"
        ops = importlib.import_module("quack.epilogue.ops")
        lane_helper = getattr(ops, "_lane_warp_info_n", None)
        if not callable(lane_helper) or tuple(
            inspect.signature(lane_helper).parameters
        ) != ("tiled_copy", "reference_src", "tidx"):
            return False, "QuACK ABI mismatch"
        for module, symbol in (
            ("quack.gemm_sm90", "GemmSm90"),
            ("quack.tile_scheduler", "TileScheduler"),
            ("quack.gemm_tvm_ffi_utils", "compile_gemm_kernel"),
        ):
            if not hasattr(importlib.import_module(module), symbol):
                return False, "QuACK ABI mismatch"
    except (
        AttributeError,
        ImportError,
        TypeError,
        ValueError,
        importlib.metadata.PackageNotFoundError,
    ):
        return False, "optional CuTe stack is unavailable"
    return True, "ready"


def _require_stack():
    available, reason = _stack_status()
    if not available:
        raise RuntimeError(
            "CuTe assignment requires quack-kernels==%s, "
            "nvidia-cutlass-dsl==%s, and cuda-python>=13,<14 (%s)"
            % (
                _QUACK_VERSION,
                _CUTLASS_DSL_VERSION,
                reason,
            )
        )


def _check_device(device):
    """Fail before compilation unless the fixed kernel is legal."""
    import torch

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("CuTe assignment requires a PyTorch CUDA tensor")
    if getattr(torch.version, "hip", None) is not None:
        raise RuntimeError("CuTe assignment requires NVIDIA CUDA SM90")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return _check_cuda_device(device)


@lru_cache(maxsize=None)
def _check_cuda_device(device):
    """Check pinned-stack and hardware metadata once per explicit CUDA device."""
    import torch

    with torch.cuda.device(device):
        _require_stack()
        capability = torch.cuda.get_device_capability(device)
        if capability != (9, 0):
            raise RuntimeError(
                "CuTe assignment requires exactly SM90; device is sm%d%d"
                % capability
            )
        properties = torch.cuda.get_device_properties(device)
        smem = getattr(properties, "shared_memory_per_block_optin", None)
        if smem is None:
            smem = getattr(properties, "shared_memory_per_block", 0)
        if smem < _MIN_OPTIN_SMEM:
            raise RuntimeError(
                "CuTe assignment requires at least %d bytes of opt-in "
                "shared memory"
                % _MIN_OPTIN_SMEM
            )


def _padded_extent(value, alignment):
    return ((int(value) + alignment - 1) // alignment) * alignment


def _check_extent(name, value):
    if value < 1 or value >= 2**31:
        raise ValueError("CuTe assignment requires 0 < %s < 2**31" % name)


def _check_padded_span(name, rows, dim):
    pitch = _padded_extent(dim, 8)
    if rows * pitch >= 2**31:
        raise ValueError("CuTe padded %s storage must fit signed int32" % name)
    return pitch


def _tensor_version(tensor):
    """Return the mutation version, or disable reuse when unavailable."""
    try:
        return tensor._version
    except (AttributeError, RuntimeError):
        return None


def _aligned_bf16(rows, dim, device):
    import torch

    pitch = _padded_extent(dim, 8)
    preferred = _padded_extent(dim, 16)
    if rows * preferred < 2**31:
        pitch = preferred
    if pitch == dim:
        return torch.empty((rows, dim), dtype=torch.bfloat16, device=device)
    return torch.zeros((rows, pitch), dtype=torch.bfloat16, device=device)[
        :, :dim
    ]


@dataclass
class _PreparedCentroids:
    source_ref: object
    source_version: int
    centroids: object
    norms: object
    padded_norms: object
    bindings: OrderedDict = field(default_factory=OrderedDict)
    spherical: bool = False
    transform: object = None
    transform_bias: object = None
    transform_token: object = None

    @property
    def num_centroids(self):
        return self.centroids.shape[0]

    @property
    def dim(self):
        return self.centroids.shape[1]


@dataclass
class _Binding:
    workspace: object
    codes: object
    scores: object
    labels: object
    launcher: object
    kernel: object


def _can_refresh(cached, centroids):
    return (
        isinstance(cached, _PreparedCentroids)
        and cached.centroids.device == centroids.device
        and cached.centroids.shape == centroids.shape
    )


def _refresh_prepared(cached, centroids, source_version, transform=None):
    import torch

    cached.centroids.copy_(centroids.detach() if transform is None else transform._center_rows(centroids))
    if cached.spherical:
        cached.norms.zero_()
    else:
        torch.sum(cached.centroids.float().square(), dim=1, out=cached.norms)
    cached.source_ref = weakref.ref(centroids)
    cached.source_version = source_version
    cached.transform = transform
    cached.transform_bias = None if transform is None else transform.shift
    cached.transform_token = None if transform is None else transform._fit_token()
    return cached


def _prepare_centroids(centroids, cached=None, spherical=False, transform=None):
    """Round one FP32 centroid state and compute norms from that mirror.

    Repeated public assignment reuses an unchanged installed centroid tensor.
    A new same-shape training readout refreshes its mirror and norms in place,
    preserving pointer-bound assignment workspaces across statistical batches.
    Refresh and assignment are ordered on the caller's current CUDA stream.
    """
    import torch

    if not isinstance(centroids, torch.Tensor):
        raise TypeError("centroids must be a tensor")
    _check_device(centroids.device)
    if centroids.ndim != 2:
        raise ValueError("centroids must be rank two")
    if (
        not centroids.is_cuda
        or centroids.dtype is not torch.float32
        or not centroids.is_contiguous()
    ):
        raise ValueError("centroids must be contiguous CUDA FP32")
    if transform is not None:
        from .e_means_quantizer import CastQuantizer
        if not isinstance(transform, CastQuantizer) or not transform.center or spherical:
            raise ValueError("CuTe centering requires a nonspherical centered CastQuantizer")
        transform._matrix(centroids)
    k, dim = centroids.shape
    _check_extent("K", k)
    _check_extent("d", dim)
    _check_padded_span("centroid", k, dim)
    padded_k = _padded_extent(k, _TILE_N)
    if padded_k >= 2**31:
        raise ValueError("CuTe padded centroid IDs must fit signed int32")
    source_version = _tensor_version(centroids)
    if (
        isinstance(cached, _PreparedCentroids)
        and cached.source_ref() is centroids
        and source_version is not None
        and cached.source_version == source_version
        and cached.spherical == spherical
        and cached.transform is transform
        and cached.transform_token == (None if transform is None else transform._fit_token())
    ):
        return cached

    with torch.cuda.device(centroids.device):
        if _can_refresh(cached, centroids):
            cached.spherical = spherical
            return _refresh_prepared(cached, centroids, source_version, transform)
        mirror = _aligned_bf16(k, dim, centroids.device)
        storage = torch.empty(
            padded_k, dtype=torch.float32, device=centroids.device
        )
        if padded_k != k:
            storage[k:].fill_(float("inf"))
        prepared = _PreparedCentroids(
            None,
            None,
            mirror,
            storage[:k],
            storage.view(1, padded_k),
            spherical=spherical,
        )
        return _refresh_prepared(prepared, centroids, source_version, transform)


def _grouped_plan(batch, prepared):
    """Choose a legal BF16 L2 plan using coarse measured work and grid size."""
    import torch

    if (batch * prepared.num_centroids < 2**31
            or prepared.centroids.dtype is not torch.bfloat16
            or prepared.spherical):
        return None
    from .e_means_cute_grouped import _plan_fits_smem
    from .e_means_cute_row_owned import _row_owned_shape_legal

    if not _row_owned_shape_legal(batch, prepared.num_centroids, prepared.dim):
        return None

    properties = torch.cuda.get_device_properties(prepared.centroids.device)
    if (properties.major, properties.minor) != (9, 0):
        return None
    # Prefer M256 with a full grid wave plus half a second wave.
    wide_grid = (batch + 255) // 256 >= (3 * properties.multi_processor_count + 1) // 2
    plans = ((256, 3), (128, 3), (128, 2)) if wide_grid else ((128, 3), (128, 2))
    for tile_m, stages in plans:
        tile_k = _padded_extent(prepared.dim, 16 if tile_m == 256 else 32)
        plan = tile_k, tile_m, stages
        if _plan_fits_smem(torch.bfloat16, plan, properties.shared_memory_per_block_optin):
            return plan
    return None


def _new_binding(batch, prepared):
    import torch

    plan = _grouped_plan(batch, prepared)
    if plan is not None:
        return _new_grouped_binding(batch, prepared, plan=plan)

    from .e_means_cute_kernel import (
        launch_tiled_gemm,
        prepare_tiled_gemm,
    )
    from .e_means_cute_row_owned import (
        launch_row_owned,
        prepare_row_owned,
        row_owned_plan,
    )

    _check_extent("B", batch)
    _check_padded_span("row", batch, prepared.dim)
    with torch.cuda.device(prepared.centroids.device):
        workspace = _aligned_bf16(
            batch, prepared.dim, prepared.centroids.device
        )
        scores = torch.empty(
            batch, dtype=torch.float32, device=workspace.device
        )
        plan = row_owned_plan(
            batch,
            prepared.num_centroids,
            prepared.dim,
            workspace.device,
        )
        if plan is not None:
            labels = torch.empty(
                batch, dtype=torch.int64, device=workspace.device
            )
            codes = None
            kernel = prepare_row_owned(
                workspace,
                prepared,
                scores,
                labels,
                plan,
                )
            launcher = launch_row_owned
        else:
            labels = None
            codes = torch.empty(
                batch, dtype=torch.int64, device=workspace.device
            )
            kernel = prepare_tiled_gemm(workspace, prepared, codes, scores)
            launcher = launch_tiled_gemm
    return _Binding(workspace, codes, scores, labels, launcher, kernel)


def _new_grouped_binding(batch, prepared, *, plan):
    """Prepare the shared BF16 L2 route with fixed grouping/output buffers."""
    import torch

    from .e_means_cute_grouped import launch_grouped, prepare_grouped

    if prepared.centroids.dtype is not torch.bfloat16 or prepared.spherical:
        raise ValueError('private grouped integration currently requires unencoded BF16 L2/FP32 scores')
    _check_extent('B', batch)
    _check_padded_span('row', batch, prepared.dim)
    with torch.cuda.device(prepared.centroids.device):
        workspace = _aligned_bf16(batch, prepared.dim, prepared.centroids.device)
        scores = torch.empty(batch, dtype=torch.float32, device=workspace.device)
        labels = torch.empty(batch, dtype=torch.int64, device=workspace.device)
        kernel = prepare_grouped(workspace, prepared, scores, labels, plan)
    return _Binding(workspace, None, scores, labels, launch_grouped, kernel)


def _binding_for(batch, prepared):
    binding = prepared.bindings.pop(batch, None)
    if binding is None:
        binding = _new_binding(batch, prepared)
    prepared.bindings[batch] = binding
    while len(prepared.bindings) > _MAX_BINDINGS:
        prepared.bindings.popitem(last=False)
    return binding


def _assign_rows(rows, prepared, distance_out=None, labels_out=None):
    """Assign represented BF16 rows into owned or caller FP32/int64 outputs."""
    import torch

    _require_stack()
    if not isinstance(prepared, _PreparedCentroids):
        raise TypeError("prepared centroids have the wrong type")
    if not isinstance(rows, torch.Tensor) or rows.ndim != 2:
        raise ValueError("rows must be a rank-two tensor")
    if (
        not rows.is_cuda
        or rows.dtype is not torch.bfloat16
        or not rows.is_contiguous()
    ):
        raise ValueError("rows must be contiguous CUDA BF16")
    if rows.device != prepared.centroids.device:
        raise ValueError("rows and centroids must be on the same device")
    if rows.shape[1] != prepared.dim:
        raise ValueError("rows and centroids must have the same dimension")
    if rows.shape[0] < 1:
        raise ValueError("rows must not be empty")

    from .e_means_cute_argmin import require_tensor

    if distance_out is None:
        distance_out = torch.empty(len(rows), dtype=torch.float32, device=rows.device)
    if labels_out is None:
        labels_out = torch.empty(len(rows), dtype=torch.int64, device=rows.device)
    require_tensor(distance_out, "distances", dtype=torch.float32, shape=(len(rows),), device=rows.device)
    require_tensor(labels_out, "labels", dtype=torch.int64, shape=(len(rows),), device=rows.device)
    with torch.cuda.device(rows.device):
        binding = _binding_for(rows.shape[0], prepared)
        binding.workspace.copy_(rows)
        if not prepared.spherical:
            torch.linalg.vector_norm(binding.workspace, dim=1, dtype=torch.float32, out=distance_out)
        binding.launcher(binding.kernel, labels_out)
        if prepared.spherical:
            distance_out.copy_(binding.scores)
        else:
            distance_out.square_().add_(binding.scores).clamp_min_(0)
    return distance_out, labels_out
