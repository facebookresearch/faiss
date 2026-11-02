# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Host adapter for E-means BF16/E4M3 CuTe assignment on SM90.

Optional CuTe dependencies are resolved when the CuTe implementation is
selected. Assignment selects between the tiled kernel and a resource-legal
low-dimension row-owned kernel; there is no fallback backend. E4M3 uses tensor-core
partial accumulation with FP32 promotion every 128 contraction coordinates.
Ranking is approximate; distances are recomputed from selected represented operands.
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


def _check_padded_span(name, rows, dim, element_size=2):
    pitch = _padded_extent(dim, 16 // element_size)
    if rows * pitch >= 2**31:
        raise ValueError("CuTe padded %s storage must fit signed int32" % name)
    return pitch


def _tensor_version(tensor):
    """Return the mutation version, or disable reuse when unavailable."""
    try:
        return tensor._version
    except (AttributeError, RuntimeError):
        return None


def _aligned_operand(rows, dim, device, dtype):
    import torch

    pitch = _padded_extent(dim, 16 // dtype.itemsize)
    if dtype is torch.bfloat16:
        preferred = _padded_extent(dim, 16)
        if rows * preferred < 2**31:
            pitch = preferred
    if pitch == dim:
        return torch.empty((rows, dim), dtype=dtype, device=device)
    return torch.zeros((rows, pitch), dtype=dtype, device=device)[
        :, :dim
    ]


@dataclass
class _PreparedCentroids:
    source_ref: object
    source_version: int
    transform: object
    transform_token: object
    centroids: object
    norms: object
    padded_norms: object
    peak: object = None
    bindings: OrderedDict = field(default_factory=OrderedDict)
    transform_bias: object = None
    spherical: bool = False

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
    row_norms: object = None
    prepared_row_norms: object = None


def _can_refresh(cached, centroids):
    return (
        isinstance(cached, _PreparedCentroids)
        and cached.centroids.device == centroids.device
        and cached.centroids.shape == centroids.shape
    )


def _transform_parameter(transform):
    if transform is None:
        return None
    from .e_means_quantizer import CastQuantizer

    return transform.shift if isinstance(transform, CastQuantizer) else transform._bias


def _transform_token(transform):
    if transform is None:
        return None
    # The prepared object retains transform, so its identity cannot be reused.
    from .e_means_quantizer import CastQuantizer

    if isinstance(transform, CastQuantizer):
        return id(transform), transform._fit_token()
    bias = transform._bias
    version = None if bias is None else _tensor_version(bias)
    return (
        id(transform),
        transform.scale,
        None
        if bias is None
        # Versionless parameters cannot prove an unchanged transform.
        else (bias.data_ptr(), version if version is not None else object(), bias.device),
    )


def _refresh_prepared(cached, centroids, source_version, transform, peak_out):
    import torch

    if cached.centroids.dtype is torch.bfloat16:
        from .e_means_cute_argmin import affine_cast

        wide = None if cached.spherical else torch.empty_like(centroids)
        value = centroids.detach() if transform is None else transform._center_rows(centroids.detach())
        affine_cast(value, None, 1.0, cached.centroids, None, squares=wide)
        torch.autograd.graph.increment_version(cached.centroids)
        if cached.spherical:
            cached.norms.zero_()
        else:
            torch.sum(wide, dim=1, out=cached.norms)
    else:
        from .e_means_cute_argmin import affine_cast
        from .e_means_fp8 import _check_fp8_range

        peak = centroids.new_zeros(()) if peak_out is None else peak_out
        cached.peak = peak
        wide = torch.empty_like(centroids)
        affine_cast(centroids.detach(), _transform_parameter(transform),
                    1.0 if transform is None else transform.scale,
                    cached.centroids, peak, squares=wide)
        torch.sum(wide, dim=1, out=cached.norms)
        if peak_out is None:
            _check_fp8_range(peak, "centroids")
    cached.source_ref = weakref.ref(centroids)
    cached.source_version = source_version
    cached.transform = transform
    # Retain this particular bias; the quantizer may replace its parameters.
    cached.transform_bias = _transform_parameter(transform)
    cached.transform_token = _transform_token(transform)
    return cached


def _prepare_centroids(centroids, cached=None, transform=None, peak_out=None, dtype=None, spherical=False):
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
    dtype = torch.bfloat16 if dtype is None else dtype
    if dtype is torch.bfloat16 and transform is not None:
        from .e_means_quantizer import CastQuantizer
        if not isinstance(transform, CastQuantizer) or not transform.center:
            raise ValueError("BF16 transform requires a centered CastQuantizer")
    if spherical and (dtype is not torch.bfloat16 or transform is not None):
        raise ValueError("spherical CuTe requires unencoded BF16 operands")
    if dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise ValueError("CuTe operands must be BF16 or E4M3")
    k, dim = centroids.shape
    _check_extent("K", k)
    _check_extent("d", dim)
    _check_padded_span("centroid", k, dim, dtype.itemsize)
    padded_k = _padded_extent(k, _TILE_N)
    if padded_k >= 2**31:
        raise ValueError("CuTe padded centroid IDs must fit signed int32")
    source_version = _tensor_version(centroids)
    transform_token = _transform_token(transform)
    if (
        isinstance(cached, _PreparedCentroids)
        and cached.source_ref() is centroids
        and source_version is not None
        and cached.source_version == source_version
        and cached.transform_bias is (_transform_parameter(transform))
        and cached.transform_token == transform_token
        and cached.centroids.dtype is dtype
        and cached.spherical == spherical
    ):
        if dtype is torch.float8_e4m3fn:
            from .e_means_fp8 import _check_fp8_range

            with torch.cuda.device(centroids.device):
                if peak_out is None:
                    _check_fp8_range(cached.peak, "centroids")
                elif peak_out is not cached.peak:
                    torch.maximum(peak_out, cached.peak, out=peak_out)
        return cached

    with torch.cuda.device(centroids.device):
        if _can_refresh(cached, centroids) and cached.centroids.dtype is dtype:
            cached.spherical = spherical
            return _refresh_prepared(cached, centroids, source_version, transform, peak_out)
        mirror = _aligned_operand(k, dim, centroids.device, dtype)
        storage = torch.empty(
            padded_k, dtype=torch.float32, device=centroids.device
        )
        if padded_k != k:
            storage[k:].fill_(float("inf"))
        prepared = _PreparedCentroids(
            None,
            None,
            None,
            None,
            mirror,
            storage[:k],
            storage.view(1, padded_k),
            spherical=spherical,
        )
        return _refresh_prepared(prepared, centroids, source_version, transform, peak_out)


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
    _check_padded_span("row", batch, prepared.dim, prepared.centroids.element_size())
    with torch.cuda.device(prepared.centroids.device):
        workspace = _aligned_operand(
            batch, prepared.dim, prepared.centroids.device, prepared.centroids.dtype
        )
        scores = torch.empty(
            batch, dtype=torch.float32, device=workspace.device
        )
        plan = row_owned_plan(
            batch,
            prepared.num_centroids,
            prepared.dim,
            workspace.device,
            workspace.dtype,
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
    row_norms = (torch.empty(batch, dtype=torch.float32, device=workspace.device)
                 if workspace.dtype is torch.float8_e4m3fn else None)
    return _Binding(workspace, codes, scores, labels, launcher, kernel, row_norms)


def _new_grouped_binding(batch, prepared, *, plan):
    """Prepare the shared BF16 L2 route with fixed grouping/output buffers."""
    import torch

    from .e_means_cute_grouped import launch_grouped, prepare_grouped

    if prepared.centroids.dtype is not torch.bfloat16 or prepared.spherical:
        raise ValueError('private grouped integration currently requires unencoded BF16 L2/FP32 scores')
    _check_extent('B', batch)
    _check_padded_span('row', batch, prepared.dim, prepared.centroids.element_size())
    with torch.cuda.device(prepared.centroids.device):
        workspace = _aligned_operand(batch, prepared.dim, prepared.centroids.device, torch.bfloat16)
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
    """Assign represented rows into owned or caller FP32/int64 outputs."""
    import torch

    _require_stack()
    if not isinstance(prepared, _PreparedCentroids):
        raise TypeError("prepared centroids have the wrong type")
    if not isinstance(rows, torch.Tensor) or rows.ndim != 2:
        raise ValueError("rows must be a rank-two tensor")
    if (
        not rows.is_cuda
        or (rows.dtype is not prepared.centroids.dtype
            and not (prepared.transform is not None and rows.dtype is torch.float32))
        or not rows.is_contiguous()
    ):
        raise ValueError("rows must be contiguous CUDA BF16/E4M3 operands or FP32 quantizer input")
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
        if prepared.centroids.dtype is torch.bfloat16:
            binding.workspace.copy_(rows)
            if not prepared.spherical:
                torch.linalg.vector_norm(binding.workspace, dim=1, dtype=torch.float32,
                                         out=distance_out)
                distance_out.square_()
        else:
            from .e_means_cute_argmin import (
                affine_cast, launch_row_norms, prepare_row_norms,
            )

            if rows.dtype is torch.float32:
                q = prepared.transform
                affine_cast(rows, q._bias, q.scale, binding.workspace, None,
                            packed_rows=True)
            else:
                binding.workspace.copy_(rows)

        binding.launcher(binding.kernel, labels_out)
        if prepared.centroids.dtype is torch.float8_e4m3fn:
            if binding.prepared_row_norms is None:
                binding.prepared_row_norms = prepare_row_norms(
                    binding.workspace, binding.row_norms, prepared.centroids)
            launch_row_norms(binding.prepared_row_norms, distance_out, labels_out)
        elif prepared.centroids.dtype is torch.bfloat16:
            if prepared.spherical:
                distance_out.copy_(binding.scores)
            else:
                distance_out.add_(binding.scores).clamp_min_(0)
        if prepared.transform is not None and prepared.centroids.dtype is not torch.bfloat16:
            # Operand-device scope retains stock ldexp, including subnormals/zero.
            torch.ldexp(distance_out, prepared.transform._distance_exponent,
                        out=distance_out)
    return distance_out, labels_out
