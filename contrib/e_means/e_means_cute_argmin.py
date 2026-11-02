# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Packed ``(FP32 score, int32 id)`` atomic argmin support for CuTe."""

from collections import OrderedDict
import math
from typing import NamedTuple

import cuda.bindings.driver as cuda
import cutlass.cute as cute
from cutlass import BFloat16, Float8E4M3FN, Float32, Int32, Int64, Uint32, const_expr
from cutlass.cute.runtime import from_dlpack
from quack import utils as quack_utils
import torch

_THREADS = 256
INITIAL_KEY = 0x7F80000000000000
_COMPILED = OrderedDict()
_AFFINE_PREPARE_COMPILED = OrderedDict()
_STAGE_ROW_NORMS_COMPILED = OrderedDict()
_PACKED_ROWS = 8


def require_tensor(
    tensor,
    name,
    *,
    dtype=None,
    shape=None,
    device=None,
    contiguous=True,
):
    """Validate one tensor passed to a raw-pointer CuTe binding."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("%s must be a tensor" % name)
    if not tensor.is_cuda:
        raise ValueError("%s must be a CUDA tensor" % name)
    if dtype is not None and tensor.dtype is not dtype:
        raise ValueError(
            "%s must be %s" % (name, str(dtype).split(".")[-1])
        )
    if shape is not None and tuple(tensor.shape) != tuple(shape):
        raise ValueError("%s must have shape %s" % (name, tuple(shape)))
    if device is not None and tensor.device != device:
        raise ValueError("%s must be on %s" % (name, device))
    if contiguous and not tensor.is_contiguous():
        raise ValueError("%s must be contiguous" % name)
    return tensor


@cute.jit
def encode_argmin_key(score: Float32, code: Int32) -> Int64:
    """Encode native FP32 ordering, canonical signed zero, then lower ID."""
    score = Float32(0.0) if score == 0.0 else score
    bits = score.bitcast(Uint32)
    ordered = ~bits if (bits >> 31) != 0 else bits ^ Uint32(0x80000000)
    return (
        (ordered.to(Int64) - Int64(0x80000000)) * Int64(0x100000000)
        + code.to(Int64)
    )


@cute.jit
def decode_argmin_score(key: Int64) -> Float32:
    """Invert the ordered FP32 word in a packed argmin key."""
    signed_order = (key >> Int64(32)).to(Int32)
    ordered = (signed_order.to(Int64) + Int64(0x80000000)).to(Uint32)
    score_bits = (
        ordered ^ Uint32(0x80000000)
        if (ordered >> 31) != 0
        else ~ordered
    )
    return score_bits.bitcast(Float32)


class _AffineCastSm90:
    """Store a BF16/E4M3 mirror with optional rounded squares and peak."""

    def __init__(
        self, centered, rows_per_block=1, with_peak=True, with_squares=False
    ):
        self.centered = centered
        self.rows_per_block = rows_per_block
        self.with_peak = with_peak
        self.with_squares = with_squares

    @cute.jit
    def __call__(
        self, x, bias, out, peak, squares, scale, dim, pitch, stream
    ):
        rows = x.shape[0] // dim
        self.kernel(x, bias, out, peak, squares, scale, dim, pitch).launch(
            grid=(
                (rows + self.rows_per_block - 1) // self.rows_per_block,
                1,
                1,
            ),
            block=(32 * self.rows_per_block, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        bias: cute.Tensor,
        out: cute.Tensor,
        peak: cute.Tensor,
        squares: cute.Tensor,
        scale: Float32,
        dim: Int64,
        pitch: Int64,
    ):
        thread, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        thread = thread.to(Int64)
        lane = thread % Int64(32)
        row = (
            block.to(Int64) * Int64(self.rows_per_block)
            + thread // Int64(32)
        )
        rows = x.shape[0] // dim
        column = lane
        local_peak = Float32(0.0)
        if row < rows:
            while column < pitch:
                output_index = row * pitch + column
                if column < dim:
                    if const_expr(out.element_type is BFloat16):
                        value = x[row * dim + column]
                    else:
                        value = x[row * dim + column] * scale
                        if const_expr(self.centered):
                            value += bias[column]
                        if const_expr(self.with_peak):
                            absolute = -value if value < 0.0 else value
                            absolute = Float32.inf if value != value else absolute
                            local_peak = (
                                absolute if absolute > local_peak else local_peak
                            )
                    rounded = value.to(out.element_type)
                    out[output_index] = rounded
                    if const_expr(self.with_squares):
                        wide = rounded.to(Float32)
                        if const_expr(out.element_type is BFloat16):
                            squares[row * dim + column] = cute.math.mul(
                                wide, wide, rounding=cute.math.RoundingMode.NEAREST_EVEN,
                                ftz=False)
                        else:
                            squares[row * dim + column] = wide * wide
                else:
                    out[output_index] = Float32(0.0).to(out.element_type)
                column += Int64(32)
        if const_expr(self.with_peak):
            for offset in (16, 8, 4, 2, 1):
                other_peak = cute.arch.shuffle_sync_bfly(
                    local_peak, offset=offset
                )
                local_peak = (
                    other_peak if other_peak > local_peak else local_peak
                )
        if const_expr(self.with_peak):
            if lane == 0:
                cute.arch.atomic_fmax(
                    quack_utils.elem_pointer(peak, 0).llvm_ptr,
                    local_peak,
                    sem="relaxed",
                    scope="gpu",
                )


class _StageRowNormsSm90:
    """Reduce direct selected-centroid E4M3 squared differences."""

    def __init__(self, rows_per_block=_PACKED_ROWS):
        self.rows_per_block = rows_per_block

    @cute.jit
    def __call__(self, x, centroids, labels, norms, dim, pitch, centroid_pitch, stream):
        rows = norms.shape[0]
        self.kernel(x, centroids, labels, norms, dim, pitch, centroid_pitch).launch(
            grid=((rows + self.rows_per_block - 1) // self.rows_per_block,
                  1, 1),
            block=(32 * self.rows_per_block, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, x: cute.Tensor, centroids: cute.Tensor, labels: cute.Tensor,
               norms: cute.Tensor, dim: Int64, pitch: Int64, centroid_pitch: Int64):
        thread, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        thread = thread.to(Int64)
        lane = thread % Int64(32)
        row = (block.to(Int64) * Int64(self.rows_per_block)
               + thread // Int64(32))
        rows = norms.shape[0]
        column = lane
        total = Float32(0.0)
        if row < rows:
            while column < dim:
                stored = x[row * pitch + column].to(Float32)
                stored -= centroids[labels[row] * centroid_pitch + column].to(Float32)
                total += stored * stored
                column += Int64(32)
        for offset in (16, 8, 4, 2, 1):
            total += cute.arch.shuffle_sync_bfly(total, offset=offset)
        if row < rows and lane == 0:
            norms[row] = total


class PreparedRowNorms(NamedTuple):
    compiled: object
    flat_mirror: torch.Tensor
    mirror: torch.Tensor
    norms: torch.Tensor
    centroids: torch.Tensor
    flat_centroids: torch.Tensor


@torch.no_grad()
def prepare_row_norms(mirror, norms, centroids):
    """Bind E4M3 operands for selected-centroid direct distances."""
    require_tensor(mirror, "E4M3 mirror", dtype=torch.float8_e4m3fn,
                   contiguous=False)
    if mirror.ndim != 2 or mirror.stride(1) != 1:
        raise ValueError("E4M3 mirror must be a row-major matrix")
    require_tensor(norms, "row norms", dtype=torch.float32,
                   shape=(len(mirror),), device=mirror.device)
    dim = mirror.shape[1]
    pitch = mirror.stride(0)
    if pitch < dim:
        raise ValueError("E4M3 mirror pitch must cover its dimension")
    storage = mirror.as_strided((len(mirror), pitch), (pitch, 1)).reshape(-1)
    require_tensor(centroids, "E4M3 centroids", dtype=mirror.dtype,
                   device=mirror.device, contiguous=False)
    if centroids.ndim != 2 or centroids.shape[1] != dim or centroids.stride(1) != 1:
        raise ValueError("centroids must match the row dimension")
    centroid_pitch = centroids.stride(0)
    flat_centroids = centroids.as_strided((len(centroids), centroid_pitch),
                                          (centroid_pitch, 1)).reshape(-1)
    labels = torch.empty(len(mirror), dtype=torch.int64, device=mirror.device)
    with torch.cuda.device(mirror.device):
        key = (_PACKED_ROWS, tuple(mirror.shape), pitch,
               tuple(centroids.shape), centroid_pitch, mirror.device.index)
        compiled = _STAGE_ROW_NORMS_COMPILED.pop(key, None)
        if compiled is None:
            compiled = cute.compile(
                _StageRowNormsSm90(),
                from_dlpack(storage, assumed_align=16),
                from_dlpack(flat_centroids, assumed_align=16),
                from_dlpack(labels, assumed_align=4),
                from_dlpack(norms, assumed_align=4),
                Int64(dim),
                Int64(pitch),
                Int64(centroid_pitch),
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            )
        _STAGE_ROW_NORMS_COMPILED[key] = compiled
        while len(_STAGE_ROW_NORMS_COMPILED) > 4:
            _STAGE_ROW_NORMS_COMPILED.popitem(last=False)
    return PreparedRowNorms(compiled, storage, mirror, norms, centroids, flat_centroids)


@torch.no_grad()
def launch_row_norms(prepared, norms_out=None, labels=None):
    """Reduce on the caller's current CUDA stream into the actual destination."""
    if not isinstance(prepared, PreparedRowNorms):
        raise TypeError("prepared row norms have the wrong type")
    norms = prepared.norms if norms_out is None else norms_out
    require_tensor(norms, "row norms", dtype=torch.float32,
                   shape=prepared.norms.shape, device=prepared.norms.device)
    require_tensor(labels, "labels", dtype=torch.int64,
                   shape=norms.shape, device=norms.device)
    with torch.cuda.device(prepared.mirror.device):
        prepared.compiled(prepared.flat_mirror,
                          prepared.flat_centroids,
                          labels, norms,
                          prepared.mirror.shape[1], prepared.mirror.stride(0),
                          prepared.centroids.stride(0))


@torch.no_grad()
def affine_cast(
    x, bias, scale, out, peak, *, packed_rows=False, squares=None
):
    """Write a BF16/E4M3 mirror, optional rounded squares and transformed peak."""
    require_tensor(x, "rows", dtype=torch.float32)
    require_tensor(out, "mirror", contiguous=False)
    if out.dtype not in (torch.float8_e4m3fn, torch.bfloat16):
        raise ValueError("mirror must be E4M3 or BF16")
    if out.dtype is torch.bfloat16 and (bias is not None or scale != 1.0 or peak is not None):
        raise ValueError("BF16 mirror requires identity scale, no bias and no peak")
    with_peak = peak is not None
    with_squares = squares is not None
    if with_squares:
        require_tensor(squares, "squares", dtype=torch.float32,
                       shape=x.shape, device=x.device)
    if with_peak:
        require_tensor(
            peak, "peak", dtype=torch.float32, shape=(), device=x.device
        )
    valid_scale = math.isfinite(scale) and scale > 0.0
    if not valid_scale or not math.log2(scale).is_integer():
        raise ValueError("FP8 assignment scale must be a power of two")
    if scale > torch.finfo(torch.float32).max:
        raise ValueError("FP8 assignment scale exceeds the FP32 range")
    if with_peak and peak.requires_grad:
        raise ValueError("FP8 peak accumulator does not match input")
    if out.device != x.device or out.shape != x.shape or out.stride(1) != 1:
        raise ValueError("mirror does not match rows")
    centered = bias is not None
    if centered:
        require_tensor(
            bias,
            "bias",
            dtype=torch.float32,
            shape=(x.shape[1],),
            device=x.device,
        )
        bound_bias = bias
    else:
        bound_bias = x[0]
    bound_peak = peak if with_peak else x[0]
    pitch = out.stride(0)
    storage = out.as_strided((len(x), pitch), (pitch, 1))
    with torch.cuda.device(x.device):
        args = (x.reshape(-1), bound_bias, storage.reshape(-1), bound_peak,
                (squares if with_squares else x).reshape(-1))
        rows_per_block = _PACKED_ROWS if packed_rows else 1
        key = (
            tuple(x.shape),
            pitch,
            centered,
            rows_per_block,
            with_peak,
            with_squares,
            out.dtype,
            x.device.index,
        )
        compiled = _AFFINE_PREPARE_COMPILED.pop(key, None)
        if compiled is None:
            operation = _AffineCastSm90(centered, rows_per_block, with_peak, with_squares)
            specs = tuple(from_dlpack(tensor, assumed_align=16 if i == 2 else 4)
                          for i, tensor in enumerate(args))
            compiled = cute.compile(
                operation, *specs, Float32(scale), Int64(x.shape[1]), Int64(pitch),
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            )
        _AFFINE_PREPARE_COMPILED[key] = compiled
        while len(_AFFINE_PREPARE_COMPILED) > 4:
            _AFFINE_PREPARE_COMPILED.popitem(last=False)
        compiled(*args, scale, x.shape[1], pitch)


class _DecodeArgminOutputsSm90:
    @cute.jit
    def __call__(
        self,
        keys: cute.Tensor,
        codes: cute.Tensor,
        scores: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(keys, codes, scores).launch(
            grid=(cute.ceil_div(codes.shape[0], _THREADS), 1, 1),
            block=(_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        keys: cute.Tensor,
        codes: cute.Tensor,
        scores: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        row = bidx * _THREADS + tidx
        if row < codes.shape[0]:
            key = keys[row]
            codes[row] = (key & Int64(0xFFFFFFFF)).to(Int32)
            scores[row] = decode_argmin_score(key)
            keys[row] = Int64(INITIAL_KEY)


class PreparedArgminDecode(NamedTuple):
    compiled: object
    keys: torch.Tensor
    codes: torch.Tensor
    scores: torch.Tensor


@torch.no_grad()
def prepare_argmin_decode(keys, codes, scores):
    """Bind stable private outputs and compile their decode/reset kernel."""
    require_tensor(keys, "keys", dtype=torch.int64)
    count = keys.numel()
    if codes.dtype not in (torch.int32, torch.int64):
        raise ValueError("codes must be int32 or int64")
    require_tensor(
        codes,
        "codes",
        shape=(count,),
        device=keys.device,
    )
    require_tensor(
        scores,
        "scores",
        dtype=torch.float32,
        shape=(count,),
        device=keys.device,
    )
    flat_keys = keys.reshape(-1)
    keys_cute = from_dlpack(flat_keys, assumed_align=8)
    codes_cute = from_dlpack(codes, assumed_align=4)
    scores_cute = from_dlpack(scores, assumed_align=4)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    cache_key = (count, keys.device.index, codes.dtype)
    compiled = _COMPILED.pop(cache_key, None)
    if compiled is None:
        compiled = cute.compile(
            _DecodeArgminOutputsSm90(),
            keys_cute,
            codes_cute,
            scores_cute,
            stream,
            options="--enable-tvm-ffi",
        )
    _COMPILED[cache_key] = compiled
    while len(_COMPILED) > 2:
        _COMPILED.popitem(last=False)
    return PreparedArgminDecode(
        compiled,
        flat_keys,
        codes,
        scores,
    )


@torch.no_grad()
def launch_argmin_decode(prepared, codes_out=None):
    """Decode on the caller's current stream and restore the key sentinel."""
    if codes_out is not None:
        require_tensor(codes_out, "codes", dtype=prepared.codes.dtype,
                       shape=prepared.codes.shape, device=prepared.codes.device)
    codes = prepared.codes if codes_out is None else codes_out
    with torch.cuda.device(prepared.scores.device):
        prepared.compiled(prepared.keys, codes, prepared.scores)
