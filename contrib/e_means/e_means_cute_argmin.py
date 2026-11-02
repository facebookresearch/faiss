# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Packed ``(FP32 score, int32 id)`` atomic argmin support for CuTe."""

from collections import OrderedDict
from typing import NamedTuple

import cuda.bindings.driver as cuda
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.cute.runtime import from_dlpack
import torch

_THREADS = 256
INITIAL_KEY = 0x7F80000000000000
_COMPILED = OrderedDict()


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
