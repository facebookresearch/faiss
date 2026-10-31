# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Native-byte helpers for INT8 E-means assignment.

Floating corpora can be converted explicitly before training::

    from faiss.contrib.e_means import Emeans
    from faiss.contrib.e_means.e_means_int8 import Int8Quantizer
    q = Int8Quantizer(x.shape[1])
    q.train(representative_sample)
    q.to(x.device)
    x_i8 = q.compute_codes(x)
    model = Emeans(x.shape[1], k, quantizer=q, assignment_backend="triton")
    model.train(x_i8)
    source_centroids = model.centroids  # already decoded FP32

The quantizer trains a per-coordinate FP32 shift and one global power-of-two
scale. One global scale preserves L2 geometry before rounding; separate scales
would reweight the coordinates. ``compute_codes`` processes internal chunks,
rounds, and clips to INT8. Finite outliers clip. Value checks are explicit
through validate; unchecked nonfinite values have no meaningful result.

With quantizer=q, E-means consumes matching native INT8 codes and returns
FP32 centroids/distances in decoded units. Floating-to-INT8 JIT is unsupported.
Without q, native INT8/UINT8 define raw numeric units; in that raw-code workflow
q.decode(model.centroids) explicitly converts signed code-coordinate outputs.
Helpers execute on CPU or CUDA; CPU E-means still requires FP32/q=None.
"""

import math

import torch

from .e_means_quantizer import _ldexp, _transform


_ENCODE_CHUNK_ELEMENTS = 8 * 1024 * 1024
_TARGET_PEAK = 112.0


def _matrix(x, name):
    if not isinstance(x, torch.Tensor):
        raise TypeError("%s must be a torch.Tensor" % name)
    if x.layout != torch.strided or x.ndim != 2:
        raise ValueError("%s must be a dense 2-D tensor" % name)
    if x.shape[1] == 0:
        raise ValueError("%s must not be empty" % name)
    if x.device.type not in ("cpu", "cuda"):
        raise ValueError("%s must be on a CPU or CUDA device" % name)
    return x.detach()


class Int8Quantizer:
    """An explicit affine preprocessor for floating-point corpora."""

    def __init__(self, d):
        self.d = int(d)
        if self.d < 1:
            raise ValueError("d must be positive")
        self.dtype = torch.int8
        self.code_size = self.d
        self.is_trained = False
        self.shift = None
        self.scale = None
        self._bias = None
        self.exponent = self._decode_exponent = self._distance_exponent = None

    def train(self, sample):
        """Train from a representative floating-point sample."""
        self.is_trained = False
        self.shift = None
        self.scale = None
        self._bias = None
        self.exponent = self._decode_exponent = self._distance_exponent = None
        sample = _matrix(sample, "sample")
        if not len(sample):
            raise ValueError("sample must not be empty")
        if sample.shape[1] != self.d:
            raise ValueError("sample has the wrong dimension")
        if sample.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise TypeError("sample must be FP32, FP16, or BF16")
        sample = sample.float()
        if not bool(torch.isfinite(sample).all()):
            raise ValueError("sample must be finite")
        shift = sample.mean(dim=0)
        if not bool(torch.isfinite(shift).all()):
            raise ValueError("the quantizer shift must be finite")
        peak = float((sample - shift).abs().max())
        if not math.isfinite(peak):
            raise ValueError("the centered sample peak must be finite")
        exponent = 0 if peak == 0.0 else math.floor(
            math.log2(_TARGET_PEAK / peak)
        )
        if exponent > 127:
            raise ValueError("the quantizer scale exceeds the FP32 range")
        scale = 2.0**exponent
        bias = -(shift * scale)
        if not bool(torch.isfinite(bias).all()):
            raise ValueError("the quantizer bias must be finite")
        self.shift = shift.contiguous()
        self.scale = float(scale)
        self._bias = bias.contiguous()
        self.exponent = exponent
        self._decode_exponent = torch.tensor(
            -exponent, dtype=torch.int32, device=sample.device
        )
        self._distance_exponent = 2 * self._decode_exponent
        self.is_trained = True

    def _validate(self, x, name):
        if not self.is_trained:
            raise RuntimeError("quantizer must be trained")
        x = _matrix(x, name)
        if x.device != self.shift.device:
            raise ValueError(
                "%s and quantizer must be on the same device" % name
            )
        if x.shape[1] != self.d:
            raise ValueError("%s has the wrong dimension" % name)
        return x

    def compute_codes(self, x):
        """Compute signed-byte codes in bounded internal chunks."""
        x = self._validate(x, "x")
        if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise TypeError("x must be FP32, FP16, or BF16")
        codes = torch.empty(x.shape, dtype=torch.int8, device=x.device)
        chunk_rows = max(1, _ENCODE_CHUNK_ELEMENTS // self.d)
        for begin in range(0, len(x), chunk_rows):
            end = begin + chunk_rows
            value = torch.add(
                self._bias, x[begin:end].float(), alpha=self.scale
            )
            value.round_().clamp_(-128, 127)
            codes[begin:end].copy_(value.to(torch.int8))
        return codes

    def decode(self, value):
        """Restore INT8 codes or FP32 learned state to source coordinates."""
        value = self._validate(value, "value")
        if value.dtype not in (torch.int8, torch.float32):
            raise TypeError("value must be INT8 or FP32")
        return _ldexp(value.float(), self._decode_exponent).add(self.shift)

    def to(self, device):
        """Move calibrated parameters in place and return self."""
        if not self.is_trained:
            raise RuntimeError("quantizer must be trained")
        device = torch.device(device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("device must be CPU or CUDA")
        self.shift = self.shift.to(device)
        self._bias = self._bias.to(device)
        self._decode_exponent = self._decode_exponent.to(device)
        self._distance_exponent = self._distance_exponent.to(device)
        return self

    def validate(self, x):
        """Explicit finite-under-transform check; finite outliers may clip."""
        x = self._validate(x, "x")
        if x.dtype == torch.int8:
            return  # Every native signed-byte code is finite.
        if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise TypeError("x must be INT8, FP32, FP16 or BF16")
        if not bool(torch.isfinite(_transform(x.float(), self)).all()):
            raise ValueError("x must be finite under the quantizer transform")


def _check_int8_support(device):
    """Reject devices outside the qualified INT8 assignment targets."""
    if device.type != "cuda":
        raise ValueError("INT8 assignment requires a CUDA tensor")
    capability = torch.cuda.get_device_capability(device)
    if (
        getattr(torch.version, "hip", None) is not None
        or capability not in ((8, 0), (9, 0), (10, 0))
    ):
        raise RuntimeError(
            "INT8 assignment requires an NVIDIA SM80, SM90, or SM100 GPU"
        )


def representation(dtype):
    if dtype == torch.int8:
        return "int8"
    if dtype == torch.uint8:
        return "uint8"
    raise TypeError("INT8 assignment requires INT8 or UINT8 input")


def quantize(value, representation, quantizer=None):
    value = _transform(value.float(), quantizer)
    if representation == "uint8":
        value = value - 128.0
    return value.round().clamp(-128, 127).to(torch.int8)


def separate_splits(donor, child, representation, quantizer=None):
    """Move child coordinate zero only when signed-code rows coincide."""
    donor_q, child_q = (quantize(x, representation, quantizer) for x in (donor, child))
    collapsed = (donor_q == child_q).all(dim=1)
    value = child_q[:, 0].to(torch.int16)
    neighbor = torch.where(value < 127, value + 1, value - 1).float()
    if quantizer is not None:
        neighbor = _ldexp(neighbor, quantizer._decode_exponent)
        neighbor = neighbor + quantizer.shift[0]
    elif representation == "uint8":
        neighbor += 128.0
    child[:, 0] = torch.where(collapsed & torch.isfinite(neighbor), neighbor, child[:, 0])
