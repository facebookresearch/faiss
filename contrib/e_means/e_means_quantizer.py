# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Stateless FP16/BF16 casts for represented-value GPU E-means.

Codes are native Torch tensors, not packed CPU bytes. Conversion and decoding
check metadata only. Call validate explicitly for finite/range checks; it may
synchronize. Helpers can execute on CPU, but CPU E-means requires quantizer=None.
"""

import torch


class CastQuantizer:
    """Round floating rows to FP16 or BF16, optionally after an explicit fit.

    ``center=False`` preserves the stateless cast and no-op train. With
    ``center=True``, train a representative sample before encoding, then place
    the fitted centre on the input device with to. Codes represent x - shift;
    decode restores original FP32 coordinates before model statistics.
    Matching model-input dtype means preencoded codes; compute_codes itself
    always takes original coordinates, even when its input already has that
    dtype. Widening rounded source data cannot recover lost information.

    The owned shift is read-only. Explicit refitting/moving invalidates models
    bound to the previous fit; do it before a new run, never during a run or
    while a graph/reader uses it. Calls share parameters and require ordering.
    Values must be finite and fit the target after centering. Fit/conversion
    do not validate values; validate explicitly. Centering is L2-only.
    """

    def __init__(self, d, dtype, *, center=False):
        self.d = int(d)
        if self.d < 1:
            raise ValueError("d must be positive")
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("dtype must be torch.float16 or torch.bfloat16")
        if not isinstance(center, bool):
            raise TypeError("center must be a bool")
        self.dtype = dtype
        self.code_size = 2 * self.d
        self._center = center
        self._shift = None
        self._generation = 0
        self._shift_version = None
        self.is_trained = not center

    @property
    def center(self):
        """Whether explicit centering is enabled (independent of its values)."""
        return self._center

    @property
    def shift(self):
        """Borrow the fitted FP32 centre; do not mutate it."""
        return self._shift

    def train(self, sample=None):
        """Fit an owned centre explicitly, without numerical value checks.

        With centering disabled this remains a no-op. Prefer original FP32
        training rows. FP64 mean accumulation is narrowed once to FP32; no
        automatic sampling, model-time fit or input-dependent policy is used.
        Nonfinite data follows Torch reduction semantics. Call validate after
        fitting for explicit finite/range checks; validation may synchronize.
        """
        if not self.center:
            return
        self.is_trained = False
        self._shift = self._shift_version = None
        self._generation += 1
        sample = self._matrix(sample, fitting=True)
        if not len(sample):
            raise ValueError("sample must not be empty")
        if sample.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise TypeError("sample must be FP32, FP16 or BF16")
        # Normal owned parameters retain a mutation version even if the caller
        # fits inside inference_mode. Input rows are detached, never changed.
        with torch.inference_mode(False), torch.no_grad():
            shift = sample.mean(dim=0, dtype=torch.float64).float().contiguous()
        self._shift = shift
        self._shift_version = shift._version
        self.is_trained = True

    def _fit_token(self):
        """Metadata-only identity of one immutable fitted parameter tensor."""
        if not self.center:
            return None
        if not self.is_trained or self.shift is None:
            raise RuntimeError("centered quantizer must be trained explicitly")
        if self.shift._version != self._shift_version:
            raise RuntimeError("quantizer centre is frozen; train explicitly to refit")
        return self._generation, id(self.shift), self.shift._version, self.shift.device

    def to(self, device):
        """Place fitted parameters explicitly; stateless casts still return self."""
        device = torch.device(device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("device must be CPU or CUDA")
        if self.center:
            self._fit_token()
            if self.shift.device != device:
                with torch.inference_mode(False), torch.no_grad():
                    shift = self.shift.to(device)
                if shift is not self.shift:
                    self._shift = shift
                    self._generation += 1
                    self._shift_version = shift._version
        return self

    def _matrix(self, x, *, fitting=False):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.layout != torch.strided or x.ndim != 2 or x.shape[1] != self.d:
            raise ValueError("x must be a dense (n, d) tensor")
        if x.device.type not in ("cpu", "cuda"):
            raise ValueError("x must be on CPU or CUDA")
        if self.center and not fitting:
            self._fit_token()
            if x.device != self.shift.device:
                raise ValueError("x and quantizer centre must be on the same device")
        return x.detach()

    def _center_rows(self, x):
        x = self._matrix(x)
        return x.float().sub(self.shift) if self.center else x

    def compute_codes(self, x):
        """Return native represented rows, without a value scan."""
        x = self._matrix(x)
        if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise TypeError("x must be FP32, FP16 or BF16")
        return (x.float().sub(self.shift) if self.center else x).to(self.dtype).contiguous()

    def decode(self, codes):
        """Decode matching codes to original FP32 rows; never decode model state."""
        codes = self._matrix(codes)
        if codes.dtype != self.dtype:
            raise TypeError("codes must have the quantizer dtype")
        value = codes.float()
        return (value.add_(self.shift) if self.center else value).contiguous()

    def validate(self, x):
        """Check codes, or original nonmatching floats after centering.

        Matching dtype means codes, just as in model input. For original rows
        already in that dtype, widen explicitly before this check. With
        centering enabled, also check that the fitted centre and decoded
        original FP32 rows are finite, including target-format rounding.
        """
        x = self._matrix(x)
        if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise TypeError("x must be FP32, FP16 or BF16")
        value = (x.float().sub(self.shift) if self.center and x.dtype != self.dtype else x.float())
        valid = (value.abs() <= torch.finfo(self.dtype).max).all()
        if self.center:
            codes = x if x.dtype == self.dtype else self.compute_codes(x)
            valid = valid & torch.isfinite(self.shift).all() & torch.isfinite(self.decode(codes)).all()
        if not bool(valid):
            message = "x must be finite and fit the quantizer range"
            if self.center:
                message += " and decode to finite original FP32 values"
            raise ValueError(message)


def _ldexp(value, exponent):
    # Stock CUDA ldexp can launch on the current device rather than the operand's.
    if value.is_cuda:
        with torch.cuda.device(value.device):
            return torch.ldexp(value, exponent)
    return torch.ldexp(value, exponent)
