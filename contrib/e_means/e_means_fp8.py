# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Fit and validate the shared affine transform for E4M3 assignment."""

import math

import torch

from .e_means_quantizer import _ldexp, _transform


_CENTER_GAIN = 1.5
_FIT_ROWS = 262144
_FORMAT_MAX = 448.0
_NEAR_PARITY = 0.001
_PROBE_CENTROIDS = 1024
_PROBE_ROWS = 2048
_TARGET_PEAK = 56.0


def _check_fp8_range(peak, name):
    if not (float(peak) <= _FORMAT_MAX):
        raise ValueError("%s exceeds the E4M3 representable range" % name)


def _check_fp8_support(device):
    """Reject devices that cannot execute the E4M3 assignment path."""
    if getattr(torch, "float8_e4m3fn", None) is None:
        raise RuntimeError("PyTorch does not provide float8_e4m3fn")
    if device.type != "cuda":
        raise ValueError("float8_e4m3fn assignment requires a CUDA tensor")
    capability = torch.cuda.get_device_capability(device)
    if (
        getattr(torch.version, "hip", None) is not None
        or capability not in ((9, 0), (10, 0), (10, 3))
    ):
        raise RuntimeError(
            "float8_e4m3fn assignment requires an NVIDIA SM90, SM100, "
            "or SM103 GPU"
        )


def _rounded(value, shift, scale):
    if shift is not None:
        value = value - shift
    return (value * scale).to(torch.float8_e4m3fn).float()


def _excess_distortion(x, centroids, shift, scale, reference, best):
    xq, cq = (_rounded(v, shift, scale).double() for v in (x, centroids))
    score = cq.square().sum(dim=1)[None, :] - 2.0 * (xq @ cq.T)
    labels = score.argmin(dim=1)
    row = torch.arange(len(x), device=x.device)
    chosen = reference[row, labels] + x.double().square().sum(dim=1)
    excess = float(chosen.clamp_min(0).sum()) - best
    return excess / best if best > 0 else float(excess > 0.0)


class Fp8Quantizer:
    """Calibrate one coordinate shift and one global power-of-two E4M3 scale.

    train uses a representative sample; to moves parameters explicitly. The
    caller must not retrain or move this quantizer during an E-means run.
    """

    def __init__(self, d):
        self.d = int(d)
        if self.d < 1:
            raise ValueError("d must be positive")
        self.dtype = getattr(torch, "float8_e4m3fn", None)
        self.code_size = self.d
        self.is_trained = False
        self.shift = self.scale = self._bias = self.exponent = None
        self._decode_exponent = self._distance_exponent = None
        self.device = None

    def _matrix(self, x, name, allow_empty=True):
        if not isinstance(x, torch.Tensor):
            raise TypeError("%s must be a torch.Tensor" % name)
        if (x.layout != torch.strided or x.ndim != 2 or x.shape[1] != self.d
                or (not allow_empty and not len(x))):
            raise ValueError("%s must be a dense (n, d) tensor" % name)
        if x.device.type not in ("cpu", "cuda"):
            raise ValueError("%s must be on CPU or CUDA" % name)
        return x.detach()

    def train(self, sample):
        """Fit the existing bounded plain/centered quality selector."""
        self.is_trained = False
        self.shift = self.scale = self._bias = self.exponent = None
        self._decode_exponent = self._distance_exponent = None
        self.device = None
        sample = self._matrix(sample, "sample", allow_empty=False)
        if sample.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise TypeError("sample must be FP32, FP16 or BF16")
        if self.dtype is None:
            raise RuntimeError("PyTorch does not provide float8_e4m3fn")
        sample = sample[:_FIT_ROWS].float()
        minimum, maximum = torch.aminmax(sample, dim=0)
        if not bool((torch.isfinite(minimum) & torch.isfinite(maximum)).all()):
            raise ValueError("sample must be finite")
        plain_peak = float(torch.maximum(minimum.abs(), maximum.abs()).max())
        shift = sample.mean(dim=0)
        if not bool(torch.isfinite(shift).all()):
            raise ValueError("the quantizer shift must be finite")
        center_peak = float(torch.maximum(
            (minimum - shift).abs(), (maximum - shift).abs()
        ).max())
        if not math.isfinite(center_peak):
            raise ValueError("the centered sample peak must be finite")
        plain_exp, center_exp = (
            0 if peak == 0 else math.floor(math.log2(_TARGET_PEAK / peak))
            for peak in (plain_peak, center_peak)
        )
        plain_scale, center_scale = 2.0**plain_exp, 2.0**center_exp
        probe = sample[:min(len(sample), _PROBE_ROWS + _PROBE_CENTROIDS)]
        if len(probe) < _PROBE_ROWS:
            plain_excess = centered_excess = None
            centered, decision = True, "small_sample_centered"
        else:
            x, centroids = probe[:-_PROBE_CENTROIDS], probe[-_PROBE_CENTROIDS:]
            # Reuse identical FP64 reference arithmetic for both candidates.
            reference = (centroids.double().square().sum(dim=1)[None, :]
                         - 2.0 * (x.double() @ centroids.double().T))
            row = torch.arange(len(x), device=x.device)
            best = float((reference[row, reference.argmin(dim=1)]
                          + x.double().square().sum(dim=1)).clamp_min(0).sum())
            plain_excess = _excess_distortion(
                x, centroids, None, plain_scale, reference, best
            )
            centered_excess = _excess_distortion(
                x, centroids, shift, center_scale, reference, best
            )
            if plain_excess * 2.0 <= _NEAR_PARITY:
                centered, decision = False, "uncentered_near_parity"
            elif centered_excess <= 0.0 < plain_excess:
                centered, decision = True, "centered_nonpositive_excess"
            elif centered_excess > 0.0 and plain_excess / centered_excess >= _CENTER_GAIN:
                centered, decision = True, "centered_gain"
            else:
                centered, decision = False, "uncentered_insufficient_gain"
            if min(plain_excess, centered_excess) > _NEAR_PARITY:
                decision += ";both_above_quality_gate"
        scale = center_scale if centered else plain_scale
        if (center_exp if centered else plain_exp) > 127:
            raise ValueError("the quantizer scale exceeds the FP32 range")
        selected_shift = shift.contiguous() if centered else None
        bias = None if selected_shift is None else -(selected_shift * scale)
        if bias is not None and not bool(torch.isfinite(bias).all()):
            raise ValueError("the quantizer bias must be finite")
        self.shift, self._bias = selected_shift, bias
        self.scale = scale
        self.exponent = center_exp if centered else plain_exp
        self.device = sample.device
        # Stage two tiny scalar parameters only during calibration/placement.
        self._decode_exponent = torch.tensor(
            -self.exponent, dtype=torch.int32, device=self.device
        )
        self._distance_exponent = 2 * self._decode_exponent
        self.centered, self.decision = centered, decision
        self.fit_rows = len(sample)
        self.sample_peak = center_peak if centered else plain_peak
        self.plain_excess, self.centered_excess = plain_excess, centered_excess
        self.is_trained = True
        self.validate(sample)

    def _validate(self, x, name):
        if not self.is_trained:
            raise RuntimeError("quantizer must be trained")
        x = self._matrix(x, name)
        if x.device != self.device:
            raise ValueError("%s and quantizer must be on the same device" % name)
        return x

    def to(self, device):
        """Move calibrated parameters in place and return self."""
        if not self.is_trained:
            raise RuntimeError("quantizer must be trained")
        device = torch.device(device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("device must be CPU or CUDA")
        if self.shift is not None:
            self.shift = self.shift.to(device)
            self._bias = self._bias.to(device)
        self._decode_exponent = self._decode_exponent.to(device)
        self._distance_exponent = self._distance_exponent.to(device)
        self.device = self._decode_exponent.device
        return self

    def compute_codes(self, x):
        """Encode floating rows, without clipping or a value scan."""
        x = self._validate(x, "x")
        if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise TypeError("x must be FP32, FP16 or BF16")
        return _transform(x.float(), self).to(self.dtype).contiguous()

    def decode(self, codes):
        """Decode E4M3 codes or floating code coordinates to FP32 units."""
        codes = self._validate(codes, "codes")
        if codes.dtype not in (self.dtype, torch.float32):
            raise TypeError("codes must be E4M3 or FP32")
        value = _ldexp(codes.float(), self._decode_exponent)
        return value if self.shift is None else value.add(self.shift)

    def validate(self, x):
        """Explicit finite/transformed-range check; may synchronize."""
        x = self._validate(x, "x")
        if x.dtype == self.dtype:
            value = x.float()
        elif x.dtype in (torch.float32, torch.float16, torch.bfloat16):
            value = _transform(x.float(), self)
        else:
            raise TypeError("x must be floating rows or E4M3 codes")
        if len(value):
            _check_fp8_range(float(value.abs().max()), "x")


def _update_fp8_peak(transformed, peak_out):
    # Retain the existing internal-centroid guard, without scanning caller rows.
    peak = transformed.abs().max()
    torch.maximum(peak_out, peak, out=peak_out)


def separate_splits(donor, child, quantizer):
    """Change child coordinate zero only for coincident E4M3 code rows."""
    donor_q, child_q = (
        _transform(x, quantizer).to(torch.float8_e4m3fn) for x in (donor, child)
    )
    # Torch has no E4M3 nextafter. Move sign/magnitude bits toward +infinity,
    # inward from +448, or from +/-0 to +2**-9. No dynamic row list/host read.
    collapsed = (donor_q.float() == child_q.float()).all(dim=1)
    bits = child_q[:, 0].contiguous().view(torch.uint8).to(torch.int16)
    adjacent = torch.where(bits >= 128, bits - 1, bits + 1)
    adjacent = torch.where(bits == 126, 125, adjacent)
    adjacent = torch.where((bits == 0) | (bits == 128), 1, adjacent)
    value = adjacent.to(torch.uint8).view(torch.float8_e4m3fn).float()
    if quantizer is not None:
        value = _ldexp(value, quantizer._decode_exponent)
        if quantizer.shift is not None:
            value = value + quantizer.shift[0]
    # A finite E4M3 neighbor may decode beyond FP32 range at tiny scales.
    # Retain the ordinary child when this single replacement is invalid.
    child[:, 0] = torch.where(collapsed & torch.isfinite(value), value, child[:, 0])
