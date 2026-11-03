# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Quantized PyTorch E-means and representation-specific execution."""

import torch

from .base import _SPLIT_EPSILON
from .e_means_torch import (
    TorchEmeans,
)

from .torch_backend import (
    _get_triton_assign_rows,
    _check_torch_abs_bound,
    _separate_splits,
)

__all__ = ["QuantizedTorchEmeans"]


def _get_triton_int8_functions():
    try:
        from .triton_assignment import assign_int8_rows, prepare_int8_centroids
    except ImportError as error:
        raise ImportError(
            "Triton assignment requires the triton package"
        ) from error
    return assign_int8_rows, prepare_int8_centroids


class QuantizedTorchEmeans(TorchEmeans):
    """E-means with quantizers or native FP8/INT8/UINT8 representations.

    CastQuantizer converts floating chunks to FP16/BF16. FP8 can encode
    floating batches per chunk; FP8 and INT8/UINT8 execution require CUDA.
    INT8 requires matching codes.
    Each object represents one run; create a fresh object for another run.
    Raw bytes retain numeric units and INT64 statistics; calibrated codecs
    return decoded units. Model coordinates and closing centroids are FP32.
    """

    def __init__(self, *args, quantizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantizer = quantizer
        self._fp8_quantizer = (
            quantizer is not None and quantizer.dtype == torch.float8_e4m3fn
        )
        self._int8_quantizer = (
            quantizer is not None and quantizer.dtype == torch.int8
        )
        self._input_dtype = quantizer.dtype if quantizer is not None else None
        self._fp8_peak = None
        self._batch_code_units = False
        self._batch_operand = self._stats_operand = None
        self._centered_cast = getattr(quantizer, "center", False)
        self._cast_fit_token = None
        if self.spherical and self._centered_cast:
            raise ValueError("spherical assignment does not support centered CastQuantizer")

    def _check_cast_fit(self):
        if self._centered_cast:
            token = self.quantizer._fit_token()
            if self._cast_fit_token is not None and token != self._cast_fit_token:
                raise RuntimeError("quantizer fit changed; create a new object before model use")

    def begin(self, global_n, init_centroids):
        if self._centered_cast:
            self.quantizer._matrix(init_centroids)
            if init_centroids.device.type == "cpu":
                raise ValueError("CPU E-means requires quantizer=None")
            self._cast_fit_token = self.quantizer._fit_token()
        return super().begin(global_n, init_centroids)

    def _require_running(self):
        super()._require_running()
        self._check_cast_fit()

    def current_centroids(self):
        self._check_cast_fit()
        return super().current_centroids()

    def _local_execution(self):
        return type(self) is QuantizedTorchEmeans

    def _finish_step(self, expected_n):
        super()._finish_step(expected_n)
        self._batch_operand = None
        self._batch_code_units = False

    def finalize(self):
        self._check_cast_fit()
        result = super().finalize()
        self._batch_operand = self._stats_operand = None
        return result

    def _start_epoch(self):
        if self._fp8_peak is not None:
            self._fp8_peak.zero_()
        super()._start_epoch()

    def _initial_centroid_values(self, init_centroids):
        rows = self._prepare_rows(init_centroids)
        # Fused CuTe JIT defers chunk encoding; seed positions still need
        # codes.
        if (
            self._assignment_backend == "cute"
            and self._fp8_quantizer
            and rows.dtype != self.quantizer.dtype
        ):
            rows = self.quantizer.compute_codes(rows)
        return (
            self.quantizer.decode(rows)
            if self.quantizer is not None
            else rows.float()
        )

    def _prepare_assignment_query(self, centroids):
        if self._input_dtype in (torch.int8, torch.uint8):
            _check_torch_abs_bound(
                "centroids", torch.finfo(torch.float32).max, centroids
            )
        self._fp8_peak = None
        represented, centroid_norm = self._prepare_torch_assignment(centroids)
        if self._input_dtype == getattr(torch, "float8_e4m3fn", None):
            from .quantizer_fp8 import _check_fp8_range

            _check_fp8_range(float(self._fp8_peak), "centroids")

        return represented, centroid_norm

    def _apply_splits(self, source, target):
        self._execution_backend.split(
            self._centroids,
            self._usage,
            source,
            target,
            self._input_dtype,
            _SPLIT_EPSILON,
            self._local_execution(),
            self.quantizer,
            separate=self._separate_split_values,
        )
        self.n_splits += len(source)
        self._invalidate_readout()
        return len(source)

    def _separate_split_values(self, donor_centroids, child_centroids):
        if self._input_dtype in (torch.int8, torch.uint8):
            from .quantizer_int8 import representation, separate_splits

            # Encoded split positions must stay finite in model coordinates.
            _check_torch_abs_bound(
                "revived centroids",
                torch.finfo(torch.float32).max,
                donor_centroids,
                child_centroids,
            )
            separate_splits(
                donor_centroids,
                child_centroids,
                representation(self._input_dtype),
                self.quantizer,
            )
        elif self._input_dtype == getattr(torch, "float8_e4m3fn", None):
            from .quantizer_fp8 import separate_splits

            separate_splits(donor_centroids, child_centroids, self.quantizer)
        else:
            _separate_splits(
                donor_centroids, child_centroids, self._input_dtype,
                self.quantizer if self._centered_cast else None,
            )

    def _assign_rows(
        self,
        x,
        centroids,
        centroid_norm=None,
        distance_out=None,
        labels_out=None,
    ):
        """Private assignment; native bytes may retain exact INT64 distances."""

        if self._assignment_backend == "cute":
            return self._execution_backend.assign(
                x,
                centroids,
                centroid_norm,
                self.spherical,
                distance_out,
                labels_out,
            )
        if self._assignment_backend == "triton":

            if self._input_dtype in (torch.int8, torch.uint8):
                from .quantizer_int8 import representation

                assign, _ = _get_triton_int8_functions()
                distance, labels = assign(
                    x,
                    centroids,
                    centroid_norm,
                    representation(self._input_dtype),
                    distance_out,
                )
            else:
                if x.shape[1] != centroids.shape[1]:
                    from .triton_assignment import _pad_columns

                    x = _pad_columns(
                        x, 64 if centroids.dtype == torch.float8_e4m3fn else 16
                    )
                distance, labels = _get_triton_assign_rows()(
                    x,
                    centroids,
                    centroid_norm,
                    **({"spherical": True} if self.spherical else {}),
                )
            if self._fp8_quantizer or self._int8_quantizer:
                from .quantizer import _ldexp

                distance = _ldexp(distance, self.quantizer._distance_exponent)
            return distance, labels

        return super()._assign_rows(
            x, centroids, centroid_norm, distance_out, labels_out
        )

    def _accumulate_core(self, x, index=None, raw=False, labels_out=None):
        # Tensor work shared by public accumulation and graph replay.
        # Raw replay may consume unshifted bounded INT32 byte totals directly.
        n = len(x) if index is None else len(index)
        pending_n = (
            0 if self._pending_stats is None else self._pending_stats["n"]
        )
        target = self._local_batch_size()
        wide = n > 0 and target > 2**24
        borrowed = False

        integer_stats = self._batch_stats.dtype == torch.int64
        shift = (
            integer_stats and x.dtype == torch.uint8 and 255 * target >= 2**31
        )
        borrowed = (
            raw
            and integer_stats
            and n > 0
            and n == target
            and not wide
            and not shift
        )
        if pending_n == 0:
            # Bound the whole contribution, not each supplied/assignment chunk.
            self._batch_code_units = self._use_code_units(n, target)
            if borrowed:
                # Complete private replay borrows storage, never public chunks.
                accumulation = self._batch_stats.view(torch.int32)[
                    :, : self.d + 1
                ]
                accumulation.zero_()
            else:
                accumulation = (
                    self._batch_stats.new_zeros(
                        self._batch_stats.shape,
                        dtype=(
                            torch.int32
                            if integer_stats and not wide
                            else torch.float64
                        ),
                    )
                    if (integer_stats and n) or wide
                    else self._batch_stats.zero_()
                )
            self._batch_objective = self._objective.new_zeros(())
            # Retain represented operands through the whole contribution.
            if self.spherical:
                self._readout(
                    force=raw
                )  # Record projection on every graph replay.
            self._batch_centroids, self._batch_norm = (
                self._prepare_torch_assignment(self._centroids)
                if n
                else (None, None)
            )
            self._batch_accumulation = accumulation
            self._batch_operand = None
        accumulation = self._batch_accumulation
        objective, centroid_norm = self._batch_objective, self._batch_norm
        centroids = self._batch_centroids
        operand = self._batch_operand
        q = self.quantizer
        code_units = self._batch_code_units
        native_stats = (
            self._local_execution()
            and self.quantizer is None
            and self._centroids.is_cuda
            and self._assignment_backend in ("triton", "cute")
            and n > 0
            and target <= 2**24
            and x.dtype in (torch.int8, torch.uint8, torch.float16)
            and not torch.are_deterministic_algorithms_enabled()
        )
        fp8_stats = (
            code_units
            and self._local_execution()
            and not torch.are_deterministic_algorithms_enabled()
        )
        if native_stats or fp8_stats:
            from .triton_assignment import accumulate_bytes
        sums, counts = accumulation[:, :-1], accumulation[:, -1]
        captured_labels = None if labels_out is None else labels_out[:n]
        if (
            captured_labels is None
            and self.capture_assignments
            and self._epoch == len(self.recipe["epoch_rows"]) - 1
        ):
            captured_labels = x.new_empty(n, dtype=torch.int64)
        chunk_size = self.recipe["assignment_chunk_size"]
        for begin in range(0, n, chunk_size):
            end = min(begin + chunk_size, n)
            rows = x[begin:end] if index is None else x[index[begin:end]]
            rows = self._prepare_rows(rows)
            distance, labels = self._assign_rows(
                rows, centroids, centroid_norm
            )
            if captured_labels is not None:
                captured_labels[begin:end] = labels
            if self.spherical:
                objective += 2 * len(rows)
            objective += distance.sum(dtype=torch.float64)
            if (
                native_stats
                and rows.dtype == x.dtype
                and accumulate_bytes(rows, labels, accumulation, shift=shift)
            ):
                continue
            # CuTe FP8 assignment retains these encoded rows for statistics;
            # consume them on the ordered stream before another assignment.
            codes = (
                centroids.bindings[len(rows)].workspace
                if self._assignment_backend == "cute" and self._fp8_quantizer
                else rows
            )
            if fp8_stats and accumulate_bytes(
                codes, labels, accumulation, fp8_code_units=True
            ):
                continue
            if operand is None:
                if self._stats_operand is None:
                    with torch.inference_mode(
                        self._batch_stats.is_inference()
                    ):
                        self._stats_operand = self._centroids.new_ones(
                            (
                                min(chunk_size, self.recipe["batch_size"]),
                                self.d + 1,
                            ),
                            dtype=(
                                torch.int32 if integer_stats else torch.float32
                            ),
                        )
                operand = self._stats_operand.to(accumulation.dtype)
            values = operand[: len(rows)]
            direct_bf16 = (
                q is None
                and index is not None
                and self._local_execution()
                and type(x) is torch.Tensor
                and type(values) is torch.Tensor
                and rows.dtype is torch.bfloat16
                and values.dtype is torch.float32
            )
            represented = (
                codes
                if code_units
                else (
                    self.quantizer.decode(codes)
                    if self.quantizer is not None
                    else rows if integer_stats or direct_bf16 else rows.float()
                )
            )
            values[:, :-1].copy_(
                represented
                if integer_stats or code_units or direct_bf16
                else represented.float()
            )
            if accumulation.dtype == torch.int32 and shift:
                values[:, :-1].sub_(128)
            accumulation.index_add_(0, labels, values)
        self._batch_objective, self._batch_operand = objective, operand
        total_n = pending_n + n
        if total_n == target and code_units:
            sums.mul_(2.0**-q.exponent)
        if (
            total_n == target
            and integer_stats
            and (not wide)
            and (not borrowed)
        ):
            from .triton_assignment import copy_byte_stats

            copy_byte_stats(accumulation, self._batch_stats, unsigned=shift)
            sums, counts = self._batch_stats[:, :-1], self._batch_stats[:, -1]
        elif (
            total_n == target
            and not borrowed
            and accumulation is not self._batch_stats
        ):
            self._batch_stats.copy_(accumulation)
            sums, counts = self._batch_stats[:, :-1], self._batch_stats[:, -1]
        complete = total_n == target or total_n == 0
        stats = {
            "sums": sums if complete else None,
            "counts": counts if complete else None,
            "objective": objective,
            "n": total_n,
        }
        if captured_labels is not None:
            stats["labels"] = captured_labels
        return stats

    def _epoch_totals(self):

        fp8 = self._input_dtype == getattr(torch, "float8_e4m3fn", None)
        if fp8:
            from .quantizer_fp8 import (
                _transform,
                _update_fp8_peak,
                _check_fp8_range,
            )

            if self._fp8_peak is None:
                self._fp8_peak = self._centroids.new_zeros(())
            _update_fp8_peak(
                _transform(self._centroids, self.quantizer), self._fp8_peak
            )
        objective = self._epoch_objective()
        report_counts = self._pass_counts.double()
        totals = torch.stack(
            (objective, torch.dot(report_counts, report_counts))
            + ((self._fp8_peak.double(),) if fp8 else ())
        ).tolist()
        objective, count_square = totals[:2]
        if fp8:
            _check_fp8_range(totals[-1], "centroids")
        return objective, count_square

    def _replay_representation_eligible(self):
        if self.quantizer is not None:
            from .quantizer import CastQuantizer
            from .quantizer_fp8 import Fp8Quantizer
            from .quantizer_int8 import Int8Quantizer

            if type(self.quantizer) not in (
                CastQuantizer,
                Fp8Quantizer,
                Int8Quantizer,
            ):
                return False
            if any(
                name in vars(self.quantizer)
                for name in ("decode", "compute_codes")
            ):
                return False
        return True

    def _use_code_units(self, n, target):
        if not self._fp8_quantizer:
            return False
        from .quantizer_fp8 import Fp8Quantizer

        q = self.quantizer
        return (
            n > 0
            and target <= 2**24
            and type(q) is Fp8Quantizer
            and q.dtype is torch.float8_e4m3fn
            and q.shift is None
            and -94 <= q.exponent <= 117
            and not any(
                name in vars(q) for name in ("decode", "compute_codes")
            )
            and not torch.are_deterministic_algorithms_enabled()
        )

    def _prepare_input(self, x, allow_empty=False, prepare=True):
        self._check_cast_fit()
        if self._centered_cast:
            self.quantizer._matrix(x)
        if prepare:
            x = x.detach()
            if x.device.type == "cpu":
                x = x.float()
            x = x.contiguous()
        if self._input_dtype is None:
            dtype = (
                self.quantizer.dtype
                if self.quantizer is not None
                else (torch.float32 if x.device.type == "cpu" else x.dtype)
            )
            if self._batch_stats is not None:
                self._select_statistics_dtype(
                    torch.int64
                    if self.quantizer is None
                    and dtype in (torch.int8, torch.uint8)
                    else torch.float32
                )
            self._input_dtype = dtype
        self._resolve_assignment_backend(x)
        if self._input_dtype in (torch.int8, torch.uint8) and self.d > 43919:
            raise ValueError("INT8 assignment requires d <= 43919")
        if self.recipe is not None:
            self.recipe.update(
                representation=str(self._input_dtype).removeprefix("torch."),
                assignment_backend=self._assignment_backend,
                assignment_backend_requested=self.assignment_backend,
                assignment_chunk_size=self._get_assignment_chunk_size(),
            )
        return x

    def _prepare_rows(self, rows, *, new_run=False):
        if self._centered_cast:
            if not new_run:
                self._check_cast_fit()
            self.quantizer._matrix(rows)
        if self._int8_quantizer:
            return rows
        if (
            self._assignment_backend == "cute"
            and self._fp8_quantizer
            and rows.dtype != self.quantizer.dtype
        ):
            return rows.float()
        if self.quantizer is None or rows.dtype == self.quantizer.dtype:
            return rows
        return self.quantizer.compute_codes(rows)

    def _prepare_torch_assignment(self, centroids):
        self._check_cast_fit()
        if self.spherical and centroids is self._centroids:
            centroids = self._readout()
        if self._assignment_backend == "cute":
            dtype = self._input_dtype or torch.bfloat16
            if dtype is torch.float8_e4m3fn and self._fp8_peak is None:
                self._fp8_peak = centroids.new_zeros(())
            return self._execution_backend.prepare_centroids(
                centroids,
                transform=(
                    self.quantizer
                    if self._fp8_quantizer or self._int8_quantizer or self._centered_cast
                    else None
                ),
                peak_out=self._fp8_peak,
                dtype=dtype,
                spherical=self.spherical,
            )
        dtype = self._input_dtype or torch.float32
        if dtype in (torch.int8, torch.uint8):
            from .quantizer import _transform
            from .quantizer_int8 import representation

            _, prepare = _get_triton_int8_functions()
            value = _transform(
                centroids, self.quantizer if self._int8_quantizer else None
            )
            return prepare(value, representation(dtype), validate=False)
        if dtype == getattr(torch, "float8_e4m3fn", None):
            from .quantizer_fp8 import (
                Fp8Quantizer,
                _transform,
                _update_fp8_peak,
            )

            if self._fp8_peak is None:
                self._fp8_peak = centroids.new_zeros(())
            if self._assignment_backend == "triton" and (
                self.quantizer is None or type(self.quantizer) is Fp8Quantizer
            ):
                from .triton_assignment import _cast_fp8_centroids

                prepared = _cast_fp8_centroids(
                    centroids, self.quantizer, self._fp8_peak
                )
                if prepared is not None:
                    return prepared
            transformed = _transform(centroids, self.quantizer)
            _update_fp8_peak(transformed, self._fp8_peak)
            represented = transformed.to(dtype)
            if self.d % 64:
                from .triton_assignment import _pad_columns

                represented = _pad_columns(represented, 64)
        else:
            represented = (
                self.quantizer.compute_codes(centroids)
                if self._centered_cast else centroids.to(dtype)
            )
        norm = (
            centroids.new_zeros(len(centroids))
            if self.spherical
            else represented[:, : self.d].float().square().sum(dim=1)
        )
        if self._assignment_backend == "triton" and dtype is torch.float32:
            represented = represented.t().contiguous().t()
        if (
            self._assignment_backend == "triton"
            and dtype in (torch.bfloat16, torch.float16)
            and self.d % 16
        ):
            from .triton_assignment import _pad_columns

            represented = _pad_columns(represented, 16)
        return represented, norm

    def _select_statistics_dtype(self, dtype):
        # Input or external statistics select the run schema.

        if dtype == self._batch_stats.dtype:
            return
        if 255 * self._n > 2**53:
            raise ValueError(
                "integer statistics require global 255*n <= 2**53"
            )
        try:
            with torch.inference_mode(self._batch_stats.is_inference()):
                self._batch_stats = self._batch_stats.new_empty(
                    self._batch_stats.shape, dtype=dtype
                )
                self._pass_counts = self._pass_counts.to(dtype)
                if self._pass_sums is not None:
                    self._pass_sums = self._pass_sums.to(dtype)
            self._stats_operand = None
        except BaseException:
            self._failed = True
            raise
