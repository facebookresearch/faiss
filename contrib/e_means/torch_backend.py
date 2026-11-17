# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Torch E-means execution backends with lazy optional kernel imports.

Backends own represented centroid layouts, assignment workspace, fused kernel
selection and capture resources. E-means owns the training state and schedule.
"""

import torch


def _check_torch_abs_bound(name, bound, *tensors):
    extrema = torch.stack(
        [value for x in tensors for value in torch.aminmax(x)]
    )
    if not bool((extrema.abs() <= bound).all()):
        raise ValueError(f"{name} must be finite with abs(value) <= {bound:g}")


def _separate_splits(donor, child, dtype=None, quantizer=None):
    """Move one coordinate only when represented split operands coincide.

    Zero donors become distinct subnormal operands; their squared distances
    can still tie. No host value read or dynamic collapsed-row list is needed.
    Centered casts try original-FP32 neighbours and re-encode before repair.
    If coordinate zero has no finite distinct representable code, it stays
    unchanged; e.g. FP16 at an FP32 origin of 2**40. No range clamping is used.
    """
    donor_q = (quantizer.compute_codes(donor) if quantizer is not None
               else donor if dtype is None else donor.to(dtype))
    child_q = (quantizer.compute_codes(child) if quantizer is not None
               else child if dtype is None else child.to(dtype))
    collapsed = (donor_q == child_q).all(dim=1)
    value = child_q[:, 0]
    neighbor = torch.nextafter(value, torch.full_like(value, float("inf")))
    inward = torch.nextafter(value, torch.full_like(value, float("-inf")))
    neighbor = torch.where(neighbor.abs() < float("inf"), neighbor, inward)
    neighbor = neighbor.float()
    if quantizer is not None:
        # A low-format nextafter can disappear during FP32 add-back.
        # Try at least one original-FP32 ULP in each direction, keeping
        # the decoded code neighbour when it requires a larger move.
        original = child[:, 0]
        shift = quantizer.shift[0]
        positive = torch.maximum(neighbor + shift, torch.nextafter(
            original, torch.full_like(original, float("inf"))))
        negative = torch.minimum(inward.float() + shift, torch.nextafter(
            original, torch.full_like(original, float("-inf"))))
        positive_q = (positive - shift).to(quantizer.dtype)
        negative_q = (negative - shift).to(quantizer.dtype)
        valid_positive = (torch.isfinite(positive) & torch.isfinite(positive_q)
                          & torch.isfinite(positive_q.float() + shift)
                          & (positive_q != donor_q[:, 0]))
        valid_negative = (torch.isfinite(negative) & torch.isfinite(negative_q)
                          & torch.isfinite(negative_q.float() + shift)
                          & (negative_q != donor_q[:, 0]))
        neighbor = torch.where(valid_positive, positive, negative)
        collapsed = collapsed & (valid_positive | valid_negative)
    child[:, 0] = torch.where(collapsed, neighbor, child[:, 0])


def _get_triton_assign_rows():
    try:
        from .triton_assignment import assign_rows
    except ImportError as error:
        raise ImportError(
            "Triton assignment requires the triton package"
        ) from error
    return assign_rows


def _get_cute_functions():
    try:
        from . import cute
    except ImportError as error:
        raise ImportError(
            "CuTe assignment requires its pinned optional dependencies"
        ) from error
    return cute._assign_rows, cute._prepare_centroids, cute._check_device


class TorchBackend:
    """Native Torch assignment, accumulation, EMA updates and splitting."""

    writes_outputs = False
    name = "native"

    def __init__(self, centroid_chunk_size):
        self.centroid_chunk_size = centroid_chunk_size
        self.clear_statistics()

    def clear_statistics(self):
        self.statistics_operand = self.batch_operand = None
        self.statistics_template = None

    def require_assignment(self, device):
        pass

    def require_training(self, device):
        self.require_assignment(device)

    def begin_batch(self, template, workspace_rows):
        self.statistics_template = template
        self.workspace_rows = workspace_rows
        self.batch_operand = None

    def chunk_size(self, centroids):
        return default_chunk_size(self.name, centroids, self.centroid_chunk_size)

    def query_centroids(self, centroids):
        return centroids.detach()

    def fork_query(self):
        import copy

        return copy.copy(self)

    def finish_query(self, backend):
        pass

    def prepare_centroids(self, centroids, dtype, spherical=False):
        represented = centroids.to(dtype or torch.float32)
        norm = (
            centroids.new_zeros(len(centroids))
            if spherical
            else represented.float().square().sum(dim=1)
        )
        return represented, norm

    def assign(
        self,
        x,
        centroids,
        centroid_norm=None,
        spherical=False,
        distance_out=None,
        labels_out=None,
    ):
        x_norm = None if spherical else x.float().square().sum(dim=1)
        best_score = torch.full(
            (len(x),), float("inf"), dtype=torch.float32, device=x.device
        )
        labels = torch.zeros(len(x), dtype=torch.int64, device=x.device)

        for begin in range(0, len(centroids), self.centroid_chunk_size):
            end = min(begin + self.centroid_chunk_size, len(centroids))
            block = centroids[begin:end]
            block_norm = (
                (block * block).sum(dim=1)
                if centroid_norm is None
                else centroid_norm[begin:end]
            )
            score = torch.addmm(
                block_norm[None, :],
                x,
                block.t(),
                alpha=-2.0,
                **(
                    {"out_dtype": torch.float32}
                    if x.dtype != torch.float32
                    else {}
                ),
            )
            block_score, block_label = score.min(dim=1)
            better = block_score < best_score
            torch.where(better, block_score, best_score, out=best_score)
            torch.where(better, block_label + begin, labels, out=labels)

        return (
            best_score if spherical else (best_score + x_norm).clamp_min(0)
        ), labels

    def accumulate(
        self, rows, labels, statistics, batch_size, indexed=False
    ):
        """Add sums and counts in place using the current batch scratch."""
        if self.batch_operand is None:
            if self.statistics_operand is None:
                with torch.inference_mode(
                    self.statistics_template.is_inference()
                ):
                    self.statistics_operand = (
                        self.statistics_template.new_ones(
                            (self.workspace_rows, statistics.shape[1]),
                            dtype=torch.float32,
                        )
                    )
            self.batch_operand = self.statistics_operand.to(statistics.dtype)
        values = self.batch_operand[: len(rows)]
        direct_bf16 = (
            indexed
            and type(values) is torch.Tensor
            and rows.dtype == torch.bfloat16
            and values.dtype == torch.float32
        )
        values[:, :-1].copy_(rows if direct_bf16 else rows.float())
        statistics.index_add_(0, labels, values)

    def update(
        self,
        all_centroids,
        all_usage,
        ids,
        sums,
        counts,
        alpha,
        decay,
        replace_usage,
    ):
        # EMA arithmetic uses FP32; the supplied statistics keep their schema.
        sums, counts = sums.float(), counts.float()
        touched = counts > 0
        # Fixed-shape masks avoid dynamic row selection/host reads. Inactive
        # lanes divide zero by one; only touched lanes divide by statistics.
        safe_counts = torch.where(touched, counts, 1)
        mean = sums.clone()
        mean.masked_fill_(~touched[:, None], 0)
        mean /= safe_counts[:, None]
        if not replace_usage:
            all_usage *= decay
        usage = all_usage if ids is None else all_usage[ids]
        centroids = all_centroids if ids is None else all_centroids[ids]
        retained = usage if not replace_usage else torch.zeros_like(usage)
        mass = retained + alpha * counts
        weight = alpha * counts / torch.where(touched, mass, 1)
        position = mean - centroids
        position.mul_(weight[:, None]).add_(centroids)
        torch.where((weight == 1)[:, None], mean, position, out=position)
        torch.where(touched[:, None], position, centroids, out=centroids)
        torch.where(touched, mass, usage, out=usage)
        if ids is not None:
            all_centroids.index_copy_(0, ids, centroids)
            all_usage.index_copy_(0, ids, usage)

    def split(
        self,
        centroids,
        usage,
        source,
        target,
        dtype,
        epsilon,
        quantizer=None,
        separate=None,
        built_in_separation=True,
    ):
        """Split positions and masses.

        ``separate`` preserves distinctions in a quantized representation.
        """
        donors = centroids[source]
        sign = torch.ones(
            donors.shape[1], dtype=donors.dtype, device=donors.device
        )
        sign[1::2] = -1.0
        offset = epsilon * sign
        donor_centroids = donors * (1.0 - offset)
        child_centroids = donors * (1.0 + offset)
        if dtype == torch.float16:
            guarded = (
                (quantizer._center_rows(donor_centroids), quantizer._center_rows(child_centroids))
                if getattr(quantizer, "center", False)
                else (donor_centroids, child_centroids)
            )
            _check_torch_abs_bound(
                "revived centroids", 65504.0, *guarded
            )
        if separate is None:
            _separate_splits(donor_centroids, child_centroids, dtype)
        else:
            separate(donor_centroids, child_centroids)
        half_usage = usage[source] / 2.0
        usage[source] = half_usage
        centroids[source] = donor_centroids
        usage[target] = half_usage
        centroids[target] = child_centroids

    def capture_state(self):
        return None

    def initialize_capture(self):
        torch.cuda.current_blas_handle()

    def finish_capture(self, state, captured):
        pass

    def capture_bindings(self, state):
        return ()

    def validate_capture(self, state):
        pass


class TritonBackend(TorchBackend):
    """Triton layouts and assignment, with optional fused tensor operations."""

    name = "triton"

    def require_assignment(self, device):
        _get_triton_assign_rows()

    def chunk_size(self, centroids):
        return default_chunk_size(self.name, centroids, self.centroid_chunk_size)

    def prepare_centroids(self, centroids, dtype, spherical=False):
        represented, norm = super().prepare_centroids(
            centroids, dtype, spherical
        )
        if represented.dtype == torch.float32:
            represented = represented.t().contiguous().t()
        if centroids.shape[1] % 16 and represented.dtype in (
            torch.bfloat16,
            torch.float16,
        ):
            from .triton_assignment import _pad_columns

            represented = _pad_columns(represented, 16)
        return represented, norm

    def assign(
        self,
        x,
        centroids,
        centroid_norm=None,
        spherical=False,
        distance_out=None,
        labels_out=None,
    ):
        if x.shape[1] != centroids.shape[1]:
            from .triton_assignment import _pad_columns

            x = _pad_columns(x, 16)
        return _get_triton_assign_rows()(
            x,
            centroids,
            centroid_norm,
            **({"spherical": True} if spherical else {}),
        )

    def accumulate(
        self, rows, labels, statistics, batch_size, indexed=False
    ):
        if (
            rows.is_cuda
            and batch_size <= 2**24
            and rows.dtype == torch.float16
            and not torch.are_deterministic_algorithms_enabled()
        ):
            from .triton_assignment import accumulate_bytes

            if accumulate_bytes(rows, labels, statistics):
                return
        super().accumulate(
            rows, labels, statistics, batch_size, indexed
        )

    def update(
        self, centroids, usage, ids, sums, counts, alpha, decay, replace_usage
    ):
        if ids is None and centroids.is_cuda:
            from .triton_assignment import ema_update

            if ema_update(
                sums, counts, centroids, usage, alpha, decay, replace_usage
            ):
                return
        super().update(
            centroids, usage, ids, sums, counts, alpha, decay, replace_usage
        )

    def split(
        self,
        centroids,
        usage,
        source,
        target,
        dtype,
        epsilon,
        quantizer=None,
        separate=None,
        built_in_separation=True,
    ):
        if (
            built_in_separation
            and centroids.is_cuda
            and centroids.shape[1] <= 1024
            and dtype
            in (None, torch.float32, torch.bfloat16, torch.float8_e4m3fn)
        ):
            from .triton_assignment import _write_float_splits

            if _write_float_splits(
                centroids, usage, source, target, dtype, epsilon, quantizer
            ):
                return
        super().split(
            centroids,
            usage,
            source,
            target,
            dtype,
            epsilon,
            quantizer,
            separate,
            built_in_separation,
        )

    def initialize_capture(self):
        pass


class CuteBackend(TritonBackend):
    """CuTe centroid and row workspace, including resources held by graphs.

    Accumulation, EMA updates and split writes use inherited Triton kernels
    with Torch fallbacks.
    """

    writes_outputs = True
    name = "cute"

    def require_assignment(self, device):
        _get_cute_functions()[2](device)

    def require_training(self, device):
        self.require_assignment(device)
        _get_triton_assign_rows()

    def __init__(self, centroid_chunk_size):
        super().__init__(centroid_chunk_size)
        self.cache = None

    def query_centroids(self, centroids):
        return centroids

    def finish_query(self, backend):
        if isinstance(backend, CuteBackend):
            self.cache = backend.cache

    def prepare_centroids(
        self, centroids, dtype, spherical=False, transform=None, peak_out=None
    ):
        _, prepare, _ = _get_cute_functions()
        self.cache = prepare(
            centroids,
            self.cache,
            dtype=dtype or torch.bfloat16,
            spherical=spherical,
            transform=transform,
            peak_out=peak_out,
        )
        return self.cache, None

    def assign(
        self,
        x,
        centroids,
        centroid_norm=None,
        spherical=False,
        distance_out=None,
        labels_out=None,
    ):
        assign, _, _ = _get_cute_functions()
        return assign(x, centroids, distance_out, labels_out)

    def capture_state(self):
        state = self.cache
        if state is not None:
            # Capture must record a refresh even after an eager warmup.
            state.source_version = None
        return state

    def finish_capture(self, state, captured):
        if state is not None:
            state.source_version = None
        if not captured:
            self.cache = None

    def capture_bindings(self, state):
        return tuple(state.bindings.values()) if state is not None else ()

    def validate_capture(self, state):
        if self.cache is not state:
            raise RuntimeError("centroid preparation changed during replay")


def select_backend(x, dtype, k, d, requested="auto"):
    """Select from device, dtype and extent, then require the selected software.

    Automatic selection still depends on dtype and extent as well as hardware.
    Missing or broken required software raises an error for that selection.
    """
    if requested == "native":
        return TorchBackend
    if requested == "triton":
        return TritonBackend
    if requested == "cute":
        return CuteBackend
    if requested != "auto":
        raise ValueError("assignment_backend must be auto, native, triton or cute")
    if x.device.type == "cpu":
        return TorchBackend
    capability = torch.cuda.get_device_capability(x.device) if torch.version.hip is None else (0, 0)
    cuda = capability >= (8, 0)
    if dtype in (torch.int8, torch.uint8):
        from .quantizer_int8 import _check_int8_support

        _check_int8_support(x.device)
        if d > 43919:
            raise ValueError("INT8 assignment requires d <= 43919")
    elif dtype == torch.float8_e4m3fn:
        from .quantizer_fp8 import _check_fp8_support

        _check_fp8_support(x.device)
    # SM90 BF16 selects CuTe within these padded extents. The checks below
    # select Triton or native Torch for other inputs, or reject them.
    padded_k, padded_d = ((k + 191) // 192) * 192, ((d + 7) // 8) * 8
    if (capability == (9, 0) and dtype is torch.bfloat16
            and k <= 2**31 - 192 and 0 < d < 2**31
            and padded_k * padded_d < 2**31):
        return CuteBackend
    if cuda and k <= 2**31 - 256:
        return TritonBackend
    floating = dtype in (torch.float32, torch.float16, torch.bfloat16)
    native = dtype is torch.float32 or (
        cuda and "out_dtype" in (torch.addmm.__doc__ or "")
    )
    if floating and native:
        return TorchBackend
    if floating:
        raise RuntimeError(
            "auto assignment has no compatible backend for this device and "
            "representation: FP16/BF16 assignment requires an NVIDIA SM80 or "
            "later GPU"
        )
    raise RuntimeError(
        "auto assignment has no supported operation for this device and representation; "
        "use QuantizedEmeans(quantizer=None) for native FP32")


def default_chunk_size(backend, centroids, centroid_chunk_size=4096):
    """Built-in chunk geometry, also used for equivalent-recipe replay."""
    if centroids.device.type == "cpu":
        return 8192
    k, d = centroids.shape
    width = max(d, min(k, centroid_chunk_size)) if backend == "native" else d
    return min(2**20, max(8192, 2**26 // width))
