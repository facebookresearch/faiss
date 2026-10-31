# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Triton nearest-centroid assignment and dense CUDA EMA for E-means.

Dispatch uses fixed shape and dtype rules without runtime autotuning. Kernels
stream centroid tiles with bounded score workspaces instead of materializing
the native backend's row-by-centroid score tensor.
"""

import torch

try:
    import triton
    import triton.language as tl
except ImportError as error:
    raise ImportError(
        "Triton assignment requires the triton package"
    ) from error


try:
    from triton.tools.tensor_descriptor import TensorDescriptor
except ImportError:
    TensorDescriptor = None

try:
    from triton.language.extra.cuda import libdevice as _fp8_libdevice
except ImportError:
    _fp8_libdevice = None


_INT32_MAX = 2**31 - 1
_SPLIT_K_BLOCK_B = 64
_SPLIT_K_BLOCK_K = 128
_SPLIT_K_BLOCK_D = 128
_SPLIT_K_MAX_PARTITIONS = 128
_SPLIT_K_NUM_WARPS = 8
_SPLIT_K_NUM_STAGES = 3
# Small empirical occupancy target; not a hardware legality constraint.
_SPLIT_K_TARGET_PROGRAMS = 512
_SPLIT_K_MIN_TILES_PER_PARTITION = 4
_DIRECT_BLOCK_B = 64
_DIRECT_BLOCK_K = 256
_DIRECT_REDUCED_BLOCK_D = 64
_DIRECT_FP32_BLOCK_D = 32
_FLOAT8_E4M3FN = getattr(torch, "float8_e4m3fn", None)
_FULL_D_MAX_D = 128
_FULL_D_CONFIG = (64, 128, 4, 3)
_MAX_K = 2**31 - max(
    _SPLIT_K_MAX_PARTITIONS,
    _SPLIT_K_BLOCK_K,
    _DIRECT_BLOCK_K,
)


def _centroid_offset_is_wide(shape, stride):
    """Return whether any physical centroid lane needs int64."""
    physical_k = max(
        triton.cdiv(shape[0], _DIRECT_BLOCK_K) * _DIRECT_BLOCK_K,
        shape[0] + _SPLIT_K_BLOCK_K - 1,
    )
    physical_d = (
        triton.cdiv(shape[1], _SPLIT_K_BLOCK_D) * _SPLIT_K_BLOCK_D
    )
    maximum = (physical_k - 1) * stride[0] + (physical_d - 1) * stride[1]
    return maximum > _INT32_MAX


@triton.jit
def _selected_fp8_distance(
    x_ptr, centroid_ptr, row, label, D,
    stride_xb, stride_xd, stride_ck, stride_cd,
    row_mask, BLOCK_D: tl.constexpr,
):
    distance = tl.full(row.shape, 0.0, tl.float32)
    for d0 in range(0, D, BLOCK_D):
        column = d0 + tl.arange(0, BLOCK_D)
        mask = row_mask[:, None] & (column[None, :] < D)
        x = tl.load(
            x_ptr + row[:, None] * stride_xb + column[None, :] * stride_xd,
            mask=mask, other=0.0,
        ).to(tl.float32)
        c = tl.load(
            centroid_ptr + label[:, None].to(tl.int64) * stride_ck
            + column[None, :].to(tl.int64) * stride_cd,
            mask=mask, other=0.0,
        ).to(tl.float32)
        delta = x - c
        distance += tl.sum(delta * delta, axis=1)
    return distance


@triton.jit
def _assign_kernel(
    x_ptr,
    centroid_ptr,
    centroid_norm_ptr,
    distance_ptr,
    label_ptr,
    B,
    K,
    D,
    stride_xb,
    stride_xd,
    stride_ck,
    stride_cd,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    FULL_D: tl.constexpr,
    REDUCED: tl.constexpr,
    FP16: tl.constexpr,
    FP8: tl.constexpr,
    WIDE_CENTROID: tl.constexpr,
    x_desc=None,
    centroid_desc=None,
    SPHERICAL: tl.constexpr = False,
):
    row_program = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, BLOCK_B).to(tl.int64)
    row = row_program * BLOCK_B + lane
    row_mask = row < B
    row_norm = tl.zeros((BLOCK_B,), tl.float32)

    if FULL_D:
        column = tl.arange(0, BLOCK_D)
        full_x = tl.load(
            x_ptr
            + row[:, None] * stride_xb
            + column[None, :] * stride_xd,
            mask=row_mask[:, None] & (column[None, :] < D),
            other=0.0,
        )
        if REDUCED and not FP8:
            full_x = (
                full_x.to(tl.float16) if FP16 else full_x.to(tl.bfloat16)
            )
        if not FP8 and not SPHERICAL:
            x_float = full_x.to(tl.float32)
            row_norm = tl.sum(x_float * x_float, axis=1)
    elif not FP8 and not SPHERICAL:
        for d0 in range(0, D, BLOCK_D):
            column = d0 + tl.arange(0, BLOCK_D)
            x = tl.load(
                x_ptr
                + row[:, None] * stride_xb
                + column[None, :] * stride_xd,
                mask=row_mask[:, None] & (column[None, :] < D),
                other=0.0,
            )
            if REDUCED and not FP8:
                x = x.to(tl.float16) if FP16 else x.to(tl.bfloat16)
            x = x.to(tl.float32)
            row_norm += tl.sum(x * x, axis=1)

    best_score = tl.full((BLOCK_B,), float("inf"), tl.float32)
    best_label = tl.zeros((BLOCK_B,), tl.int32)
    # Only paired-descriptor FP8 uses two halves; pointer tiles stay unchanged.
    halves: tl.constexpr = FP8 and not FULL_D and centroid_desc is not None
    score_k: tl.constexpr = BLOCK_K // 2 if halves else BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        centroid = k0 + tl.arange(0, score_k)
        centroid_mask = centroid < K
        if FULL_D:
            column = tl.arange(0, BLOCK_D)
            column_mask = column < D
            if WIDE_CENTROID:
                centroid_offset = (
                    centroid[:, None].to(tl.int64) * stride_ck
                    + column[None, :].to(tl.int64) * stride_cd
                )
            else:
                centroid_offset = (
                    centroid[:, None] * stride_ck
                    + column[None, :] * stride_cd
                )
            c = tl.load(
                centroid_ptr + centroid_offset,
                mask=centroid_mask[:, None] & column_mask[None, :],
                other=0.0,
            )
            if FP8:
                score = tl.dot(
                    full_x,
                    tl.trans(c),
                    out_dtype=tl.float32,
                    max_num_imprecise_acc=max(32, BLOCK_D),
                )
            else:
                score = tl.dot(
                    full_x,
                    tl.trans(c),
                    out_dtype=tl.float32,
                )
        else:
            score = tl.zeros((BLOCK_B, score_k), tl.float32)
            if halves:
                upper_score = tl.zeros((BLOCK_B, score_k), tl.float32)
            for d0 in range(0, D, BLOCK_D):
                column = d0 + tl.arange(0, BLOCK_D)
                column_mask = column < D
                if x_desc is not None:
                    x = x_desc.load([(row_program * BLOCK_B).to(tl.int32), d0])
                    c = centroid_desc.load([k0, d0])
                else:
                    x = tl.load(
                        x_ptr
                        + row[:, None] * stride_xb
                        + column[None, :] * stride_xd,
                        mask=row_mask[:, None] & column_mask[None, :],
                        other=0.0,
                    )
                    if WIDE_CENTROID:
                        centroid_offset = (
                            centroid[:, None].to(tl.int64) * stride_ck
                            + column[None, :].to(tl.int64) * stride_cd
                        )
                    else:
                        centroid_offset = (
                            centroid[:, None] * stride_ck
                            + column[None, :] * stride_cd
                        )
                    c = tl.load(
                        centroid_ptr + centroid_offset,
                        mask=centroid_mask[:, None] & column_mask[None, :],
                        other=0.0,
                    )
                if REDUCED and not FP8:
                    x = x.to(tl.float16) if FP16 else x.to(tl.bfloat16)
                if FP8:
                    score = tl.dot(
                        x,
                        tl.trans(c),
                        acc=score,
                        out_dtype=tl.float32,
                        max_num_imprecise_acc=max(32, BLOCK_D),
                    )
                    if halves:
                        u = centroid_desc.load([k0 + BLOCK_K // 2, d0])
                        upper_score = tl.dot(
                            x, tl.trans(u), acc=upper_score, out_dtype=tl.float32,
                            max_num_imprecise_acc=max(32, BLOCK_D),
                        )
                elif REDUCED:
                    score = tl.dot(
                        x,
                        tl.trans(c),
                        acc=score,
                        out_dtype=tl.float32,
                    )
                else:
                    score = tl.dot(
                        x,
                        tl.trans(c),
                        acc=score,
                        input_precision="ieee",
                        out_dtype=tl.float32,
                    )

        for half in tl.static_range(0, 2 if halves else 1):
            if halves:
                if half == 1:
                    score = upper_score
                centroid = k0 + half * score_k + tl.arange(0, score_k)
                centroid_mask = centroid < K
            norm = tl.load(
                centroid_norm_ptr + centroid,
                mask=centroid_mask,
                other=float("inf"),
            )
            score = norm[None, :] - 2.0 * score
            block_score, block_label = tl.min(
                score,
                axis=1,
                return_indices=True,
                return_indices_tie_break_left=True,
            )
            block_label = block_label.to(tl.int32) + k0
            if halves:
                block_label += half * score_k
            better = block_score < best_score
            best_score = tl.where(better, block_score, best_score)
            best_label = tl.where(better, block_label, best_label)

    if SPHERICAL:
        distance = best_score
    elif FP8:
        distance = _selected_fp8_distance(
            x_ptr, centroid_ptr, row, best_label, D,
            stride_xb, stride_xd, stride_ck, stride_cd, row_mask, BLOCK_D,
        )
    else:
        distance = tl.maximum(best_score + row_norm, 0.0)
    tl.store(distance_ptr + row, distance, mask=row_mask)
    tl.store(label_ptr + row, best_label.to(tl.int64), mask=row_mask)


@triton.jit
def _assign_split_k_kernel(
    x_ptr,
    centroid_ptr,
    centroid_norm_ptr,
    partial_score_ptr,
    partial_label_ptr,
    B,
    K,
    D,
    stride_xb,
    stride_xd,
    stride_ck,
    stride_cd,
    PARTITIONS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    FP16: tl.constexpr,
    FP8: tl.constexpr,
    WIDE_CENTROID: tl.constexpr,
):
    row_program = tl.program_id(0).to(tl.int64)
    partition = tl.program_id(1)
    lane = tl.arange(0, BLOCK_B).to(tl.int64)
    row = row_program * BLOCK_B + lane
    row_mask = row < B
    partition_size = (K - 1) // PARTITIONS + 1
    centroid_start = partition * partition_size
    centroid_end = tl.minimum(centroid_start + partition_size, K)
    best_score = tl.full((BLOCK_B,), float("inf"), tl.float32)
    best_label = tl.zeros((BLOCK_B,), tl.int32)

    k0 = centroid_start
    while k0 < centroid_end:
        centroid = k0 + tl.arange(0, BLOCK_K)
        centroid_mask = centroid < centroid_end
        score = tl.zeros((BLOCK_B, BLOCK_K), tl.float32)

        for d0 in range(0, D, BLOCK_D):
            column = d0 + tl.arange(0, BLOCK_D)
            column_mask = column < D
            x = tl.load(
                x_ptr
                + row[:, None] * stride_xb
                + column[None, :] * stride_xd,
                mask=row_mask[:, None] & column_mask[None, :],
                other=0.0,
            )
            if not FP8:
                x = x.to(tl.float16) if FP16 else x.to(tl.bfloat16)
            if WIDE_CENTROID:
                centroid_offset = (
                    centroid[:, None].to(tl.int64) * stride_ck
                    + column[None, :].to(tl.int64) * stride_cd
                )
            else:
                centroid_offset = (
                    centroid[:, None] * stride_ck
                    + column[None, :] * stride_cd
                )
            c = tl.load(
                centroid_ptr + centroid_offset,
                mask=centroid_mask[:, None] & column_mask[None, :],
                other=0.0,
            )
            if FP8:
                score = tl.dot(
                    x,
                    tl.trans(c),
                    acc=score,
                    out_dtype=tl.float32,
                    max_num_imprecise_acc=max(32, BLOCK_D),
                )
            else:
                score = tl.dot(
                    x,
                    tl.trans(c),
                    acc=score,
                    out_dtype=tl.float32,
                )

        norm = tl.load(
            centroid_norm_ptr + centroid,
            mask=centroid_mask,
            other=float("inf"),
        )
        score = norm[None, :] - 2.0 * score
        block_score, block_label = tl.min(
            score,
            axis=1,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        block_label = block_label.to(tl.int32) + k0
        better = block_score < best_score
        best_score = tl.where(better, block_score, best_score)
        best_label = tl.where(better, block_label, best_label)
        k0 += BLOCK_K

    offset = row * PARTITIONS + partition
    tl.store(partial_score_ptr + offset, best_score, mask=row_mask)
    tl.store(partial_label_ptr + offset, best_label, mask=row_mask)


@triton.jit
def _reduce_split_k_kernel(
    x_ptr,
    centroid_ptr,
    partial_score_ptr,
    partial_label_ptr,
    distance_ptr,
    label_ptr,
    B,
    D,
    stride_xb,
    stride_xd,
    stride_ck,
    stride_cd,
    PARTITIONS: tl.constexpr,
    BLOCK_PARTITIONS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    FP16: tl.constexpr,
    FP8: tl.constexpr,
    SPHERICAL: tl.constexpr = False,
):
    row = tl.program_id(0).to(tl.int64)
    if not FP8 and not SPHERICAL:
        row_norm = tl.zeros((), tl.float32)
        for d0 in range(0, D, BLOCK_D):
            column = d0 + tl.arange(0, BLOCK_D)
            x = tl.load(
                x_ptr + row * stride_xb + column * stride_xd,
                mask=column < D,
                other=0.0,
            )
            x = x.to(tl.float16) if FP16 else x.to(tl.bfloat16)
            x = x.to(tl.float32)
            row_norm += tl.sum(x * x, axis=0)

    partition = tl.arange(0, BLOCK_PARTITIONS)
    mask = partition < PARTITIONS
    offset = row * PARTITIONS + partition
    score = tl.load(
        partial_score_ptr + offset,
        mask=mask,
        other=float("inf"),
    )
    # Partitions are in centroid order, so a left tie is the lowest label.
    best_score, best_partition = tl.min(
        score,
        axis=0,
        return_indices=True,
        return_indices_tie_break_left=True,
    )
    partial_label = tl.load(partial_label_ptr + offset, mask=mask, other=0)
    best_label = tl.sum(
        tl.where(partition == best_partition, partial_label, 0), axis=0
    )
    if SPHERICAL:
        distance = best_score
    elif FP8:
        rows = row + tl.arange(0, 1).to(tl.int64)
        labels = best_label + tl.arange(0, 1)
        distance = tl.sum(_selected_fp8_distance(
            x_ptr, centroid_ptr, rows, labels, D,
            stride_xb, stride_xd, stride_ck, stride_cd, rows < B, BLOCK_D,
        ), axis=0)
    else:
        distance = tl.maximum(best_score + row_norm, 0.0)
    tl.store(distance_ptr + row, distance)
    tl.store(label_ptr + row, best_label.to(tl.int64))


def _use_split_k(b, k, d, is_reduced, direct_block_d=_DIRECT_REDUCED_BLOCK_D):
    """Add parallelism only to a small grid after accounting for padding work."""
    if not is_reduced or _split_k_partitions(b, k) == 1:
        return False
    rows = triton.cdiv(b, _DIRECT_BLOCK_B)
    direct_span = triton.cdiv(d, direct_block_d) * direct_block_d
    split_span = triton.cdiv(d, _SPLIT_K_BLOCK_D) * _SPLIT_K_BLOCK_D
    # About one wave of the original four-wave target. A wider reduction tile
    # can double work for short rows; avoid it on an already adequate row grid.
    return rows * split_span < (_SPLIT_K_TARGET_PROGRAMS // 4) * direct_span


def _split_k_partitions(b, k):
    """Bound parallelism and ensure each partition has several K tiles.

    The program target amortizes the extra reduction launch on undersubscribed
    row grids; four tiles per partition avoid trivial partial searches. Both
    are conservative performance defaults, not a measured portable crossover.
    """
    k_blocks = triton.cdiv(k, _SPLIT_K_BLOCK_K)
    b_blocks = triton.cdiv(b, _SPLIT_K_BLOCK_B)
    maximum = max(1, min(
        k_blocks // _SPLIT_K_MIN_TILES_PER_PARTITION, _SPLIT_K_MAX_PARTITIONS
    ))
    partitions = 1
    while (2 * partitions <= maximum
           and b_blocks * partitions < _SPLIT_K_TARGET_PROGRAMS):
        partitions *= 2
    return partitions


def _launch_assignment(
    x, centroids, centroid_norm, distance, labels, wide_centroid, spherical=False
):
    b, d = x.shape
    k = centroids.shape[0]
    is_reduced = centroids.dtype != torch.float32
    is_fp16 = centroids.dtype == torch.float16
    is_fp8 = centroids.dtype == _FLOAT8_E4M3FN
    direct_block_d = _DIRECT_REDUCED_BLOCK_D if is_reduced else _DIRECT_FP32_BLOCK_D
    # E4M3 benefits from a wider contraction tile when it adds no padded work.
    # This replaces the old fitted D>=512 gate with the actual padded span.
    if (is_fp8 and triton.cdiv(d, 128) * 128
            == triton.cdiv(d, _DIRECT_REDUCED_BLOCK_D) * _DIRECT_REDUCED_BLOCK_D):
        direct_block_d = 128
    split_k = _use_split_k(b, k, d, is_reduced, direct_block_d)
    if not split_k:
        full_d = (
            is_reduced
            and _DIRECT_REDUCED_BLOCK_D < d <= _FULL_D_MAX_D
        )
        use_descriptors = (
            TensorDescriptor is not None
            and not full_d
            and is_reduced
            and d > _FULL_D_MAX_D
            and x.dtype == centroids.dtype
            and torch.cuda.get_device_capability(x.device) == (9, 0)
            and all(
                t.stride(1) == 1
                and t.data_ptr() % 16 == 0
                and t.stride(0) >= t.shape[1]
                and 0 < t.stride(0) * t.element_size() < 2**40
                and t.stride(0) * t.element_size() % 16 == 0
                and all(0 < dim <= 2**31 for dim in t.shape)
                for t in (x, centroids)
            )
        )
        if full_d:
            block_b, block_k, num_warps, num_stages = _FULL_D_CONFIG
            block_d = max(16, triton.next_power_of_2(d))
        else:
            # Match Triton's D/element-stride divisibility and pointer alignment.
            block_b = (128 if is_reduced and not is_fp8 and d > _FULL_D_MAX_D and d % 16 == 0
                       and all(t.stride(1) == 1 and t.stride(0) % 16 == 0
                               and t.data_ptr() % 16 == 0 for t in (x, centroids))
                       else _DIRECT_BLOCK_B)
            block_k = _DIRECT_BLOCK_K if is_reduced else 128
            block_d = direct_block_d
            num_warps = 8
            num_stages = 3
        if is_fp8 and use_descriptors:
            block_b = 128
        descriptors = {}
        if use_descriptors:
            descriptors = dict(
                x_desc=TensorDescriptor.from_tensor(x, [block_b, block_d]),
                centroid_desc=TensorDescriptor.from_tensor(
                    centroids, [block_k // 2 if is_fp8 else block_k, block_d],
                ),
            )
        _assign_kernel[(triton.cdiv(b, block_b),)](
            x,
            centroids,
            centroid_norm,
            distance,
            labels,
            b,
            k,
            d,
            x.stride(0),
            x.stride(1),
            centroids.stride(0),
            centroids.stride(1),
            BLOCK_B=block_b,
            BLOCK_K=block_k,
            BLOCK_D=block_d,
            FULL_D=full_d,
            REDUCED=is_reduced,
            FP16=is_fp16,
            FP8=is_fp8,
            WIDE_CENTROID=wide_centroid,
            SPHERICAL=spherical,
            **descriptors,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return

    partitions = _split_k_partitions(b, k)
    partial_score = torch.empty(
        (b, partitions), dtype=torch.float32, device=x.device
    )
    partial_label = torch.empty(
        (b, partitions), dtype=torch.int32, device=x.device
    )
    _assign_split_k_kernel[
        (triton.cdiv(b, _SPLIT_K_BLOCK_B), partitions)
    ](
        x,
        centroids,
        centroid_norm,
        partial_score,
        partial_label,
        b,
        k,
        d,
        x.stride(0),
        x.stride(1),
        centroids.stride(0),
        centroids.stride(1),
        PARTITIONS=partitions,
        BLOCK_B=_SPLIT_K_BLOCK_B,
        BLOCK_K=_SPLIT_K_BLOCK_K,
        BLOCK_D=_SPLIT_K_BLOCK_D,
        FP16=is_fp16,
        FP8=is_fp8,
        WIDE_CENTROID=wide_centroid,
        num_warps=_SPLIT_K_NUM_WARPS,
        num_stages=_SPLIT_K_NUM_STAGES,
    )
    _reduce_split_k_kernel[(b,)](
        x,
        centroids,
        partial_score,
        partial_label,
        distance,
        labels,
        b,
        d,
        x.stride(0),
        x.stride(1),
        centroids.stride(0),
        centroids.stride(1),
        PARTITIONS=partitions,
        BLOCK_PARTITIONS=triton.next_power_of_2(partitions),
        BLOCK_D=128,
        FP16=is_fp16,
        FP8=is_fp8,
        SPHERICAL=spherical,
        num_warps=4,
    )


def _pad_columns(value, multiple):
    """Return unit-stride rows whose width and stride are multiples."""
    if value.stride(1) == 1 and not (value.shape[1] % multiple or value.stride(0) % multiple):
        return value
    d = value.shape[1]
    result = value.new_zeros((len(value), d + -d % multiple))
    result[:, :d] = value
    return result


# Triton 3.8 mispacks E4M3 stores when d == 1 is specialized as a constant.
@triton.jit(do_not_specialize=["d"])
def _cast_fp8_centroids_kernel(
    source,
    out,
    squares,
    peak,
    bias,
    k,
    d,
    width: tl.constexpr,
    scale,
    QUANTIZED: tl.constexpr,
    AFFINE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lane = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    row, column = lane // width, lane % width
    mask = (row < k) & (column < d)
    value = tl.load(source + row * d + column, mask=mask, other=0.0)
    if AFFINE:
        shift = tl.load(bias + column, mask=mask, other=0.0)
        value = tl.inline_asm_elementwise(
            "fma.rn.f32 $0, $1, $2, $3;",
            constraints="=f,f,f,f",
            args=[value, scale, shift],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    elif QUANTIZED:
        value = tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[value, scale],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    value = tl.where(mask, value, 0.0)
    # Absolute finite bits are ordered; unsigned maximum also retains any prior NaN.
    maximum = tl.max(tl.abs(value).to(tl.uint32, bitcast=True), 0)
    tl.atomic_max(peak.to(tl.pointer_type(tl.uint32)), maximum)
    code = value.to(out.dtype.element_ty, fp_downcast_rounding="rtne")
    tl.store(out + lane, code, mask=row < k)
    code = code.to(tl.float32)
    tl.store(squares + row * d + column, code * code, mask=mask)


def _cast_fp8_centroids(source, quantizer, peak):
    """FP8-only preparation; logical contiguous squares retain the ATen reduction."""
    if (type(source) is not torch.Tensor or not source.is_cuda
            or source.dtype is not torch.float32 or not source.is_contiguous()):
        return None
    k, d = source.shape
    bias = None if quantizer is None else quantizer._bias
    if (
        quantizer is not None and type(quantizer.scale) is not float
    ) or (
        bias is not None
        and (
            type(bias) is not torch.Tensor
            or bias.dtype is not torch.float32
            or bias.device != source.device
            or bias.shape != (d,)
            or not bias.is_contiguous()
        )
    ):
        return None
    width = d + (-d % 64)
    out, squares = source.new_empty((k, width), dtype=_FLOAT8_E4M3FN), source.new_empty((k, d))
    with torch.cuda.device(source.device):
        _cast_fp8_centroids_kernel[(triton.cdiv(k * width, 2048),)](
            source, out, squares, peak.view(torch.int32), source if bias is None else bias,
            k, d, width, 1.0 if quantizer is None else quantizer.scale,
            QUANTIZED=quantizer is not None, AFFINE=bias is not None,
            BLOCK=2048, num_warps=4, enable_fp_fusion=False)
    torch.autograd.graph.increment_version(peak)
    return out, squares.sum(dim=1)


@triton.jit
def _byte_stats_kernel(rows, labels, stats, n, d: tl.constexpr, row_stride, label_stride,
                       stats_stride, BLOCK: tl.constexpr):
    lane = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    row, col = lane // (d + 1), lane % (d + 1)
    valid = row < n
    label = tl.load(labels + row * label_stride, mask=valid, other=0).to(tl.int64)
    value = tl.load(rows + row * row_stride + col,
                    mask=valid & (col < d), other=0.0).to(tl.float32)
    value = tl.where(col == d, 1.0, value)
    tl.atomic_add(stats + label * stats_stride + col, value,
                  mask=valid, sem="relaxed")


def accumulate_bytes(rows, labels, stats, *, fp8_code_units=False):
    """Accumulate finite FP16 rows with RN order-dependent FP32 sums.
    Counts are exact under the caller's whole-batch n<=2**24 bound.
    Internal labels are valid and unit inner stride is required.
    """
    if ((rows.dtype not in (torch.float16,)
         and not (fp8_code_units and rows.dtype is torch.float8_e4m3fn))
            or rows.stride(1) != 1):
        return False
    n, d = rows.shape
    with torch.cuda.device(rows.device):
        _byte_stats_kernel[(triton.cdiv(n * (d + 1), 512),)](
            rows, labels, stats, n, d, rows.stride(0), labels.stride(0), stats.stride(0),
            BLOCK=512,
        )
    torch.autograd.graph.increment_version(stats)
    return True


@triton.jit
def _ema_update_kernel(
    sums, counts, centroids, usage, k, d, sum_stride, count_stride,
    alpha, decay, DECAY: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
    SCALARS: tl.constexpr = False,
):
    if SCALARS:
        alpha = tl.load(alpha)
        decay = tl.load(decay)
    row = (tl.program_id(0) * BLOCK_K + tl.arange(0, BLOCK_K)).to(tl.int64)
    valid = row < k
    count = tl.load(counts + row * count_stride, mask=valid, other=0.0)
    old_usage = tl.load(usage + row, mask=valid, other=0.0)
    touched = count > 0
    # Explicit RN operations preserve subnormals without FMA/reassociation.
    retained = tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;", constraints="=f,f,f",
        args=[old_usage, decay], dtype=tl.float32, is_pure=True, pack=1,
    ) if DECAY else 0.0
    step, mass = tl.inline_asm_elementwise(
        "{ .reg .f32 increment, total; mul.rn.f32 increment, $2, $3; "
        "add.rn.f32 total, $4, increment; mov.b32 $0, increment; mov.b32 $1, total; }",
        constraints="=f,=f,f,f,f", args=[alpha, count, retained],
        dtype=(tl.float32, tl.float32), is_pure=True, pack=1,
    )
    weight = tl.math.div_rn(step, tl.where(touched, mass, 1.0))
    divisor = tl.where(touched, count, 1.0)[:, None]
    for begin in range(0, d, BLOCK_D):
        col = (begin + tl.arange(0, BLOCK_D)).to(tl.int64)
        moving = valid[:, None] & touched[:, None] & (col < d)[None, :]
        value = tl.load(sums + row[:, None] * sum_stride + col[None, :],
                        mask=moving, other=0.0)
        offset = row[:, None] * d + col[None, :]
        centroid = tl.load(centroids + offset, mask=moving, other=0.0)
        mean = tl.math.div_rn(value, divisor)
        position = tl.inline_asm_elementwise(
            "{ .reg .f32 delta, scaled; sub.rn.f32 delta, $1, $2; "
            "mul.rn.f32 scaled, delta, $3; add.rn.f32 $0, scaled, $2; }",
            constraints="=f,f,f,f", args=[mean, centroid, weight[:, None]],
            dtype=tl.float32, is_pure=True, pack=1,
        )
        tl.store(centroids + offset, tl.where((weight == 1)[:, None], mean, position),
                 mask=moving)
    # Retained usage can be reloaded per layout: finish every read before writing.
    tl.debug_barrier()
    tl.store(usage + row, tl.where(touched, mass, retained if DECAY else old_usage),
             mask=valid)


def ema_update(sums, counts, centroids, usage, alpha, decay, replace_usage):
    """Apply dense FP32 CUDA EMA, or retain eager handling for unsafe metadata.

    State is owned/contiguous; statistics have checked shape/dtype/device.
    Rates are independently computed host floats or owned 0-dim FP32 device views.
    Device rates are loaded in the kernel; no GPU values are scanned.
    """
    if sums.stride(1) != 1:
        return False
    mutable = {centroids.untyped_storage().data_ptr(), usage.untyped_storage().data_ptr()}
    if any(value.untyped_storage().data_ptr() in mutable for value in (sums, counts)):
        return False
    k, d = centroids.shape
    block_d = min(triton.next_power_of_2(d), 1024)
    block_k = max(1, 2048 // block_d)
    sms = torch.cuda.get_device_properties(centroids.device).multi_processor_count
    while block_k > 1 and triton.cdiv(k, block_k) < 2 * sms:
        block_k //= 2
    with torch.cuda.device(centroids.device):
        _ema_update_kernel[(triton.cdiv(k, block_k),)](
            sums, counts, centroids, usage, k, d, sums.stride(0), counts.stride(0),
            alpha, decay, DECAY=not replace_usage, SCALARS=isinstance(alpha, torch.Tensor),
            BLOCK_K=block_k, BLOCK_D=block_d,
            num_warps=4, enable_fp_fusion=False,
        )
    torch.autograd.graph.increment_version((centroids, usage))
    return True


@triton.jit
def _write_float_splits_kernel(
    centroids,
    usage,
    source,
    target,
    d,
    epsilon,
    scale,
    decode_exponent,
    BF16: tl.constexpr,
    FP8: tl.constexpr,
    SCALED: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # One pair owns both rows; hold donor operands in SSA until final stores.
    donor = tl.load(source + tl.program_id(0)).to(tl.int64)
    child = tl.load(target + tl.program_id(0)).to(tl.int64)
    col = tl.arange(0, BLOCK_D).to(tl.int64)
    value = tl.load(centroids + donor * d + col, mask=col < d, other=0.0)
    low, high = tl.inline_asm_elementwise(
        "{ .reg .f32 offset, minus, plus, low, high; mul.rn.f32 offset, $3, $4; "
        "sub.rn.f32 minus, 0f3F800000, offset; add.rn.f32 plus, 0f3F800000, offset; "
        "mul.rn.f32 low, $2, minus; mul.rn.f32 high, $2, plus; "
        "mov.b32 $0, low; mov.b32 $1, high; }",
        constraints="=f,=f,f,f,f", args=[value, epsilon, tl.where(col % 2 == 0, 1.0, -1.0)],
        dtype=(tl.float32, tl.float32), is_pure=True, pack=1,
    )
    if FP8:
        low_code, high_code = low, high
        if SCALED:
            low_code = tl.inline_asm_elementwise(
                "mul.rn.f32 $0, $1, $2;",
                constraints="=f,f,f",
                args=[low, scale],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )
            high_code = tl.inline_asm_elementwise(
                "mul.rn.f32 $0, $1, $2;",
                constraints="=f,f,f",
                args=[high, scale],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )
        low_bits = low_code.to(tl.float8e4nv, fp_downcast_rounding="rtne").to(tl.uint8, bitcast=True).to(tl.uint32)
        high_bits = high_code.to(tl.float8e4nv, fp_downcast_rounding="rtne").to(tl.uint8, bitcast=True).to(tl.uint32)
    elif BF16:
        low_bits, high_bits = tl.inline_asm_elementwise(
            "cvt.rn.bf16.f32 $0, $2; cvt.rn.bf16.f32 $1, $3;",
            constraints="=h,=h,f,f", args=[low, high],
            dtype=(tl.uint16, tl.uint16), is_pure=True, pack=1,
        )
        low_bits, high_bits = low_bits.to(tl.uint32), high_bits.to(tl.uint32)
    else:
        low_bits, high_bits = low.to(tl.uint32, bitcast=True), high.to(tl.uint32, bitcast=True)
    sign: tl.constexpr = 0x80 if FP8 else (0x8000 if BF16 else 0x80000000)
    magnitude: tl.constexpr = sign - 1
    infinity: tl.constexpr = 0x7F if FP8 else (0x7F80 if BF16 else 0x7F800000)
    low_abs, high_abs = low_bits & magnitude, high_bits & magnitude
    equal = ((low_bits == high_bits) | ((low_abs == 0) & (high_abs == 0)))
    equal = equal & (low_abs <= infinity) & (high_abs <= infinity)
    if FP8:
        equal = equal & (low_abs < infinity) & (high_abs < infinity)
    collapsed = tl.sum(tl.where((col < d) & ~equal, 1, 0), 0) == 0
    # Represented nextafter toward +inf, otherwise inward; keep NaNs as NaNs.
    negative = (high_bits & sign) != 0
    up = tl.where(high_abs == 0, 1, tl.where(negative, high_bits - 1, high_bits + 1))
    inward = tl.where(high_abs == 0, sign | 1,
                      tl.where(negative, high_bits + 1, high_bits - 1))
    moved = tl.where((up & magnitude) < infinity, up, inward)
    moved = tl.where(high_abs > infinity, high_bits, moved)
    if FP8:
        neighbor = moved.to(tl.uint8).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        if SCALED:
            neighbor = _fp8_libdevice.ldexp(neighbor, tl.load(decode_exponent))
        collapsed = collapsed & (tl.abs(neighbor) < float("inf"))
    else:
        neighbor = (moved << 16 if BF16 else moved).to(tl.float32, bitcast=True)
    high = tl.where((col == 0) & collapsed, neighbor, high)
    half = tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, 0f40000000;", constraints="=f,f",
        args=[tl.load(usage + donor)], dtype=tl.float32, is_pure=True, pack=1,
    )
    tl.store(centroids + donor * d + col, low, mask=col < d)
    tl.store(centroids + child * d + col, high, mask=col < d)
    tl.store(usage + donor, half)
    tl.store(usage + child, half)


def _write_float_splits(centroids, usage, source, target, dtype, epsilon, quantizer=None):
    fp8 = _FLOAT8_E4M3FN is not None and dtype is _FLOAT8_E4M3FN
    if quantizer is not None:
        from .e_means_fp8 import Fp8Quantizer

        if (not fp8 or _fp8_libdevice is None or type(quantizer) is not Fp8Quantizer
                or quantizer.shift is not None or quantizer._bias is not None
                or any(name in vars(quantizer) for name in ("compute_codes", "decode"))):
            return False
    if (dtype not in (None, torch.float32, torch.bfloat16, _FLOAT8_E4M3FN)
            or not centroids.is_cuda or centroids.shape[1] > 1024
            or centroids.dtype != torch.float32 or usage.dtype != torch.float32
            or not centroids.is_contiguous() or not usage.is_contiguous()):
        return False
    with torch.cuda.device(centroids.device):
        _write_float_splits_kernel[(len(source),)](
            centroids, usage, source, target, centroids.shape[1], epsilon,
            1.0 if quantizer is None else quantizer.scale,
            None if quantizer is None else quantizer._decode_exponent,
            BF16=dtype is torch.bfloat16, FP8=fp8, SCALED=quantizer is not None,
            BLOCK_D=triton.next_power_of_2(centroids.shape[1]),
            num_warps=4, enable_fp_fusion=False,
        )
    torch.autograd.graph.increment_version((centroids, usage))
    return True


def assign_rows(x, centroids, centroid_norm, *, spherical=False):
    """Return nearest-centroid squared distances and indices."""
    if (
        x.device.type != "cuda"
        or centroids.device != x.device
        or centroid_norm.device != x.device
    ):
        raise ValueError(
            "Triton assignment requires tensors on one CUDA device"
        )
    if x.dtype not in (torch.float32, centroids.dtype) or centroids.dtype not in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
        _FLOAT8_E4M3FN,
    ):
        raise TypeError(
            "Triton rows must be FP32 or match floating centroid dtype; E4M3 must match"
        )
    is_fp8 = centroids.dtype == _FLOAT8_E4M3FN
    if spherical and is_fp8:
        raise ValueError("spherical Triton requires unencoded floating operands")
    if is_fp8:
        from .e_means_fp8 import _check_fp8_support

        _check_fp8_support(x.device)
        if x.dtype != _FLOAT8_E4M3FN:
            raise TypeError("FP8 assignment requires native E4M3 rows")
    if centroid_norm.dtype != torch.float32:
        raise TypeError("centroid_norm must be float32")
    if x.ndim != 2 or centroids.ndim != 2 or centroid_norm.ndim != 1:
        raise ValueError("Triton assignment expects matrices and a norm vector")
    b, d = x.shape
    k, centroid_d = centroids.shape
    if k > _MAX_K:
        raise ValueError("Triton assignment requires k <= 2**31 - 256")
    if not centroid_norm.is_contiguous():
        raise ValueError("centroid_norm must be contiguous")
    if d != centroid_d or centroid_norm.shape[0] != k:
        raise ValueError("Triton assignment shapes do not match")
    with torch.cuda.device(x.device):
        if is_fp8 or (d % 16 and centroids.dtype != torch.float32):
            # Zero columns add exact zeros at unchanged tile positions.
            x = _pad_columns(x, 64 if is_fp8 else 16)
            centroids = _pad_columns(centroids, 64 if is_fp8 else 16)
        distance = torch.empty(b, dtype=torch.float32, device=x.device)
        labels = torch.empty(b, dtype=torch.int64, device=x.device)
        wide_centroid = _centroid_offset_is_wide(
            centroids.shape, centroids.stride()
        )
        _launch_assignment(
            x,
            centroids,
            centroid_norm,
            distance,
            labels,
            wide_centroid,
            spherical=spherical,
        )
    return distance, labels
