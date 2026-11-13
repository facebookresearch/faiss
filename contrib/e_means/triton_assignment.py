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
_INT8_BLOCK_B = 128
_INT8_BLOCK_K = 256
_INT8_BLOCK_D = 128
_INT8_PREPARE_BLOCK_D = 256
_INT8_MAX_D = 43919
_INT8_D_ALIGNMENT = 16
_MAX_K = 2**31 - max(
    _SPLIT_K_MAX_PARTITIONS,
    _SPLIT_K_BLOCK_K,
    _DIRECT_BLOCK_K,
)


_int8_libdevice = None


def _load_int8_libdevice():
    """Load the INT8 rounding dependency only when INT8 is selected."""
    global _int8_libdevice
    if _int8_libdevice is None:
        try:
            from triton.language.extra import libdevice
        except ImportError as error:
            raise ImportError(
                "INT8 assignment requires Triton libdevice support"
            ) from error
        if not hasattr(libdevice, "rint"):
            raise ImportError(
                "INT8 assignment requires Triton libdevice.rint support"
            )
        _int8_libdevice = libdevice
    return _int8_libdevice


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
def _merge_second(score, block_score, block_label, best_score, second):
    """Running exact second-best score: a tile's best beating the best demotes it."""
    column = tl.arange(0, score.shape[1])[None, :]
    block_second = tl.min(tl.where(column == block_label[:, None], float("inf"), score), axis=1)
    return tl.where(block_score < best_score, tl.minimum(best_score, block_second),
                    tl.minimum(second, block_score))


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
    second_ptr=None,
    SECOND: tl.constexpr = 0,
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
    second = best_score
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
            if SECOND:
                second = _merge_second(score, block_score, block_label,
                                       best_score, second)
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
    if SECOND:
        if not SPHERICAL:
            second = tl.maximum(second + row_norm, 0.0)
        tl.store(second_ptr + row, second, mask=row_mask)


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
    partial_second_ptr=None,
    SECOND: tl.constexpr = 0,
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
    second = best_score

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
        if SECOND:
            second = _merge_second(score, block_score, block_label,
                                   best_score, second)
        block_label = block_label.to(tl.int32) + k0
        better = block_score < best_score
        best_score = tl.where(better, block_score, best_score)
        best_label = tl.where(better, block_label, best_label)
        k0 += BLOCK_K

    offset = row * PARTITIONS + partition
    tl.store(partial_score_ptr + offset, best_score, mask=row_mask)
    tl.store(partial_label_ptr + offset, best_label, mask=row_mask)
    if SECOND:
        tl.store(partial_second_ptr + offset, second, mask=row_mask)


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
    partial_second_ptr=None,
    second_ptr=None,
    SECOND: tl.constexpr = 0,
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
    if SECOND:
        # Exact merge: the winner's own second, or another partition's best.
        second = tl.min(tl.where(partition == best_partition, tl.load(
            partial_second_ptr + offset, mask=mask, other=float("inf")), score), axis=0)
        if not SPHERICAL:
            second = tl.maximum(second + row_norm, 0.0)
        tl.store(second_ptr + row, second)


@triton.jit
def _stage_int8_rows_kernel(
    x_ptr,
    prepared_ptr,
    D,
    stride_xb,
    stride_xd,
    stride_pb,
    UNSIGNED: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    for d0 in range(0, D, BLOCK_D):
        column = d0 + tl.arange(0, BLOCK_D)
        mask = column < D
        value = tl.load(
            x_ptr + row * stride_xb + column * stride_xd,
            mask=mask,
            other=0,
        )
        if UNSIGNED:
            value = value.to(tl.int16) - 128
        value = tl.where(mask, value, 0).to(tl.int8)
        tl.store(
            prepared_ptr + row * stride_pb + column,
            value,
            mask=column < stride_pb,
        )


@triton.jit
def _prepare_int8_kernel(
    centroid_ptr,
    prepared_ptr,
    norm_ptr,
    D,
    stride_ck,
    stride_cd,
    stride_pk,
    UNSIGNED: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    centroid = tl.program_id(0).to(tl.int64)
    norm = tl.zeros((), tl.int32)
    for d0 in range(0, D, BLOCK_D):
        column = d0 + tl.arange(0, BLOCK_D)
        mask = column < D
        value = tl.load(
            centroid_ptr + centroid * stride_ck + column * stride_cd,
            mask=mask,
            other=0.0,
        )
        if UNSIGNED:
            value = _int8_libdevice.rint(value - 128.0)
        else:
            value = _int8_libdevice.rint(value)
        value = tl.maximum(-128.0, tl.minimum(127.0, value))
        value = tl.where(mask, value, 0).to(tl.int8)
        tl.store(
            prepared_ptr + centroid * stride_pk + column,
            value,
            mask=column < stride_pk,
        )
        value = value.to(tl.int32)
        norm += tl.sum(value * value, axis=0)
    tl.store(norm_ptr + centroid, norm)


@triton.jit
def _assign_int8_kernel(
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
    UNSIGNED: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    WIDE_CENTROID: tl.constexpr,
    FULL_D: tl.constexpr,
    PACKED: tl.constexpr,
    x_desc=None,
    centroid_desc=None,
):
    row_program = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, BLOCK_B).to(tl.int64)
    row = row_program * BLOCK_B + lane
    row_mask = row < B
    row_norm = tl.zeros((BLOCK_B,), tl.int32)

    for d0 in range(0, D, BLOCK_D):
        column = d0 + tl.arange(0, BLOCK_D)
        column_mask = column < D
        value = tl.load(
            x_ptr
            + row[:, None] * stride_xb
            + column[None, :] * stride_xd,
            mask=row_mask[:, None] & column_mask[None, :],
            other=0,
        )
        if UNSIGNED:
            value = value.to(tl.int16) - 128
        value = tl.where(column_mask[None, :], value, 0).to(tl.int8)
        value = value.to(tl.int32)
        row_norm += tl.sum(value * value, axis=1)

    if FULL_D:
        # Keep the narrow row operand resident across the centroid search.
        column = tl.arange(0, BLOCK_D)
        full_x = tl.load(
            x_ptr + row[:, None] * stride_xb + column[None, :] * stride_xd,
            mask=row_mask[:, None] & (column[None, :] < D), other=0,
        )
        if UNSIGNED:
            full_x = full_x.to(tl.int16) - 128
        full_x = tl.where(column[None, :] < D, full_x, 0).to(tl.int8)

    best_score = tl.full((BLOCK_B,), 2147483647, tl.int32)
    best_label = tl.zeros((BLOCK_B,), tl.int32)
    for k0 in tl.range(0, K, BLOCK_K, flatten=x_desc is not None):
        centroid = k0 + tl.arange(0, BLOCK_K)
        centroid_mask = centroid < K
        score = tl.zeros((BLOCK_B, BLOCK_K), tl.int32)
        if FULL_D:
            column = tl.arange(0, BLOCK_D)
            if WIDE_CENTROID:
                centroid_offset = (
                    centroid[:, None].to(tl.int64) * stride_ck
                    + column[None, :].to(tl.int64) * stride_cd
                )
            else:
                centroid_offset = (
                    centroid[:, None] * stride_ck + column[None, :] * stride_cd
                )
            c = tl.load(
                centroid_ptr + centroid_offset,
                mask=centroid_mask[:, None] & (column[None, :] < D), other=0,
            )
            score = tl.dot(full_x, tl.trans(c), out_dtype=tl.int32)
        else:
            for d0 in range(0, D, BLOCK_D):
                if x_desc is None:
                    column = d0 + tl.arange(0, BLOCK_D)
                    column_mask = column < D
                    value = tl.load(
                        x_ptr
                        + row[:, None] * stride_xb
                        + column[None, :] * stride_xd,
                        mask=row_mask[:, None] & column_mask[None, :],
                        other=0,
                    )
                    if UNSIGNED:
                        value = value.to(tl.int16) - 128
                    value = tl.where(column_mask[None, :], value, 0).to(tl.int8)
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
                        other=0,
                    )
                else:
                    value = x_desc.load([tl.program_id(0) * BLOCK_B, d0])
                    c = centroid_desc.load([k0, d0])
                    if UNSIGNED:
                        # Unstaged UINT8: zero-filled centroid tails cancel -128.
                        value = (value.to(tl.int16) - 128).to(tl.int8)
                score = tl.dot(
                    value,
                    tl.trans(c),
                    acc=score,
                    out_dtype=tl.int32,
                )

        norm = tl.load(
            centroid_norm_ptr + centroid,
            mask=centroid_mask,
            other=0 if PACKED else 2147483647,
        )
        if PACKED:
            # Masked norms are zero, so every intermediate stays in range.
            base = tl.where(centroid_mask,
                            norm * BLOCK_K + tl.arange(0, BLOCK_K), 2147483647)
            key = tl.min(base[None, :] - 2 * BLOCK_K * score, axis=1)
            local = key & (BLOCK_K - 1)
            block_score = (key - local) // BLOCK_K
            block_label = local + k0
        else:
            score = norm[None, :] - 2 * score
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

    distance = best_score.to(tl.int64) + row_norm.to(tl.int64)
    distance = tl.maximum(distance, 0).to(distance_ptr.dtype.element_ty)
    tl.store(distance_ptr + row, distance, mask=row_mask)
    tl.store(label_ptr + row, best_label.to(tl.int64), mask=row_mask)


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


def _descriptor_layout(tensor):
    byte_pitch = tensor.stride(0) * tensor.element_size()
    return (tensor.stride(1) == 1 and tensor.data_ptr() % 16 == 0
            and 0 < byte_pitch < 2**40 and byte_pitch % 16 == 0
            and all(0 < dim <= 2**31 for dim in tensor.shape))


def _descriptor_from_tensor():
    factory = getattr(TensorDescriptor, "from_tensor", None)
    if not callable(factory):
        raise ImportError("selected Triton assignment requires callable "
                          "triton.tools.tensor_descriptor.TensorDescriptor.from_tensor")
    return factory


def _launch_assignment(
    x, centroids, centroid_norm, distance, labels, wide_centroid, spherical=False,
    second=None,
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
            not full_d
            and is_reduced
            and d > _FULL_D_MAX_D
            and x.dtype == centroids.dtype
            and torch.cuda.get_device_capability(x.device) == (9, 0)
            and all(
                t.stride(0) >= t.shape[1] and _descriptor_layout(t)
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
            from_tensor = _descriptor_from_tensor()
            descriptors = dict(
                x_desc=from_tensor(x, [block_b, block_d]),
                centroid_desc=from_tensor(
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
            second_ptr=second,
            SECOND=second is not None,
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
    partial_second = None if second is None else torch.empty_like(partial_score)
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
        partial_second_ptr=partial_second,
        SECOND=second is not None,
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
        partial_second_ptr=partial_second,
        second_ptr=second,
        SECOND=second is not None,
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


def _int8_unsigned(source):
    if source == "int8":
        return False
    if source == "uint8":
        return True
    raise ValueError("unsupported INT8 representation %r" % source)


def _int8_physical_d(d):
    return (
        (d + _INT8_D_ALIGNMENT - 1)
        // _INT8_D_ALIGNMENT
        * _INT8_D_ALIGNMENT
    )


def _check_int8_source(x, source):
    from .quantizer_int8 import _check_int8_support

    _check_int8_support(x.device)
    return _int8_unsigned(source)


def prepare_int8_centroids(centroids, representation, *, validate=True):
    """Return signed-byte centroids and their matched INT32 norms."""
    unsigned = _check_int8_source(centroids, representation)
    _load_int8_libdevice()
    if centroids.dtype != torch.float32 or centroids.ndim != 2:
        raise TypeError("INT8 centroid preparation requires an FP32 matrix")
    if validate and not bool(torch.isfinite(centroids).all()):
        raise ValueError("INT8 centroids must be finite")
    k, d = centroids.shape
    if d > _INT8_MAX_D:
        raise ValueError("INT8 assignment requires d <= 43919")
    with torch.cuda.device(centroids.device):
        physical_d = _int8_physical_d(d)
        prepared = torch.empty(
            (k, physical_d), dtype=torch.int8, device=centroids.device
        )
        norm = torch.empty(k, dtype=torch.int32, device=centroids.device)
        _prepare_int8_kernel[(k,)](
            centroids,
            prepared,
            norm,
            d,
            centroids.stride(0),
            centroids.stride(1),
            prepared.stride(0),
            UNSIGNED=unsigned,
            BLOCK_D=_INT8_PREPARE_BLOCK_D,
            num_warps=4,
        )
    return prepared, norm


def assign_int8_rows(
    x,
    centroids,
    centroid_norm,
    representation,
    distance_out=None,
):
    """Return exact-ranking INT8 distances and centroid indices.

    centroid_norm must be the squared norms of the matching signed-byte centroids.
    Distances default to FP32; a caller INT64 vector retains exact integer values.
    """
    if (
        x.device.type != "cuda"
        or centroids.device != x.device
        or centroid_norm.device != x.device
    ):
        raise ValueError(
            "Triton assignment requires tensors on one CUDA device"
        )
    unsigned = _check_int8_source(x, representation)
    expected_dtype = {
        "int8": (torch.int8,),
        "uint8": (torch.uint8,),
    }[representation]
    if x.dtype not in expected_dtype or centroids.dtype != torch.int8:
        raise TypeError(
            "INT8 assignment input does not match its representation"
        )
    if centroid_norm.dtype != torch.int32:
        raise TypeError("INT8 centroid norms must be int32")
    if x.ndim != 2 or centroids.ndim != 2 or centroid_norm.ndim != 1:
        raise ValueError("INT8 assignment expects matrices and a norm vector")
    b, d = x.shape
    k, centroid_d = centroids.shape
    if d > _INT8_MAX_D:
        raise ValueError("INT8 assignment requires d <= 43919")
    if k > _MAX_K:
        raise ValueError("Triton assignment requires k <= 2**31 - 256")
    physical_d = _int8_physical_d(d)
    if physical_d != centroid_d or centroid_norm.shape[0] != k:
        raise ValueError("INT8 assignment shapes do not match")
    if not centroid_norm.is_contiguous():
        raise ValueError("centroid_norm must be contiguous")
    if distance_out is not None and (
        distance_out.dtype not in (torch.float32, torch.int64)
        or distance_out.device != x.device
        or distance_out.shape != (b,)
        or not distance_out.is_contiguous()
    ):
        raise ValueError("distance output must be a same-device contiguous FP32/INT64 vector")

    stage_rows = physical_d != d
    full_d = physical_d <= 256
    # The staged width is bounded and16-aligned by _int8_physical_d;
    # its CUDA allocation has aligned unit-stride rows. Check before staging.
    use_descriptors = (not full_d
                       and torch.cuda.get_device_capability(x.device) == (9, 0)
                       and _descriptor_layout(centroids)
                       and (0 < b <= 2**31 if stage_rows else _descriptor_layout(x)))
    from_tensor = _descriptor_from_tensor() if use_descriptors else None

    with torch.cuda.device(x.device):
        if stage_rows:
            staged = torch.empty(
                (b, physical_d), dtype=torch.int8, device=x.device
            )
            _stage_int8_rows_kernel[(b,)](
                x,
                staged,
                d,
                x.stride(0),
                x.stride(1),
                staged.stride(0),
                UNSIGNED=unsigned,
                BLOCK_D=_INT8_PREPARE_BLOCK_D,
                num_warps=4,
            )
            x = staged
            unsigned = False
        distance = (torch.empty(b, dtype=torch.float32, device=x.device)
                    if distance_out is None else distance_out)
        labels = torch.empty(b, dtype=torch.int64, device=x.device)
        wide_centroid = _centroid_offset_is_wide(
            centroids.shape, centroids.stride()
        )
        # Wider operands retain the original launch configuration.
        block_b, block_k, num_warps = (
            (128, 128, 4) if full_d else (_INT8_BLOCK_B, _INT8_BLOCK_K, 8)
        )
        descriptors = {}
        if use_descriptors:
            descriptors = dict(
                x_desc=from_tensor(x, [block_b, _INT8_BLOCK_D]),
                centroid_desc=from_tensor(centroids, [block_k, _INT8_BLOCK_D]),
            )
        _assign_int8_kernel[(triton.cdiv(b, block_b),)](
            x,
            centroids,
            centroid_norm,
            distance,
            labels,
            b,
            k,
            physical_d,
            x.stride(0),
            x.stride(1),
            centroids.stride(0),
            centroids.stride(1),
            UNSIGNED=unsigned,
            BLOCK_B=block_b,
            BLOCK_K=block_k,
            BLOCK_D=(triton.next_power_of_2(max(_INT8_BLOCK_D, physical_d))
                     if full_d else _INT8_BLOCK_D),
            WIDE_CENTROID=wide_centroid,
            FULL_D=full_d,
            PACKED=48896 * physical_d * block_k + block_k - 1 < 2**31,
            **descriptors,
            num_warps=num_warps,
            num_stages=3,
        )
    return distance, labels


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
                       stats_stride, SHIFT: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    row, col = lane // (d + 1), lane % (d + 1)
    valid = row < n
    label = tl.load(labels + row * label_stride, mask=valid, other=0).to(tl.int64)
    value = tl.load(rows + row * row_stride + col,
                    mask=valid & (col < d), other=0.0).to(stats.dtype.element_ty)
    if SHIFT:
        value -= 128
    value = tl.where(col == d, 1, value)
    tl.atomic_add(stats + label * stats_stride + col, value,
                  mask=valid, sem="relaxed")


def accumulate_bytes(rows, labels, stats, *, fp8_code_units=False, shift=False):
    """Add admitted typed rows to packed stats under whole-batch bounds.

    Internal assignment labels are valid; unit inner stride is required.
    Whole-batch n<=2**24 keeps counts exact. INT8 n<=131072 or UINT8 n<=65793
    makes every FP32 sum addition exact. Raw bytes use bounded INT32 scratch;
    shift subtracts 128 from UINT8 when 255*n could exceed INT32.
    Finite native FP16 remains lattice-safe.
    These lattices never reach FP32 subnormals. The caller excludes quantizers
    and deterministic mode.
    Opt-in finite E4M3 code units require a zeroed table and whole-batch n<=2**24;
    FP32 RN sums have unspecified addition order, while counts remain exact.
    """
    if ((rows.dtype not in (torch.int8, torch.uint8, torch.float16)
         and not (fp8_code_units and rows.dtype is torch.float8_e4m3fn))
            or rows.stride(1) != 1):
        return False
    n, d = rows.shape
    with torch.cuda.device(rows.device):
        _byte_stats_kernel[(triton.cdiv(n * (d + 1), 512),)](
            rows, labels, stats, n, d, rows.stride(0), labels.stride(0), stats.stride(0),
            SHIFT=shift, BLOCK=512,
        )
    torch.autograd.graph.increment_version(stats)
    return True


@triton.jit
def _copy_byte_stats_kernel(source, out, k, d: tl.constexpr, source_stride, out_stride,
                            UNSIGNED: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    row, col = lane // (d + 1), lane % (d + 1)
    valid = row < k
    value = tl.load(source + row * source_stride + col, mask=valid, other=0).to(tl.int64)
    if UNSIGNED:
        count = tl.load(source + row * source_stride + d, mask=valid, other=0).to(tl.int64)
        value += tl.where(col < d, 128 * count, 0)
    tl.store(out + row * out_stride + col, value, mask=valid)


def copy_byte_stats(source, out, *, unsigned):
    """Reconstruct raw totals from disjoint bounded INT32 sums/counts."""
    k, width = source.shape
    with torch.cuda.device(source.device):
        _copy_byte_stats_kernel[(triton.cdiv(k * width, 512),)](
            source, out, k, width - 1, source.stride(0), out.stride(0),
            UNSIGNED=unsigned, BLOCK=512)
    torch.autograd.graph.increment_version(out)


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
    count = tl.load(counts + row * count_stride, mask=valid, other=0.0).to(tl.float32)
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
                        mask=moving, other=0.0).to(tl.float32)
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
    """Apply FP32 CUDA EMA from FP32, raw INT64 or bounded INT32 replay statistics.

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
    count=None,
):
    # One pair owns both rows; hold donor operands in SSA until final stores.
    # With a device count, programs past it write nothing.
    live = True if count is None else tl.program_id(0) < tl.load(count)
    donor = tl.load(source + tl.program_id(0)).to(tl.int64)
    child = tl.load(target + tl.program_id(0)).to(tl.int64)
    col = tl.arange(0, BLOCK_D).to(tl.int64)
    value = tl.load(centroids + donor * d + col, mask=(col < d) & live, other=0.0)
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
        args=[tl.load(usage + donor, mask=live, other=0.0)], dtype=tl.float32, is_pure=True, pack=1,
    )
    tl.store(centroids + donor * d + col, low, mask=(col < d) & live)
    tl.store(centroids + child * d + col, high, mask=(col < d) & live)
    tl.store(usage + donor, half, mask=live)
    tl.store(usage + child, half, mask=live)


def _write_float_splits(centroids, usage, source, target, dtype, epsilon, quantizer=None, count=None):
    fp8 = _FLOAT8_E4M3FN is not None and dtype is _FLOAT8_E4M3FN
    if quantizer is not None:
        from .quantizer_fp8 import Fp8Quantizer

        if (not fp8 or type(quantizer) is not Fp8Quantizer
                or quantizer.shift is not None or quantizer._bias is not None
                or any(name in vars(quantizer) for name in ("compute_codes", "decode"))):
            return False
    if (dtype not in (None, torch.float32, torch.bfloat16, _FLOAT8_E4M3FN)
            or not centroids.is_cuda or centroids.shape[1] > 8192
            or centroids.dtype != torch.float32 or usage.dtype != torch.float32
            or not centroids.is_contiguous() or not usage.is_contiguous()):
        return False
    if quantizer is not None and not callable(getattr(_fp8_libdevice, "ldexp", None)):
        raise ImportError("scaled FP8 centroid splitting requires callable "
                          "triton.language.extra.cuda.libdevice.ldexp")
    with torch.cuda.device(centroids.device):
        _write_float_splits_kernel[(len(source),)](
            centroids, usage, source, target, centroids.shape[1], epsilon,
            1.0 if quantizer is None else quantizer.scale,
            None if quantizer is None else quantizer._decode_exponent,
            BF16=dtype is torch.bfloat16, FP8=fp8, SCALED=quantizer is not None,
            BLOCK_D=triton.next_power_of_2(centroids.shape[1]),
            count=count, num_warps=max(4, triton.next_power_of_2(centroids.shape[1]) // 256), enable_fp_fusion=False,
        )
    torch.autograd.graph.increment_version((centroids, usage))
    return True


def assign_rows(x, centroids, centroid_norm, *, spherical=False, return_score=False,
                second_out=None):
    """Return nearest-centroid squared distances and indices.

    ``second_out`` receives exact second-nearest distances (not with E4M3).
    It must be a same-device contiguous FP32 vector of length ``len(x)``.

    ``return_score=True`` returns the ranking score ``||c||² - 2*x.c``
    without adding the row norm or clamping. This preserves cross-shard
    ranking, including negative scores, when evaluation recomputes distances
    from the selected original FP32 operands. Centroid norms are supplied by
    the caller in both modes.
    """
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
    if (return_score or second_out is not None) and is_fp8:
        raise ValueError("ranking scores and second distances require unencoded floating operands")
    if spherical and is_fp8:
        raise ValueError("spherical Triton requires unencoded floating operands")
    if is_fp8:
        from .quantizer_fp8 import _check_fp8_support

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
    if second_out is not None:
        if not isinstance(second_out, torch.Tensor):
            raise TypeError("second output must be a tensor")
        if (second_out.dtype != torch.float32 or second_out.device != x.device
                or second_out.shape != (b,) or not second_out.is_contiguous()):
            raise ValueError("second output must be a same-device contiguous FP32 vector of shape (%d,)" % b)
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
            # The kernel's spherical flag only suppresses row norms/clamping;
            # the supplied centroid norms still determine L2 ranking.
            spherical=spherical or return_score,
            second=second_out,
        )
    return distance, labels


@triton.jit
def _mul_rn(a, b):  # FP32, never contracted to an FMA (as removal_update's enable_fp_fusion=False)
    return tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32,
                                     is_pure=True, pack=1)


@triton.jit
def _add_rn(a, b):
    return tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32,
                                     is_pure=True, pack=1)


@triton.jit
def _removal_stats_kernel(rows, order, offsets, distance, second, axes, centroids, stats, removal, d,
                          row_stride, stats_stride, removal_stride, alpha, decay, BLOCK_R: tl.constexpr,
                          BLOCK_D: tl.constexpr, EMA: tl.constexpr, SCALARS: tl.constexpr, LONG: tl.constexpr):
    # Centroid j owns its segment of label-sorted rows (one read of each row, no atomics) and
    # adds [x | 1] to stats and [p y | d2 - d1] to removal, y = x - c, p = y.v; with EMA, removal is
    # the EMA itself: [U | R] <- decay [U | R] + alpha (0 + this batch), V <- U / |U| where |U| > 0,
    # for every centroid (as removal_update on a zero batch table). More than LONG rows: _long_pieces_kernel.
    j = tl.program_id(0).to(tl.int64)
    start = tl.load(offsets + j)
    end = tl.load(offsets + j + 1)
    col = tl.arange(0, BLOCK_D)
    cmask = col < d
    axis = tl.load(axes + j * d + col, mask=cmask, other=0.0)
    oja = tl.zeros((BLOCK_D,), tl.float32)
    margin = tl.zeros((BLOCK_R,), tl.float32)
    short = end - start <= LONG
    if (start < end) & short:
        center = tl.load(centroids + j * d + col, mask=cmask, other=0.0)
        total = tl.zeros((BLOCK_D,), tl.float32)
        for begin in range(start, end, BLOCK_R):
            lane = begin + tl.arange(0, BLOCK_R)
            rmask = lane < end
            row = tl.load(order + lane, mask=rmask, other=0).to(tl.int64)
            value = tl.load(rows + row[:, None] * row_stride + col[None, :],
                            mask=rmask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
            total += tl.sum(value, axis=0)
            y = tl.where(rmask[:, None] & cmask[None, :], value - center[None, :], 0.0)
            oja += tl.sum(tl.sum(y * axis[None, :], axis=1)[:, None] * y, axis=0)
            gap = tl.load(second + row, mask=rmask, other=0.0) - tl.load(distance + row, mask=rmask, other=0.0)
            gap = tl.where(gap == float("-inf"), -3.4028234663852886e38, gap)  # as nan_to_num(0, 0)
            margin += tl.where(rmask & (gap == gap) & (gap < float("inf")), gap, 0.0)
        out = stats + j * stats_stride
        tl.store(out + col, tl.load(out + col, mask=cmask) + total, mask=cmask)
        tl.store(out + d, tl.load(out + d) + (end - start).to(tl.float32))
        if not EMA:
            out = removal + j * removal_stride
            tl.store(out + col, tl.load(out + col, mask=cmask) + oja, mask=cmask)
            tl.store(out + d, tl.load(out + d) + tl.sum(margin))
    if EMA:
        if SCALARS:
            alpha = tl.load(alpha)
            decay = tl.load(decay)
        if short:
            out = removal + j * removal_stride
            u = _add_rn(_mul_rn(tl.load(out + col, mask=cmask, other=0.0), decay), _mul_rn(alpha, 0.0 + oja))
            tl.store(out + col, u, mask=cmask)
            tl.store(out + d, _add_rn(_mul_rn(tl.load(out + d), decay), _mul_rn(alpha, 0.0 + tl.sum(margin))))
            u = tl.where(cmask, u, 0.0).to(tl.float64)  # FP64: |U|^2 can overflow FP32
            norm = tl.sqrt(tl.sum(u * u))
            tl.store(axes + j * d + col, (u / norm).to(tl.float32), mask=cmask & (norm > 0))


def accumulate_removal(rows, labels, stats, distance, second, axes, centroids, removal, ema=None,
                       statistics_rows=None):
    """Add [x | 1] to FP32 stats and [p y | d2 - d1] to removal by label (y = x - centroids,
    p = y.axes), deterministically, from label-sorted rows.

    ema = (state, alpha, decay), only for a batch accumulated in this one call into a zero `removal` table
    (left untouched): removal_update's EMA is applied to state and axes, and the result is "folded" (up to
    d = 2048; above, True and removal_update as usual). alpha and decay are floats or 0-d tensors."""
    if statistics_rows is not None:
        rows = statistics_rows
    k, width = stats.shape
    fits = not (not rows.is_cuda or not rows.dtype.is_floating_point or rows.stride(1) != 1 or width > 4097
                or width != rows.shape[1] + 1 or stats.stride(1) != 1 or removal.stride(1) != 1
                or not stats.dtype == removal.dtype == axes.dtype == centroids.dtype == torch.float32
                or not axes.is_contiguous() or not centroids.is_contiguous())
    if not fits:
        return False
    dtype = torch.int16 if k < 2**15 else torch.int32  # positions run to k inclusive
    sorted_labels, order = torch.sort(labels.to(dtype))
    offsets = torch.searchsorted(sorted_labels, torch.arange(k + 1, device=rows.device, dtype=dtype))
    with torch.cuda.device(rows.device):
        block_d = triton.next_power_of_2(rows.shape[1])
        # Short segments are latency-bound: small programs keep more of them in flight.
        block_r, warps = (4, 1) if block_d <= 128 else (2, 2 if block_d <= 512 else 4 if block_d <= 2048 else 8)
        # Same arithmetic as removal_update; from 4 warps (d > 512) its batch sums may differ by an ULP.
        fold = ema is not None and block_d <= 2048
        state, alpha, decay = ema if fold else (removal, 0.0, 0.0)
        scalars = isinstance(alpha, torch.Tensor)
        _removal_stats_kernel[(k,)](
            rows, order, offsets, distance, second, axes, centroids, stats, state, rows.shape[1],
            rows.stride(0), stats.stride(0), state.stride(0), alpha, decay, BLOCK_R=block_r,
            BLOCK_D=block_d, EMA=fold, SCALARS=scalars, LONG=_LONG_SEGMENT, num_warps=warps)
        pieces, partial, counter = _pieces(rows, stats, 2 * block_d + 1)
        _long_pieces_kernel[(pieces,)](
            rows, order, offsets, sorted_labels, partial, counter, distance, second, axes, centroids, stats, state,
            rows.shape[1], len(rows), rows.stride(0), stats.stride(0), state.stride(0), alpha, decay,
            BLOCK_R=block_r, BLOCK_D=block_d, LONG=_LONG_SEGMENT, EMA=fold, SCALARS=scalars, num_warps=warps)
        torch.autograd.graph.increment_version((stats, state, axes) if fold else (stats, removal))
        return "folded" if fold else True


@triton.jit
def _nearest_operands_kernel(centroids, hi, lo, norms, d, BLOCK_D: tl.constexpr):
    # hi = BF16(c), lo = BF16(c - hi) and |c|^2, one row per program.
    j = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK_D)
    c = tl.load(centroids + j * d + col, mask=col < d, other=0.0)
    high = c.to(tl.bfloat16)
    tl.store(hi + j * d + col, high, mask=col < d)
    tl.store(lo + j * d + col, (c - high.to(tl.float32)).to(tl.bfloat16), mask=col < d)
    tl.store(norms + j, tl.sum(c * c))


@triton.jit
def _nearest_screen_kernel(hi, lo, norms, ids, count, tile_min, k, d, tiles, row_offset, rows,
                           query_hi, query_lo, centroid_begin,
                           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
                           SPLIT: tl.constexpr, EXTERNAL: tl.constexpr):
    # Scores |c|^2 - 2 q.c of rows ids[:count] from BF16 products (SPLIT: c = hi + lo, three
    # products); one minimum per centroid tile, self excluded. Programs stride over the
    # (row block, tile) pairs of the live rows, so a few candidates still fill the GPU.
    live = tl.minimum(tl.maximum(tl.load(count) - row_offset, 0), rows)
    for unit in range(tl.program_id(0), tl.cdiv(live, BLOCK_M) * tiles, tl.num_programs(0)):
        tile = unit % tiles
        row = row_offset + (unit // tiles) * BLOCK_M + tl.arange(0, BLOCK_M)
        rmask = row < row_offset + live
        query = tl.load(ids + row, mask=rmask, other=0).to(tl.int64)
        col = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        cmask = col < k
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for d0 in range(0, d, BLOCK_D):
            dim = d0 + tl.arange(0, BLOCK_D)
            dmask = dim < d
            qindex = row if EXTERNAL else query
            a = qindex[:, None] * d + dim[None, :]
            b = col[:, None].to(tl.int64) * d + dim[None, :]
            amask, bmask = rmask[:, None] & dmask[None, :], cmask[:, None] & dmask[None, :]
            a_hi = tl.load(query_hi + a, mask=amask, other=0.0)
            b_hi = tl.load(hi + b, mask=bmask, other=0.0)
            acc = tl.dot(a_hi, tl.trans(b_hi), acc)
            if SPLIT:
                acc = tl.dot(a_hi, tl.trans(tl.load(lo + b, mask=bmask, other=0.0)), acc)
                acc = tl.dot(tl.load(query_lo + a, mask=amask, other=0.0), tl.trans(b_hi), acc)
        score = tl.load(norms + col, mask=cmask, other=float("inf"))[None, :] - 2.0 * acc
        score = tl.where((col[None, :] + centroid_begin == query[:, None]) | ~cmask[None, :], float("inf"), score)
        tl.store(tile_min + (row - row_offset).to(tl.int64) * tiles + tile, tl.min(score, axis=1), mask=rmask)


@triton.jit
def _nearest_refine_kernel(centroids, ids, count, tile_min, norms, peak, labels, k, d, tiles, row_offset,
                           grow, product, gamma_n, gamma_d, root_d, queries, query_norms, distances,
                           centroid_begin, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
                           BLOCK_T: tl.constexpr, EXTERNAL: tl.constexpr):
    # FP64 |q - c|^2 of the FP32 centroids over the tiles whose screened minimum is within
    # the band; every tile when the band or a screened minimum is not finite.
    row = row_offset + tl.program_id(0)
    if row < tl.load(count):
        query = tl.load(ids + row).to(tl.int64)
        lane = tl.arange(0, BLOCK_T)
        real = lane < tiles
        minima = tl.load(tile_min + tl.program_id(0).to(tl.int64) * tiles + lane, mask=real, other=0.0)
        # Screening error bound (see nearest_other); not finite: rescan every tile.
        qindex = row if EXTERNAL else query
        qn = tl.sqrt(tl.load(query_norms + qindex).to(tl.float64)) * grow
        pn = tl.sqrt(tl.load(peak).to(tl.float64)) * grow
        band = (2 * (product * qn * pn + gamma_n * pn * pn + 2.0**-22 * (pn * pn + 2 * qn * pn)) + 2 * gamma_d
                * (qn + pn) * (qn + pn) + 8 * d * 2.0**-126 + (qn + pn) * root_d * 2.0**-65 * 2.0**-66) * (1 + 2.0**-6)
        limit = tl.min(tl.where(real, minima, float("inf"))) + band
        broken = (minima != minima) | (tl.abs(minima) == float("inf")) | (limit != limit) | (limit == float("inf"))
        broken = tl.sum((broken & real).to(tl.int32)) > 0
        best = tl.full((), float("inf"), tl.float64)
        label = ((query == centroid_begin) & (k > 1)).to(tl.int64)
        if EXTERNAL:
            label = tl.where((k == 1) & (query == centroid_begin), 0x7fffffffffffffff, label)
        pending = real & (broken | ~(minima > limit))
        # Visit the band's tiles in ascending order; usually one or two.
        for _ in range(0, tl.sum(pending.to(tl.int32))):
            tile = tl.min(tl.where(pending, lane, BLOCK_T))
            pending = pending & (lane != tile)
            col = tile * BLOCK_N + tl.arange(0, BLOCK_N)
            cmask = col < k
            dist = tl.zeros((BLOCK_N,), tl.float64)
            for d0 in range(0, d, BLOCK_D):
                dim = d0 + tl.arange(0, BLOCK_D)
                dmask = dim < d
                q = tl.load(queries + qindex * d + dim, mask=dmask, other=0.0).to(tl.float64)
                diff = tl.load(centroids + col[:, None].to(tl.int64) * d + dim[None, :],
                               mask=cmask[:, None] & dmask[None, :], other=0.0).to(tl.float64) - q[None, :]
                dist += tl.sum(diff * diff, axis=1)
            dist = tl.where((col + centroid_begin == query) | ~cmask | (dist != dist), float("inf"), dist)
            tile_best, tile_label = tl.min(dist, axis=0, return_indices=True,
                                           return_indices_tie_break_left=True)
            if tile_best < best:
                best = tile_best
                label = (tile * BLOCK_N + tile_label).to(tl.int64)
        tl.store(labels + row, label)
        if EXTERNAL:
            tl.store(distances + row, best)


def _gamma(n):
    # Bound on the relative error of n chained FP32 operations, rounding either way.
    return n * 2.0**-23 / (1 - n * 2.0**-23)


def prepare_nearest_other(centroids):
    """Prepare a frozen centroid table once for several query chunks."""
    centroids = centroids.contiguous()
    k, d = centroids.shape
    hi = torch.empty(k, d, dtype=torch.bfloat16, device=centroids.device)
    lo = torch.empty_like(hi)
    norms = torch.empty(k, dtype=torch.float32, device=centroids.device)
    with torch.cuda.device(centroids.device):
        _nearest_operands_kernel[(k,)](centroids, hi, lo, norms, d,
                                       BLOCK_D=triton.next_power_of_2(d), num_warps=4)
    return centroids, hi, lo, norms, norms.max()


def nearest_other(centroids, ids, count, *, queries=None, centroid_begin=0, prepared=None):
    """Nearest other centroid of centroids[ids[:count]] by FP64 |q - c|^2 of the FP32 rows.

    BF16 tensor-core scores keep one minimum per tile of 128 centroids; a
    rigorous bound on their error selects the tiles that can hold the
    nearest, which are rescanned in FP64 direct differences (all tiles when
    a bound or score is not finite). Ties go to the lower index; count is a
    device scalar; entries past count are 0. No host reads. Workspace is
    O(k) per chunk of queries, never O(k^2).

    With external queries, ids are their global centroid IDs and centroid_begin
    locates the local candidate shard. Return FP64 distances and local IDs;
    a shard with only the query itself returns inf and INT64_MAX.
    """
    external = queries is not None
    k, d = centroids.shape
    centroids, hi, lo, norms, peak = prepare_nearest_other(centroids) if prepared is None else prepared
    tiles = triton.cdiv(k, 128)
    split = d >= 128  # Split operands keep the band narrow: one BF16 product's error grows with |q||c|.
    if external:
        queries = queries.contiguous()
        query_hi = torch.empty_like(queries, dtype=torch.bfloat16)
        query_lo = torch.empty_like(query_hi)
        query_norms = torch.empty(len(queries), device=centroids.device, dtype=torch.float32)
        with torch.cuda.device(centroids.device):
            _nearest_operands_kernel[(len(queries),)](
                queries, query_hi, query_lo, query_norms, d,
                BLOCK_D=triton.next_power_of_2(d), num_warps=4)
    else:
        queries, query_hi, query_lo, query_norms = centroids, hi, lo, norms
    # s = |c|^2 - 2 q.c is screened as fl(n~ - 2 acc); with u = 2^-8 (BF16) per operand,
    # products err by 2u + u^2 (split hi/lo pairs: 4u^2), the FP32 accumulation of the
    # products by gamma(terms), the norms by gamma(d), the subtraction by one rounding.
    # The rescan of D = |q - c|^2 errs by at most gamma(d + 4) D, D <= (|q| + max|c|)^2.
    # Values below FP32's normal range (2^-126) may flush to zero: an absolute term covers them.
    # Band (in the rescan kernel): 2 (P q p + gamma(d + 2) p^2 + 2^-22 (p^2 + 2 q p))  [P = product]
    # + 2 gamma(d + 4) (q + p)^2 + the underflow terms, with q = |q| and p = max|c| grown.
    u = 2.0**-8
    accumulate = _gamma((3 if split else 1) * d + 2)
    product = 2 * ((4 * u * u if split else 2 * u + u * u) * (1 + accumulate) + accumulate * (1 + u) ** 2)
    grow = (1 + _gamma(d + 2)) ** 0.5 * (1 + 2.0**-20)
    # Queries run in chunks so the per-tile minima stay O(k).
    chunk = max(128, min(triton.cdiv(len(ids), 128) * 128, (2**24 // tiles) // 128 * 128, 2**16))
    chunk = max(chunk, 4096) if len(ids) > 4096 else chunk
    tile_min = torch.empty((chunk, tiles), dtype=torch.float32, device=centroids.device)
    labels = torch.zeros(len(ids), dtype=torch.int64, device=centroids.device)
    distances = torch.full((len(ids),), torch.inf, dtype=torch.float64, device=centroids.device) if external else None
    block_d = 32 if split else 64 if d >= 64 else max(16, triton.next_power_of_2(d))
    stages = (1 if torch.cuda.get_device_capability(centroids.device)[0] >= 10 else 3) if split else 2
    programs = 4 * torch.cuda.get_device_properties(centroids.device).multi_processor_count
    with torch.cuda.device(centroids.device):
        for offset in range(0, len(ids), chunk):
            rows = min(chunk, len(ids) - offset)
            _nearest_screen_kernel[(programs,)](
                hi, lo, norms, ids, count, tile_min, k, d, tiles, offset, rows,
                query_hi, query_lo, centroid_begin, BLOCK_M=128, BLOCK_N=128,
                BLOCK_D=block_d, SPLIT=split, EXTERNAL=external, num_warps=8, num_stages=stages)
            _nearest_refine_kernel[(rows,)](
                centroids, ids, count, tile_min, norms, peak, labels, k, d, tiles, offset, grow, product,
                _gamma(d + 2), _gamma(d + 4), d**0.5, queries, query_norms, distances, centroid_begin,
                BLOCK_N=128, BLOCK_D=min(64, triton.next_power_of_2(d)),
                BLOCK_T=triton.next_power_of_2(tiles), EXTERNAL=external, num_warps=4)
    return (distances, labels) if external else labels


@triton.jit
def _removal_update_kernel(stats, batch, axes, alpha, decay, k, d, stats_stride, batch_stride,
                           ROWS: tl.constexpr, BLOCK_W: tl.constexpr, SCALARS: tl.constexpr):
    # [U | R] <- decay [U | R] + alpha [sum p y | sum d2 - d1], which is then zeroed for the next
    # batch; V <- U / |U| where |U| > 0.
    if SCALARS:
        alpha = tl.load(alpha)
        decay = tl.load(decay)
    j = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    col = tl.arange(0, BLOCK_W)
    live = (j < k)[:, None] & (col <= d)[None, :]
    s = stats + j[:, None] * stats_stride + col[None, :]
    b = batch + j[:, None] * batch_stride + col[None, :]
    value = tl.load(s, mask=live) * decay + alpha * tl.load(b, mask=live)
    tl.store(s, value, mask=live)
    tl.store(b, tl.zeros_like(value), mask=live)
    axis = tl.where(col[None, :] < d, value, 0.0).to(tl.float64)  # FP64: |U|^2 can overflow FP32
    norm = tl.sqrt(tl.sum(axis * axis, axis=1))
    tl.store(axes + j[:, None] * d + col[None, :], (axis / norm[:, None]).to(tl.float32),
             mask=live & (col < d)[None, :] & (norm > 0)[:, None])


def removal_update(stats, batch, axes, alpha, decay):
    """Fused removal-statistics EMA, zeroing batch: stats bitwise as the torch ops; V from an FP64 |U|."""
    if not (stats.is_cuda and stats.dtype == batch.dtype == axes.dtype == torch.float32
            and stats.stride(1) == batch.stride(1) == 1 and axes.is_contiguous() and stats.shape[1] <= 8193):
        return False
    block_w = triton.next_power_of_2(stats.shape[1])
    rows = max(1, 2048 // block_w)
    with torch.cuda.device(stats.device):
        _removal_update_kernel[(triton.cdiv(len(stats), rows),)](
            stats, batch, axes, alpha, decay, len(stats), axes.shape[1], stats.stride(0), batch.stride(0),
            ROWS=rows, BLOCK_W=block_w, SCALARS=isinstance(alpha, torch.Tensor), num_warps=4,
            enable_fp_fusion=False)
    torch.autograd.graph.increment_version((stats, batch, axes))
    return True


@triton.jit
def _removal_split_kernel(stats, axes, source, target, width, d, stats_stride,
                          moved, BLOCK_W: tl.constexpr, BLOCK_D: tl.constexpr, count=None):
    # Donor halves its removal statistics; the moved centroid copies them and the axis.
    pair = tl.program_id(0)
    live = True if count is None else pair < tl.load(count)
    s = tl.load(source + pair).to(tl.int64)
    t = tl.load(target + pair).to(tl.int64)
    col = tl.arange(0, BLOCK_W)
    half = tl.load(stats + s * stats_stride + col, mask=(col < width) & live) / 2
    tl.store(stats + s * stats_stride + col, half, mask=(col < width) & live)
    tl.store(stats + t * stats_stride + col, half, mask=(col < width) & live)
    dim = tl.arange(0, BLOCK_D)
    tl.store(axes + t * d + dim, tl.load(axes + s * d + dim, mask=(dim < d) & live), mask=(dim < d) & live)
    tl.store(moved + s, 1, mask=live)  # Both ends close from their EMA position.
    tl.store(moved + t, 1, mask=live)


def removal_split(stats, axes, source, target, count, moved):
    """stats[source] /= 2; stats[target] = stats[source]; axes[target] = axes[source] (disjoint ids).

    A device ``count`` limits the pairs to source[:count], target[:count] (None: all pairs).
    ``moved`` marks both ends of each pair."""
    if not (stats.is_cuda and stats.stride(1) == 1 and axes.is_contiguous()
            and stats.dtype == axes.dtype == torch.float32 and stats.shape[1] <= 8194):
        return False
    with torch.cuda.device(stats.device):
        _removal_split_kernel[(len(source),)](
            stats, axes, source, target, stats.shape[1], axes.shape[1], stats.stride(0),
            BLOCK_W=triton.next_power_of_2(stats.shape[1]), BLOCK_D=triton.next_power_of_2(axes.shape[1]),
            count=count, moved=moved, num_warps=4)
    torch.autograd.graph.increment_version((stats, axes))
    return True


@triton.jit
def _check_pair_kernel(nearest, same, position, count, order, donors, donor, splits, victims, chosen,
                       moves_out, kept, blocked, victim, targets, k, BLOCK: tl.constexpr):
    # One program: kept candidates, kept victims in order, donors that are not victims in
    # gain order, and the pair count.
    m = tl.load(count).to(tl.int32)
    zero = tl.sum(tl.zeros((BLOCK,), tl.int32))
    for b in range(0, k, BLOCK):
        i = b + tl.arange(0, BLOCK)
        # Preserve raw backward positions; refill on every call and graph replay.
        target = tl.load(position + tl.load(nearest + i, mask=i < m, other=0), mask=i < m, other=m)
        tl.store(targets + i, target, mask=i < m)
        tl.store(kept + i, (i < m).to(tl.int8), mask=i < m)
        tl.store(victim + i, tl.zeros((BLOCK,), tl.int8), mask=i < k)
    tl.debug_barrier()
    # Greedy conflicts in both directions: no kept candidate's nearest is also kept.
    # Each parallel round settles a prefix; after 8 rounds the rest runs sequentially (O(m)).
    changed = m > 0
    rounds = zero
    first = zero
    while changed & (rounds < 8):
        for b in range(0, m + 1, BLOCK):
            i = b + tl.arange(0, BLOCK)
            tl.store(blocked + i, tl.zeros((BLOCK,), tl.int8), mask=i <= m)
        tl.debug_barrier()
        for b in range(0, m, BLOCK):
            i = b + tl.arange(0, BLOCK)
            live = (i < m) & (tl.load(kept + i, mask=i < k, other=0) != 0)
            target = tl.load(targets + i, mask=i < m, other=m)
            # Include unkept active candidates, or a conflicting backedge can resurrect.
            earlier = (i < m) & (target < i) & (tl.load(same + i, mask=i < m, other=0) == 0)
            back = tl.load(kept + target, mask=earlier, other=0) != 0
            tl.store(blocked + i, tl.full((BLOCK,), 1, tl.int8), mask=earlier & back)
            tl.store(blocked + target, tl.full((BLOCK,), 1, tl.int8), mask=live & (target > i) & (target < m))
        tl.debug_barrier()
        low = m + zero
        for b in range(0, m, BLOCK):
            i = b + tl.arange(0, BLOCK)
            was = tl.load(kept + i, mask=i < m, other=0) != 0
            fresh = tl.load(blocked + i, mask=i < m, other=1) == 0
            low = tl.minimum(low, tl.min(tl.where((was != fresh) & (i < m), i, m)))
        tl.debug_barrier()
        changed = low < m
        if changed:
            for b in range(0, m, BLOCK):
                i = b + tl.arange(0, BLOCK)
                fresh = (tl.load(blocked + i, mask=i < k, other=1) == 0) & (i < m)
                tl.store(kept + i, fresh.to(tl.int8), mask=i < k)
        tl.debug_barrier()
        rounds += 1
        first = low + 1  # The first change's parents did not change: it and all before are final.
    if changed:
        for b in range(0, m + 1, BLOCK):
            i = b + tl.arange(0, BLOCK)
            tl.store(blocked + i, tl.zeros((BLOCK,), tl.int8), mask=i <= m)
        tl.debug_barrier()
        for b in range(0, first, BLOCK):
            i = b + tl.arange(0, BLOCK)
            live = (i < first) & (tl.load(kept + i, mask=i < k, other=0) != 0)
            target = tl.load(targets + i, mask=i < first, other=m)
            tl.store(blocked + target, tl.full((BLOCK,), 1, tl.int8), mask=live & (target > i) & (target < m))
        tl.debug_barrier()
        for p in range(first, m):
            target = tl.load(targets + p)
            free = tl.load(blocked + p) == 0
            if free and target < p and tl.load(same + p) == 0:
                free = tl.load(kept + target) == 0
            tl.store(kept + p, free.to(tl.int8))
            if free and target > p and target < m:
                tl.store(blocked + target, 1)
        tl.debug_barrier()
    taken = zero
    for b in range(0, m, BLOCK):
        i = b + tl.arange(0, BLOCK)
        live = (i < m) & (tl.load(kept + i, mask=i < k, other=0) != 0)
        c = tl.load(order + i, mask=live, other=0)
        tl.store(victims + taken + tl.cumsum(live.to(tl.int32), 0) - 1, c, mask=live)
        tl.store(victim + c, tl.full((BLOCK,), 1, tl.int8), mask=live)
        taken += tl.sum(live.to(tl.int32))
    tl.debug_barrier()
    donated = zero
    for b in range(0, k, BLOCK):
        i = b + tl.arange(0, BLOCK)
        valid = i < k
        c = tl.load(donors + i, mask=valid, other=0)
        live = valid & (tl.load(donor + c, mask=valid, other=0) != 0)
        live = live & (tl.load(victim + c, mask=live, other=1) == 0)
        tl.store(chosen + donated + tl.cumsum(live.to(tl.int32), 0) - 1, c, mask=live)
        donated += tl.sum(live.to(tl.int32))
    tl.debug_barrier()
    moves = tl.minimum(taken, donated)
    tl.store(moves_out, moves)
    tl.store(splits, tl.load(splits) + moves)


def check_pair(nearest, position, count, order, donors, donor, splits, centroids=None):
    """Greedy victims with no nearest also a victim (but an exact duplicate), their donors and count (one CTA)."""
    k = len(order)
    same = torch.zeros(k, dtype=torch.int8, device=order.device)  # Equal rows, by chunks of 2^16 candidates.
    for s in range(0, 0 if centroids is None else k, 1 << 16):
        same[s:s + (1 << 16)] = (centroids[order[s:s + (1 << 16)]] == centroids[nearest[s:s + (1 << 16)]]).all(1)
    victims, chosen = torch.empty_like(order), torch.empty_like(order)
    moves = torch.empty((), dtype=torch.int64, device=order.device)
    kept = torch.empty(k, dtype=torch.int8, device=order.device)
    blocked = torch.empty(k + 1, dtype=torch.int8, device=order.device)
    victim = torch.empty(k, dtype=torch.int8, device=order.device)
    targets = torch.empty_like(position)
    with torch.cuda.device(order.device):
        _check_pair_kernel[(1,)](nearest, same, position, count, order, donors, donor, splits, victims, chosen,
                                 moves, kept, blocked, victim, targets, k, BLOCK=2048, num_warps=16)
    return victims, chosen, moves


@triton.jit
def _long_pieces(rows, order, offsets, sorted_labels, partial, counter, stats, d, n, row_stride, stats_stride,
                 BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr, LONG: tl.constexpr,
                 distance=None, second=None, axes=None, centroids=None, removal=None, removal_stride=0,
                 alpha=0.0, decay=0.0, REMOVAL: tl.constexpr = False, EMA: tl.constexpr = False):
    # Program q takes the pieces of segments longer than LONG in label-sorted rows q LONG..(q + 1) LONG (the
    # first and last segments there); the last of a segment's pieces adds them up in row order and flushes them
    # as the per-centroid programs would: deterministic.
    # REMOVAL: also [p y | d2 - d1] to removal, as _removal_stats_kernel.
    # With EMA: removal_update's EMA of removal instead.
    q = tl.program_id(0).to(tl.int64)
    lo = q * LONG
    hi = tl.minimum(lo + LONG, n)
    col = tl.arange(0, BLOCK_D)
    cmask = col < d
    width = 2 * BLOCK_D + 1 if REMOVAL else BLOCK_D  # partial sums per piece
    first = tl.load(sorted_labels + lo)
    for side in tl.static_range(2):  # the chunk's first segment, then its last if another one
        j = tl.load(sorted_labels + lo + side * (hi - 1 - lo))
        start = tl.load(offsets + j)
        end = tl.load(offsets + j + 1)
        j = j.to(tl.int64)
        if (end - start > LONG) & ((side == 0) | (j != first)):
            total = tl.zeros((BLOCK_D,), tl.float32)
            if REMOVAL:
                axis = tl.load(axes + j * d + col, mask=cmask, other=0.0)
                center = tl.load(centroids + j * d + col, mask=cmask, other=0.0)
                oja = tl.zeros((BLOCK_D,), tl.float32)
                margin = tl.zeros((BLOCK_R,), tl.float32)
            for begin in range(tl.maximum(start, lo), tl.minimum(end, hi), BLOCK_R):
                lane = begin + tl.arange(0, BLOCK_R)
                rmask = lane < tl.minimum(end, hi)
                row = tl.load(order + lane, mask=rmask, other=0).to(tl.int64)
                value = tl.load(rows + row[:, None] * row_stride + col[None, :],
                                mask=rmask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
                total += tl.sum(value, axis=0)
                if REMOVAL:
                    y = tl.where(rmask[:, None] & cmask[None, :], value - center[None, :], 0.0)
                    oja += tl.sum(tl.sum(y * axis[None, :], axis=1)[:, None] * y, axis=0)
                    gap = tl.load(second + row, mask=rmask, other=0.0) - tl.load(distance + row, mask=rmask, other=0.0)
                    gap = tl.where(gap == float("-inf"), -3.4028234663852886e38, gap)  # as nan_to_num(0, 0)
                    margin += tl.where(rmask & (gap == gap) & (gap < float("inf")), gap, 0.0)
            # Piece slots: 2 p for chunks p = start // LONG..(end - 1) // LONG, 2 p + 1 if it starts inside chunk p.
            slot = partial + (q * 2 + (start > lo).to(tl.int64)) * width
            tl.store(slot + col, total)
            if REMOVAL:
                tl.store(slot + BLOCK_D + col, oja)
                tl.store(slot + 2 * BLOCK_D, tl.sum(margin))
            # This condition is CTA-uniform: publish only after every warp's stores.
            tl.debug_barrier()
            if tl.atomic_add(counter + j, 1) == (end - 1) // LONG - start // LONG:  # the last piece
                total = tl.zeros((BLOCK_D,), tl.float32)
                if REMOVAL:
                    oja = tl.zeros((BLOCK_D,), tl.float32)
                    gaps = 0.0
                for p in range(start // LONG, (end - 1) // LONG + 1):
                    slot = partial + (p * 2 + (start > p * LONG).to(tl.int64)) * width
                    total += tl.load(slot + col)
                    if REMOVAL:
                        oja += tl.load(slot + BLOCK_D + col)
                        gaps += tl.load(slot + 2 * BLOCK_D)
                out = stats + j * stats_stride
                tl.store(out + col, tl.load(out + col, mask=cmask) + total, mask=cmask)
                tl.store(out + d, tl.load(out + d) + (end - start).to(tl.float32))
                if REMOVAL:
                    out = removal + j * removal_stride
                    if EMA:
                        u = _add_rn(_mul_rn(tl.load(out + col, mask=cmask, other=0.0), decay),
                                    _mul_rn(alpha, 0.0 + oja))
                        tl.store(out + col, u, mask=cmask)
                        tl.store(out + d, _add_rn(_mul_rn(tl.load(out + d), decay), _mul_rn(alpha, 0.0 + gaps)))
                        u = tl.where(cmask, u, 0.0).to(tl.float64)  # FP64: |U|^2 can overflow FP32
                        norm = tl.sqrt(tl.sum(u * u))
                        tl.store(axes + j * d + col, (u / norm).to(tl.float32), mask=cmask & (norm > 0))
                    else:
                        tl.store(out + col, tl.load(out + col, mask=cmask) + oja, mask=cmask)
                        tl.store(out + d, tl.load(out + d) + gaps)


@triton.jit
def _long_pieces_kernel(rows, order, offsets, sorted_labels, partial, counter, distance, second, axes, centroids,
                        stats, removal, d, n, row_stride, stats_stride, removal_stride, alpha, decay,
                        BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr, LONG: tl.constexpr, EMA: tl.constexpr,
                        SCALARS: tl.constexpr):
    # accumulate_removal's long segments, in a launch of their own: inside _removal_stats_kernel, _long_pieces'
    # registers would cost the per-centroid programs occupancy.
    if EMA:
        if SCALARS:
            alpha = tl.load(alpha)
            decay = tl.load(decay)
    _long_pieces(rows, order, offsets, sorted_labels, partial, counter, stats, d, n, row_stride, stats_stride, BLOCK_R,
                 BLOCK_D, LONG, distance, second, axes, centroids, removal, removal_stride, alpha, decay, True, EMA)


_LONG_SEGMENT = 256  # rows; longer segments are summed in pieces (no single-program tail)


def _pieces(rows, stats, width):
    """(number of LONG-row pieces, their partial sums, zeroed per-centroid piece counters of this call)."""
    pieces = triton.cdiv(len(rows), _LONG_SEGMENT)
    return pieces, stats.new_empty(2 * pieces, width), torch.zeros(len(stats), dtype=torch.int32, device=rows.device)


@triton.jit
def _segment_sums_kernel(rows, order, offsets, sorted_labels, partial, counter, stats, d, n, pieces, row_stride,
                         stats_stride, BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr, LONG: tl.constexpr):
    # Program pieces + j: centroid j owns its segment of label-sorted rows (one read of each row, no atomics)
    # unless it has more than LONG rows: programs < pieces sum those (_long_pieces).
    if tl.program_id(0) < pieces:
        _long_pieces(rows, order, offsets, sorted_labels, partial, counter, stats, d, n, row_stride, stats_stride,
                     BLOCK_R, BLOCK_D, LONG)
        return
    j = (tl.program_id(0) - pieces).to(tl.int64)
    start = tl.load(offsets + j)
    end = tl.load(offsets + j + 1)
    if (start < end) & (end - start <= LONG):
        col = tl.arange(0, BLOCK_D)
        total = tl.zeros((BLOCK_D,), tl.float32)
        for begin in range(start, end, BLOCK_R):
            lane = begin + tl.arange(0, BLOCK_R)
            row = tl.load(order + lane, mask=lane < end, other=0).to(tl.int64)
            total += tl.sum(tl.load(rows + row[:, None] * row_stride + col[None, :],
                                    mask=(lane < end)[:, None] & (col < d)[None, :], other=0.0).to(tl.float32), 0)
        out = stats + j * stats_stride
        tl.store(out + col, tl.load(out + col, mask=col < d) + total, mask=col < d)
        tl.store(out + d, tl.load(out + d) + (end - start).to(tl.float32))


def accumulate_segments(rows, labels, stats):
    """Add [x | 1] rows to FP32 stats by label, deterministically: rows sorted by label."""
    k, width = stats.shape
    if (not rows.is_cuda or stats.dtype != torch.float32 or not rows.dtype.is_floating_point
            or rows.stride(1) != 1 or stats.stride(1) != 1 or width != rows.shape[1] + 1 or width > 4097):
        return False
    dtype = torch.int16 if k < 2**15 else torch.int32  # positions run to k inclusive
    sorted_labels, order = torch.sort(labels.to(dtype))
    offsets = torch.searchsorted(sorted_labels, torch.arange(k + 1, device=rows.device, dtype=dtype))
    block_d = triton.next_power_of_2(rows.shape[1])
    # Short segments are latency-bound: small programs keep more of them in flight.
    block_r, warps = ((16, 4) if len(rows) >= 8 * k else (4, 1) if block_d <= 128
                      else (2, 2 if block_d <= 512 else 4))
    pieces, partial, counter = _pieces(rows, stats, block_d)
    with torch.cuda.device(rows.device):
        _segment_sums_kernel[(pieces + k,)](
            rows, order, offsets, sorted_labels, partial, counter, stats, rows.shape[1], len(rows), pieces,
            rows.stride(0), stats.stride(0), BLOCK_R=block_r, BLOCK_D=block_d, LONG=_LONG_SEGMENT, num_warps=warps)
    torch.autograd.graph.increment_version(stats)
    return True
