# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#
# Selection/recovery adapted from cute_kmeans/cute/fitted_grouped.py and
# grouped_recovery.py at f858b07bb23ff8b86af5c6fadee242ff35deef8d.

"""Private SM90 tile selection and grouped recovery, with owned scan reuse.

Plans are explicit preparation metadata; this module does not choose routes.
The shared kernels retain BF16/FP8/INT8 score structure. BF16 L2 is the initial
integration qualification; other representations are not newly qualified here.
"""
from typing import NamedTuple

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, const_expr
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import LayoutEnum
import cutlass.utils.hopper_helpers as hopper_utils
import torch
from quack import layout_utils, sm90_utils

from .e_means_cute_argmin import require_tensor
from .e_means_cute_row_owned import (
    TILE_N, _RowOwnedAssignSm90, _assert_candidate_order,
    _required_smem, _row_owned_shape_legal, _validate_operand,
)

_RECOVERY_M = 256
_RECOVERY_THREADS = 256
_COMPILED = {}


class _TileSelectSm90(_RowOwnedAssignSm90):
    def __init__(self, operand_type, tile_k, tile_m, stages):
        # Integer selection carries raw keys, rather than packed centroid IDs.
        super().__init__(operand_type, tile_k, tile_m, stages,
                         1 if tile_m == 64 else 2)
        self.score_type = Float32

    @cute.jit
    def fold(self, accumulator, n_smem, coordinates, row_min, row_id, block, stage):
        values = layout_utils.reshape_acc_to_mn(accumulator)
        fragment = cute.make_rmem_tensor_like(values[0, None])
        for column in cutlass.range_constexpr(cute.size(values, mode=[1])):
            fragment[column] = n_smem[coordinates[0, column][1], stage]
        norms = fragment.load()
        if const_expr(self.score_type is Int32):
            for row in cutlass.range_constexpr(cute.size(row_min)):
                best = Int32(0x7FFFFFFF)
                for column in cutlass.range_constexpr(cute.size(values, mode=[1])):
                    score = norms[column] - Int32(2) * values[row, column]
                    best = score if score < best else best
                take = best < row_min[row]
                row_min[row] = best if take else row_min[row]
                row_id[row] = block if take else row_id[row]
        else:
            scales = cute.make_rmem_tensor_like(values[0, None])
            scales.fill(Float32(-2.0))
            for row in cutlass.range_constexpr(cute.size(row_min)):
                scores = cute.math.fma(scales.load(), values[row, None].load(), norms)
                # minimumNumber skips NaNs; no FTZ preserves subnormal order.
                # A balanced tree limits the local dependency depth.
                width = const_expr(cute.size(values, mode=[1]))
                if const_expr(width < 1 or width > TILE_N or width & (width - 1)):
                    raise RuntimeError("row-owned score fragment must have power-of-two width")
                minima = cute.make_rmem_tensor((width,), Float32)
                for column in cutlass.range_constexpr(width):
                    minima[column] = scores[column]
                for distance in (1, 2, 4, 8, 16, 32, 64):
                    if const_expr(distance < width):
                        for column in cutlass.range_constexpr(0, width, 2 * distance):
                            minima[column] = cute.arch.fmin(
                                minima[column], minima[column + distance], nan=False, ftz=False)
                # All-NaN lanes become +Inf, so strict updates retain valid ID0.
                best = cute.arch.fmin(Float32.inf, minima[0], nan=False, ftz=False)
                take = best < row_min[row]
                row_min[row] = best if take else row_min[row]
                row_id[row] = block if take else row_id[row]

    @cute.jit
    def store(self, q, output_values, output_labels,
              coordinates, row_min, row_id):
        row_block, _, _ = cute.arch.block_idx()
        if cute.arch.lane_idx() % 4 == 0:
            for local in cutlass.range_constexpr(cute.size(row_id)):
                row = row_block * self.tile_m + coordinates[local, 0][0]
                if row < q.shape[0]:
                    output_labels[row] = row_id[local]


class _RecoverTilesSm90(_TileSelectSm90):
    """Gather fixed-size query groups and resolve their selected centroid tile."""
    def __init__(self, operand_type, logical_d, tile_k):
        super().__init__(operand_type, tile_k, _RECOVERY_M, 1)
        self.logical_d = logical_d

    @cute.jit
    def __call__(self, rows, centroids, norms, order, offsets, codes, values, stream):
        # Validated base/pitches are16-byte aligned; retain that fact after
        # dynamic row indexing without asserting divisibility of logical d.
        alignment = 128 // self.operand_type.width
        rows = cute.make_tensor(rows.iterator, cute.make_layout(rows.shape, stride=(
            cute.assume(rows.stride[0], divby=alignment), rows.stride[1])))
        centroids = cute.make_tensor(centroids.iterator, cute.make_layout(centroids.shape, stride=(
            cute.assume(centroids.stride[0], divby=alignment), centroids.stride[1])))
        mma = hopper_utils.make_trivial_tiled_mma(
            self.operand_type, self.operand_type,
            cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.K,
            self.score_type, atom_layout_mnk=(2, 1, 1), tiler_mn=(64, TILE_N),
        )
        q_layout = sm90_utils.make_smem_layout(
            self.operand_type, LayoutEnum.ROW_MAJOR, (_RECOVERY_M, self.tile_k), None)
        c_layout = sm90_utils.make_smem_layout(
            self.operand_type, LayoutEnum.ROW_MAJOR, (TILE_N, self.tile_k), None)
        n_layout = cute.make_layout(TILE_N)

        @cute.struct
        class Storage:
            q: cute.struct.Align[cute.struct.MemRange[self.operand_type, cute.cosize(q_layout)], 1024]
            c: cute.struct.Align[cute.struct.MemRange[self.operand_type, cute.cosize(c_layout)], 1024]
            n: cute.struct.Align[cute.struct.MemRange[self.score_type, TILE_N], 128]

        self.kernel(rows, centroids, norms, order, offsets, codes, values,
                    mma, q_layout, c_layout, n_layout, Storage).launch(
            grid=(centroids.shape[0] // TILE_N, 1, 1),
            block=(_RECOVERY_THREADS, 1, 1), stream=stream, min_blocks_per_mp=1)

    @cute.kernel
    def kernel(self, rows, centroids, norms, order, offsets, codes, values,
               mma, q_layout, c_layout, n_layout, Storage: cutlass.Constexpr):
        tile, _, _ = cute.arch.block_idx()
        begin, end = Int32(offsets[tile]), Int32(offsets[tile + 1])
        thread, _, _ = cute.arch.thread_idx()
        storage = cutlass.utils.SmemAllocator().allocate(Storage)
        q_smem = storage.q.get_tensor(q_layout.outer, swizzle=q_layout.inner)
        c_smem = storage.c.get_tensor(c_layout.outer, swizzle=c_layout.inner)
        n_smem = storage.n.get_tensor(n_layout)
        vector_width = 128 // self.operand_type.width
        vectors = self.logical_d // vector_width
        tail = vectors * vector_width
        copy = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.operand_type,
                                   num_bits_per_copy=128)
        zero = cute.make_rmem_tensor((vector_width,), self.operand_type)
        zero.fill(self.operand_type(0))
        if const_expr(vectors > 0):
            linear = thread
            while linear < TILE_N * vectors:
                row, vector = linear // vectors, linear % vectors
                source = cute.local_tile(centroids[tile * TILE_N + row, None],
                                         (vector_width,), (vector,))
                target = cute.local_tile(c_smem[row, None], (vector_width,), (vector,))
                cute.copy(copy, source, target)
                linear += _RECOVERY_THREADS
        if const_expr(tail < self.tile_k):
            linear = thread
            while linear < TILE_N * (self.tile_k - tail):
                row = linear // (self.tile_k - tail)
                column = tail + linear % (self.tile_k - tail)
                if column < self.logical_d:
                    c_smem[row, column] = centroids[tile * TILE_N + row, column]
                else:
                    c_smem[row, column] = self.operand_type(0)
                linear += _RECOVERY_THREADS
        if thread < TILE_N:
            n_smem[thread] = norms[tile * TILE_N + thread]
        cute.arch.fence_view_async_shared()
        cute.arch.sync_threads()

        warpgroup = cute.arch.make_warp_uniform(thread // 128)
        group_mma = mma.get_slice(cute.make_layout(2, stride=128)(warpgroup))
        _, q_fragment, c_fragment = sm90_utils.partition_fragment_ABC(
            group_mma, (_RECOVERY_M, TILE_N, self.tile_k), q_smem, c_smem)
        coordinates = layout_utils.reshape_acc_to_mn(mma.get_slice(thread).partition_C(
            cute.make_identity_tensor((_RECOVERY_M, TILE_N))))
        _assert_candidate_order(coordinates, mma, _RECOVERY_M)
        position = begin
        while position < end:
            if const_expr(vectors > 0):
                linear = thread
                while linear < _RECOVERY_M * vectors:
                    row, vector = linear // vectors, linear % vectors
                    target = cute.local_tile(q_smem[row, None], (vector_width,), (vector,))
                    if position + row < end:
                        source = cute.local_tile(rows[order[position + row], None],
                                                 (vector_width,), (vector,))
                        cute.copy(copy, source, target)
                    else:
                        cute.copy(copy, zero, target)
                    linear += _RECOVERY_THREADS
            if const_expr(tail < self.tile_k):
                linear = thread
                while linear < _RECOVERY_M * (self.tile_k - tail):
                    row = linear // (self.tile_k - tail)
                    column = tail + linear % (self.tile_k - tail)
                    if position + row < end and column < self.logical_d:
                        q_smem[row, column] = rows[order[position + row], column]
                    else:
                        q_smem[row, column] = self.operand_type(0)
                    linear += _RECOVERY_THREADS
            cute.arch.fence_view_async_shared()
            cute.arch.sync_threads()
            accumulator = cute.make_rmem_tensor(
                mma.partition_shape_C((_RECOVERY_M, TILE_N)), self.score_type)
            self.gemm(mma, accumulator, q_fragment, c_fragment, zero_init=True, wg_wait=0)
            acc = layout_utils.reshape_acc_to_mn(accumulator)
            best = cute.make_rmem_tensor((cute.size(coordinates, mode=[0]),), self.score_type)
            ids = cute.make_rmem_tensor_like(best, Int32)
            best.fill(0x7FFFFFFF if const_expr(self.score_type is Int32) else Float32.inf)
            # The native fallback sentinel is centroid0, never INT32_MAX.
            ids.fill(0)
            for column in cutlass.range_constexpr(cute.size(acc, mode=[1])):
                local = coordinates[0, column][1]
                centroid = tile * TILE_N + local
                norm = n_smem[local]
                for row in cutlass.range_constexpr(cute.size(best)):
                    score = (norm - Int32(2) * acc[row, column]
                             if const_expr(self.score_type is Int32)
                             else cute.math.fma(Float32(-2.0), acc[row, column], norm))
                    take = (score < best[row]) | ((score == best[row]) & (centroid < ids[row]))
                    best[row] = score if take else best[row]
                    ids[row] = centroid if take else ids[row]
            for offset in (2, 1):
                for row in cutlass.range_constexpr(cute.size(best)):
                    other = cute.arch.shuffle_sync_bfly(best[row], offset=offset)
                    other_id = cute.arch.shuffle_sync_bfly(ids[row], offset=offset)
                    take = (other < best[row]) | ((other == best[row]) & (other_id < ids[row]))
                    best[row] = other if take else best[row]
                    ids[row] = other_id if take else ids[row]
            if cute.arch.lane_idx() % 4 == 0:
                for row in cutlass.range_constexpr(cute.size(best)):
                    entry = position + coordinates[row, 0][0]
                    if entry < end:
                        codes[order[entry]] = ids[row]
                        values[order[entry]] = best[row]
            cute.arch.sync_threads()
            position += _RECOVERY_M


class PreparedGrouped(NamedTuple):
    selection: object
    recovery: object
    rows: torch.Tensor
    centroids: torch.Tensor
    norms: torch.Tensor
    codes: torch.Tensor
    values: torch.Tensor
    labels: torch.Tensor
    grouping: object

    @property
    def workspace_bytes(self):
        return self.codes.numel() * self.codes.element_size() + self.grouping.workspace_bytes

    @property
    def grouping_provider(self):
        return self.grouping.provider


def _plan_fits_smem(dtype, plan, smem):
    tile_k, tile_m, stages = plan
    recovery_smem = (_RECOVERY_M + TILE_N) * tile_k * dtype.itemsize + TILE_N * 4 + 2048
    return max(_required_smem(tile_k, tile_m, stages, dtype.itemsize), recovery_smem) <= smem


def _validate_plan(batch, clusters, dim, dtype, plan, smem):
    if dtype is not torch.bfloat16:
        raise ValueError("grouped operands require BF16")
    if (not isinstance(plan, tuple) or len(plan) != 3
            or not all(type(value) is int for value in plan)):
        raise ValueError('grouped plan requires explicit (tile_k, tile_m, stages)')
    tile_k, tile_m, stages = plan
    if (not _row_owned_shape_legal(batch, clusters, dim)
            or tile_k < dim or tile_k % (16 if dtype is torch.bfloat16 else 32)
            or tile_m not in (64, 128, 256) or stages not in (2, 3)):
        raise ValueError('grouped plan does not cover legal operands')
    if not _plan_fits_smem(dtype, plan, smem):
        raise ValueError('grouped plan exceeds opt-in shared memory')


def _compile(kind, operation, operands, key):
    key = (kind,) + key
    compiled = _COMPILED.get(key)
    if compiled is None:
        tensors = tuple(None if value is None else from_dlpack(
            value, assumed_align=16 if value.ndim == 2 or index == 2 else value.element_size())
            for index, value in enumerate(operands))
        for index in (0, 1):
            tensors = tensors[:index] + (tensors[index].mark_layout_dynamic(leading_dim=1),) + tensors[index + 1:]
        compiled = cute.compile(operation, *tensors,
                                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                                options='--enable-tvm-ffi')
        _COMPILED[key] = compiled
        while len(_COMPILED) > 8:
            _COMPILED.pop(next(iter(_COMPILED)))
    return compiled


@torch.no_grad()
def prepare_grouped(rows, prepared, values, labels, plan):
    """Bind an explicitly planned raw-key path; retain all fixed grouping buffers."""
    from .e_means_cute import _require_stack
    from .e_means_grouping import prepare_grouping

    _require_stack()
    centroids, norms = prepared.centroids, prepared.norms
    operand_type = BFloat16
    _validate_operand(rows, 'grouped rows', rows.device)
    _validate_operand(centroids, 'grouped centroids', rows.device)
    if prepared.spherical or centroids.dtype is not rows.dtype or rows.shape[1] != centroids.shape[1]:
        raise ValueError('grouped L2 requires matching operand dtypes and logical widths')
    score_dtype = torch.int32 if rows.dtype is torch.int8 else torch.float32
    for value, name, dtype, shape in (
        (norms, 'centroid norms', score_dtype, (len(centroids),)),
        (values, 'raw keys', score_dtype, (len(rows),)),
        (labels, 'labels', torch.int64, (len(rows),)),
    ):
        require_tensor(value, name, dtype=dtype, shape=shape, device=rows.device)
        if value.data_ptr() % (16 if name == 'centroid norms' else value.element_size()):
            raise ValueError(name + ' pointer alignment is invalid')
    for value in (rows, centroids):
        if len(value) * value.stride(0) >= 2**31:
            raise ValueError('grouped operand storage must fit signed int32')
    properties = torch.cuda.get_device_properties(rows.device)
    if (properties.major, properties.minor) != (9, 0):
        raise RuntimeError('owned grouped kernels require exactly SM90')
    _validate_plan(len(rows), len(centroids), rows.shape[1], rows.dtype, plan,
                   properties.shared_memory_per_block_optin)
    tile_k, tile_m, stages = plan
    with torch.cuda.device(rows.device):
        codes = torch.empty(len(rows), dtype=torch.int32, device=rows.device)
        grouping = prepare_grouping(codes, len(centroids) // TILE_N)
        require_tensor(grouping.order, 'group order', dtype=torch.int32,
                       shape=(len(rows),), device=rows.device)
        require_tensor(grouping.offsets, 'group offsets', dtype=torch.uint64,
                       shape=(len(centroids) // TILE_N + 1,), device=rows.device)
        key = (rows.dtype, plan, tuple(rows.shape), tuple(centroids.shape),
               rows.stride(0), centroids.stride(0), rows.device.index)
        selection = _compile('select', _TileSelectSm90(operand_type, *plan),
                             (rows, centroids, norms, None, codes), key)
        recovery = _compile('recover', _RecoverTilesSm90(operand_type, rows.shape[1], tile_k),
                            (rows, centroids, norms, grouping.order, grouping.offsets, codes, values), key)
    return PreparedGrouped(selection, recovery, rows, centroids, norms,
                           codes, values, labels, grouping)


@torch.no_grad()
def launch_grouped(prepared, labels_out=None):
    from .e_means_grouping import launch_grouping

    labels = prepared.labels if labels_out is None else labels_out
    require_tensor(labels, 'labels', dtype=torch.int64,
                   shape=prepared.labels.shape, device=prepared.labels.device)
    with torch.cuda.device(prepared.rows.device):
        prepared.selection(prepared.rows, prepared.centroids, prepared.norms,
                           None, prepared.codes)
        launch_grouping(prepared.grouping)
        prepared.recovery(prepared.rows, prepared.centroids, prepared.norms,
                          prepared.grouping.order, prepared.grouping.offsets,
                          prepared.codes, prepared.values)
        labels.copy_(prepared.codes)
