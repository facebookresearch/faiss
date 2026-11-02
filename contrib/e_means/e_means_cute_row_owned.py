# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#
# Adapted from PR #280 cute_kmeans/cute/row_owned.py at
# e8e288285e885064b477a882e237fcc47da8a9b2.

"""Low-dimension SM90 floating and integer row-owned assignment kernel.

Floating operands rank FP32 scores; native bytes rank exact INT32 scores and
return reconstructed FP32 distances. A compile-time layout proof establishes
increasing centroid visitation order, and the cross-lane reduction explicitly
breaks ties by centroid ID.
"""

from typing import NamedTuple

import cutlass
import cutlass.cute as cute
from cutlass import (
    BFloat16,
    Float8E4M3FN,
    Float32,
    Int8,
    Int32,
    Int64,
    const_expr,
    pipeline,
)
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils import LayoutEnum
import cutlass.utils.hopper_helpers as hopper_utils
import torch

from quack import copy_utils, layout_utils, pipeline as quack_pipeline
from quack import sm90_utils

from .e_means_cute_argmin import require_tensor


TILE_N = 128
LOCAL_ID_BITS = 7
_MAX_SCORE_TERM = 48_896  # max(c*c - 2*x*c), x/c in [-128,127]; P06 bound.
_PRODUCER_THREADS = 128
_WARPGROUP_THREADS = 128
_SMEM_OVERHEAD = 2048


def _assert_candidate_order(coordinates, tiled_mma, tile_m):
    """Prove this geometry's row slots visit strictly increasing centroid IDs.

    Adapted from ``e_means_cute_kernel._assert_candidate_order`` for the
    row-owned accumulator. Thread zero's slice carries the whole geometry:
    its layout is asserted equal to the compiled thread's, so a real thread's
    candidate is thread zero's plus one offset that is constant over the
    fragment. Two properties therefore transfer to every thread.

    Each slot keeps one row, and all slots see the same centroid per column.
    Each slot's centroid IDs also strictly increase, so PR #280's
    ``score < row_min`` retains the lowest ID without a per-candidate tie
    comparison. Centroid blocks are scanned in increasing order, extending
    that property to the full scan. The cross-lane butterfly still breaks
    ties explicitly by ID.
    """
    static = layout_utils.reshape_acc_to_mn(
        tiled_mma.get_slice(0).partition_C(
            cute.make_identity_tensor((tile_m, TILE_N))
        )
    )
    if str(static.layout) != str(coordinates.layout):
        raise RuntimeError("row-owned candidates are thread dependent")
    for row in range(cute.size(static, mode=[0])):
        for column in range(cute.size(static, mode=[1])):
            slot, candidate = static[row, column]
            if type(slot) is not int or type(candidate) is not int:
                raise RuntimeError("row-owned candidates are not static")
            if slot != static[row, 0][0]:
                raise RuntimeError("a row-owned slot spans several rows")
            if candidate != static[0, column][1]:
                raise RuntimeError("row-owned slots disagree on candidates")
            if column and candidate <= static[row, column - 1][1]:
                raise RuntimeError("centroid candidates are not ordered by ID")


def _row_owned_shape_legal(batch, clusters, dim, packed_integer=False):
    return (
        batch >= 1
        and clusters >= TILE_N
        and clusters % TILE_N == 0
        and dim >= 1
        and (not packed_integer or _MAX_SCORE_TERM * dim * TILE_N + TILE_N - 1 < 2**31)
    )


def _row_owned_tile_k(dim):
    return max(128, 64 * _ceil_div(dim, 64))


class _RowOwnedAssignSm90:
    """Compute exact row minima without materializing B-by-K scores."""

    def __init__(
        self,
        operand_type,
        packed_integer,
        tile_k,
        tile_m,
        stages,
        mma_warpgroups,
    ):
        self.operand_type = operand_type
        self.score_type = Int32 if packed_integer else Float32
        self.packed_integer = packed_integer
        self.tile_k = tile_k
        self.tile_m = tile_m
        self.stages = stages
        self.mma_warpgroups = mma_warpgroups

    @cute.jit
    def __call__(
        self, q, centroids, norms, row_norms, values, labels, stream
    ):
        tiled_mma = hopper_utils.make_trivial_tiled_mma(
            self.operand_type,
            self.operand_type,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
            self.score_type,
            atom_layout_mnk=(self.mma_warpgroups, 1, 1),
            tiler_mn=(64, TILE_N),
        )
        # FP8 uses the same per-128-coordinate FP32 promotion as stock QuACK.
        # The synchronous centroid-block schedule follows.
        q_layout = sm90_utils.make_smem_layout(
            self.operand_type,
            LayoutEnum.ROW_MAJOR,
            (self.tile_m, self.tile_k),
            None,
        )
        c_layout = sm90_utils.make_smem_layout(
            self.operand_type,
            LayoutEnum.ROW_MAJOR,
            (TILE_N, self.tile_k),
            self.stages,
        )
        n_layout = cute.make_layout(
            (TILE_N, self.stages), stride=(1, TILE_N)
        )
        q_bytes = cute.size_in_bytes(
            self.operand_type, cute.select(q_layout, mode=[0, 1])
        )
        c_bytes = cute.size_in_bytes(
            self.operand_type, cute.select(c_layout, mode=[0, 1])
        )
        n_bytes = TILE_N * self.score_type.width // 8
        q_atom, q_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), q, q_layout,
            (self.tile_m, self.tile_k),
        )
        c_atom, c_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), centroids,
            cute.select(c_layout, mode=[0, 1]),
            (TILE_N, self.tile_k), 1,
        )
        n_atom, n_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), norms,
            cute.select(n_layout, mode=[0]), (TILE_N,), 1,
        )

        @cute.struct
        class SharedStorage:
            q_barriers: cute.struct.MemRange[cutlass.Int64, 2]
            c_barriers: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
            q: cute.struct.Align[
                cute.struct.MemRange[
                    self.operand_type, cute.cosize(q_layout)
                ], 1024
            ]
            centroids: cute.struct.Align[
                cute.struct.MemRange[
                    self.operand_type, cute.cosize(c_layout)
                ], 1024
            ]
            norms: cute.struct.Align[
                cute.struct.MemRange[
                    self.score_type, cute.cosize(n_layout)
                ], 128
            ]

        self.kernel(
            q_atom,
            q_tensor,
            c_atom,
            c_tensor,
            n_atom,
            n_tensor,
            row_norms,
            values,
            labels,
            tiled_mma,
            q_layout,
            c_layout,
            n_layout,
            SharedStorage,
            q_bytes,
            c_bytes + n_bytes,
        ).launch(
            grid=(cute.ceil_div(q.shape[0], self.tile_m), 1, 1),
            block=(
                _PRODUCER_THREADS
                + self.mma_warpgroups * _WARPGROUP_THREADS,
                1, 1,
            ),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        q_atom,
        q,
        c_atom,
        centroids,
        n_atom,
        norms,
        row_norms,
        values,
        labels,
        tiled_mma,
        q_layout,
        c_layout,
        n_layout,
        SharedStorage: cutlass.Constexpr,
        q_bytes: cutlass.Constexpr,
        c_bytes: cutlass.Constexpr,
    ):
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp == 0:
            cpasync.prefetch_descriptor(q_atom)
            cpasync.prefetch_descriptor(c_atom)
            cpasync.prefetch_descriptor(n_atom)
        allocator = cutlass.utils.SmemAllocator()
        storage = allocator.allocate(SharedStorage)
        producer = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        consumers = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.mma_warpgroups * _WARPGROUP_THREADS // cute.arch.WARP_SIZE,
        )
        q_pipe = quack_pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.q_barriers.data_ptr(),
            num_stages=1,
            producer_group=producer,
            consumer_group=consumers,
            tx_count=q_bytes,
            defer_sync=True,
        )
        c_pipe = quack_pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.c_barriers.data_ptr(),
            num_stages=self.stages,
            producer_group=producer,
            consumer_group=consumers,
            tx_count=c_bytes,
            defer_sync=True,
        )
        pipeline_init_arrive(cluster_shape_mn=(1, 1), is_relaxed=True)
        q_smem = storage.q.get_tensor(q_layout.outer, swizzle=q_layout.inner)
        c_smem = storage.centroids.get_tensor(
            c_layout.outer, swizzle=c_layout.inner
        )
        n_smem = storage.norms.get_tensor(n_layout)
        pipeline_init_wait(cluster_shape_mn=(1, 1))
        if warp < 4:
            cute.arch.setmaxregister_decrease(24)
            self.load(
                q,
                centroids,
                norms,
                q_smem,
                c_smem,
                n_smem,
                q_atom,
                c_atom,
                n_atom,
                q_pipe,
                c_pipe,
            )
        else:
            cute.arch.setmaxregister_increase(240)
            thread, _, _ = cute.arch.thread_idx()
            self.mma(
                q,
                centroids,
                row_norms,
                values,
                labels,
                q_smem,
                c_smem,
                n_smem,
                tiled_mma,
                q_pipe,
                c_pipe,
                thread - _PRODUCER_THREADS,
            )

    @cute.jit
    def load(
        self,
        q,
        centroids,
        norms,
        q_smem,
        c_smem,
        n_smem,
        q_atom,
        c_atom,
        n_atom,
        q_pipe,
        c_pipe,
    ):
        if cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4 == 0:
            row_block, _, _ = cute.arch.block_idx()
            q_tile = cute.local_tile(
                q, (self.tile_m, self.tile_k), (row_block, 0)
            )
            c_tile = cute.local_tile(
                centroids, (TILE_N, self.tile_k), (None, 0)
            )
            n_tile = cute.local_tile(norms, (TILE_N,), (None,))
            load_q, _, _ = copy_utils.tma_get_copy_fn(
                q_atom, 0, cute.make_layout(1), q_tile, q_smem,
                single_stage=True,
            )
            load_c, _, _ = copy_utils.tma_get_copy_fn(
                c_atom, 0, cute.make_layout(1), c_tile, c_smem
            )
            load_n, _, _ = copy_utils.tma_get_copy_fn(
                n_atom, 0, cute.make_layout(1), n_tile, n_smem
            )
            load_c = copy_utils.tma_producer_copy_fn(load_c, c_pipe)
            load_n = copy_utils.tma_producer_copy_fn(load_n, c_pipe)
            q_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 1
            )
            q_pipe.producer_acquire(q_state)
            load_q(tma_bar_ptr=q_pipe.producer_get_barrier(q_state))
            q_pipe.producer_commit(q_state)
            c_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.stages
            )
            for block in cutlass.range(centroids.shape[0] // TILE_N, unroll=1):
                c_pipe.producer_acquire(c_state)
                load_c(src_idx=block, producer_state=c_state)
                load_n(src_idx=block, producer_state=c_state)
                c_pipe.producer_commit(c_state)
                c_state.advance()

    @cute.jit
    def fold(
        self, accumulator, n_smem, coordinates, row_min, row_id, block, stage
    ):
        """Fold one centroid block's scores into every row slot's best."""
        values = layout_utils.reshape_acc_to_mn(accumulator)
        width = const_expr(cute.size(values, mode=[1]))
        if const_expr(self.packed_integer):
            folded = cute.make_rmem_tensor((width,), Int32)
            for column in cutlass.range_constexpr(width):
                local = coordinates[0, column][1]
                folded[column] = (
                    n_smem[local, stage] * Int32(TILE_N) + local
                )
            for row in cutlass.range_constexpr(cute.size(row_min)):
                best = Int32(0x7FFFFFFF)
                for column in cutlass.range_constexpr(width):
                    key = (
                        folded[column]
                        - Int32(2 * TILE_N) * values[row, column]
                    )
                    best = key if key < best else best
                take = (
                    (best >> Int32(LOCAL_ID_BITS))
                    < (row_min[row] >> Int32(LOCAL_ID_BITS))
                )
                row_min[row] = best if take else row_min[row]
                row_id[row] = block if take else row_id[row]
        else:
            for column in cutlass.range_constexpr(width):
                local = coordinates[0, column][1]
                centroid = block * TILE_N + local
                norm = n_smem[local, stage]
                for row in cutlass.range_constexpr(cute.size(row_min)):
                    score = cute.math.fma(
                        Float32(-2.0), values[row, column], norm
                    )
                    take = score < row_min[row]
                    row_min[row] = score if take else row_min[row]
                    row_id[row] = centroid if take else row_id[row]

    @cute.jit
    def gemm(self, tiled_mma, accumulator, q_fragment, c_fragment,
             zero_init: cutlass.Constexpr[bool] = True,
             wg_wait: cutlass.Constexpr[int] = 0):
        if const_expr(self.operand_type is Float8E4M3FN):
            promoted = cute.make_rmem_tensor(accumulator.layout, Float32)
            promoted.fill(0.0)
            mma_atom = cute.make_mma_atom(tiled_mma.op)
            # Four k32 atoms form stock QuACK's k128 FP8 promotion interval.
            for begin in cutlass.range_constexpr(0, cute.size(q_fragment.shape[2]), 4):
                cute.nvgpu.warpgroup.fence()
                mma_atom.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
                for k in cutlass.range_constexpr(begin, min(begin + 4, cute.size(q_fragment.shape[2]))):
                    cute.gemm(mma_atom, accumulator, q_fragment[None, None, k],
                              c_fragment[None, None, k], accumulator)
                    mma_atom.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                cute.nvgpu.warpgroup.commit_group()
                cute.nvgpu.warpgroup.wait_group(0)
                promoted.store(promoted.load() + accumulator.load())
            accumulator.store(promoted.load())
        else:
            sm90_utils.gemm(tiled_mma, accumulator, q_fragment, c_fragment,
                            zero_init=zero_init, wg_wait=wg_wait)

    @cute.jit
    def mma(
        self,
        q,
        centroids,
        row_norms,
        output_values,
        output_labels,
        q_smem,
        c_smem,
        n_smem,
        tiled_mma,
        q_pipe,
        c_pipe,
        thread: Int32,
    ):
        warpgroup = cute.arch.make_warp_uniform(
            thread // _WARPGROUP_THREADS
        )
        group_layout = cute.make_layout(
            self.mma_warpgroups, stride=_WARPGROUP_THREADS
        )
        thread_mma = tiled_mma.get_slice(thread)
        group_mma = tiled_mma.get_slice(group_layout(warpgroup))
        _, q_fragment, c_fragment = sm90_utils.partition_fragment_ABC(
            group_mma,
            (self.tile_m, TILE_N, self.tile_k),
            q_smem,
            c_smem,
        )
        coordinates = layout_utils.reshape_acc_to_mn(
            thread_mma.partition_C(
                cute.make_identity_tensor((self.tile_m, TILE_N))
            )
        )
        _assert_candidate_order(coordinates, tiled_mma, self.tile_m)
        row_shape = (cute.size(coordinates, mode=[0]),)
        row_min = cute.make_rmem_tensor(row_shape, self.score_type)
        row_id = cute.make_rmem_tensor(row_shape, Int32)
        if const_expr(self.score_type is Int32):
            row_min.fill(0x7FFFFFFF)
            row_id.fill(0x7FFFFFFF if const_expr(self.packed_integer) else 0)
        else:
            row_min.fill(Float32.inf)
            row_id.fill(0)
        q_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, 1
        )
        q_pipe.consumer_wait(q_state, q_pipe.consumer_try_wait(q_state))
        c_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.stages
        )
        for block in cutlass.range(
            centroids.shape[0] // TILE_N, unroll=1
        ):
            c_pipe.consumer_wait(
                c_state, c_pipe.consumer_try_wait(c_state)
            )
            accumulator = cute.make_rmem_tensor(
                tiled_mma.partition_shape_C((self.tile_m, TILE_N)),
                self.score_type,
            )
            self.gemm(
                tiled_mma,
                accumulator,
                q_fragment,
                c_fragment[None, None, None, c_state.index],
                zero_init=True,
                wg_wait=0,
            )
            self.fold(
                accumulator,
                n_smem,
                coordinates,
                row_min,
                row_id,
                block,
                c_state.index,
            )
            c_pipe.consumer_release(c_state)
            c_state.advance()
        for offset in (2, 1):
            for row in cutlass.range_constexpr(cute.size(row_min)):
                other_key = cute.arch.shuffle_sync_bfly(
                    row_min[row], offset=offset
                )
                other_id = cute.arch.shuffle_sync_bfly(
                    row_id[row], offset=offset
                )
                if const_expr(self.packed_integer):
                    other_score = other_key >> Int32(LOCAL_ID_BITS)
                    current_score = row_min[row] >> Int32(LOCAL_ID_BITS)
                    take = (other_score < current_score) | (
                        (other_score == current_score)
                        & (
                            (other_id < row_id[row])
                            | (
                                (other_id == row_id[row])
                                & (other_key < row_min[row])
                            )
                        )
                    )
                else:
                    take = (other_key < row_min[row]) | (
                        (other_key == row_min[row])
                        & (other_id < row_id[row])
                    )
                row_min[row] = other_key if take else row_min[row]
                row_id[row] = other_id if take else row_id[row]
        self.store(q, row_norms, output_values, output_labels,
                   coordinates, row_min, row_id)
        q_pipe.consumer_release(q_state)

    @cute.jit
    def store(self, q, row_norms, output_values, output_labels,
              coordinates, row_min, row_id):
        """Default native output; grouped selection overrides only this store."""
        lane = cute.arch.lane_idx()
        row_block, _, _ = cute.arch.block_idx()
        if lane % 4 == 0:
            for local in cutlass.range_constexpr(cute.size(row_min)):
                row = row_block * self.tile_m + coordinates[local, 0][0]
                if row < q.shape[0]:
                    if const_expr(self.packed_integer):
                        code = (
                            row_id[local] * Int32(TILE_N)
                            + (row_min[local] & Int32(TILE_N - 1))
                        )
                        score = row_min[local] >> Int32(LOCAL_ID_BITS)
                        distance = (
                            score.to(Int64) + row_norms[row].to(Int64)
                        )
                        output_values[row] = distance.to(output_values.element_type)
                        output_labels[row] = code.to(Int64)
                    else:
                        output_values[row] = row_min[local]
                        output_labels[row] = row_id[local].to(Int64)



def _ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def _required_smem(tile_k, tile_m, stages, element_bytes):
    per_stage = TILE_N * tile_k * element_bytes + TILE_N * 4
    return (
        tile_m * tile_k * element_bytes
        + stages * per_stage
        + _SMEM_OVERHEAD
    )


def _operand_type(dtype):
    if dtype is torch.int8:
        return Int8
    if dtype is torch.bfloat16:
        return BFloat16
    if dtype is torch.float8_e4m3fn:
        return Float8E4M3FN
    raise ValueError("row-owned operands must be BF16, E4M3 or INT8")


def row_owned_plan(batch, clusters, dim, device, dtype=torch.bfloat16):
    """Choose an existing tile with at least one CTA per SM when possible.

    Prefer the larger row tile only when its grid covers the device and its
    three-stage workspace fits. This is a conservative resource/occupancy rule,
    not a measured performance crossover or a batch-size admission table.
    """
    _operand_type(dtype)
    if not _row_owned_shape_legal(batch, clusters, dim, dtype is torch.int8):
        return None
    properties = torch.cuda.get_device_properties(device)
    smem = properties.shared_memory_per_block_optin
    tile_k = _row_owned_tile_k(dim)
    stages = 3
    tile_m = 128 if _ceil_div(batch, 128) >= properties.multi_processor_count else 64
    if _required_smem(tile_k, tile_m, stages, dtype.itemsize) > smem:
        tile_m = 64
        if _required_smem(tile_k, tile_m, stages, dtype.itemsize) > smem:
            return None
    return tile_k, tile_m, stages


class PreparedRowOwned(NamedTuple):
    compiled: object
    rows: torch.Tensor
    centroids: torch.Tensor
    norms: torch.Tensor
    row_norms: object
    values: torch.Tensor
    labels: torch.Tensor


def _validate_operand(tensor, name, device):
    require_tensor(tensor, name, device=device, contiguous=False)
    _operand_type(tensor.dtype)
    if tensor.ndim != 2 or tensor.stride(1) != 1:
        raise ValueError("%s must be rank two with unit inner stride" % name)
    if (
        tensor.stride(0) < tensor.shape[1]
        or tensor.stride(0) * tensor.element_size() % 16
        or tensor.data_ptr() % 16
    ):
        raise ValueError("%s must retain a 16-byte-aligned row pitch" % name)


@torch.no_grad()
def prepare_row_owned(
    rows,
    prepared,
    values,
    labels,
    plan,
    *,
    row_norms=None,
):
    """Compile and bind one floating or native-byte row-owned operation.

    Floating operands return FP32 scores, matching the tiled floating contract.
    Native bytes rank INT32 scores and return reconstructed FP32/INT64 distances.
    """
    from .e_means_cute import _require_stack

    centroids = prepared.centroids
    operand_type = _operand_type(rows.dtype)
    packed_integer = rows.dtype is torch.int8
    _require_stack()
    _validate_operand(rows, "row-owned rows", rows.device)
    _validate_operand(centroids, "row-owned centroids", rows.device)
    if centroids.dtype is not rows.dtype:
        raise ValueError("row-owned operands must have the same dtype")
    if rows.shape[1] != centroids.shape[1]:
        raise ValueError("row-owned operands must have equal dimensions")
    require_tensor(
        prepared.norms, "row-owned centroid norms",
        dtype=torch.int32 if packed_integer else torch.float32,
        shape=(len(centroids),), device=rows.device,
    )
    if packed_integer:
        require_tensor(row_norms, "row-owned row norms", dtype=torch.int32,
                       shape=(len(rows),), device=rows.device)
    elif row_norms is not None:
        raise ValueError("floating row-owned assignment takes no row norms")
    value_dtype = values.dtype
    if value_dtype is not torch.float32 and not (packed_integer and value_dtype is torch.int64):
        raise ValueError("row-owned values require FP32 or raw INT64")
    require_tensor(
        values, "row-owned values", dtype=value_dtype,
        shape=(len(rows),), device=rows.device,
    )
    require_tensor(
        labels, "row-owned labels", dtype=torch.int64,
        shape=(len(rows),), device=rows.device,
    )
    if (
        prepared.norms.data_ptr() % 16
        or values.data_ptr() % values.element_size()
        or labels.data_ptr() % 8
    ):
        raise ValueError("row-owned tensors do not meet pointer alignment")
    if (
        not isinstance(plan, tuple)
        or len(plan) != 3
        or not all(isinstance(value, int) for value in plan)
    ):
        raise ValueError("plan does not match the row-owned operands")
    tile_k, tile_m, stages = plan
    if (
        not _row_owned_shape_legal(len(rows), len(centroids), rows.shape[1], packed_integer)
        or tile_k != _row_owned_tile_k(rows.shape[1])
        or tile_m not in (64, 128)
        or stages != 3
    ):
        raise ValueError("plan does not match the row-owned operands")
    properties = torch.cuda.get_device_properties(rows.device)
    required_smem = _required_smem(
        tile_k, tile_m, stages, rows.element_size()
    )
    if required_smem > properties.shared_memory_per_block_optin:
        raise ValueError("row-owned plan exceeds opt-in shared memory")
    mma_warpgroups = 1 if tile_m == 64 else 2
    # Keep singleton axes in the TMA layout; static folding corrupts query loads.
    tensors = (
        from_dlpack(rows, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        from_dlpack(centroids, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        from_dlpack(prepared.norms, assumed_align=16),
        from_dlpack(row_norms, assumed_align=4) if packed_integer else None,
        from_dlpack(values, assumed_align=values.element_size()),
        from_dlpack(labels, assumed_align=8),
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    key = (
        rows.dtype,
        tile_k,
        tile_m,
        stages,
        rows.shape[0],
        centroids.shape[0],
        rows.shape[1],
        rows.stride(0),
        centroids.stride(0),
        rows.device.index,
    )
    if value_dtype is torch.int64:
        key += (value_dtype,)
    compiled = _COMPILED.get(key)
    if compiled is None:
        compiled = cute.compile(
            _RowOwnedAssignSm90(
                operand_type,
                packed_integer,
                tile_k,
                tile_m,
                stages,
                mma_warpgroups,
            ),
            *tensors,
            stream,
            options="--enable-tvm-ffi",
        )
        _COMPILED[key] = compiled
        while len(_COMPILED) > 8:
            _COMPILED.pop(next(iter(_COMPILED)))
    return PreparedRowOwned(
        compiled,
        rows,
        centroids,
        prepared.norms,
        row_norms,
        values,
        labels,
    )


_COMPILED = {}


@torch.no_grad()
def launch_row_owned(prepared, labels_out=None, distance_out=None):
    labels = prepared.labels if labels_out is None else labels_out
    if labels_out is not None:
        require_tensor(labels, "labels", dtype=torch.int64,
                       shape=prepared.labels.shape, device=prepared.labels.device)
    values = prepared.values if distance_out is None else distance_out
    require_tensor(values, "values", dtype=prepared.values.dtype,
                   shape=prepared.values.shape, device=prepared.values.device)
    if values.data_ptr() % values.element_size():
        raise ValueError("row-owned values do not meet pointer alignment")
    with torch.cuda.device(prepared.values.device):
        prepared.compiled(prepared.rows, prepared.centroids, prepared.norms,
                          prepared.row_norms, values, labels)
