# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#
# Adapted from PR #280 cute_kmeans/cute/row_owned.py at
# e8e288285e885064b477a882e237fcc47da8a9b2.

"""Low-dimension SM90 BF16 row-owned assignment kernel.

BF16 operands rank FP32 scores. A compile-time layout proof establishes
increasing centroid visitation order, and the cross-lane reduction explicitly
breaks ties by centroid ID.
"""

from typing import NamedTuple

import cutlass
import cutlass.cute as cute
from cutlass import (
    BFloat16,
    Float32,
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


def _row_owned_shape_legal(batch, clusters, dim):
    return (
        batch >= 1
        and clusters >= TILE_N
        and clusters % TILE_N == 0
        and dim >= 1
    )


def _row_owned_tile_k(dim):
    return max(128, 64 * _ceil_div(dim, 64))


class _RowOwnedAssignSm90:
    """Compute exact row minima without materializing B-by-K scores."""

    def __init__(
        self,
        tile_k,
        tile_m,
        stages,
        mma_warpgroups,
    ):
        self.tile_k = tile_k
        self.tile_m = tile_m
        self.stages = stages
        self.mma_warpgroups = mma_warpgroups

    @cute.jit
    def __call__(self, q, centroids, norms, values, labels, stream):
        tiled_mma = hopper_utils.make_trivial_tiled_mma(
            BFloat16,
            BFloat16,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(self.mma_warpgroups, 1, 1),
            tiler_mn=(64, TILE_N),
        )
        # QuACK's fp8_fast_accum=True is a mainloop policy, not an MMA-atom
        # option: it disables the separate per-K-tile acc_slow promotion.
        # sm90_utils.gemm below uses the same supported fast construction,
        # accumulating all K slices directly into this FP32 fragment.
        q_layout = sm90_utils.make_smem_layout(
            BFloat16,
            LayoutEnum.ROW_MAJOR,
            (self.tile_m, self.tile_k),
            None,
        )
        c_layout = sm90_utils.make_smem_layout(
            BFloat16,
            LayoutEnum.ROW_MAJOR,
            (TILE_N, self.tile_k),
            self.stages,
        )
        n_layout = cute.make_layout(
            (TILE_N, self.stages), stride=(1, TILE_N)
        )
        q_bytes = cute.size_in_bytes(
            BFloat16, cute.select(q_layout, mode=[0, 1])
        )
        c_bytes = cute.size_in_bytes(
            BFloat16, cute.select(c_layout, mode=[0, 1])
        )
        n_bytes = TILE_N * Float32.width // 8
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
                    BFloat16, cute.cosize(q_layout)
                ], 1024
            ]
            centroids: cute.struct.Align[
                cute.struct.MemRange[
                    BFloat16, cute.cosize(c_layout)
                ], 1024
            ]
            norms: cute.struct.Align[
                cute.struct.MemRange[
                    Float32, cute.cosize(n_layout)
                ], 128
            ]

        self.kernel(
            q_atom,
            q_tensor,
            c_atom,
            c_tensor,
            n_atom,
            n_tensor,
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
    def mma(
        self,
        q,
        centroids,
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
        row_min = cute.make_rmem_tensor(row_shape, Float32)
        row_id = cute.make_rmem_tensor(row_shape, Int32)
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
                Float32,
            )
            sm90_utils.gemm(
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
                take = (other_key < row_min[row]) | (
                    (other_key == row_min[row]) & (other_id < row_id[row])
                )
                row_min[row] = other_key if take else row_min[row]
                row_id[row] = other_id if take else row_id[row]
        self.store(q, output_values, output_labels, coordinates, row_min, row_id)
        q_pipe.consumer_release(q_state)

    @cute.jit
    def store(self, q, output_values, output_labels, coordinates, row_min, row_id):
        """Native floating output; grouped selection overrides this store."""
        lane = cute.arch.lane_idx()
        row_block, _, _ = cute.arch.block_idx()
        if lane % 4 == 0:
            for local in cutlass.range_constexpr(cute.size(row_min)):
                row = row_block * self.tile_m + coordinates[local, 0][0]
                if row < q.shape[0]:
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


def row_owned_plan(batch, clusters, dim, device):
    """Choose an existing tile with at least one CTA per SM when possible.

    Prefer the larger row tile only when its grid covers the device and its
    three-stage workspace fits. This is a conservative resource/occupancy rule,
    not a measured performance crossover or a batch-size admission table.
    """
    if not _row_owned_shape_legal(batch, clusters, dim):
        return None
    properties = torch.cuda.get_device_properties(device)
    smem = properties.shared_memory_per_block_optin
    tile_k = _row_owned_tile_k(dim)
    stages = 3
    tile_m = 128 if _ceil_div(batch, 128) >= properties.multi_processor_count else 64
    if _required_smem(tile_k, tile_m, stages, 2) > smem:
        tile_m = 64
        if _required_smem(tile_k, tile_m, stages, 2) > smem:
            return None
    return tile_k, tile_m, stages


class PreparedRowOwned(NamedTuple):
    compiled: object
    rows: torch.Tensor
    centroids: torch.Tensor
    norms: torch.Tensor
    values: torch.Tensor
    labels: torch.Tensor


def _validate_operand(tensor, name, device):
    require_tensor(
        tensor,
        name,
        dtype=torch.bfloat16,
        device=device,
        contiguous=False,
    )
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
):
    """Compile and bind one BF16 row-owned assignment operation.

    It returns bare FP32 scores, matching the tiled BF16 contract.
    """
    from .e_means_cute import _require_stack

    centroids = prepared.centroids
    _require_stack()
    _validate_operand(rows, "row-owned rows", rows.device)
    _validate_operand(centroids, "row-owned centroids", rows.device)
    if centroids.dtype is not rows.dtype:
        raise ValueError("row-owned operands must have the same dtype")
    if rows.shape[1] != centroids.shape[1]:
        raise ValueError("row-owned operands must have equal dimensions")
    require_tensor(
        prepared.norms, "row-owned centroid norms", dtype=torch.float32,
        shape=(len(centroids),), device=rows.device,
    )
    require_tensor(
        values, "row-owned values", dtype=torch.float32,
        shape=(len(rows),), device=rows.device,
    )
    require_tensor(
        labels, "row-owned labels", dtype=torch.int64,
        shape=(len(rows),), device=rows.device,
    )
    if (
        prepared.norms.data_ptr() % 16
        or values.data_ptr() % 4
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
        not _row_owned_shape_legal(len(rows), len(centroids), rows.shape[1])
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
        from_dlpack(values, assumed_align=4),
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
    compiled = _COMPILED.get(key)
    if compiled is None:
        compiled = cute.compile(
            _RowOwnedAssignSm90(
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
        values,
        labels,
    )


_COMPILED = {}


@torch.no_grad()
def launch_row_owned(prepared, labels_out=None):
    labels = prepared.labels if labels_out is None else labels_out
    if labels_out is not None:
        require_tensor(labels, "labels", dtype=torch.int64,
                       shape=prepared.labels.shape, device=prepared.labels.device)
    with torch.cuda.device(prepared.values.device):
        prepared.compiled(prepared.rows, prepared.centroids, prepared.norms,
                          prepared.values, labels)
