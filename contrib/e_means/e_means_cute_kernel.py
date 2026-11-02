# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#
# NativeRowVecLoad below is adapted from QuACK 0.6.5 RowVecLoad.begin_loop.
# Copyright (c) 2025-2026, Han Guo, Tri Dao. Licensed under Apache-2.0.
# Meta modified it to preserve int32 metadata and removed unrelated modes.

"""One fixed SM90 BF16 GEMM-plus-argmin assignment kernel.

The only schedule is CTA (M,N,K) = (128,192,64), cluster (2,1), with
non-serpentine 33-cluster N panels. It ranks native FP32 tensor-core scores
and atomically reduces packed ``(score, centroid id)`` keys per input row.
"""

from functools import partial
import hashlib
from pathlib import Path
from typing import NamedTuple, Optional

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32, Int32, Int64, const_expr
import torch

from quack import layout_utils, utils as quack_utils
from quack.cache import jit_cache
from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.cute_dsl_utils import (
    get_device_capacity,
    get_max_active_clusters,
    mlir_namedtuple,
    torch2cute_dtype_map,
)
from quack.epilogue.mixin import ComposableEpiMixin
from quack.epilogue.ops import (
    EpiOp,
    EpiSmemBytes,
    RowVecLoad,
    _lane_warp_info_n,
)
from quack.gemm_sm90 import GemmSm90
from quack.gemm_tvm_ffi_utils import (
    compile_gemm_kernel,
    get_major,
    make_fake_gemm_tensors,
    make_fake_varlen_args,
    make_scheduler_args,
    make_varlen_args,
)
from quack.rounding import RoundingMode
from quack.sm90_utils import partition_for_epilogue
from quack.tile_scheduler import (
    RasterOrderOption,
    TileScheduler,
    TileSchedulerOptions,
)

from .e_means_cute_argmin import (
    INITIAL_KEY,
    encode_argmin_key,
    launch_argmin_decode,
    prepare_argmin_decode,
    require_tensor,
)


TILE_M = 128
TILE_N = 192
TILE_K = 64
CLUSTER_M = 2
CLUSTER_N = 1
BLOCKED_N_CLUSTERS = 33


def _persistent_cache_source_key():
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).with_name("e_means_cute_argmin.py"),
    ):
        source = path.read_bytes()
        digest.update(path.name.encode() + b"\0")
        digest.update(len(source).to_bytes(8, "big"))
        digest.update(source)
    return digest.hexdigest()


# Every downstream Faiss module inlined into the artifact must participate.
_PERSISTENT_CACHE_SOURCE_KEY = _persistent_cache_source_key()


class NativeRowVecLoad(RowVecLoad):
    """QuACK row-vector load modified to retain the source element type."""

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        source, _ = super().begin(gemm, param, smem_tensor, ctx)
        source_subtile = cute.group_modes(source, 3, cute.rank(source))[
            None, None, None, 0
        ]
        registers = cute.make_rmem_tensor(
            source_subtile.layout, source_subtile.element_type
        )
        return [source, registers]

    @cute.jit
    def begin_loop(self, gemm, state, epi_coord):
        source, registers = state
        should_load = Boolean(True)
        if const_expr(gemm.epi_m_major):
            should_load = epi_coord[0] == 0
        if should_load:
            current = cute.group_modes(source, 3, cute.rank(source))[
                None, None, None, epi_coord
            ]
            loaded = cute.make_rmem_tensor(
                current.layout, current.element_type
            )
            cute.autovec_copy(
                cute.filter_zeros(current), cute.filter_zeros(loaded)
            )
            registers.store(loaded.load())
        return registers


def _blocked_scheduler_args(max_active):
    return TileSchedulerOptions(
        max_active_clusters=Int32(max_active),
        raster_order=RasterOrderOption.AlongN,
        max_swizzle_size=Int32(BLOCKED_N_CLUSTERS),
    )


@cute.jit
def _blocked_swizzle(params, cluster_id_in_problem: Int32):
    n_group, within_n_group = divmod(
        cluster_id_in_problem, params.num_clusters_in_group_fdd
    )
    cid_m, cid_n = Int32(0), Int32(0)
    if n_group < params.num_groups_regular:
        cid_m, n_in = divmod(within_n_group, params.group_size_fdd)
        cid_n = n_group * Int32(params.group_size_fdd.divisor) + n_in
    else:
        cid_m, n_in = divmod(within_n_group, params.group_size_tail_fdd)
        cid_n = (
            params.num_groups_regular * Int32(params.group_size_fdd.divisor)
            + n_in
        )
    return cid_m, cid_n


class _BlockedTileScheduler(TileScheduler):
    """Persistent monotonic-M traversal over 33-cluster N panels."""

    @cute.jit
    def _swizzle_cta(self, cluster_id_in_problem: Int32, *, loc=None, ip=None):
        return _blocked_swizzle(self.params, cluster_id_in_problem)


class _AtomicRowArgMin(EpiOp):
    """Reduce a 192-centroid tile and publish one packed key per row."""

    def param_fields(self):
        return [(self.name, object, None)]

    def to_params(self, gemm, args):
        return {self.name: getattr(args, self.name)}

    def epi_m_major_score(self, arg_tensor, gemm):
        return -1

    def smem_bytes(
        self, arg_tensor, cta_tile_shape_mnk, epi_tile, warp_shape_mnk=None
    ):
        return EpiSmemBytes()

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        vector_layout = cute.make_layout(
            (ctx.tile_M, ctx.tile_N), stride=(1, 0)
        )
        state_layout = ctx.partition_for_epilogue_fn(
            cute.make_rmem_tensor(vector_layout, Float32)
        ).layout
        return (
            cute.make_rmem_tensor(state_layout, Float32),
            cute.make_rmem_tensor(state_layout, Int32),
        )

    @cute.jit
    def begin_loop(self, gemm, state, epi_coord):
        minima, indices = state
        current_min = minima[None, None, None, epi_coord[0], epi_coord[1]]
        current_idx = indices[None, None, None, epi_coord[0], epi_coord[1]]
        if const_expr(epi_coord[1] == 0):
            cute.filter_zeros(current_min).fill(Float32.inf)
            cute.filter_zeros(current_idx).fill(0x7FFFFFFF)
        return current_min, current_idx

    @cute.jit
    def end_loop_stage(
        self,
        gemm,
        param,
        state,
        epi_coord,
        epi_tile,
        tiled_copy_t2r,
        tiled_copy_r2s,
        tidx,
    ):
        epi_tile_shape = cute.zipped_divide(
            cute.make_layout(gemm.cta_tile_shape_mnk[:2]), epi_tile
        ).shape[1]
        if const_expr(epi_coord[1] != epi_tile_shape[1] - 1):
            return None
        return (
            False,
            (
                state,
                epi_coord,
                epi_tile,
                tiled_copy_t2r,
                tiled_copy_r2s,
                tidx,
            ),
        )

    @cute.jit
    def end_loop_finish(
        self,
        gemm,
        param,
        staged,
        tile_coord_mnkl,
        varlen_manager,
    ):
        (
            state,
            epi_coord,
            epi_tile,
            tiled_copy_t2r,
            tiled_copy_r2s,
            tidx,
        ) = staged
        minima, indices = state
        current_min = minima[None, None, None, epi_coord[0], epi_coord[1]]
        current_idx = indices[None, None, None, epi_coord[0], epi_coord[1]]
        tiled_copy = (
            tiled_copy_t2r if tiled_copy_t2r is not None else tiled_copy_r2s
        )
        reference_src = tiled_copy_t2r is None
        lanes_in_n, warps_in_n, _, is_lane_n_leader = _lane_warp_info_n(
            tiled_copy, reference_src, tidx
        )
        if const_expr(lanes_in_n > 1):
            values = cute.filter_zeros(current_min)
            ids = cute.filter_zeros(current_idx)
            for i in cutlass.range(cute.size(values), unroll_full=True):
                columns = lanes_in_n // 2
                while columns > 0:
                    other_value = cute.arch.shuffle_sync_bfly(
                        values[i], offset=columns
                    )
                    other_id = cute.arch.shuffle_sync_bfly(
                        ids[i], offset=columns
                    )
                    take_other = (other_value < values[i]) | (
                        (other_value == values[i]) & (other_id < ids[i])
                    )
                    values[i] = other_value if take_other else values[i]
                    ids[i] = other_id if take_other else ids[i]
                    columns //= 2
        assert warps_in_n == 1

        partition = partial(
            partition_for_epilogue,
            epi_tile=epi_tile,
            tiled_copy=tiled_copy,
            tidx=tidx,
            reference_src=reference_src,
        )
        coordinates = partition(
            cute.make_identity_tensor(gemm.cta_tile_shape_mnk[:2])
        )[None, None, None, epi_coord[0], epi_coord[1]]
        row_min = layout_utils.convert_layout_zero_stride(
            current_min, current_min.layout
        )[None, 0]
        row_idx = layout_utils.convert_layout_zero_stride(
            current_idx, current_min.layout
        )[None, 0]
        row_coordinates = layout_utils.convert_layout_zero_stride(
            coordinates, current_min.layout
        )[None, 0]

        tile_m = gemm.cta_tile_shape_mnk[0]
        batch_idx = tile_coord_mnkl[3]
        limit_m = min(
            varlen_manager.len_m(batch_idx) - tile_coord_mnkl[0] * tile_m,
            tile_m,
        )
        batch_keys = param[batch_idx, None, 0]
        output = cute.local_tile(
            batch_keys, (tile_m,), (tile_coord_mnkl[0],)
        )
        if is_lane_n_leader:
            for m in cutlass.range(cute.size(row_coordinates, mode=[0])):
                local_row = row_coordinates[m][0]
                if local_row < limit_m:
                    key = encode_argmin_key(row_min[m], row_idx[m])
                    cute.arch.atomic_min(
                        quack_utils.elem_pointer(output, local_row).llvm_ptr,
                        key,
                        sem="relaxed",
                        scope="gpu",
                    )


class _GemmAssignMinSm90(ComposableEpiMixin, GemmSm90):
    _epi_ops = (
        RowVecLoad("mCentroidNorms"),
        NativeRowVecLoad("mCentroidIds"),
        _AtomicRowArgMin("mLocalReduce"),
    )
    _extra_param_fields = ()

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        mCentroidNorms: Optional[cute.Tensor] = None
        mCentroidIds: Optional[cute.Tensor] = None
        mLocalReduce: Optional[cute.Tensor] = None

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        self.rounding_mode = RoundingMode.RN
        return self.EpilogueParams(**self._epi_ops_to_params_dict(args))

    def get_scheduler_class(self, varlen_m=False):
        if varlen_m:
            raise ValueError("the fixed scheduler does not support varlen M")
        return _BlockedTileScheduler


def _flatten_layout(shape, stride):
    if not isinstance(shape, tuple):
        if not isinstance(shape, int) or not isinstance(stride, int):
            raise RuntimeError("epilogue layout is not compile-time constant")
        return [(shape, stride)]
    if not isinstance(stride, tuple) or len(shape) != len(stride):
        raise RuntimeError("epilogue layout shape and stride differ")
    pairs = []
    for sub_shape, sub_stride in zip(shape, stride):
        pairs.extend(_flatten_layout(sub_shape, sub_stride))
    return pairs


def _static_offsets(layout):
    offsets = [0]
    for extent, stride in _flatten_layout(
        tuple(layout.shape), tuple(layout.stride)
    ):
        offsets = [
            base + step * stride
            for step in range(extent)
            for base in offsets
        ]
    return offsets


def _assert_candidate_order(state_layout, id_layout):
    """Prove each aliased row slot visits strictly increasing IDs."""
    state = _static_offsets(state_layout)
    ids = _static_offsets(id_layout)
    if len(state) != len(ids):
        raise RuntimeError("score and ID fragments have different sizes")
    last_by_row = {}
    for row, candidate in zip(state, ids):
        previous = last_by_row.get(row)
        if previous is not None and candidate <= previous:
            raise RuntimeError("centroid candidates are not ordered by ID")
        last_by_row[row] = candidate
    return len(state), len(last_by_row)


class _OrderedGemmAssignMinSm90(_GemmAssignMinSm90):
    _schedule_guard_ran = False
    _order_guard_ran = False

    def _setup_attributes(self, epilogue_args):
        super()._setup_attributes(epilogue_args)
        actual = (
            tuple(self.cta_tile_shape_mnk),
            tuple(self.cluster_shape_mnk),
            tuple(self.atom_layout_mnk),
            tuple(self.epi_tile),
            tuple(self.epi_tile_shape),
            bool(self.epi_m_major),
        )
        expected = (
            (TILE_M, TILE_N, TILE_K),
            (CLUSTER_M, CLUSTER_N, 1),
            (1, 1, 1),
            (64, 32),
            (2, 6),
            True,
        )
        if actual != expected:
            raise RuntimeError(
                "fixed CuTe assignment schedule changed: %r" % (actual,)
            )
        type(self)._schedule_guard_ran = True

    @cute.jit
    def epi_visit_subtile(self, params, tensors, accumulator, c=None):
        norms = tensors.get("mCentroidNorms")
        centroid_ids = tensors.get("mCentroidIds")
        tile_min, tile_idx = tensors.get("mLocalReduce")
        _assert_candidate_order(tile_min.layout, centroid_ids.layout)
        type(self)._order_guard_ran = True
        for i in cutlass.range(cute.size(tile_min), unroll_full=True):
            score = norms[i] - 2.0 * accumulator[i]
            take = score < tile_min[i]
            tile_min[i] = score if take else tile_min[i]
            tile_idx[i] = centroid_ids[i] if take else tile_idx[i]
        return ()


@jit_cache
def _compile_ordered(a_major, b_major, device_capacity, source_cache_key):
    del source_cache_key  # Used by jit_cache when selecting a disk entry.
    operand = torch2cute_dtype_map[torch.bfloat16]
    a, b, d, c, m, _, _, batch_axis = make_fake_gemm_tensors(
        operand, operand, None, None, a_major, b_major, None, None
    )
    epi_n = cute.sym_int()
    epilogue = _GemmAssignMinSm90.EpilogueArguments(
        mCentroidNorms=fake_tensor(
            Float32, (batch_axis, epi_n), leading_dim=1, divisibility=4
        ),
        mCentroidIds=fake_tensor(
            Int32, (batch_axis, epi_n), leading_dim=1, divisibility=4
        ),
        mLocalReduce=fake_tensor(
            Int64, (batch_axis, m, 1), leading_dim=2, divisibility=1
        ),
    )
    _OrderedGemmAssignMinSm90._schedule_guard_ran = False
    _OrderedGemmAssignMinSm90._order_guard_ran = False
    compiled = compile_gemm_kernel(
        _OrderedGemmAssignMinSm90,
        operand,
        (TILE_M, TILE_N, TILE_K),
        (CLUSTER_M, CLUSTER_N, 1),
        True,
        True,
        False,
        False,
        device_capacity,
        a,
        b,
        d,
        c,
        epilogue,
        _blocked_scheduler_args(1),
        make_fake_varlen_args(False, False, False, None),
    )
    if not _OrderedGemmAssignMinSm90._schedule_guard_ran:
        raise RuntimeError("QuACK did not invoke the fixed schedule guard")
    if not _OrderedGemmAssignMinSm90._order_guard_ran:
        raise RuntimeError("candidate order was not proved during compilation")
    return compiled


def _centroid_metadata(prepared):
    k = prepared.num_centroids
    padded_k = ((k + TILE_N - 1) // TILE_N) * TILE_N
    require_tensor(
        prepared.norms,
        "centroid norms",
        dtype=torch.float32,
        shape=(k,),
        device=prepared.centroids.device,
    )
    require_tensor(
        prepared.padded_norms,
        "padded centroid norms",
        dtype=torch.float32,
        shape=(1, padded_k),
        device=prepared.centroids.device,
    )
    if prepared.padded_norms[:, :k].data_ptr() != prepared.norms.data_ptr():
        raise ValueError("padded norms must alias the logical norm prefix")
    ids = torch.full(
        (1, padded_k),
        0x7FFFFFFF,
        dtype=torch.int32,
        device=prepared.centroids.device,
    )
    ids[:, :k] = torch.arange(
        k, dtype=torch.int32, device=prepared.centroids.device
    )[None]
    return ids


class PreparedTiledGemm(NamedTuple):
    compiled: object
    a: torch.Tensor
    b: torch.Tensor
    epilogue: object
    scheduler: object
    varlen: object
    decode: object


@torch.no_grad()
def prepare_tiled_gemm(rows, prepared, codes, scores):
    """Compile and bind one stable private full-or-tail workspace."""
    centroids = prepared.centroids
    if (
        rows.ndim != 2
        or centroids.ndim != 2
        or rows.shape[1] != centroids.shape[1]
    ):
        raise ValueError("rows and centroids must be rank two with equal d")
    if (
        rows.dtype is not torch.bfloat16
        or centroids.dtype is not torch.bfloat16
        or rows.device != centroids.device
        or not rows.is_cuda
    ):
        raise ValueError("CuTe operands must be same-device CUDA BF16")
    if rows.stride(-1) != 1 or centroids.stride(-1) != 1:
        raise ValueError("CuTe operands must have unit inner stride")
    if rows.stride(-2) % 8 or centroids.stride(-2) % 8:
        raise ValueError("CuTe BF16 row pitches must be 16-byte aligned")
    ids = _centroid_metadata(prepared)
    keys = torch.full(
        (1, rows.shape[0], 1),
        INITIAL_KEY,
        dtype=torch.int64,
        device=rows.device,
    )
    a = rows.unsqueeze(0)
    b = centroids.unsqueeze(0)
    compiled = _compile_ordered(
        get_major(a, "m", "k"),
        get_major(b, "n", "k"),
        get_device_capacity(rows.device),
        _PERSISTENT_CACHE_SOURCE_KEY,
    )
    return PreparedTiledGemm(
        compiled,
        a,
        b,
        _GemmAssignMinSm90.EpilogueArguments(
            mCentroidNorms=prepared.padded_norms,
            mCentroidIds=ids,
            mLocalReduce=keys,
        ),
        make_scheduler_args(
            get_max_active_clusters(CLUSTER_M * CLUSTER_N),
            BLOCKED_N_CLUSTERS,
            None,
        ),
        make_varlen_args(None, None, None),
        prepare_argmin_decode(keys, codes, scores),
    )


@torch.no_grad()
def launch_tiled_gemm(prepared, labels_out=None):
    """Launch the one assignment scan and its decode/reset kernel."""
    prepared.compiled(
        prepared.a,
        prepared.b,
        None,
        None,
        prepared.epilogue,
        prepared.scheduler,
        prepared.varlen,
        None,
        None,
    )
    launch_argmin_decode(prepared.decode, labels_out)
