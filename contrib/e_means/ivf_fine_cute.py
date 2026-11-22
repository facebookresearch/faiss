# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""CuTe SM90 kernel of the fused IVF fine search (see ivf_fine).

One persistent CTA per SM walks the work items (one list, up to 128 routed
pairs). A producer warpgroup gathers the pairs' query rows with cp.async and
loads the list's candidate tile with TMA, one 64-column contraction chunk per
stage of a 4-stage ring. Two consumer warpgroups run wgmma over each chunk
(BF16 operands, FP32 accumulation, 16-wide steps in contraction order), then
fold every candidate tile into a running lexicographic top-2 per row with
score = fma(-2, inner, norm) and key (score, position); positions follow ID
order inside a list, so the order equals the (score, ID) order of the padded
BMM path. Rows and lists whose width is not a multiple of 8 are fed as copies
padded with zero columns, which add exact zeros. The candidate-tile width
(128 or 256) changes timing only and is tuned once per shape class.
"""

import math

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, const_expr, pipeline
from cutlass.cute.nvgpu import cpasync
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils import LayoutEnum
import cutlass.utils.hopper_helpers as hopper_utils
from quack import copy_utils, layout_utils, pipeline as quack_pipeline
from quack import sm90_utils
import torch

from . import ivf_fine

_PRODUCER_THREADS = 128
_WARPGROUP_THREADS = 128
_TM = 128   # routed pairs (query rows) per work item
_TN = 256   # candidates per tile
_BK = 64    # columns per K chunk
_STAGES = 4
_IMAX32 = 0x7FFFFFFF


class _FineSearchSm90:
    def __init__(self, w, tn=256):
        self.nk = (w + _BK - 1) // _BK
        self.tn = tn  # the last K chunk may be partial: cp.async and TMA zero-fill past the width

    @cute.jit
    def __call__(self, x, c16, cn, ids, lims, brow, porder, item_list, item_start, item_count, total, o_sc, o_id,
                 nsm: Int32, stream):
        tiled_mma = hopper_utils.make_trivial_tiled_mma(
            BFloat16, BFloat16, cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.K,
            Float32, atom_layout_mnk=(2, 1, 1), tiler_mn=(64, self.tn))
        q_layout = sm90_utils.make_smem_layout(BFloat16, LayoutEnum.ROW_MAJOR, (_TM, _BK), _STAGES)
        c_layout = sm90_utils.make_smem_layout(BFloat16, LayoutEnum.ROW_MAJOR, (self.tn, _BK), _STAGES)
        tma_atom_c, tma_c = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), c16,
                                                        cute.slice_(c_layout, (None, None, 0)), (self.tn, _BK))

        @cute.struct
        class SharedStorage:
            barriers: cute.struct.MemRange[cutlass.Int64, _STAGES * 2]
            q: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(q_layout)], 1024]
            c: cute.struct.Align[cute.struct.MemRange[BFloat16, cute.cosize(c_layout)], 1024]

        self.kernel(x, cn, ids, lims, brow, porder, item_list, item_start, item_count, total, o_sc, o_id, tiled_mma, q_layout,
                    c_layout, SharedStorage, tma_atom_c, tma_c).launch(
            grid=(nsm, 1, 1), block=(_PRODUCER_THREADS + 2 * _WARPGROUP_THREADS, 1, 1), stream=stream, min_blocks_per_mp=1)

    @cute.kernel
    def kernel(self, x, cn, ids, lims, brow, porder, item_list, item_start, item_count, total, o_sc, o_id, tiled_mma,
               q_layout, c_layout, SharedStorage: cutlass.Constexpr, tma_atom_c, tma_c):
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        allocator = cutlass.utils.SmemAllocator()
        storage = allocator.allocate(SharedStorage)
        if warp == 0:
            cpasync.prefetch_descriptor(tma_atom_c)
        pipe = quack_pipeline.PipelineTmaCpAsync.create(
            barrier_storage=storage.barriers.data_ptr(), num_stages=_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1 + _PRODUCER_THREADS),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 2 * _WARPGROUP_THREADS // cute.arch.WARP_SIZE),
            tx_count=self.tn * _BK * 2, cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)), defer_sync=True)
        pipeline_init_arrive(cluster_shape_mn=(1, 1), is_relaxed=True)
        q_smem = storage.q.get_tensor(q_layout.outer, swizzle=q_layout.inner)
        c_smem = storage.c.get_tensor(c_layout.outer, swizzle=c_layout.inner)
        pipeline_init_wait(cluster_shape_mn=(1, 1))
        if warp < 4:
            cute.arch.setmaxregister_decrease(40)
            self.produce(x, lims, brow, item_list, item_start, item_count, total, q_smem, c_smem, pipe, tma_atom_c, tma_c, warp)
        else:
            cute.arch.setmaxregister_increase(232)
            thread, _, _ = cute.arch.thread_idx()
            self.consume(cn, ids, lims, porder, item_list, item_start, item_count, total, o_sc, o_id, q_smem, c_smem, tiled_mma,
                         pipe, thread - _PRODUCER_THREADS)

    @cute.jit
    def produce(self, x, lims, brow, item_list, item_start, item_count, total, q_smem, c_smem, pipe, tma_atom_c, tma_c, warp):
        tidx = cute.arch.thread_idx()[0]
        block, _, _ = cute.arch.block_idx()
        grid, _, _ = cute.arch.grid_dim()
        per_row = const_expr(_BK // 8)
        tiled_copy = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), BFloat16, num_bits_per_copy=128),
            cute.make_layout((_PRODUCER_THREADS // per_row, per_row), stride=(per_row, 1)), cute.make_layout((1, 8)))
        thr_copy = tiled_copy.get_slice(tidx)
        state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, _STAGES)
        for item in cutlass.range(block, total[0], grid, unroll=1):
            L = item_list[item]
            start = item_start[item]
            count = item_count[item]
            lo = lims[L]
            size = lims[L + 1] - lo
            qidx = cute.local_tile(cute.domain_offset((start,), brow), (_TM,), (0,))
            copy_q = copy_utils.gather_m_get_copy_fn(thr_copy, x, q_smem, qidx, limit_m=count, limit_k=x.shape[1])
            for nt in cutlass.range(((size - 1) // self.tn + 1) if size > 0 else 0, unroll=1):   # no size + tn overflow
                gC = cute.local_tile(cute.domain_offset((lo + nt * self.tn, 0), tma_c), (self.tn, _BK), (0, None))
                copy_c, _, _ = copy_utils.tma_get_copy_fn(tma_atom_c, 0, cute.make_layout(1), gC, c_smem)
                for kc in cutlass.range(self.nk, unroll=1):
                    pipe.producer_acquire(state, None, warp == 0)
                    if warp == 0:
                        copy_c(kc, state.index, tma_bar_ptr=pipe.producer_get_barrier(state))
                    copy_q(kc, state.index, pred=True)
                    pipe.producer_cpasync_commit(state)
                    state.advance()

    @cute.jit
    def consume(self, cn, ids, lims, porder, item_list, item_start, item_count, total, o_sc, o_id, q_smem, c_smem, tiled_mma,
                pipe, thread):
        warpgroup = cute.arch.make_warp_uniform(thread // _WARPGROUP_THREADS)
        group_layout = cute.make_layout(2, stride=_WARPGROUP_THREADS)
        thread_mma = tiled_mma.get_slice(thread)
        group_mma = tiled_mma.get_slice(group_layout(warpgroup))
        _, q_fragment, c_fragment = sm90_utils.partition_fragment_ABC(group_mma, (_TM, self.tn, _BK), q_smem, c_smem)
        coordinates = layout_utils.reshape_acc_to_mn(thread_mma.partition_C(cute.make_identity_tensor((_TM, self.tn))))
        row_shape = (cute.size(coordinates, mode=[0]),)
        block, _, _ = cute.arch.block_idx()
        grid, _, _ = cute.arch.grid_dim()
        lane = cute.arch.lane_idx()
        state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, _STAGES)
        mma_atom = cute.make_mma_atom(tiled_mma.op)
        mma_atom.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)   # accumulators are zeroed before each K loop
        for item in cutlass.range(block, total[0], grid, unroll=1):
            L = item_list[item]
            start = item_start[item]
            n_pairs = item_count[item]
            lo = lims[L]
            size = lims[L + 1] - lo
            b = cute.make_rmem_tensor(row_shape, Float32)
            bp = cute.make_rmem_tensor(row_shape, Int32)
            s = cute.make_rmem_tensor(row_shape, Float32)
            sp = cute.make_rmem_tensor(row_shape, Int32)
            b.fill(Float32.inf)
            bp.fill(_IMAX32)
            s.fill(Float32.inf)
            sp.fill(_IMAX32)
            for nt in cutlass.range(((size - 1) // self.tn + 1) if size > 0 else 0, unroll=1):   # no size + tn overflow
                acc = cute.make_rmem_tensor(tiled_mma.partition_shape_C((_TM, self.tn)), Float32)
                acc.fill(0.0)
                for _kc in cutlass.range(self.nk, unroll=1):
                    pipe.consumer_wait(state, pipe.consumer_try_wait(state))
                    cute.nvgpu.warpgroup.fence()
                    for k in cutlass.range_constexpr(_BK // 16):
                        cute.gemm(mma_atom, acc, q_fragment[None, None, k, state.index], c_fragment[None, None, k, state.index], acc)
                    cute.nvgpu.warpgroup.commit_group()
                    cute.nvgpu.warpgroup.wait_group(0)
                    pipe.consumer_release(state)
                    state.advance()
                values = layout_utils.reshape_acc_to_mn(acc)
                for column in cutlass.range_constexpr(cute.size(values, mode=[1])):
                    local = nt * self.tn + coordinates[0, column][1]
                    norm = cn[lo + (local if local < size else 0)]
                    norm = norm if local < size else Float32.inf
                    for row in cutlass.range_constexpr(cute.size(b)):
                        score = cute.math.fma(Float32(-2.0), values[row, column], norm)
                        fin = (score < Float32.inf) & (score > -Float32.inf)  # non-finite: missing
                        lt_b = fin & ((score < b[row]) | ((score == b[row]) & (local < bp[row])))
                        lt_s = fin & ((score < s[row]) | ((score == s[row]) & (local < sp[row])))
                        s[row] = b[row] if lt_b else (score if lt_s else s[row])
                        sp[row] = bp[row] if lt_b else (local if lt_s else sp[row])
                        b[row] = score if lt_b else b[row]
                        bp[row] = local if lt_b else bp[row]
            for offset in (2, 1):   # quad butterfly: merge two lexicographic top-2 states (positions are unique)
                for row in cutlass.range_constexpr(cute.size(b)):
                    ob = cute.arch.shuffle_sync_bfly(b[row], offset=offset)
                    obp = cute.arch.shuffle_sync_bfly(bp[row], offset=offset)
                    os_ = cute.arch.shuffle_sync_bfly(s[row], offset=offset)
                    osp = cute.arch.shuffle_sync_bfly(sp[row], offset=offset)
                    other_first = (ob < b[row]) | ((ob == b[row]) & (obp < bp[row]))
                    lb = b[row] if other_first else ob          # loser of the two firsts
                    lbp = bp[row] if other_first else obp
                    nb = ob if other_first else b[row]
                    nbp = obp if other_first else bp[row]
                    # second = smallest of (loser, own second, other second)
                    c1 = (os_ < s[row]) | ((os_ == s[row]) & (osp < sp[row]))
                    ms = os_ if c1 else s[row]
                    msp = osp if c1 else sp[row]
                    c2 = (lb < ms) | ((lb == ms) & (lbp < msp))
                    s[row] = lb if c2 else ms
                    sp[row] = lbp if c2 else msp
                    b[row] = nb
                    bp[row] = nbp
            if lane % 4 == 0:
                for local in cutlass.range_constexpr(cute.size(b)):
                    p = coordinates[local, 0][0]
                    if p < n_pairs:
                        pair = porder[start + p].to(Int64)
                        o_sc[pair * 2] = b[local]
                        o_sc[pair * 2 + 1] = s[local]
                        if b[local] < Float32.inf:
                            o_id[pair * 2] = ids[lo + bp[local]].to(Int64)
                        if s[local] < Float32.inf:
                            o_id[pair * 2 + 1] = ids[lo + sp[local]].to(Int64)


_COMPILED = {}
_TUNED = {}


def _compiled(width, tile):
    from quack.compile_utils import make_fake_tensor

    if (width, tile) not in _COMPILED:
        dyn = cute.sym_int
        vec = lambda dtype: make_fake_tensor(dtype, (dyn(),), divisibility=1, leading_dim=0)  # noqa: E731
        fakes = (make_fake_tensor(BFloat16, (dyn(), width), divisibility=8, leading_dim=1),
                 make_fake_tensor(BFloat16, (dyn(), width), divisibility=8, leading_dim=1),
                 vec(Float32), vec(Int32), vec(Int32), vec(Int32), vec(Int32), vec(Int32), vec(Int32), vec(Int32),
                 vec(Int32), vec(Float32), vec(Int64))
        _COMPILED[width, tile] = cute.compile(
            _FineSearchSm90(width, tile), *fakes, 132, cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi")
    return _COMPILED[width, tile]


def _prepare(index, x, coarse_idx, tile):
    n, d = x.shape
    P = int(coarse_idx.shape[1])
    device = x.device
    width = (d + 7) // 8 * 8  # 16-byte rows for cp.async and TMA
    x16 = x.to(torch.bfloat16).contiguous()
    rows = x16 if width == d else torch.nn.functional.pad(x16, (0, width - d))
    if rows.data_ptr() % 16:  # cp.async moves 16-byte row pieces: realign a view that starts mid-row
        rows = rows.clone()
    vecs = index.vectors_lowp if index.vectors_lowp is not None else index.vectors
    nv = vecs.shape[0]
    if width == d and vecs.is_contiguous() and vecs.data_ptr() % 16 == 0:  # TMA zero-fills rows past the end
        lists = vecs
    else:
        lists = torch.zeros((nv + _TN, width), dtype=torch.bfloat16, device=device)
        lists[:nv, :d] = vecs
    norms = torch.full((nv + _TN,), torch.inf, dtype=torch.float32, device=device)
    norms[:nv] = index.vectors_norms
    ids = torch.zeros((nv + _TN,), dtype=torch.int32, device=device)
    ids[:nv] = index.ids.to(torch.int32)
    item_list, item_start, item_count, pair_row, pair_id, slots, total = ivf_fine._file_pairs(index, coarse_idx, _TM)
    p_sc, p_id = ivf_fine._partials(n, P, device)
    args = None
    if slots:
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        args = (rows, lists, norms, ids, index.xb_lims.to(torch.int32), pair_row, pair_id, item_list.to(torch.int32),
                item_start.to(torch.int32), item_count.to(torch.int32), total.to(torch.int32), p_sc.view(-1), p_id.view(-1),
                sms)
    return dict(n=n, P=P, width=width, tile=tile, args=args, p_sc=p_sc, p_id=p_id)


def _run(state):
    if state["args"] is not None:
        _compiled(state["width"], state["tile"])(*state["args"])


def _tuned_tile(index, x, coarse_idx):
    """Time both candidate-tile widths once per shape class and keep the faster; both give identical results."""
    n, d = x.shape
    P = int(coarse_idx.shape[1])
    mean_list = int(index.vectors_norms.shape[0]) / max(1, int(index.nlist))
    key = ((d + 7) // 8 * 8, P, round(math.log2(max(1.0, mean_list))), round(math.log2(max(1, n))))
    if key not in _TUNED:
        best = None
        for tile in (128, 256):
            times = []
            for _ in range(3):
                state = _prepare(index, x, coarse_idx, tile)
                start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                _run(state)
                stop.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(stop))
            elapsed = sorted(times)[1]  # the first run includes compilation
            if best is None or elapsed < best[0]:
                best = (elapsed, tile)
        _TUNED[key] = best[1]
    return _TUNED[key]


def search_2nn(index, x, coarse_idx, tile=None):
    """As ivf_fine.search_2nn, on the CuTe kernel."""
    if tile not in (None, 128, 256):
        raise ValueError("tile must be None, 128 or 256")
    if tile is None:
        tile = _tuned_tile(index, x, coarse_idx)
    state = _prepare(index, x, coarse_idx, tile)
    _run(state)
    return ivf_fine._merge(state["p_sc"], state["p_id"], state["n"], state["P"])
