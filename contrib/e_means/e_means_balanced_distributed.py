# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Multi-GPU balanced E-means: the capacity cap over a global batch.

The cap keeps any cluster from taking more than ``cap`` rows of one batch, so
it ranks the rows of every over-full cluster over the WHOLE batch. With rows
split over ranks (data parallel) no rank holds the whole batch. These helpers
make the decision anyway, exactly: every rank gathers a small per-row record
(code, margin, runner-up, runner-up score) of the global batch, in rank order,
and runs the same deterministic ranking on it. Identical inputs give identical
decisions on every rank without a coordinator, and the result is the one a
single process takes on the concatenated batch. Only the rows themselves stay
local: a repair round re-assigns each rank's own evicted rows and gathers the
answers.

``group=None`` runs the same code on one process; it is the reference the
multi-rank result must equal.

Rankings are stable and break ties by batch position, so they do not depend
on the sort implementation.
"""

import torch
import torch.distributed as dist

from .e_means_balanced import (
    _PAIR_WORK_LIMIT, _REPAIR_MIN_ROWS, EmeansBalanced, cap_first_pass, cap_round_evict)
from .e_means_distributed import _check_group
from .e_means_torch import _torch_generator

try:
    from .balanced_triton import FusedCap, GroupSearch
except ImportError:  # No Triton: the host path only.
    FusedCap = GroupSearch = None


def _world(group):
    return 1 if group is None else dist.get_world_size(group)


def _rank(group):
    return 0 if group is None else dist.get_rank(group)


def gather_rows(local, group):
    """Concatenate equal-length per-rank tensors in rank order."""
    if group is None:
        return local
    out = local.new_empty((len(local) * dist.get_world_size(group), *local.shape[1:]))
    gather = getattr(dist, "all_gather_into_tensor", None)
    if gather is not None and dist.get_backend(group) == "nccl":
        gather(out, local.contiguous(), group=group)
    else:
        dist.all_gather(list(out.chunk(dist.get_world_size(group))), local.contiguous(),
                        group=group)
    return out


def gather_ragged(local, counts, group):
    """Concatenate per-rank tensors of known lengths ``counts`` (host ints)."""
    if group is None:
        return local
    width = max(max(counts), 1)
    padded = local.new_zeros((width, *local.shape[1:]))
    padded[:len(local)] = local
    full = gather_rows(padded, group)
    return torch.cat([full[r * width:r * width + c] for r, c in enumerate(counts)])


def cap_rounds(x, codes, margin, second, second_val, cap, k, n_rounds, assign_masked,
               group=None, stats=None):
    """The settled recipe's cap: first pass, then up to ``n_rounds`` repairs.

    Per rank: ``x`` its rows (b, d), ``codes`` / ``second`` (b,) first choice
    and runner-up, ``margin`` / ``second_val`` (b,) float32.
    ``assign_masked(rows, full)`` returns (labels, scores) of ``rows`` against
    the centroids with the ``full`` clusters excluded. Returns this rank's
    capped (b,) int64 codes. ``stats`` (a dict) receives the counters.

    Communication: one all-gather of 16 bytes per global row, then per repair
    round one host sync and one gather of the evicted rows' answers.
    """
    b = len(codes)
    world, rank = _world(group), _rank(group)
    record = torch.stack((codes.to(torch.int32), margin.float().view(torch.int32),
                          second.to(torch.int32), second_val.float().view(torch.int32)), 1)
    record = gather_rows(record, group)  # (B, 4)
    gcodes = record[:, 0].long()
    gmargin = record[:, 1].contiguous().view(torch.float32)
    gsecond = record[:, 2].long()
    score = record[:, 3].contiguous().view(torch.float32).clone()  # (B,)
    gcodes, arrived = cap_first_pass(gcodes, gmargin, gsecond, cap, k)
    if stats is not None:
        stats["moved"] = arrived.sum()
        stats["rounds"] = 0
    for _ in range(n_rounds):
        evict, full = cap_round_evict(gcodes, arrived, score, cap, k)
        evict = evict.nonzero().flatten()  # (e,) ascending; one host sync
        if len(evict) == 0:
            break
        owner = evict // b
        counts = torch.bincount(owner, minlength=world).tolist()
        mine = evict[owner == rank] - rank * b
        labels, val = assign_masked(x[mine], full)
        answer = torch.stack((labels.to(torch.int32), val.float().view(torch.int32)), 1)
        answer = gather_ragged(answer, counts, group)  # (e, 2)
        gcodes[evict] = answer[:, 0].long()
        score[evict] = answer[:, 1].contiguous().view(torch.float32)
        arrived[evict] = True
        if stats is not None:
            stats["rounds"] += 1
    return gcodes[rank * b:(rank + 1) * b]


_SENTINEL = torch.iinfo(torch.int64).max


def reduce_top2(v1, i1, v2, i2, begin, local_k, group):
    """Global best and runner-up from per-shard top-2 (sharded centroids).

    Each rank scored every row against its own ``local_k`` centroids starting
    at global id ``begin``: (B,) best ``v1``/``i1`` and runner-up ``v2``/``i2``
    (local ids, lower score is better). Ties go to the lower global id, as on
    one GPU. Four all-reduces of B values. Returns (best, winner, second
    score, runner-up), global ids int64.
    """
    best = v1.clone()
    dist.all_reduce(best, op=dist.ReduceOp.MIN, group=group)
    gi1 = i1.long() + begin
    winner = torch.where(v1 == best, gi1, _SENTINEL)
    dist.all_reduce(winner, op=dist.ReduceOp.MIN, group=group)
    own = winner == gi1
    cand_v = torch.where(own, v2, v1)
    cand_i = torch.where(own, i2.long() + begin, gi1)
    cand_i = torch.where(own & (i2 < 0), _SENTINEL, cand_i)
    second = cand_v.clone()
    dist.all_reduce(second, op=dist.ReduceOp.MIN, group=group)
    runner = torch.where(cand_v == second, cand_i, _SENTINEL)
    dist.all_reduce(runner, op=dist.ReduceOp.MIN, group=group)
    return best, winner, second, runner


class DistributedEmeansBalanced(EmeansBalanced):
    """Balanced E-means over a process group: rows split, centroids replicated.

    Every rank holds the whole FP32 state (centroids, usage, bias, lam, mu,
    D) and 1/R of the rows; ``train(x)`` takes this rank's rows. A global
    batch of B rows is B/R rows from every rank, in rank order; each rank
    shuffles its own rows every pass. One step:

    - each rank assigns its rows (biased, with runner-ups);
    - the capacity cap is taken on the whole batch: a 16-byte record per row
      (code, margin, runner-up, runner-up score) is all-gathered and every
      rank runs the same ranking on it (on CUDA with 16-bit rows, the fused
      kernels of the single-GPU class); in a repair round every rank searches
      its own evicted rows against the masked codebook and the answers are
      shared (an all-reduce on CUDA, an all-gather otherwise);
    - the batch sums and counts (a (k, d+1) table) and the FP64 objective are
      all-reduced, so D, the bias step, the dead test, the in-place revivals,
      the donors and the EMA see exactly the global batch and stay identical
      on every rank (the largest-usage donors are a deterministic function of
      the usage; the proportional draw uses the same seeded generator
      everywhere).

    The decisions are those of one process on the concatenated batch;
    centroids match it up to FP32 summation order (the table is reduced in a
    different order). On 16-bit CUDA rows a step needs no host
    synchronization and is replayed as a CUDA graph (the single-GPU device
    path, with the collectives captured); otherwise steps are eager.
    Rows, global batches and every pass must divide evenly by the group size.
    ``group`` is caller-owned; CPU tensors need Gloo.

    Initial centroids default to the rows one process would draw from the
    concatenated rows (one global permutation of ``seed``). ``train``
    returns the centroids and this rank's last-pass labels.
    """

    def __init__(self, d, k, *args, group, **kwargs):
        ranks = _check_group(group)
        if kwargs.get("max_points_per_centroid") is not None:
            raise ValueError("max_points_per_centroid is not supported; subsample x instead")
        super().__init__(d, k, *args, **kwargs)
        self._group, self._group_size = group, ranks
        self._rank = dist.get_rank(group)
        self._batch_shards = ranks
        self._dp_totals = None
        self._dp_cap = None

    # On CUDA with 16-bit rows the steps run on the single-GPU class's device
    # path: fixed shapes, the collectives captured in CUDA graphs (the two
    # ``*_device`` overrides below reduce its fused statistics). With the fused
    # global cap a step never waits for the host, and the graphs remove the
    # launch overhead that dominated small local batches: DEEP 10M, K = 65,536,
    # 8 H100s, 3.80 ms per training step eagerly against 1.70 ms (public
    # DistributedEmeans: 1.68 ms). ``_allow_device_path = False`` on the
    # instance keeps eager steps.
    _allow_device_path = True

    def _reports(self):
        return self.verbose and self._rank == 0

    def _describe_parallel(self):
        cuda = self._centroids is not None and self._centroids.is_cuda
        return "Using %d %s with class %s" % (
            self._group_size, "GPUs" if cuda else "CPU processes", type(self).__name__)

    # ----------------------------------------------------------- lifecycle

    def _global_rows(self, local_rows):
        return local_rows * self._group_size

    def begin(self, global_n, init_centroids):
        super().begin(global_n, init_centroids)
        if self._centroids.device.type == "cpu" and dist.get_backend(self._group) != "gloo":
            self._failed = True
            raise ValueError("CPU DistributedEmeansBalanced requires a Gloo group")
        ranks, batch = self._group_size, self.recipe["batch_size"]
        if global_n % ranks or batch % ranks:
            self._failed = True
            raise ValueError(
                "rows (%d) and the batch (%d) must divide by the %d ranks; pass batch_size="
                % (global_n, batch, ranks))
        # Rank 0's state is the run's state, as in DistributedEmeans.
        dist.broadcast(self._centroids, src=dist.get_global_rank(self._group, 0),
                       group=self._group)
        self._dp_totals = torch.zeros(2, dtype=torch.int64, device=self._centroids.device)
        self._dp_cap = None
        self.recipe.update(data_parallel=ranks, local_batch_size=batch // ranks)

    def _draw_initial_centroids(self, x):
        """The k rows one process would draw from the concatenated rows."""
        n_local = len(x)
        n = n_local * self._group_size
        ids = torch.randperm(n, generator=_torch_generator(x.device, self.seed),
                             device=x.device)[:self.k]  # (k,) global ids
        mine = (ids // n_local) == self._rank
        table = torch.zeros(self.k, self.d, dtype=torch.float32, device=x.device)
        if bool(mine.any()):
            table[mine] = self._initial_centroid_values(x[ids[mine] - self._rank * n_local])
        dist.all_reduce(table, group=self._group)  # (k, d), one owner per row
        return table

    def _step_at(self, x, index, step):
        """One eager step at global step ``step`` on this rank's rows ``x[index]``."""
        training, scale, alpha, cap = self._schedule(step)
        return self._train_step(x, index, training, scale, alpha, cap, self._revives(step), None)

    # ---------------------------------------------------------------- step

    def _batch_statistics(self, x, index, rows_all, codes, true_dist, training):
        """This rank's statistics, all-reduced into the global batch's."""
        stats, counts, objective, _ = super()._batch_statistics(
            x, index, rows_all, codes, true_dist, training)
        objective = objective.reshape(1)
        dist.all_reduce(stats if training else counts, group=self._group)  # (k, d+1) / (k,)
        dist.all_reduce(objective, group=self._group)
        objective = objective[0]
        return stats, counts, objective, objective / self.recipe["batch_size"]

    def _batch_statistics_device(self, x, index, rows_all, codes, dist_, shift, training,
                                 state):
        """Device path: this rank's statistics and distance sum, all-reduced in place.

        Core's launch adds this rank's distance sum to the epoch objective;
        the objective is rebuilt from the global sum instead.
        """
        before = self._objective.clone()
        stats, counts = super()._batch_statistics_device(
            x, index, rows_all, codes, dist_, shift, training, state)
        fused = state["bias"]
        dist.all_reduce(fused.batch_sum, group=self._group)  # () FP64
        self._objective.copy_(before + fused.batch_sum)
        dist.all_reduce(stats if training else counts, group=self._group)
        return stats, counts

    def _bias_step_device(self, counts, scale, training, state):
        """The fused band step reads the batch size for the fair share and the mean
        distance: give it the global batch (its distance launch keeps the local rows)."""
        fused = state["bias"]
        local, fused.n = fused.n, self.recipe["batch_size"]
        try:
            super()._bias_step_device(counts, scale, training, state)
        finally:
            fused.n = local

    def _cap_rounds_step(self, x, index, rows_all, codes, dist_, second_d, second_l, row_norm,
                         cap, prepared, norm, shift, bias, state):
        """The rounds cap on the global batch; returns this rank's codes and distances."""
        if self._fused(x):
            return self._fused_cap(x, index, rows_all, codes, dist_, second_d, second_l,
                                   row_norm, cap, prepared, norm, state)
        k, n = self.k, len(codes)
        world, rank = self._group_size, self._rank
        lo = rank * n
        margin = second_d - dist_
        score = second_d - row_norm  # kernel score at the runner-up (row norm dropped)
        record = torch.stack((codes.to(torch.int32), margin.view(torch.int32),
                              second_l.to(torch.int32), score.view(torch.int32)), 1)
        record = gather_rows(record, self._group)  # (B, 4)
        gcodes, moved = cap_first_pass(record[:, 0], record[:, 1].contiguous().view(
            torch.float32), record[:, 2], cap, k)
        gscore = record[:, 3].contiguous().view(torch.float32).clone()  # (B,)
        dist_ = torch.where(moved[lo:lo + n], second_d, dist_)
        arrived = moved
        self._dp_totals[0] += moved.sum()  # device count, no sync
        self._mark("cap_first")
        self.cap_stats["steps"] += 1
        for _ in range(self.cap_rounds):
            evict, full = cap_round_evict(gcodes, arrived, gscore, cap, k)
            ids = evict.nonzero().flatten()  # (e,) ascending; one host sync per round
            if len(ids) == 0:
                break
            self._dp_totals[1] += len(ids)
            owner = ids // n
            sizes = torch.bincount(owner, minlength=world).tolist()
            mine = ids[owner == rank] - lo  # (e_r,) local positions
            answer = torch.empty((len(mine), 2), dtype=torch.int32, device=codes.device)
            if len(mine):
                masked = torch.where(full, float("inf"), self._biased_norm(prepared, norm))
                if norm is None:
                    saved = prepared.norms.clone()
                    prepared.norms.copy_(masked)
                width = max(_REPAIR_MIN_ROWS, 1 << (len(mine) - 1).bit_length())
                padded = torch.cat((mine, mine[-1:].expand(width - len(mine))))
                rows = self._prepare_rows(x[index[padded]])
                d1, l1, _, _ = self._assign_chunk(rows, prepared,
                                                  None if norm is None else masked, second=False)
                if norm is None:
                    prepared.norms.copy_(saved)
                d1, l1 = d1[:len(mine)], l1[:len(mine)]
                dist_[mine] = d1
                answer[:, 0] = l1.to(torch.int32)
                answer[:, 1] = (d1 - self._row_norms(rows[:len(mine)])).view(torch.int32)
            answer = gather_ragged(answer, sizes, self._group)  # (e, 2)
            gcodes[ids] = answer[:, 0].long()
            gscore[ids] = answer[:, 1].contiguous().view(torch.float32)
            arrived[ids] = True
            self._mark("cap_round")
        return gcodes[lo:lo + n], dist_

    # ------------------------------------------------------ fused global cap

    def _fused(self, x):
        """The fused cap runs on CUDA with 16-bit rows (the repair search reads them)."""
        return (FusedCap is not None and x.is_cuda
                and self._input_dtype in (torch.bfloat16, torch.float16))

    def _fused_state(self, n, device):
        """Global-batch buffers of the fused cap, allocated at the first step."""
        state = getattr(self, "_dp_cap", None)
        total = n * self._group_size
        if state is None or state["fused"].n != total:
            step = torch.zeros(1, dtype=torch.int64, device=device)
            state = self._dp_cap = dict(
                fused=FusedCap(total, self.k, device, self.recipe["total_steps"], step),
                search=GroupSearch(n, total, device), step=step, planned=0,
                cap=torch.zeros((), dtype=torch.int64, device=device),
                codes=torch.empty(total, dtype=torch.int64, device=device),
                second_l=torch.empty(total, dtype=torch.int32, device=device),
                dist=torch.zeros(total, dtype=torch.float32, device=device),
                second_d=torch.zeros(total, dtype=torch.float32, device=device),
                row_norm=torch.zeros(total, dtype=torch.float32, device=device))
            self._dp_plan = dict(first="pair", round=["pair"] * self.cap_rounds)
        return state

    def _fused_cap(self, x, index, rows_all, codes, dist_, second_d, second_l, row_norm, cap,
                   prepared, norm, state):
        """The cap on the global batch with the single-GPU fused kernels.

        Every rank gathers the 16-byte records of the global batch and runs the
        same fused bookkeeping on them (identical inputs, identical decisions; it
        costs a few small launches, whatever the number of ranks). In a repair
        round each rank searches the evicted rows of its own slice and an
        all-reduce (minimum) of the (score, label) words gives every rank every
        answer. No host synchronization.
        """
        n = len(codes)
        lo = self._rank * n
        cap_state = self._fused_state(n, codes.device)
        fused, plan = cap_state["fused"], self._dp_plan
        if not torch.is_tensor(cap):  # Eager steps: the kernels read the cap on the device.
            cap = cap_state["cap"].fill_(cap)
        margin = second_d - dist_
        score = second_d - row_norm  # kernel score at the runner-up (row norm dropped)
        record = torch.stack((codes.to(torch.int32), margin.view(torch.int32),
                              second_l.to(torch.int32), score.view(torch.int32)), 1)
        record = gather_rows(record, self._group)  # (B, 4)
        gcodes, gsecond = cap_state["codes"], cap_state["second_l"]
        gcodes.copy_(record[:, 0])
        gsecond.copy_(record[:, 2])
        fused.margin.copy_(record[:, 1].contiguous().view(torch.float32))
        fused.score.copy_(record[:, 3].contiguous().view(torch.float32))
        # Distances and row norms are only needed for this rank's rows.
        gdist, gnorm = cap_state["dist"], cap_state["row_norm"]
        gdist[lo:lo + n] = dist_
        cap_state["second_d"][lo:lo + n] = second_d
        gnorm[lo:lo + n] = row_norm
        fused.first_pass(gcodes, gdist, cap_state["second_d"], gsecond, gnorm, cap,
                         plan["first"], keys=True)
        self._mark("cap_first")
        mirror = self._mirror_rows(prepared, norm)
        biased = self._biased_norm(prepared, norm)
        if state is not None:
            rows, rows_index = state["rows"], None
        elif rows_all is not None:
            rows, rows_index = rows_all, None
        elif x.dtype == self._input_dtype:
            rows, rows_index = x, index
        else:
            rows, rows_index = self._prepare_rows(x[index]), None

        def search(positions, count, packed):
            cap_state["search"].search(positions, count, packed, lo, rows, rows_index, mirror,
                                       biased, fused.held, cap)
            dist.all_reduce(packed, op=dist.ReduceOp.MIN, group=self._group)

        for r in range(self.cap_rounds):
            fused.round(gcodes, gdist, gnorm, cap, None, mirror, biased, r, plan["round"][r],
                        search=search, ordered=True)
            self._mark("cap_round")
        cap_state["step"].add_(1)
        return gcodes[lo:lo + n], gdist[lo:lo + n]

    def _update_cap_plan(self, state, begin, end):
        """Ranking methods of the fused global cap, from its steps since the last choice.

        The records carry the runner-up labels, so the assignment always returns them.
        """
        cap_state = getattr(self, "_dp_cap", None)
        if cap_state is not None:
            now = int(cap_state["step"])
            first = cap_state["planned"]
            if now > first:
                stats = cap_state["fused"].stats[first:now].double().cpu()
                work = stats[-max(1, len(stats) // 4):, :3].mean(dim=0)  # recent steps
                self._dp_plan = dict(
                    first="pair" if work[0] <= _PAIR_WORK_LIMIT else "sort",
                    round=["pair" if work[1 + min(r, 1)] <= _PAIR_WORK_LIMIT else "sort"
                           for r in range(self.cap_rounds)])
                cap_state["planned"] = now
        if self._device_mode:
            self._cap_plan = dict(self._dp_plan if cap_state is not None else self._cap_plan,
                                  bucket=[0] * self.cap_rounds, second="labels")
            self._cap_plans.append((end, self._cap_plan))

    def _start_epoch(self):
        super()._start_epoch()
        if not self._device_mode and getattr(self, "_dp_cap", None) is not None:
            self._update_cap_plan(None, 0, 0)  # one read per epoch; the epoch end syncs anyway

    def finalize(self):
        if self._dp_totals is not None and self.centroids is None:
            cap_state = getattr(self, "_dp_cap", None)
            if cap_state is not None:
                fused = cap_state["fused"]
                self._dp_totals[0] += fused.totals[0]
                self._dp_totals[1] += fused.stats[:, 3:5].sum()
                self.cap_stats["steps"] = int(cap_state["step"])
                self._dp_cap = None
            moved, evicted = self._dp_totals.tolist()
            self.cap_stats.update(moved=moved, evicted=evicted)
        return super().finalize()
