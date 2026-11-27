# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Balanced E-means with centroids sharded over a process group.

Each rank owns a contiguous block of k / R centroids and their EMA sums; the
per-cluster vectors that the balancing reads (bias, lam, mu, usage) are only
k numbers each, so every rank keeps all of them. As in
``EmeansShardedCentroids``, every step all-gathers the global batch's rows,
every rank scores all of them against its own block, and reductions give the
global winner. Because every rank then knows every row's code, distance and
runner-up, everything the recipe decides from the batch (the capacity cap,
the counts, D, the bias step, the dead test, the in-place revivals, the donors) is computed
identically on every rank with no further communication. The only other
collectives are the repair rounds' masked re-assignment (two reductions over
the evicted rows), the usage blocks after each update, and the donor rows of
a split.
"""

import torch
import torch.distributed as dist

from .e_means_balanced import (
    _REPAIR_MIN_ROWS, EmeansBalanced, select_inplace, cap_first_pass, cap_round_evict, split_rows)
from .e_means_balanced_distributed import gather_ragged, gather_rows, reduce_top2, _SENTINEL
from .e_means_distributed import _check_group
from .e_means_torch import Emeans


class EmeansBalancedSharded(EmeansBalanced):
    """Train rank-local rows against sharded centroids, with the bias.

    ``k`` is global. ``centroids`` after training is this rank's block of
    ``k / R`` rows, starting at global id ``centroid_begin``; ``bias`` is
    global (k,). ``train(x)`` takes this rank's rows; the global batch is
    B / R rows from every rank in rank order, as in
    ``DistributedEmeansBalanced``, and the decisions equal one process on that
    batch. ``assign`` is collective: every rank passes its own rows (any
    count) and gets global centroid ids back. Rows, batch and k must divide by
    the group size.
    """

    def __init__(self, d, k, *args, group, **kwargs):
        ranks = _check_group(group)
        if kwargs.get("max_points_per_centroid") is not None:
            raise ValueError("max_points_per_centroid is not supported; subsample x instead")
        if k < ranks or k % ranks:
            raise ValueError("k must be divisible by the group size")
        super().__init__(d, k, *args, **kwargs)
        self.global_k = int(k)
        self._group, self._group_size = group, ranks
        self._rank = dist.get_rank(group)
        self.local_k = self.global_k // ranks
        self.centroid_begin = self.local_k * self._rank
        self._batch_shards = ranks

    def _reports(self):
        return self.verbose and self._rank == 0

    def _describe_parallel(self):
        return "Using %d %s with class %s (%d centroids each)" % (
            self._group_size, "GPUs" if self._usage.is_cuda else "CPU processes",
            type(self).__name__, self.local_k)

    def _block(self):
        return slice(self.centroid_begin, self.centroid_begin + self.local_k)

    # ----------------------------------------------------------- lifecycle

    def begin(self, global_n, init_centroids):
        """``init_centroids`` is the full (k, d) table (every rank the same)."""
        super().begin(global_n, init_centroids)
        if self._usage.device.type == "cpu" and dist.get_backend(self._group) != "gloo":
            self._failed = True
            raise ValueError("CPU EmeansBalancedSharded requires a Gloo group")
        ranks, batch = self._group_size, self.recipe["batch_size"]
        if global_n % ranks or batch % ranks:
            self._failed = True
            raise ValueError("rows (%d) and the batch (%d) must divide by the %d ranks"
                             % (global_n, batch, ranks))
        dist.broadcast(self._centroids, src=dist.get_global_rank(self._group, 0),
                       group=self._group)
        self._centroids = self._centroids[self._block()].clone()
        self._batch_stats = self._centroids.new_empty((self.local_k, self.d + 1))
        self.recipe.update(centroid_shards=ranks, local_k=self.local_k,
                           centroid_begin=self.centroid_begin, local_batch_size=batch // ranks,
                           batch_transport="all_gather")

    # Core's training loop serves a rank-local shard through these.
    _allow_device_path = False

    def _global_rows(self, local_rows):
        return local_rows * self._group_size

    def _draw_initial_centroids(self, x):
        from .e_means_balanced_distributed import DistributedEmeansBalanced

        return DistributedEmeansBalanced._draw_initial_centroids(self, x)

    def _assign_biased(self, x):
        """The dropped tail of the last pass, against the live state (collective)."""
        return self._assign(x, live=True)

    def _step_at(self, x, index, step):
        training, scale, alpha, cap = self._schedule(step)
        return self._train_step(x, index, training, scale, alpha, cap, self._revives(step), None)

    # ---------------------------------------------------------- assignment

    def _block_assignment(self, frozen):
        """This rank's prepared block with the GLOBAL shift folded in."""
        shift = self.bias.max()
        prepared, norm = self._prepared_mirror(self._centroids, frozen)
        local = self.bias[self._block()]
        if norm is not None:
            return prepared, norm + (shift - local), shift
        torch.add(self._base_norm, shift - local, out=prepared.norms)
        return prepared, None, shift

    def _scan(self, rows, prepared, norm, second):
        """Local best (and runner-up) of every row on this block, in chunks."""
        n = len(rows)
        chunk = self.recipe["assignment_chunk_size"]
        out = [torch.empty(n, dtype=torch.float32, device=rows.device),
               torch.empty(n, dtype=torch.int64, device=rows.device)]
        if second:
            out += [torch.empty(n, dtype=torch.float32, device=rows.device),
                    torch.empty(n, dtype=torch.int64, device=rows.device)]
        for begin in range(0, n, chunk):
            d1, l1, d2, l2 = self._assign_chunk(rows[begin:begin + chunk], prepared, norm,
                                                second and self.local_k > 1)
            out[0][begin:begin + len(d1)], out[1][begin:begin + len(d1)] = d1, l1
            if second:
                if self.local_k > 1:
                    out[2][begin:begin + len(d1)], out[3][begin:begin + len(d1)] = d2, l2
                else:
                    out[2][begin:begin + len(d1)] = float("inf")
                    out[3][begin:begin + len(d1)] = -1
        return out

    def _reduce_best(self, value, label):
        best = value.clone()
        dist.all_reduce(best, op=dist.ReduceOp.MIN, group=self._group)
        winner = torch.where(value == best, label.long() + self.centroid_begin, _SENTINEL)
        dist.all_reduce(winner, op=dist.ReduceOp.MIN, group=self._group)
        return best, winner

    # ---------------------------------------------------------------- step

    def _train_step(self, x, index, training, scale, alpha, cap, revive, state):
        """One step: gather the batch, score it on every block, reduce, then the
        recipe on replicated codes; the block owners update their centroids."""
        self._mark("begin")
        k = self.k
        block = self._block()
        rows = gather_rows(self._prepare_rows(x[index]).contiguous(), self._group)  # (B, d)
        n = len(rows)
        bias = self.bias
        prepared, norm, shift = self._block_assignment(frozen=not training)
        self._mark("prepare")
        capping = self._cap_active(training) and cap < n
        local = self._scan(rows, prepared, norm, capping)
        if capping:
            dist_, codes, second_d, second_l = reduce_top2(*local, self.centroid_begin,
                                                           self.local_k, self._group)
        else:
            dist_, codes = self._reduce_best(local[0], local[1])
        self._mark("assign")
        if capping:
            row_norm = self._row_norms(rows)
            codes, moved = cap_first_pass(codes, second_d - dist_, second_l, cap, k)
            dist_ = torch.where(moved, second_d, dist_)
            score = second_d - row_norm
            arrived = moved
            self.cap_stats["steps"] += 1
            self.cap_stats["moved"] = self.cap_stats["moved"] + moved.sum()
            self._mark("cap_first")
            for _ in range(self.cap_rounds):
                evict, full = cap_round_evict(codes, arrived, score, cap, k)
                ids = evict.nonzero().flatten()  # one host sync per round
                if len(ids) == 0:
                    break
                self.cap_stats["evicted"] += len(ids)
                masked = torch.where(full[block], float("inf"), self._biased_norm(prepared, norm))
                if norm is None:
                    saved = prepared.norms.clone()
                    prepared.norms.copy_(masked)
                width = max(_REPAIR_MIN_ROWS, 1 << (len(ids) - 1).bit_length())
                padded = torch.cat((ids, ids[-1:].expand(width - len(ids))))
                d1, l1 = self._scan(rows[padded], prepared, None if norm is None else masked,
                                    False)
                if norm is None:
                    prepared.norms.copy_(saved)
                d1, winner = self._reduce_best(d1[:len(ids)], l1[:len(ids)])
                codes[ids] = winner
                dist_[ids] = d1
                score[ids] = d1 - row_norm[ids]
                arrived[ids] = True
                self._mark("cap_round")
        self._mark("cap")
        objective = (dist_ - (shift - bias[codes])).sum(dtype=torch.float64)
        return self._finish_step(rows, codes, objective, training, scale, alpha, revive,
                                 len(index))

    def _finish_step(self, rows, codes, objective, training, scale, alpha, revive, local_n):
        """Statistics, revival, bias step and EMA from the batch's global codes."""
        k, n = self.k, len(rows)
        block = self._block()
        counts = torch.zeros(k, dtype=torch.float32, device=codes.device)
        counts.index_add_(0, codes, torch.ones(n, dtype=torch.float32, device=codes.device))
        if training:
            owned = (codes >= block.start) & (codes < block.stop)
            stats = self._batch_stats.zero_()
            mine = owned.nonzero().flatten()  # (n_owned,) rows won by this block
            chunk = min(self.recipe["assignment_chunk_size"], n)
            self._execution_backend.begin_batch(stats, chunk)
            for begin in range(0, len(mine), chunk):
                ids = mine[begin:begin + chunk]
                self._execution_backend.accumulate(rows[ids], codes[ids] - block.start, stats, n)
        self._objective += objective
        self._pass_counts += counts
        self._mark("stats")
        if revive:
            self._revive()
        self._mark("revive")
        self._bias_step(counts, objective / n, scale, training)
        self._mark("bias")
        if training:
            usage = self._usage[block]  # a view: the owner updates its block in place
            self._execution_backend.update(
                self._centroids, usage, None, stats[:, :-1], stats[:, -1],
                alpha, 1.0 - alpha, alpha == 1)
            self._usage.copy_(gather_rows(usage.clone(), self._group))
            self._mirror = None
        self._mark("ema")
        return codes[self._rank * local_n:(self._rank + 1) * local_n]

    def _revive(self):
        """``EmeansBalanced._revive`` on replicated vectors; donor rows travel."""
        usage = self._usage
        threshold = usage.sum() * self._unused_threshold / self.k
        dead = usage < threshold
        if not bool(dead.any()):
            return
        if self.inplace_quantile is not None:
            inplace, reference = select_inplace(dead, self.bias, self.inplace_quantile)
            count = int(inplace.sum())
            if count:
                usage[inplace] = usage.mean()
                self._lam[inplace] = (-reference).clamp(min=0.0)
                self._mu[inplace] = reference.clamp(min=0.0)
                self.bias[inplace] = reference
                self._inplace_host += count
            rest = dead & ~inplace
        else:
            rest = dead
        source, target = self._select_donors(usage, rest, dead)
        if source is None:
            return
        block = self._block()
        donors = self._centroids.new_zeros((len(source), self.d))  # (s, d)
        mine = (source >= block.start) & (source < block.stop)
        donors[mine] = self._centroids[source[mine] - block.start]
        dist.all_reduce(donors, group=self._group)  # one owner per row
        donor, child = split_rows(donors, self.split_eps)
        self._centroids[source[mine] - block.start] = donor[mine]
        owned = (target >= block.start) & (target < block.stop)
        self._centroids[target[owned] - block.start] = child[owned]
        half = usage[source] / 2
        usage[source] = half
        usage[target] = half
        for state in (self.bias, self._lam, self._mu):
            state[target] = state[source]
        self._n_splits += len(source)
        torch.autograd.graph.increment_version((self._centroids, self._usage))

    # ------------------------------------------------------------ queries

    def gather_centroids(self):
        """The full (k, d) centroid table on every rank (collective)."""
        return gather_rows(self.centroids.contiguous(), self._group)

    def _prepare_assignment_query(self, centroids):
        backend = self._execution_backend
        if getattr(backend, "cache", None) is not None:
            backend.cache.source_version = None  # may hold biased norms
        prepared, norm = backend.prepare_centroids(centroids, self._input_dtype)
        if not getattr(self, "_query_biased", True):
            return prepared, norm
        shift = self.bias.max()
        local = self.bias[self._block()]
        if norm is not None:
            return prepared, norm + (shift - local)
        prepared.norms.add_(shift - local)
        prepared.source_version = None  # the next query recomputes the plain norms
        return prepared, None

    def _assign(self, x, live=False):
        """Collective: every rank's rows against every block; global ids back."""
        x = self._prepare_input(x)
        size = torch.tensor([len(x)], dtype=torch.int64, device=x.device)
        sizes = gather_rows(size, self._group).tolist()
        rows = gather_ragged(x.contiguous(), sizes, self._group)
        if live:  # training state (the dropped tail of the last pass)
            prepared, norm, shift = self._block_assignment(frozen=False)
            value, label = self._scan(rows, prepared, norm, False)
        else:
            value, label = Emeans._assign(self, rows)
        best, winner = self._reduce_best(value, label)
        if getattr(self, "_query_biased", True) or live:
            best = best - (self.bias.max() - self.bias[winner])
        begin = sum(sizes[:self._rank])
        return best[begin:begin + len(x)], winner[begin:begin + len(x)]
