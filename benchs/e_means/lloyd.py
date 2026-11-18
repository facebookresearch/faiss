# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Epoch-wise Lloyd updates using the SSD demo's sharded assignment engines.

Only sums and counts persist across batches. Assignment is frozen for an
epoch; IVF lists are rebuilt once per epoch, with a fixed coarse quantizer.
Statistics use original FP32 rows, FP32 sums and FP64 epoch counts. Empty
centroids retain their previous positions. No EMA or cluster revival runs.
"""

import torch
import torch.distributed as dist

from faiss.contrib.e_means import EmeansShardedCentroids, EmeansShardedCentroidsIVF


class _LloydEpoch:
    def __init__(self, *args, **kwargs):
        kwargs.update(quantizer=None, unused_threshold=0, capture_assignments=False)
        super().__init__(*args, **kwargs)

    def begin(self, n, initial):
        super().begin(n, initial)
        self.recipe.update(name="lloyd", algorithm="kmeans", centroid_update="epoch_mean")
        for key in (
            "alpha0", "alpha0_source", "alpha_min_factor", "check_unused_every",
            "unused_threshold", "split_epsilon", "init_usage", "revival_protection",
        ):
            self.recipe.pop(key, None)

    def _resolve_run_recipe(self, n):
        super()._resolve_run_recipe(n)
        # E-means reserves a full pass for closing after the fractional pass.
        # Lloyd needs no closing pass, so consume its fractional budget last.
        self.recipe["epoch_rows"].sort(reverse=True)
        batch = self.recipe["batch_size"]
        self.recipe["epoch_steps"] = [
            (rows + batch - 1) // batch for rows in self.recipe["epoch_rows"]
        ]

    def _report_header(self):
        self._say(
            f"Lloyd clustering {self._n} points in {self.d}D to "
            f"{self.global_k} clusters, {self.nepochs} epochs, "
            f"processing batch {self.recipe['batch_size']}, "
            f"{self._group_size} centroid shards"
        )

    def _start_epoch(self):
        super()._start_epoch()
        if self._pass_sums is None:
            self._pass_sums = torch.zeros_like(self._centroids)
        self._row_square = self._objective.new_zeros(())
        self._lloyd_prepared = None

    def _prepare_assignment(self):
        if self._lloyd_prepared is None:
            self._lloyd_prepared = super()._prepare_assignment()
        return self._lloyd_prepared

    @torch.no_grad()
    def accumulate(self, rows):
        stats = super().accumulate(rows)
        # Large logical batches use FP64 packed statistics internally. Keep
        # their exact counts instead of the public FP32 count view.
        if self._batch_accumulation.dtype == torch.float64:
            stats["counts"] = self._batch_accumulation[:, -1]
        # Each reader contributes its own rows once, before all-gather.
        # Bound the FP64 temporary to 8 MiB; no pairwise FP64 distances.
        chunk = max(1, (8 << 20) // (8 * self.d))
        for block in rows.split(chunk):
            self._row_square.add_(block.double().square().sum())
        return stats

    @torch.no_grad()
    def update(self, stats):
        self._require_running()
        expected = min(
            self.recipe["batch_size"],
            self.recipe["epoch_rows"][self._epoch] - self._epoch_rows,
        )
        if self._pending_stats is None or stats["n"] != expected:
            raise RuntimeError("consume one complete processing batch before update")
        try:
            _, sums, counts, objective = self._prepare_statistics(stats, expected)
            self._pass_sums.add_(sums)
            self._pass_counts.add_(counts)
            self._objective.add_(objective)
            self._step += 1
            self._epoch_steps += 1
            self._epoch_rows += expected
            self._pending_stats = None
            self._batch_accumulation = self._batch_objective = None
            self._batch_centroids = self._batch_norm = self._batch_operand = None
        except BaseException:
            self._failed = True
            raise

    @torch.no_grad()
    def end_epoch(self):
        rows = self.recipe["epoch_rows"][self._epoch]
        if (self._pending_stats is not None or self._epoch_rows != rows
                or self._epoch_steps != self.recipe["epoch_steps"][self._epoch]):
            raise RuntimeError("epoch is incomplete; consume all prescribed batches")
        self._require_running()
        try:
            explained = self._objective.new_zeros(())
            chunk = max(1, (8 << 20) // (8 * self.d))
            for begin in range(0, self.k, chunk):
                end = min(begin + chunk, self.k)
                sums = self._pass_sums[begin:end].double()
                counts = self._pass_counts[begin:end]
                means = sums / counts.clamp_min(1)[:, None]
                explained.add_((sums * means).sum())
                self._centroids[begin:end].copy_(torch.where(
                    (counts > 0)[:, None], means.float(), self._centroids[begin:end]
                ))
            # Fixed-assignment SSE after taking means, without another scan:
            # sum_i ||x_i||^2 - sum_j ||S_j||^2 / N_j.
            objective = self._row_square - explained
            dist.all_reduce(objective, group=self._group)
            assignment_mse = objective.clamp_min(0).item() / rows
            self._lloyd_prepared = None
            super().end_epoch()
            self.iteration_stats[-1]["assignment_mse"] = assignment_mse
        except BaseException:
            self._failed = True
            raise

    def finalize(self):
        if self.centroids is not None:
            return self.centroids
        if self._epoch != len(self.recipe["epoch_rows"]):
            raise RuntimeError("finish all epochs before finalize")
        self._lloyd_prepared = None
        # Keep the means computed with FP64 counts, including counts > 2**24.
        # Use the common finalizer to package objectives and release buffers.
        centroids = self._centroids
        super().finalize()
        self.centroids = centroids
        self._row_square = None
        if hasattr(self, "_ivf"):
            self._ivf = None
        return self.centroids


class LloydShardedCentroids(_LloydEpoch, EmeansShardedCentroids):
    """BF16 exhaustive assignment with original FP32 row statistics."""

    def begin(self, n, initial):
        super().begin(n, initial)
        self.recipe.update(fine_dtype="bf16", statistics_dtype="float32")

    def _prepare_torch_assignment(self, centroids):
        # Keep the input representation FP32 for communication/statistics;
        # only the frozen assignment operands use BF16 tensor cores.
        return self._execution_backend.prepare_centroids(
            centroids, torch.bfloat16, spherical=self.spherical
        )

    def _local_assignment(self, rows, prepared, second_out=None):
        # The caller retains its FP32 gathered rows for owner accumulation.
        return super()._local_assignment(
            rows.to(torch.bfloat16), prepared, second_out=second_out
        )


class LloydShardedCentroidsIVF(_LloydEpoch, EmeansShardedCentroidsIVF):
    """Lloyd training with BF16 IVF search and original FP32 row statistics."""
