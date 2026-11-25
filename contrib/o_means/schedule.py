# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The training schedule of o-means, decided from (n, d, k, nepochs) alone.

The batch size, initial step size, cosine annealing, rows per epoch and
revival cadence are e-means', taken from faiss.contrib.e_means.base. o-means
adds the coarse cells and probes of the two-level assignment, the cadence at
which the cells are rebuilt, and the weight of each batch in the closing
mean.
"""

import math
from dataclasses import dataclass
from typing import Optional

from faiss.contrib.e_means.base import (
    _CHECK_UNUSED_EVERY,
    _epoch_rows,
    _resolve_recipe,
    _resolve_unused_threshold,
    _scheduled_alpha,
)

# Below this many epochs the closing mean weights every batch of the run by
# its recency; from here up it is e-means' mean over the last pass.
RECENCY_CLOSE_BELOW = 2.0

# The e-means paper's benchmarks use the throughput batch law up to this
# many rows and the quality law above. o-means was tuned and measured with
# the same rule.
THROUGHPUT_MAX_N = 2 * 10**6


def num_cells(k):
    """Coarse cells for k centroids: k // 128 rounded up to a multiple of 16."""
    return max(16, -(-(k // 128) // 16) * 16)


def num_probes(d, k):
    """Cells each row probes.

    The smallest count that kept the rate of rows missing their nearest
    centroid at about 1% or less on the datasets the rule was fitted on.
    """
    if k > 16384:
        return 24
    return 12 if d >= 512 else 16


def resolve_batch_law(n, batch_law):
    if batch_law == "auto":
        return "throughput" if n <= THROUGHPUT_MAX_N else "quality"
    if batch_law not in ("throughput", "quality"):
        raise ValueError('batch_law must be "auto", "throughput" or "quality"')
    return batch_law


@dataclass(frozen=True)
class Step:
    epoch: int
    b0: int                  # the batch is rows b0 .. b1 - 1 of this epoch's permutation
    b1: int
    alpha: float
    check: bool              # run a revival check after this update
    omega: Optional[float]   # weight in the closing mean, None if not part of it


@dataclass(frozen=True)
class Schedule:
    n: int
    d: int
    k: int
    nepochs: float
    batch_law: str
    batch_size: int
    alpha0: float
    epoch_rows: tuple
    steps: tuple
    unused_threshold: float
    recency_close: bool
    nlist: int
    nprobe: int
    rebuild_every: int


def make_schedule(n, d, k, nepochs, batch_law="auto", nlist=None, nprobe=None):
    nepochs = float(nepochs)
    if not nepochs > 0:
        raise ValueError("nepochs must be > 0")
    if not 1 <= k <= n:
        raise ValueError("need 1 <= k <= n, got k=%d for n=%d" % (k, n))
    law = resolve_batch_law(n, batch_law)
    # Below one epoch the one-epoch batch and step are kept, so that a
    # shorter run is a prefix of the one-epoch schedule.
    batch, alpha0 = _resolve_recipe(n, k, max(nepochs, 1.0), law, None, None)
    rows = tuple(_epoch_rows(n, nepochs))
    tau = _resolve_unused_threshold(None, d)
    recency = nepochs < RECENCY_CLOSE_BELOW

    if nlist is None:
        nlist = min(num_cells(k), k)
    if not 1 <= nlist <= k:
        raise ValueError("nlist must be between 1 and k")
    if nprobe is None:
        nprobe = min(num_probes(d, k), nlist)
    if not 1 <= nprobe <= nlist:
        raise ValueError("nprobe must be between 1 and nlist")
    steps_per_pass = -(-n // batch)
    rebuild_every = max(_CHECK_UNUSED_EVERY, -(-steps_per_pass // 2))

    total_steps = sum(-(-r // batch) for r in rows)
    total_rows = sum(rows)
    last = len(rows) - 1
    steps, t, done = [], 0, 0
    for epoch, epoch_rows in enumerate(rows):
        for b0 in range(0, epoch_rows, batch):
            b1 = min(b0 + batch, epoch_rows)
            done += b1 - b0
            if recency:
                # Weight exp(-(rows processed after this batch) / n). With
                # fewer than 2 epochs this covers the whole run.
                omega = math.exp(-((total_rows - done) / n))
            else:
                omega = 1.0 if epoch == last else None
            steps.append(Step(epoch, b0, b1, _scheduled_alpha(alpha0, t, total_steps),
                              tau > 0 and (t + 1) % _CHECK_UNUSED_EVERY == 0, omega))
            t += 1
    return Schedule(n=n, d=d, k=k, nepochs=nepochs, batch_law=law, batch_size=batch,
                    alpha0=alpha0, epoch_rows=rows, steps=tuple(steps), unused_threshold=tau,
                    recency_close=recency, nlist=nlist, nprobe=nprobe,
                    rebuild_every=rebuild_every)
