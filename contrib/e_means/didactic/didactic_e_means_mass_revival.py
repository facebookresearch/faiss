# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""A small, readable E-means example. Run: python didactic_e_means_mass_revival.py

Faiss supplies NumPy data; every vector computation below uses CPU Torch.
No production E-means code is imported. See didactic_e_means_mass_revival.md for a walkthrough.
Paper: https://arxiv.org/abs/2611.09501
"""

import math
import time

import torch
from faiss.contrib.datasets import SyntheticDataset


# 1. Choose a small experiment that runs on a laptop.
# Edit these constants to try a different dataset size or number of clusters.
torch.set_default_device("cpu")
torch.set_default_dtype(torch.float32)
torch.set_num_threads(4)
torch.set_grad_enabled(False)  # Clustering uses explicit updates, not autograd.
rng = torch.Generator(device="cpu").manual_seed(0)
n, d, k, epochs = 50_000, 32, 128, 3
dataset = SyntheticDataset(d=d, nt=n, nb=0, nq=2_000, seed=1234)
train = torch.from_numpy(dataset.get_train())
test = torch.from_numpy(dataset.get_queries())
# SyntheticDataset generates separate training and query rows from the same
# distribution. The query rows are used only to measure generalization.
print(f"Data: train={tuple(train.shape)}, test={tuple(test.shape)}, CPU float32")


# 2. Couple the batch size and EMA step using the fitted quality recipe.
# These are the coefficients shipped in Faiss, rather than Table 2's fits.
# Round the batch to a power of two and cap it at half the training set.
raw_batch = 0.00439175 * k**0.134749 * n**0.876473 * epochs**0.813344
batch_size = min(2 ** round(math.log2(raw_batch)), n // 2)
alpha0 = 0.8244 * k**-0.4617 * batch_size**0.9003 * n**-0.4498 * epochs**-0.1057
alpha0 = min(1.0, max(1e-4, alpha0))
steps_per_epoch = math.ceil(n / batch_size)
total_steps = epochs * steps_per_epoch
print(f"Recipe: k={k}, epochs={epochs}, batch={batch_size}, alpha0={alpha0:.4f}")


def assign(rows, centroids):
    """Return the exact nearest centroid, evaluating all k candidates."""
    # ||x-c||² = ||x||² + ||c||² - 2 x.c. Only a (batch, k) table is needed.
    distances = (
        rows.square().sum(1, keepdim=True)
        + centroids.square().sum(1)[None, :]
        - 2 * rows @ centroids.T
    )
    labels = distances.argmin(1)
    # Recompute the selected distances by direct differences for reporting.
    # This avoids cancellation in the expanded formula above.
    squared_errors = (rows - centroids[labels]).square().sum(1)
    return labels, squared_errors


def batch_statistics(rows, labels):
    """For each cluster j, collect the batch's vector sum s_j and count n_j."""
    sums = torch.zeros(k, d)
    sums.index_add_(0, labels, rows)
    counts = torch.bincount(labels, minlength=k).float()
    return sums, counts


def mse(rows, centroids):
    """Fresh exhaustive mean squared L2 error per vector, evaluated in chunks."""
    error = torch.zeros(())
    for chunk in rows.split(batch_size):
        _, squared_errors = assign(chunk, centroids)
        error += squared_errors.sum()
    return (error / len(rows)).item()


# 3. Initialize centroids from distinct training rows.
# S_j and N_j are the EMA vector sum and mass; c_j = S_j / N_j.
# A tiny initial mass makes the first assigned batch almost replace c_j.
initial_rows = torch.randperm(n, generator=rng)[:k]
centroids = train[initial_rows].clone()
mass = torch.full((k,), 2**-16)
ema_sums = centroids * mass[:, None]
split_size = 0.01 * train.std().item()
revivals = 0
step = 0
print(f"Initialization: {k} sampled centroids, test MSE={mse(test, centroids):.4f}")
started = time.perf_counter()


# 4. Visit the training rows once per epoch, in a new random order.
for epoch in range(epochs):
    order = torch.randperm(n, generator=rng)
    moving_error = torch.zeros(())
    final_pass = epoch == epochs - 1
    if final_pass:
        # These are ordinary totals over the last pass, with no EMA decay.
        # They will remove the remaining EMA weighting at the end.
        final_sums = torch.zeros(k, d)
        final_counts = torch.zeros(k)

    for indices in order.split(batch_size):
        rows = train[indices]

        # 5. Assign this batch against the current, still fixed centroids.
        labels, squared_errors = assign(rows, centroids)
        sums, counts = batch_statistics(rows, labels)
        moving_error += squared_errors.sum()
        if final_pass:
            final_sums += sums
            final_counts += counts

        # 6. Anneal alpha from alpha0 towards 0.1 * alpha0 over the whole run.
        # All clusters decay, including ones that receive no rows this batch.
        cosine = 0.5 * (1 + math.cos(math.pi * step / total_steps))
        alpha = alpha0 * (0.1 + 0.9 * cosine)
        ema_sums = (1 - alpha) * ema_sums + alpha * sums
        mass = (1 - alpha) * mass + alpha * counts
        centroids = ema_sums / mass[:, None]
        step += 1

        # 7. Minimal revival: every five updates, replace at most one weak
        # cluster by splitting the cluster with the largest EMA mass.
        # This intentionally omits removal costs, Oja axes and split history.
        # No revival in the last pass: cluster identities must remain stable
        # while we accumulate final_sums and final_counts.
        if not final_pass and step % 5 == 0:
            victim = mass.argmin().item()
            if mass[victim] < 0.1 * mass.mean():
                donor = mass.argmax().item()
                center = centroids[donor].clone()
                direction = torch.randn(d, generator=rng)
                offset = split_size * direction / direction.norm()
                centroids[donor] = center + offset
                centroids[victim] = center - offset
                mass[donor] /= 2
                mass[victim] = mass[donor]
                ema_sums[donor] = mass[donor] * centroids[donor]
                ema_sums[victim] = mass[victim] * centroids[victim]
                revivals += 1

    # Moving MSE uses each batch's pre-update centroids; test MSE uses the
    # current centroids and fresh assignments. They are different measures.
    print(
        f"Epoch {epoch + 1}/{epochs}: batches={steps_per_epoch}, "
        f"moving train MSE={moving_error.item() / n:.4f}, "
        f"test MSE={mse(test, centroids):.4f}, revivals={revivals}, "
        f"elapsed={time.perf_counter() - started:.2f}s"
    )


# 8. Finalize with unweighted means under the last pass's assignments.
# The centroids moved during that pass; this is not a Lloyd iteration with
# assignments to one fixed centroid table. Empty clusters keep their position.
occupied = final_counts > 0
centroids[occupied] = final_sums[occupied] / final_counts[occupied, None]
print(
    f"Final means: occupied={occupied.sum().item()}/{k}, "
    f"fresh train MSE={mse(train, centroids):.4f}, "
    f"test MSE={mse(test, centroids):.4f}, revivals={revivals}, "
    f"elapsed={time.perf_counter() - started:.2f}s"
)
