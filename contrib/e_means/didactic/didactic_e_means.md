# E-means, one batch at a time

[didactic_e_means.py](didactic_e_means.py) is a standalone, commented example of
the main training loop in [E-Means: Scaling Laws for Mini-Batch
K-Means](https://arxiv.org/abs/2611.09501). It imports no Faiss E-means machinery.
Faiss provides a synthetic NumPy dataset; all vector arithmetic, assignments,
statistics and random training draws use CPU Torch in float32, without autograd.

Run it with Python, PyTorch and Faiss installed:

```bash
python contrib/e_means/didactic/didactic_e_means.py
```

The script takes no arguments. Its eight numbered stages correspond to the
sections below. Change the constants at the top to explore other sizes.

## 1. Create training and test data

[Source: CPU settings and dataset](didactic_e_means.py#L20-L33).

```python
rng = torch.Generator(device="cpu").manual_seed(0)
n, d, k, epochs = 50_000, 32, 128, 3
dataset = SyntheticDataset(d=d, nt=n, nb=0, nq=2_000, seed=1234)
train = torch.from_numpy(dataset.get_train())
test = torch.from_numpy(dataset.get_queries())
```

`SyntheticDataset` creates a nonlinear, low-dimensional distribution embedded
in a larger vector space. The example uses 50,000 training vectors in 32
dimensions, 128 centroids, three epochs and 2,000 separate test vectors.

`get_train()` supplies the clustering data; `get_queries()` supplies the test
data. Both come from the same synthetic distribution. Test vectors never enter
the training loop. No database vectors or nearest-neighbor ground truth are
needed. One seeded Torch generator controls initialization, permutations and
revival directions.

## 2. Choose the batch size and EMA step together

[Source: coupled recipe](didactic_e_means.py#L36-L45).

```python
raw_batch = 0.00439175 * k**0.134749 * n**0.876473 * epochs**0.813344
batch_size = min(2 ** round(math.log2(raw_batch)), n // 2)
alpha0 = 0.8244 * k**-0.4617 * batch_size**0.9003 * n**-0.4498 * epochs**-0.1057
alpha0 = min(1.0, max(1e-4, alpha0))
steps_per_epoch = math.ceil(n / batch_size)
total_steps = epochs * steps_per_epoch
```

E-means couples the batch size to the initial update weight. The example uses
the fitted **quality** recipe shipped in Faiss:

$$
B \approx 0.00439175\,k^{0.134749}n^{0.876473}T^{0.813344},
$$

rounded to a power of two and capped at half the training set, followed by

$$
\alpha_0 = 0.8244\,k^{-0.4617}B^{0.9003}n^{-0.4498}T^{-0.1057},
$$

clamped to $[10^{-4},1]$. Here $n$ is the training size, $k$ the number
of centroids and $T$ the epoch budget. These are empirical fits; the shipped
coefficients differ from the paper's Table 2. They are not optimality guarantees
for this small synthetic experiment.

## 3. Initialize positions, sums and masses

[Source: initial state](didactic_e_means.py#L80-L91).

```python
initial_rows = torch.randperm(n, generator=rng)[:k]
centroids = train[initial_rows].clone()
mass = torch.full((k,), 2**-16)
ema_sums = centroids * mass[:, None]
```

Sample distinct training rows for the initial centroids. Maintain two quantities
per cluster:

| Code | Meaning | Shape |
|---|---|---|
| `ema_sums` | Exponentially weighted vector sums $S_j$ | $k\times d$ |
| `mass` | Exponentially weighted counts $N_j$ | $k$ |
| `centroids` | Positions $c_j=S_j/N_j$ | $k\times d$ |

Initialize $N_j=2^{-16}$ and $S_j=N_jc_j$. This small positive mass
keeps every initial position defined and gives the first assigned rows a large
influence.

## 4. Shuffle once per epoch

[Source: epoch permutation](didactic_e_means.py#L94-L98) and
[batch iteration](didactic_e_means.py#L105-L106).

```python
for epoch in range(epochs):
    order = torch.randperm(n, generator=rng)
```

Each epoch generates a new permutation and visits every training row once.
The final, potentially shorter batch is included. This example deliberately
uses integer epochs and keeps the whole dataset in RAM.

## 5. Assign a batch and collect its statistics

[Source: assignment](didactic_e_means.py#L48-L60),
[batch statistics](didactic_e_means.py#L63-L68), and their
[use in the training loop](didactic_e_means.py#L108-L114).

```python
sums = torch.zeros(k, d)
sums.index_add_(0, labels, rows)
counts = torch.bincount(labels, minlength=k).float()
return sums, counts
```

For each row $x$, find the nearest current centroid by squared L2 distance:

$$
a(x)=\arg\min_j\|x-c_j\|^2.
$$

`assign()` evaluates every candidate using
$\|x\|^2+\|c\|^2-2x\cdot c$. This needs a $B\times k$ distance
matrix, rather than a $B\times k\times d$ array of differences. There is
no IVF or quantization. Assignment is exhaustive float32 computation, subject
to floating-point rounding. Ties go to the first centroid index.

`batch_statistics()` collects

$$
s_j=\sum_{x:a(x)=j}x,\qquad n_j=\#\{x:a(x)=j\}.
$$

`index_add_` accumulates vector sums and `bincount` accumulates counts. Reported
distances are recomputed from direct differences to the selected centroids.

## 6. Update the sum-and-count EMAs

[Source: cosine schedule and EMA update](didactic_e_means.py#L116-L123).

```python
cosine = 0.5 * (1 + math.cos(math.pi * step / total_steps))
alpha = alpha0 * (0.1 + 0.9 * cosine)
ema_sums = (1 - alpha) * ema_sums + alpha * sums
mass = (1 - alpha) * mass + alpha * counts
centroids = ema_sums / mass[:, None]
step += 1
```

At update $t$, out of $U$ total updates, use cosine decay:

$$
\alpha_t=\alpha_0\left[0.1+0.9\frac{1+\cos(\pi t/U)}2\right].
$$

The updates start at $t=0$, so the last update approaches the floor rather
than landing exactly on it. Update every cluster with

$$
S_j\leftarrow(1-\alpha_t)S_j+\alpha_ts_j,\qquad
N_j\leftarrow(1-\alpha_t)N_j+\alpha_tn_j,\qquad
c_j\leftarrow S_j/N_j.
$$

This is an EMA of **sums and counts**, whose ratio defines the position.
For an occupied cluster, the equivalent centroid update is

$$
c_j^{\mathrm{new}}=c_j^{\mathrm{old}}+
\frac{\alpha_t n_j}{N_j^{\mathrm{new}}}
\left(\frac{s_j}{n_j}-c_j^{\mathrm{old}}\right).
$$

Thus the effective movement depends on the cluster's current batch count and
historical mass. If a cluster receives no rows, both its sum and mass decay by
the same factor, keeping its position unchanged.

## 7. Revive one weak cluster

[Source: simplified revival](didactic_e_means.py#L125-L143).

```python
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
```

Every five updates, find the cluster with the smallest EMA mass. If its mass
is below one tenth of the average, replace it by a child of the cluster with
the largest mass:

1. Draw a random unit direction and a small offset.
2. Place the donor and replacement on opposite sides of the donor's position.
3. Give each half of the donor's mass and reconstruct their EMA sums.

The offset length is 1% of the training data's scalar standard deviation.
At most one cluster is revived per check. This makes the example readable;
the direction and donor choice are intentionally crude.

Revival is disabled during the final pass. Otherwise a centroid ID could
refer to different clusters while accumulating final totals. The production
implementation permits final-pass revival and handles the affected assignments.

## 8. Finish with ordinary last-pass means

[Source: last-pass totals](didactic_e_means.py#L99-L114) and
[final means](didactic_e_means.py#L155-L165).

```python
occupied = final_counts > 0
centroids[occupied] = final_sums[occupied] / final_counts[occupied, None]
```

During the last pass, collect `final_sums` and `final_counts` without EMA
weighting. At the end, replace each occupied centroid with their ratio. Keep
the positions of empty clusters.

These assignments were made against **moving** centroids as batches arrived.
The final means therefore differ from a Lloyd iteration, which assigns a whole
pass against one fixed centroid table before updating it. The finalization
removes exponential weighting from the last pass's assigned vectors; it does
not perform another assignment pass.

The [evaluation helper](didactic_e_means.py#L71-L77),
[epoch log](didactic_e_means.py#L145-L152), and
[final log](didactic_e_means.py#L160-L165) report three clearly distinguished errors:

- **Moving train MSE:** batch errors against the pre-update centroids, averaged
  over the epoch.
- **Fresh train MSE:** exhaustive reassignment of training rows to the final
  centroids, measured only after finalization.
- **Test MSE:** exhaustive assignment of separate test rows to the current or
  final centroids.

All are mean squared L2 distance **per vector**, without dividing by dimension.
Evaluation never changes the centroids.

## What this example simplifies

The sum-and-count EMA, coupled recipe, cosine schedule and last-pass means
show the core E-means loop. Revival here uses mass and a small random split;
it omits the production removal-cost criterion, Oja direction estimates and
Gaussian split size. Freezing revival during the last pass also avoids split
bookkeeping. The resulting numbers should not be presented as a reproduction
of the paper or of current Faiss.

The code has one resident dataset, one CPU process and exhaustive assignment.
It has no class hierarchy, lifecycle hooks, kernels, GPU code, distributed
collectives or SSD pipeline. The largest assignment temporary contains
`batch_size * k` float32 values. Evaluation and initialization are included in
the example's work; the printed elapsed time starts after initialization and
includes subsequent evaluations.
