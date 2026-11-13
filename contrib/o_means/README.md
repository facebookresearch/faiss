# o-means: e-means with a two-level assignment

o-means is a GPU k-means trainer built on [e-means](../e_means/README.md). It keeps e-means' objective, starting centroids, mini-batch EMA update, batch and step laws and split revival, and changes two things: each training row is compared only with the centroids of a few nearby coarse cells instead of with all k centroids, and short runs end with a recency-weighted mean. On one RTX 5090 it reached the error of 1- to 3-epoch e-means runs 4.5 times faster on average, over eight held-out datasets.

## When to use it

o-means is for training large k (thousands to tens of thousands of centroids) on one NVIDIA GPU, with the data held there as a float32 PyTorch tensor and a budget of a few epochs. In that setting nearest-centroid assignment is almost all of e-means' cost, and that is the part o-means makes cheaper. It was evaluated with k from 8,192 to 65,536, d from 96 to 3,072 and n from 240,000 to 10 million.

Use e-means instead when you need the CPU, data that does not fit in GPU memory (e-means lets you feed the batches yourself), several GPUs, spherical clustering, FP8 or INT8 data, or a search over all k centroids for every row during training. With k of about 2,000 or less every row probes all or most cells, so there is nothing to gain.

## Quick start

o-means needs CUDA-enabled PyTorch and Triton (the Linux CUDA builds of PyTorch include Triton), on an NVIDIA GPU with compute capability 8.0 or newer. It was developed and measured on an RTX 5090.

```python
import numpy as np
import torch
from faiss.contrib.o_means import Omeans

x = np.random.rand(1_000_000, 96).astype("float32")
xg = torch.from_numpy(x).cuda()               # float32, already on the GPU

om = Omeans(96, 16384, nepochs=2)
centroids, labels = om.train(xg)              # (16384, 96) float32, last-pass labels
distances, nearest = om.assign(xg[:10_000])   # full search over all k centroids, in FP16
print(om.recipe["nlist"], om.recipe["nprobe"], om.recipe["batch_size"])
```

`Omeans` takes the same leading arguments as `Emeans` and returns the same things from `train` and `assign`, so switching an existing e-means script is usually a matter of replacing `Emeans(d, k, nepochs=...)` with `Omeans(d, k, nepochs=...)`. `train(x, init_centroids)` accepts the starting centroids explicitly; without them it draws the same k rows that `Emeans` would draw with the same seed. For a single call, `train_omeans(x, k, nepochs=2)` returns the centroids.

## How it works

The training loop is e-means'. Each epoch visits the rows in a fresh random order, in batches whose size comes from e-means' batch law. A batch is assigned to its nearest centroids, each centroid that received rows moves towards their mean with an EMA step that follows a cosine schedule, and every five updates, centroids whose running count has fallen below a tenth of the mean are revived by splitting a busy centroid in two. The README of e-means describes these parts in detail.

**Two-level assignment.** The k centroids are grouped into `nlist` coarse cells by running a few Lloyd iterations on the centroids themselves, and the centroid table is kept sorted by cell so that each cell is a contiguous block. A row is first scored against the `nlist` cell centres; it then takes its `nprobe` nearest cells and is compared with the centroids in those cells only. With k = 16,384 there are 128 cells of 128 centroids on average, so a row probing 16 cells evaluates about 128 + 16 × 128 = 2,176 distances instead of 16,384. On the GPU this is three kernel launches per batch: one scores the cells and sorts the (row, cell) pairs into per-cell buckets with atomic counters, one scans each probed cell's centroid block on tensor cores, and one keeps each row's best result. No value is read back to the host during training.

The search is approximate: a row whose nearest centroid sits in a cell it does not probe gets the best centroid of the cells it does probe. On DEEP 1M, SIFT 1M, arXiv-Nomic-768 and Landmark-Nomic-768, 0.5 to 2.4% of rows missed their nearest centroid at the default `nprobe` (GloVe-200 did worse, see Limitations).

**Rebuilding the cells.** The centroids move during training, so the cells are rebuilt periodically, starting from the previous cell centres. After the first pass this happens every half pass, with two Lloyd iterations. During the first pass revival is very active (thousands of splits on a 16,384-centroid problem), and a new child centroid is written into the table slot of the dead centroid it replaces, which belongs to a cell far from where the child now is. So during the first pass the cells are rebuilt after every revival check, without Lloyd iterations: the centroids are only re-sorted into the existing cells. Without these early rebuilds, 10 to 28% of the first pass's rows missed their nearest centroid. Between two rebuilds, each split also records the child as a link on its donor, and the assignment compares every row with the child and grandchild of its best match, so rows near a donor can reach the centroids split from it.

**Closing mean.** e-means ends by setting each centroid to the mean of the rows assigned to it during the last pass. Below 2 epochs, o-means instead weights every batch of the run by exp(−r/n), where r is the number of rows processed after that batch, and sets each centroid to the weighted mean of its rows. The last pass still dominates, but the rows of the earlier part of the run are not wasted, which matters when there is only one pass or less. When a centroid is split, donor and child each keep half of the donor's accumulated mass. At 1 epoch this lowered the error by 0.34 to 0.70% compared with the last-pass mean, on the development datasets listed under Results. At 2 epochs and above the last-pass mean was better, so o-means uses it there.

**Numerics.** Rows and centroids are compared in FP16, with FP32 accumulation. Data whose largest magnitude is outside [2^-8, 2^15] is first rescaled by a power of two, which is exact. Batch sums are accumulated as int64 fixed-point values (every FP16 number is a multiple of 2^-24), so they are exact and do not depend on the order of the atomic additions. The centroids themselves are float32.

## Parameters

Only `d` and `k` are required. The defaults of `nlist` and `nprobe` are the rule o-means was tuned and measured with.

| Setting | Default | What it does |
|---|---|---|
| `d` | required | Dimension of the vectors |
| `k` | required | Number of clusters |
| `nepochs` | `5` | Passes over the data, can be fractional. Below 1, the batch size and step size of a 1-epoch run are used |
| `batch_law` | `"auto"` | e-means' batch law: `"throughput"`, `"quality"`, or `"auto"`, which picks throughput up to 2 million rows and quality above, as in the e-means paper's benchmarks. This differs from `Emeans`, whose `"auto"` decides by dimension |
| `seed` | `1234` | Seeds the starting centroids, the shuffling, the revivals and the first cell build |
| `nlist` | k // 128 rounded up to a multiple of 16 (at least 16) | Number of coarse cells |
| `nprobe` | 16; 12 if d ≥ 512; 24 if k > 16,384 | Cells each row probes |
| `capture_assignments` | `True` | Keeps each row's label from the last pass (8 bytes per row) |

After `train`, the model holds `centroids`, `assignments` (the labels also returned by `train`) and `recipe`, a dictionary with the batch size, step size, rows per epoch, cells, probes, rebuild cadence and closing mean actually used. The labels come from the last pass, while the centroids were still moving and with the approximate search; call `assign` for labels from a full search over all k centroids (in FP16, like training). When `nepochs` is fractional below 1, rows the run did not visit get label -1.

Besides the input, training keeps an FP16 copy of the data (half the size of the float32 input), about 22 bytes per centroid coordinate, and per-cell buckets of 4 × `nlist` × batch size bytes. Peak GPU memory on the held-out datasets below was at most 7.5 GB, data included. If the FP16 copy would not fit, each batch is gathered and converted on the fly instead, with identical results.

Runs are bitwise reproducible for a given input, seed and GPU when `torch.use_deterministic_algorithms(True)` is set. Otherwise the float32 `index_add_` used in the cell build can change the last bits of the cell centres between runs, and through them, rarely, the cell of a centroid. The centroids then differ at the level of floating-point noise (about 1e-6 relative in error in our runs).

## Results

All numbers below were measured on one RTX 5090 (32 GB). e-means ran its Triton kernels, because its CuTe kernels need an H100, with FP32 data converted to BF16 as in its own benchmarks and the same batch law as o-means. Both started from the same centroids. Speedup at T epochs is e-means' time for a T-epoch run divided by the time o-means needs to reach the same error, interpolated between the efforts o-means was run at (0.8, 1, 1.6, 2, 2.5, 3.2, 4, 5, 8 and 12.8 epochs). Errors are measured on held-out rows with exact float32 assignment. Times and errors are averaged over 5 seeds (3 for SPACEV 10M); times include everything the training call does but not loading the data.

o-means was developed and tuned on nine datasets: DEEP 1M and 10M, SIFT 1M and 10M, GloVe-200, arXiv-Nomic-768, Landmark-Nomic-768, CC-News-768 and CelebA-2048. The eight datasets below were held out during that work and were run once, on the final version.

| Dataset | n | d | k | Speedup at 1 epoch | at 2 | at 3 | Error vs e-means at e-means' 5-epoch time | Distance evaluations vs e-means, at 1 / 2 / 3 |
|---|---|---|---|---|---|---|---|---|
| SPACEV 1M | 1,000,000 | 100 | 16,384 | 3.77 | 3.74 | 3.77 | −0.13% | 0.14 / 0.17 / 0.15 |
| SPACEV 10M | 10,000,000 | 100 | 65,536 | 7.25 | 6.28 | 6.18 | −0.24% | 0.06 / 0.07 / 0.08 |
| ImageNet-CLIP | 1,000,000 | 512 | 16,384 | 5.22 | 4.84 | 5.11 | −0.12% | 0.11 / 0.11 / 0.15 |
| GooAQ | 1,455,024 | 768 | 16,384 | 4.86 | 4.56 | 3.55 | −0.19% | 0.11 / 0.14 / 0.19 |
| CodeSearchNet | 1,354,067 | 768 | 16,384 | 4.69 | 3.94 | 3.28 | −0.16% | 0.11 / 0.18 / 0.19 |
| AG-News | 749,382 | 1,024 | 16,384 | 4.53 | 4.75 | 4.42 | −0.49% | 0.11 / 0.11 / 0.15 |
| Landmark-DINO | 740,757 | 768 | 16,384 | 4.58 | 4.63 | 5.00 | −0.18% | 0.11 / 0.13 / 0.14 |
| SimpleWiki | 240,372 | 3,072 | 8,192 | 4.84 | 3.96 | 2.54 | −0.19% | 0.22 / 0.22 / 0.30 |

The geometric mean of the 24 speedups is 4.49, and the smallest is 2.54. Counting only the efforts o-means was actually run at, without interpolating between them, the geometric mean is 4.08. Given e-means' 5-epoch time, o-means' error was not worse than e-means' on any dataset and was lower by 0.12 to 0.49%; the noise of this comparison, measured by running e-means against itself with a different seed, is about 0.16%, so the margins below that (SPACEV 1M and ImageNet-CLIP) are within it. The last column is the number of point-to-centroid distances o-means computed to reach e-means' error, as a fraction of e-means' n × k × T: it does not depend on the hardware. The time saved is smaller than that fraction suggests: about 7 times fewer distance evaluations gave about 3 times less time on this GPU, because the two-level kernels gather scattered rows and the cells have to be rebuilt. How that ratio carries over to other hardware was not measured.

Part of the speedup comes from o-means' kernels rather than from the algorithm. Running e-means' own algorithm, with dense assignment, on o-means' FP16 kernels was 1.43 times faster than e-means on these datasets (1.51 on the first five development datasets listed above and 1.53 on all nine), and o-means was 3.13 times faster than that (2.94 and 3.02 on the development datasets). On an H100, where e-means uses its CuTe kernels, the first factor would likely be smaller.

## Limitations

The assignment during training is approximate, and there is no worst-case bound on how many rows miss their nearest centroid. Evaluated with exact assignment, the returned centroids were not worse than e-means' at e-means' 5-epoch time on any of the 17 real datasets tried (9 during development and the 8 above), but that is an empirical result.

The rule for `nprobe` was fitted on a handful of datasets. On GloVe-200 (L2-normalized word vectors), 4 to 13% of rows missed their nearest centroid at 16 probes, which cost about 0.1% in error. On a synthetic mixture of 65,536 Gaussian modes in 96 dimensions with k = 16,384 (the default data of the benchmark script), o-means' error at equal epochs was 1.2 to 1.7% above e-means' from 2 to 5 epochs with the default 16 probes. At 3 epochs, probing all 128 cells reduced the gap from 1.65% to 0.23%. At equal time o-means' error was still lower than e-means'. If your data behaves like that, a larger `nprobe` trades speed for accuracy.

On synthetic, well-separated mixtures in 1,024 dimensions and more, the Lloyd iterations on the centroids can pull a few cell centres between many groups, and those cells collect most of the centroids. Rows then scan a large part of k and the speedup over dense assignment mostly disappears (1.04 to 1.16 times on a 1M × 2048 mixture); the error is unaffected. This was not seen on any real dataset.

The gain is smallest at small k. The three datasets with k = 8,192 (SimpleWiki above, and CC-News and CelebA during development) had the lowest speedups, 3.1 to 3.7 on average per dataset: with 64 cells, a row probing 12 of them still covers about a fifth of the centroids.

o-means runs on one CUDA GPU only, takes float32 input held on that GPU, and has no caller-driven batch loop, spherical mode or quantized formats. Values of k above 65,536 and dimensions of 16 or less were not evaluated. All measurements come from one GPU model.

## Tests and benchmark

```bash
python -m unittest discover -s tests -p 'test_o_means.py'          # schedule, no GPU needed
python -m unittest discover -s tests -p 'torch_test_o_means.py'    # kernels and training, CUDA + Triton
python benchs/o_means/bench_o_means.py --nepochs 1 2 3             # o-means and e-means, time and error
```

The GPU tests are skipped when CUDA or Triton is not available.
