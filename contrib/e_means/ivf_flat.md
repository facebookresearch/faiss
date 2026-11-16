# IVF assignment

## 1. IVF and big-batch search

An inverted file (IVF) index divides the vectors to search into `nlist`
lists. A coarse quantizer, trained by k-means, defines these lists: each
indexed vector goes into the list of its nearest coarse centroid. To answer
a query, the index finds its `nprobe` nearest coarse centroids and searches
only their lists. Increasing `nprobe` trades more work for better recall.
IVF-Flat stores the vectors themselves, without product quantization.

For k-means assignment, the **data vectors are queries** and the **k-means
centroids are the indexed vectors**. These k-means centroids are different
from the IVF coarse centroids. With `k` indexed centroids and balanced
lists, each query searches about `k * nprobe / nlist` candidates, plus the
coarse-routing work. Limited probing makes assignment approximate, even
when the distances within a list are computed in FP32.

### Big-batch IVF trained on data vectors

When assigning a large batch, queries can be grouped by the lists they
probe. All queries in a group search the same candidate vectors, so their
dot products can be computed together as a matrix multiplication. Results
are then merged across the probed lists. This replaces many independent
query scans with operations that reuse candidate data across queries.

A useful variant for clustering trains the coarse quantizer on a sample of
the **data vectors**, rather than the indexed k-means centroids. The coarse
partition then follows the query distribution. Data vectors are routed to
their nearest coarse cells, while the moving k-means centroids are inserted
into the corresponding inverted lists. With a fixed coarse quantizer,
geometric query routing can be reused; centroid list membership must be
rebuilt as the k-means centroids move. Filtering empty lists can also require
refreshing the probes.

The matrix search organization and the choice of coarse training data are
separate decisions. The e-means trainers described below use big-batch
matrix search with the Torch engine, but currently train their coarse
quantizer on a sample of the **initial k-means centroids**. They do not
train it on the input data vectors. The lower-level engines accept an
externally trained coarse quantizer, including one trained on data.

## 2. CPU and GPU implementations

The shared assignment layer in [ivf_assignment.py](ivf_assignment.py)
provides two engines:

| Engine | CPU | GPU | Search organization |
|---|---|---|---|
| `TorchIVFEngine` | FP32 | BF16 fine search | Pack vectors by list, group queries, compute batched dot products |
| `FaissIVFEngine` | FP32 | Unsupported | Native `faiss.IndexIVFFlat` inverted-list scans with OpenMP |

Both support routing, externally supplied probe IDs, and the best one or
two distinct candidates. They return search scores with the query norm
omitted:

```text
score(q, c) = ||c||² - 2 q·c
```

Adding `||q||²` gives squared L2 distance. Omitting this term preserves the
ordering for a query and lets different centroid owners compare scores.
Missing results have score `inf` and ID `-1`.

### CPU

The Torch engine uses the packed-list implementation in
[ivf_flat.py](ivf_flat.py), with native FP32 operands. Queries and candidate
lists are processed in chunks; padded batched matrix multiplication handles
groups with different list lengths. This shares the search organization
with the GPU implementation.

The Faiss engine uses a fixed `IndexFlatL2` coarse quantizer. Rebuilding the
index computes list memberships and inserts vectors with `add_core`;
search uses `search_preassigned` to scan the selected lists. It scans all
candidates in those lists (`max_codes=0`). This engine uses native Faiss
distance kernels and OpenMP rather than Torch's matrix search.

The caller sets Torch and Faiss thread counts and CPU affinity. The trainers
do not change the process-wide thread pools. Torch resolves equal scores
by lowest candidate ID. Faiss scanning requires an explicit
`ivf_tie_policy="faiss"`: a tied lower-ID candidate may be omitted at the
scanner's heap boundary, so sorting the returned pair cannot provide the
same tie guarantee as Torch.

### GPU

The Torch engine assigns indexed vectors to coarse lists with FP16 Triton
assignment, then uses BF16 operands for fine search. Ordinary matrix and
batched matrix products produce FP32 dot outputs. Triton kernels compute
the distance epilogue and merge candidates across list chunks and probes;
top-2 search computes each fine dot product once.

The lower-level index also offers ragged grouped matrix multiplication via
`search_1nn_batched(queries, grouped=True)`. Ordinary `search_1nn` uses the
padded path. Grouped multiplication retains its selected output precision:
FP16 output can overflow and BF16 output can round near ties. Converting
that output to FP32 afterwards does not recover the lost precision.

The raw `IndexIVFFlat` defaults to FP16 reduced operands. `TorchIVFEngine`
selects native FP32 on CPU and BF16 fine search on CUDA. For direct index
use, `implem < 3` selects native operands; `implem >= 3` selects `dtype`,
which must be `"fp16"` or `"bf16"`. `lowp_scope="fine"` keeps query routing
native, while `"all"` also uses reduced operands for routing. Changing the
stored candidate representation requires another `add` call.

`search_workspace_bytes` bounds the fine-search planner's workspace, not
total memory use. Index storage, coarse routing, query sorting, statistics
and allocator overhead are additional. A contiguous FP32 coarse table may
alias Torch index storage and must remain fixed. Rebuilds and searches on
the same index must be ordered by the caller.

## 3. Distribution: replicas and shards

In both layouts, workers share a coarse quantizer and split the training
data. The main difference is whether each worker keeps all k-means
centroids or owns only a subset.

### Replicated centroids

Each worker holds a complete centroid table and IVF index, and assigns its
own data vectors locally. Workers then reduce centroid sums, counts and
revival statistics before applying one global EMA update. Thus every
replica advances from the same global batch. The coordinator selects
revivals and broadcasts the selected donor and target IDs.

`DistributedEmeansIVF` implements this layout on CPU with Gloo. Group rank
zero supplies the initial centroid values, trains the coarse quantizer and
broadcasts it. Common initial Oja directions and synchronized revival keep
the numerical state consistent across replicas. `transport="dense"` reduces
full statistics tables; `"sparse"` merges occupied centroid IDs and their
statistics. `"auto"` chooses from the existing payload-size rule.

Replication avoids exchanging query vectors during assignment. Its cost is
a complete centroid/index/statistics copy per worker, plus statistics
communication at each update. It suits centroid tables that fit comfortably
on each worker.

### Sharded centroids

Each worker owns a contiguous range of global centroid IDs and indexes only
that shard. Every owner must search the queries for the global batch.
Readers route their local data first, then gather data vectors and probe IDs
to the owners. List occupancy is reduced globally so a reader can probe a
list that is empty on its own shard but populated elsewhere.

Owners search their local candidates and reduce the best score and global
ID. With the Torch engine, equal scores select the lowest global ID. For
cost-based revival, local top-2 results also supply the global second-best
score through an additional MIN reduction. Each owner accumulates and
updates the statistics of its winning centroids.

`EmeansShardedCentroidsIVF` implements this layout with CPU/Gloo or
CUDA/NCCL. The coarse quantizer is trained collectively on the initial
centroid shards. Revival selection gathers scalar summaries to the
coordinator; cross-owner splits transfer the donor's centroid and revival
state to the target owner. Each worker retains only its shard of centroid,
removal and Oja state.

Sharding reduces centroid memory per worker, at the cost of query exchange
and winner reductions. CPU uses FP32 search and state. CUDA uses BF16 fine
search and split separation, with FP32 statistical rows and state.

## 4. E-means with IVF

### Classes and shared assignment

```text
Emeans
└── QuantizedEmeans
    ├── EmeansIVF                         + _IVFAssignmentMixin
    │   └── DistributedEmeansIVF          + _ReplicatedCentroids
    └── EmeansShardedCentroids
        └── EmeansShardedCentroidsIVF     + _IVFAssignmentMixin
```

`IVFAssignment` owns coarse training, engine creation, index rebuilds and
invalidation. `_IVFAssignmentMixin` connects it to the local and sharded
trainers. The existing trainer classes own the epoch schedule, sufficient
statistics, EMA updates, revival and closing. The replicated class reuses
the shared `_ReplicatedCentroids` communication and revival hooks.

### Local CPU or GPU training

Here `x` and `x_test` are FP32 Torch tensors on the same device:

```python
from faiss.contrib.e_means import EmeansIVF

d = x.shape[1]
k = 32768
model = EmeansIVF(d, k, nepochs=2.6, seed=1,
                  nlist=256, nprobe=4, ivf_engine="torch")
centroids, labels = model.train(x)
squared_distances, ids = model.assign(x_test)
```

For a single GPU, pass CUDA tensors, for example `x.cuda()` and
`x_test.cuda()`. No process group is needed. The input rows remain FP32;
the GPU engine prepares BF16 operands for fine assignment internally.

To use native Faiss scanning on CPU:

```python
model = EmeansIVF(d, k, nepochs=2.6, seed=1,
                  nlist=256, nprobe=4,
                  ivf_engine="faiss", ivf_tie_policy="faiss")
centroids, labels = model.train(x)
```

### Distributed streaming training

Initialize a caller-owned Gloo group for CPU or NCCL group for GPU. For
replicated CPU training, every rank passes an initial tensor of shape
`(k, d)`; group rank zero supplies its values:

```python
from faiss.contrib.e_means import DistributedEmeansIVF

model = DistributedEmeansIVF(
    d, k, group=group, transport="dense", nepochs=2.6, seed=1,
    batch_size=global_batch, nlist=256, nprobe=4)
model.begin(global_n, initial_centroids)
```

For CPU or GPU centroid ownership, pass a local initial shard of shape
`(k // ranks, d)`, where `k` is the global centroid count:

```python
from faiss.contrib.e_means import EmeansShardedCentroidsIVF

model = EmeansShardedCentroidsIVF(
    d, k, group=group, nepochs=2.6, seed=1,
    batch_size=global_batch, nlist=256, nprobe=4)
model.begin(global_n, local_initial_centroids)
```

Both models use the following loop. `x_local` is this rank's equally sized
data shard, with `global_n = ranks * len(x_local)`. All ranks use matching
recipes; global batches and epoch budgets must be divisible by `ranks`.

```python
import torch
import torch.distributed as dist

ranks = dist.get_world_size(group)
local_batch = model.recipe["batch_size"] // ranks
for budget in model.recipe["epoch_rows"]:
    local_budget = budget // ranks
    order = torch.randperm(len(x_local), device=x_local.device)
    for start in range(0, local_budget, local_batch):
        stop = min(start + local_batch, local_budget)
        rows = x_local[order[start:stop]]
        model.update(model.accumulate(rows))
    model.end_epoch()
centroids = model.finalize()
```

Replicated `finalize` returns the full centroid table; sharded `finalize`
returns the local shard. `DistributedEmeansIVF` uses this streaming lifecycle
and has no resident `train` method. Data can also be streamed from SSD rather
than held in `x_local`; supply the same prescribed number of rows per rank.
Smaller `accumulate` calls can build a local batch incrementally before
`update`. Replicated partial accumulation does not communicate. Sharded
accumulation gathers queries and reduces winners, so ranks must make
matching calls with equal row counts.

### A data-trained coarse quantizer

The lower-level API can implement the data-trained big-batch variant from
section 1. Here `x` and `candidates` are FP32 CPU tensors; `candidates` are
the current k-means centroids:

```python
import faiss
import torch
from faiss.contrib.e_means.ivf_assignment import TorchIVFEngine

coarse_training = faiss.Kmeans(x.shape[1], 256, niter=10, seed=1)
coarse_training.train(x.numpy())
coarse = torch.from_numpy(coarse_training.centroids)
index = TorchIVFEngine(coarse, nprobe=4)
index.add(candidates)
scores, ids = index.search_2nn(x)
# After changing candidates, call index.add(candidates) again.
```

This supplies an external coarse quantizer to the search engine. The
e-means trainer constructors currently expose `nlist`, `nprobe` and
`coarse_niter`, but no external coarse-quantizer argument.

### Implementation notes

- The trainers train the coarse quantizer once, from a seeded sample of
  initial k-means centroids. CPU coarse training uses FP32 Torch assignment;
  CUDA uses FP16 Triton assignment. List membership is rebuilt on the first
  positive accumulation of each logical batch, and stays frozen across
  partial accumulations of that batch.
- Default revival (`unused_threshold=None`) uses the margin between the
  best two distinct centroids in the probed lists, together with Oja donor
  gains. These costs are approximate under limited probing. A missing
  finite second candidate contributes zero removal cost. An explicit
  `unused_threshold` selects mass-based revival; zero disables revival.
- Nearest-neighbor protection queries updated centroid positions against
  the frozen, pre-update index, excludes self, and merges results across
  owners when sharded. A candidate with no other finite neighbor in its
  probes protects no one. There is no exhaustive fallback. The next batch
  rebuilds the index.
- CUDA owner-statistics kernels can accumulate sums, counts, SSE and
  removal/Oja contributions directly from gathered rows. Removal updates
  normalize axes and clear batch sums in one kernel. CPU uses the Torch
  statistics path, with bounded FP64 workspace for axis norms and
  normalization. Distances and squared residuals remain FP32; scalar SSE
  totals are summed in FP64.
- `EmeansIVF.assign` builds separate query storage and reports FP32 squared
  distances to the selected centroids. It can run between partial
  accumulations without rebuilding the training index. Its labels still
  come from approximate IVF search; an exhaustive final MSE evaluation
  requires a separate flat index.
- Fractional epoch counts use the ordinary e-means row budgets and closing
  rules. IVF trainers reject spherical clustering and external quantizers.
  IVF executes eagerly, with CUDA graph capture disabled. Package exports
  are lazy; importing the package does not itself require Torch.

The [CPU benchmark report](../../benchs/e_means/bench_synthetic.md)
compares exact and IVF assignment, and one 128-thread worker with four
32-thread replicas on the same machine. For the GPU/SSD commands and scale
measurements, see [the benchmark README](../../benchs/e_means/README.md).
