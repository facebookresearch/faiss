# E-means

Implements e-means, a mini-batch k-means algorithm that updates FP32 centroid
positions and masses with exponential moving averages and revives underused clusters by
splitting active ones. `Emeans` has a Faiss-style `train`/`assign` interface:

```python
from faiss.contrib.e_means import Emeans

km = Emeans(d, k, niter=25)
km.train(x)
distances, labels = km.assign(x)
```

The default statistical batch size and EMA step are selected together; set
`batch_size` and `alpha0` together to override them. With the default
`spherical=False`, NumPy input uses exact Faiss L2 assignment. PyTorch input
stays on its CPU or CUDA device and does not call Faiss; PyTorch is optional.
The native backend is the reference path for dense Euclidean clustering.
See `demos/demo_e_means.py` in the source checkout for a DEEP1M example.

With `spherical=True`, explicitly normalize input rows before conversion or
prequantization for `train`, `accumulate`, `assign` and any frozen closing scan;
keep zero rows zero. Inputs are not normalized internally. `assign` returns
represented inner-product similarity (higher is better), approximating cosine
after operand rounding; accuracy follows
the backend's floating-point accumulation. `obj`, `iteration_stats`
report the moving-pass sum of `2 - 2 * similarity` (lower is better),
without clamping rounding error. EMA updates retain raw FP32 coordinates;
assignment, readout and closing output use projected directions, with zero
directions staying zero. `current_centroids()` borrows one reusable effective
FP32 direction cache under the existing read-window and stream contract. The
cache requires `4 * k * d` additional bytes. This prefix supports native FP32
NumPy and Torch input, including the existing eligible resident CUDA replay.

For caller-driven batches, supply FP32 initial centroids and the global row
count. `train` uses the same numerical operations with seeded initialization
and shuffling:

```python
km.begin(global_n, init_centroids)
for epoch, rows in enumerate(km.recipe["epoch_rows"]):
    for batch in batches(epoch, rows, km.recipe["batch_size"]):
        km.update(km.accumulate(batch))
    km.end_epoch()
centroids = km.finalize()
```

During an active run, `km.current_centroids()` borrows the evolving FP32 model
coordinates without copying their data. `km.centroids` remains final-only.
The getter is available after `begin`, including completed epochs, until
`finalize`; before `begin`, after `reset` and after `finalize` it raises
`RuntimeError`. NumPy returns a non-writeable view; Torch returns a detached
alias on the model device. Neither is immutable or safe for concurrent mutation.
Do not write through it. Finish readers before `update`, `end_epoch`, `finalize`,
`reset` or a new `begin`, then reacquire; old handles are not valid current
state after reset/finalize/new begin. Use `.copy()` or `.clone()` explicitly for
an independent snapshot. CUDA callers order producer, reader and subsequent
mutation streams themselves; the getter adds no synchronization or collective.
These are FP32 model coordinates, not rounded or encoded assignment operands;
external assignment engines remain responsible for their representation policy.

Each epoch has `ceil(recipe["epoch_rows"][epoch] / batch_size)` updates,
including its final remainder; `total_steps` is their sum. Integer budgets
keep full passes and the fitted recipe `(N,E)`. Fractional `niter=2.4` runs
`N + round(.4*N) + N` rows, with the fractional prefix before the full moving
closing pass. Below one pass, only `round(niter*N)` rows form moving closing.
Round the fractional float product with Python's ties-to-even rule, omit a
zero remainder and reject a budget below one that rounds to zero rows.
Nonfinite and nonpositive budgets are rejected. Resident training selects
each prefix from that epoch's seeded permutation; caller-driven training
supplies the prescribed counts. Assignment, EMA and revival continue during
closing. Finalize uses collected means with no new scan. Unvisited resident
captured IDs are `-1` when closing is short. Every rounded epoch count must
divide by the group size when a derived model uses equal contributions.
The caller owns loading, ordering, transfers and prefetch. Supplied
input chunks can be smaller than one statistical batch:

```python
for chunks in statistical_batches:
    for rows in chunks:
        stats = km.accumulate(rows)
    km.update(stats)  # One update for the prescribed batch, including its tail.
```

Supply each intended row position once, in order; do not refit or change
centroids between chunks. Finish assignment/statistics readers before reusing
an input staging buffer. Centroids and masses remain resident. Positive partial
results report cumulative `n` and FP64 `objective`, with `sums` and `counts`
set to `None`. This replaces the older partial-result behavior that exposed
arrays although `update` rejected them. At completion, the mapping contains
model-owned FP32 sums/counts. Consume it once with `update`; borrowed statistics
expire on extension/update. Copy complete values for independent retention,
and keep their provenance yourself; there is no stale-copy token validation.
An earlier Torch objective tensor can also change on extension; its receipt's
host `n` remains the count at that call. Copy/clone complete retained statistics.

Overfill raises `ValueError` before tensor work; positive input after completion
raises `RuntimeError` until update. Incomplete update raises `ValueError` even
for an external replacement. These preflight errors preserve pending progress.
Empty calls validate metadata without clearing/converting pending totals.
Before positive input they return zeros and allow a complete external update.
`end_epoch`/`finalize` reject incomplete state. Reset/new begin discard it;
execution failure invalidates the run until reset/new begin, without rollback,
automatic retry or a successful-call synchronization.
CPU callers can sum all four statistics fields
with their own communication and call local `update`. CUDA accepts native FP32
and requires finite caller-validated values; metadata is always checked, with
no implicit CUDA value scan. NumPy/Torch CPU validation still checks finiteness.
Whole statistical batches above 2²⁴ rows use temporary FP64 sums/counts
accumulation across supplied chunks, with one FP32 conversion at completion.
This requires additional memory and computation; there is no extra K*d table
for ordinary batches.

## Train results and last-pass labels

Capture defaults on; `capture_assignments=False` opts out. Enabled capture returns caller-owned INT64
`stats["labels"]` during only the final caller-driven epoch. These labels cover
only the current supplied chunk, in supplied order (empty for an empty call),
and later calls cannot overwrite them. They stay on the execution device;
there is no implicit host copy, gather or dataset-wide caller-loop retention.

After successful resident `train`, `km.assignments` contains the last training
pass's labels in original input order. It is `None` before completion, on reset
or failure, and after starting a new train attempt, including failed preflight.
Caller-driven runs leave it `None`. Resident `train` returns `(centroids, assignments)`;
labels are None with capture disabled. Objective history remains in `obj`;
`finalize` still returns centroids. Last-pass labels are the codes used for closing means,
not necessarily nearest to the returned centroids: call `assign` explicitly
for that additional search.

Default resident capture retains8N bytes plus staging/workspace; the explicit opt-out adds no label output/storage. Capture enabled still
permits eligible resident CUDA replay: one stable 8*batch byte label buffer is
written in captured earlier epochs too, followed by final-epoch scatter to the
8*N byte retained output. Caller loops need 8*supplied_rows bytes per retained
chunk output. These costs are separate from existing assignment workspace and
input permutation; input chunking does not remove centroid-state memory.

Default resident CUDA `train` may replay full batches through a CUDA graph.
Capture and instantiation are part of each training call. The graph and private
pool are retained across epoch reports, then retired
at training-call teardown. Exceptional teardown drains pending work before
graph reset and owner release. Explicit
overrides, custom models, tails and very wide accumulation stay eager.
Caller-driven loops remain eager. Capture/execution errors propagate, without retry.
CUDA replay was validated with Torch 2.14; support on older versions is unestablished.
ROCm/HIP graph replay is also unqualified. An explicit
`assignment_chunk_size` or the caller-driven batch loop selects the eager path;
this is not a claim of general older-Torch or ROCm compatibility.

Untouched centroid positions survive mass decay. Revival keeps ordinary
multiplicative splits; a collapsed FP32 pair changes one coordinate to an
adjacent finite value. Distinct operands do not guarantee both children get
rows (a zero donor's subnormal distance can underflow). Final FP32 centroids
are means of the final pass's own assignments in ordinary mode; never-assigned
centroids retain their current positions. `obj` remains the moving-pass objective, rather than a score against the
closing means. `train` returns the centroids and last-pass assignments.

CPU or GPU callers use `DistributedEmeans(d, k, group=group, transport="auto", ...)`
with an initialized, caller-owned process group. Group and transport are fixed
at construction. `begin(global_n, init_centroids)` takes matching NumPy FP32
arrays or Torch tensors on each rank and broadcasts its owned FP32 centroid
state from group rank zero. CPU state requires Gloo; CUDA
retains the supported backend/representation combinations. Root values are
authoritative; caller arrays are never modified.
The remaining `accumulate`/`update`/`end_epoch`/`finalize` lifecycle is shared
with `Emeans`; distributed `train(x)` is rejected. Centroid state is replicated;
the caller supplies matching recipes/codecs and disjoint rank-local contributions
to each prescribed global batch. Every batch, including the final tail, must
divide evenly by group size; this is checked at `begin` before broadcasting.
Each update checks that local `stats["n"]` equals its prescribed global size
divided by group size. Every rank participates in the same updates and epoch
closes. Local `Emeans.update(stats)` has no group or transport arguments;
`DistributedEmeans` requires an explicit group, not an implicit world group.

Dense transport all-reduces FP32 statistics. Sparse transport packs occupied
IDs, all-gathers and merges in group-rank order. It can synchronize during
packing/size discovery; padded receive storage is approximately
`ranks * max_local_occupied * (d+1) * 4` bytes plus IDs. Automatic selection
uses known batch/table/rank sizes, conservatively bounds padding and amortizes
packing on large payloads; it is not a portable network crossover. Use an
explicit `"dense"` or `"sparse"` override where appropriate, and ensure the
state and transport workspace fit. No batch size is silently reduced.

One scalar FP64 epoch collective sums local moving-pass objective contributions.
Already-global FP64 pass counts and FP32 closing sums are captured once. Equal
row counts can still have different occupied-centroid counts, which sparse
transport discovers separately. Divergent collective schedules remain a caller
error; this API does not recover from them. Revival IDs are
selected by group rank zero and broadcast; every replica applies the same
split. NumPy and Torch CPU use the same class lifecycle and dense/sparse/auto
transports with Gloo. External assignment engines can call `update` with LOCAL
FP32 sums/counts, an FP64 objective and the prescribed LOCAL `n`, without
calling `accumulate`. Statistics must match the model's array type. NumPy
buffers are staged in owned storage; sparse results remain NumPy arrays. Local
`Emeans.update` still performs no communication. Unequal row contributions are
unsupported. See
`demos/demo_e_means_distributed.py` in the source checkout for a small
`torchrun` example.

## Numerical limits

The training objective follows the moving-pass assignments, not the returned
closing centroids. When precision matters, also assess the final centroids on
held-out data in the original coordinates. Validation is caller-controlled;
there is no automatic scan, centering or precision fallback.

## Running the public tests

From a source checkout with its built or installed Faiss package:

```sh
python -m unittest discover -s tests -p 'test_e_means.py'
python -m unittest discover -s tests -p 'torch_test_e_means*.py'
```

Both filename patterns are needed: default discovery omits the Torch files.
Torch is required for the second command. GPU/backend and two-/three-GPU
cases report explicit skips when their dependencies or devices are unavailable;
those skips do not establish qualification of the unavailable scopes.

## Train API defaults

Resident train returns (centroids, last_pass_assignments), with capture on by
default and capture_assignments=False yielding None labels. Objective history
remains in obj. Labels precede closing means; no extra assignment search.
Outputs alias owned model results; copy before mutating. Capture retains8N
bytes plus staging/workspace. Caller loops retain only supplied-chunk labels.
