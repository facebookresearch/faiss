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
The native backend is the reference path for
dense Euclidean clustering; supported low-precision and byte formats are
described below. See `demos/demo_e_means.py` in the source checkout for a DEEP1M
example.

With `spherical=True`, explicitly normalize input rows before conversion or
prequantization for `train`, `accumulate`, `assign` and any frozen closing scan;
keep zero rows zero. Inputs are not normalized internally. `assign` returns
represented inner-product similarity (higher is better), approximating cosine
after operand rounding; FP16/BF16 norms are not repaired, and accuracy follows
the selected backend's floating-point accumulation. `obj`, `iteration_stats`
report the moving-pass sum of `2 - 2 * similarity` (lower is better),
without clamping rounding error. EMA updates retain raw FP32 coordinates;
assignment, readout and closing output use projected directions, with zero
directions staying zero. `current_centroids()` borrows one reusable effective
FP32 direction cache under the existing read-window and stream contract. The
cache requires `4 * k * d` additional bytes. This mode supports native floating
input and cast quantizers, including existing eligible resident CUDA replay.

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
model-owned sums/counts in the run's FP32 or raw-byte INT64 schema. Consume it
once with `update`; borrowed statistics
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
with their own communication and call local `update`. CUDA requires finite
caller-validated values fitting the chosen format; metadata is always checked, with
no implicit CUDA value scan. NumPy/Torch CPU validation still checks finiteness.
Floating statistical batches above 2²⁴ rows use temporary FP64 sums/counts
accumulation across supplied chunks, with one FP32 conversion at completion.
This requires additional memory and computation; there is no extra K*d table
for ordinary batches.

Floating or explicitly quantized inputs retain FP32 statistics. Raw CUDA
INT8/UINT8 with `quantizer=None` instead returns matching INT64 sums/counts,
including small, empty and tail batches. Native input metadata or the first
external matching pair fixes the run's schema. Later incompatible statistics
raise "statistics dtype must match the run's fixed statistics schema"; input
metadata must also match the selected schema and native format. Previous FP32
contributions are not silently widened into an integer run.
Raw integer runs require declared global `255*global_n <= 2**53`. Whole local
batches up to 2²⁴ rows use bounded INT32 scratch, reconstructing UINT8 totals
in INT64 once at completion. Larger batches retain exact FP64 integer scatter
under that bound, then convert directly to INT64. Scratch and UINT8 shift
selection cover the whole prescribed batch, not each supplied chunk.
Integer final-pass totals are divided in FP64 before returning FP32 centroids.
EMA still rounds batch sums/counts once to FP32 at its existing update boundary.
Callers combining statistics remain responsible for valid totals within the bound;
final floating division/output rounding is not an exact rational or code contract.

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
Capture and instantiation are part of each training call. The graph, private
pool and captured bindings are retained across epoch reports, then retired
at training-call teardown. Exceptional teardown drains pending work before
graph reset and owner release. Explicit
overrides, custom models/codecs, tails and very wide accumulation stay eager.
Caller-driven loops remain eager. Capture/execution errors propagate, without retry.
CUDA replay was validated with Torch 2.14; support on older versions is unestablished.
ROCm/HIP graph replay is also unqualified. An explicit
`assignment_chunk_size` or the caller-driven batch loop selects the eager path;
this is not a claim of general older-Torch or ROCm compatibility.

Untouched centroid positions survive mass decay. Revival keeps ordinary
multiplicative splits; a collapsed represented pair changes one coordinate to an
adjacent finite value. Distinct operands do not guarantee both children get
rows (a zero donor's subnormal distance can underflow). Final FP32 centroids
are means of the final pass's own assignments in ordinary mode; never-assigned
centroids retain their current positions. `obj` remains the moving-pass objective, rather than a score against the
closing means. `train` returns the centroids and last-pass assignments.

CUDA native Torch accepts FP32, FP16 and BF16. With `quantizer=None`, the
first batch (including a typed empty batch) fixes the native dtype for the run.
FP32 initial centroids do not select that dtype. Norms, dot products and FP32
statistics use the same represented rows; centroid operands follow that format.
FP32 centroid state and final means remain unquantized. Distances refer to the
rounded centroid operands, and native FP32 honors the caller's Torch matmul
settings. FP16/BF16 assignment requires NVIDIA SM80 or later and Torch
`addmm` support for FP32 output.

For explicit batchwise conversion, use
`CastQuantizer(d, torch.float16)` or `CastQuantizer(d, torch.bfloat16)` from
`faiss.contrib.e_means.e_means_quantizer`, passed as `quantizer=q`. Its `train` is a no-op;
`compute_codes` returns native typed Torch rows and `decode` widens them to
FP32. Conversion/decoding check metadata without CUDA value scans. Call
`q.validate(x)` explicitly to check finite/range hazards; unchecked overflow
follows Torch cast semantics. The existing internally generated FP16 split
range guard remains. CPU computes supplied represented floating values in FP32
and requires `quantizer=None`, even though the cast helper can execute on CPU.

`assignment_backend="triton"` explicitly selects optional Triton CUDA assignment
instead of automatic backend selection. It requires NVIDIA SM80 or
later, supports the same FP32/FP16/BF16 represented inputs and cast quantizer,
and retains the shared lifecycle/statistics. FP32 dots request IEEE precision.
It uses bounded centroid tiles and split-K partial results without runtime
autotuning or backend fallback. `centroid_chunk_size` applies only to native
Torch; Triton chooses its internal tile geometry. Conservative known-size
parallelism/work-per-partition rules are performance defaults, not hardware
legality or a portable crossover. Padded physical index/label bounds still apply.

CPU or GPU callers use `DistributedEmeans(d, k, group=group, transport="auto", ...)`
with an initialized, caller-owned process group. Group and transport are fixed
at construction. `begin(global_n, init_centroids)` takes matching NumPy FP32
arrays or Torch tensors on each rank and broadcasts its owned FP32 centroid
state from group rank zero. CPU state requires Gloo and quantizer=None; CUDA
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

Dense transport all-reduces the matching FP32 or INT64 statistics. All ranks
must use the same fixed schema. Sparse transport packs occupied
IDs, all-gathers and merges in group-rank order. It can synchronize during
packing/size discovery; padded receive storage is approximately
`ranks * max_local_occupied * (d+1) * element_size` bytes plus IDs, where
element_size is 4 for FP32 or 8 for INT64. Automatic selection
uses actual element size and known batch/table/rank sizes, bounds padding and amortizes
packing on large payloads; it is not a portable network crossover. Use an
explicit `"dense"` or `"sparse"` override where appropriate, and ensure the
state and transport workspace fit. No batch size is silently reduced.

One scalar FP64 epoch collective sums local moving-pass objective contributions.
Already-global matching integer totals, or FP64 counts/FP32 floating closing sums,
are captured once before EMA conversion. Equal
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

For calibrated E4M3 on qualified Triton CUDA GPUs, use an explicit quantizer:

```python
from faiss.contrib.e_means.e_means_fp8 import Fp8Quantizer
q = Fp8Quantizer(x.shape[1])
q.train(representative_sample)
q.to(x.device)
q.validate(x)  # optional explicit finite/range scan
km = Emeans(x.shape[1], k, quantizer=q, assignment_backend="triton")
km.train(x)  # floating chunks encode during assignment
# Alternatively: km.train(q.compute_codes(x))
centroids = km.centroids  # already decoded FP32; do not decode again
```

FP8 codes with `quantizer=None` define raw numeric code units. FP8 input requires
NVIDIA SM90, SM100 or SM103. Conversion does not clip or scan values: unchecked
out-of-range/nonfinite values can produce NaN codes and invalid results. The
existing internal-centroid range check runs at epoch close and during assign.
Partial calls retain the fitted transform and frozen centroid operands. Eligible
uncentered FP8 batches accumulate in code units and apply the decode scale once
at completion; other batches retain the existing decoded-row accumulation.
Only complete statistics expose FP32 sums/counts in the selected model units.

Native INT8/UINT8 assignment uses exact INT32 ranking (dimension at most
43,919), with FP32 returned distances, on qualified NVIDIA SM80/SM90/SM100.
An explicit INT8 quantizer returns decoded centroids and distances:

```python
from faiss.contrib.e_means.e_means_int8 import Int8Quantizer

q = Int8Quantizer(d)
q.train(representative_sample)
q.to(x.device)
codes = q.compute_codes(x)  # rounds and clips finite outliers, in bounded chunks
km = Emeans(d, k, quantizer=q, assignment_backend="triton")
km.train(codes)  # matching native codes only; no floating-to-INT8 JIT
```

Without a quantizer, native signed/unsigned bytes retain their numeric units.
Calibration, device placement and validation are explicit; unchecked nonfinite
values have no meaningful encoding result. Use `q.validate(x)` when needed.

CuTe BF16/E4M3 assignment is an explicit optional backend on exactly NVIDIA SM90,
requiring `quack-kernels==0.6.5`, `nvidia-cutlass-dsl==4.7.1` and
`cuda-python>=13,<14`, plus Triton for dense CUDA EMA updates.
Native Torch/CPU/NumPy updates do not require Triton. Supply native BF16 rows with `quantizer=None`, or use
`CastQuantizer(d, torch.bfloat16)` for per-chunk floating conversion; select
`assignment_backend="cute"`. It uses resource-constrained tiled/row-owned
assignment, native FP32 score ranking and no runtime autotuning; the only
fallback is the warned dependency-absent grouping provider described below.
For BF16 L2, a coarse `B * K >= 2**31` work threshold and legal grid/resources
select tile selection plus grouping and recovery. This uses actual local
assignment chunk B and local K across resident, caller and replica
operations; centered casts use the same prepared-operand rule.
Grouping prefers `cuda-cccl`
(`cuda.compute` CountingIterator, SortOrder, make_lower_bound and
make_radix_sort; tested with 1.2.0). When that Python distribution is genuinely
absent, CuTe emits one `RuntimeWarning` and uses stable Torch sort/search grouping,
measured about 0.6–2.4% slower in complete large/caller calls. Install
`cuda-cccl` for maximum performance. An installed but broken CCCL import raises;
there is no public provider knob or shape/sharding-dependent provider choice. No
input-content scan is added. BF16 CuTe workspaces prefer 32B-aligned row pitches while retaining
legal 16B storage at the existing span limit; this affects every BF16 CuTe route.
CCCL grouping adds 16B per row plus 8B per tile-prefix entry and queried scratch;
Torch grouping retains 16B per row plus 12B per prefix and uses opaque internal
sort/search scratch. Bindings expose their selected provider for diagnostics and
share mutable cached storage under the existing ordered-stream/capture lifetime
contract.

The work threshold is a coarse empirical bracket, not a densely measured
crossover or an address bound. Small calls retain the incumbent even when
warmed grouped timing wins; one DEEP1M first-fit control cost 6.22s more with
grouping. At the same large geometry, all-one-tile and 90%-one-tile replay were
57% and 50% slower respectively; their frequency is unknown. This concentration
limitation is accepted for this release, with no recovery-row-stripe change.
Specialized K-tail, additional grouped dtype, wide addressing and broader graph
work remain unfinished; this release does not claim those gaps repaired.
Spherical BF16 keeps the incumbent kernels; grouped routing for that metric
remains deferred despite sharing the represented dot-product primitive.
Bit-equal scores select the lower centroid ID; FP64-only ties can differ.
E4M3 uses native codes or caller-calibrated `Fp8Quantizer` batch conversion,
with decoded FP32 state/statistics and stock exponent-scaled distances. FP8
tensor-core partial accumulation uses FP32 promotion every 128 contraction
coordinates. Ranking is approximate; distances are recomputed from selected
represented operands. Unchecked invalid
JIT inputs have no meaningful result and need not match Torch cast overflow.
Calls on one model share mutable workspace; the caller must order stream switching.
Partial accumulation retains one prepared centroid snapshot until its completed
contribution is consumed. Eligible resident replay records label staging
independently of the epoch of capture, then scatters last-pass labels before the
next replay. Graph-bound CuTe bindings remain owned until after graph reset;
clearing pending-batch operands does not retire those bindings.
CuTe FP8 JIT conversion consumes each assignment's encoded workspace for
statistics on the ordered stream before the next assignment reuses it. Pending
code-unit totals retain one unit choice and decode scale only at completion;
seed initialization still explicitly encodes and decodes its selected rows.

## Numerical limits

Low precision can round distinct rows or centroid operands to the same values.
Calibration can reduce this loss on some data, but it cannot guarantee distinct
assignments or recover information already lost in the supplied representation.
Use native FP32 input with `quantizer=None` and native Torch or Triton assignment
when representation loss is unacceptable; CuTe FP32 is unsupported. These CUDA
paths use expanded L2 ranking (`centroid_norm - 2*dot`) in ordinary mode, so a
large common offset can cancel small distance differences. Native FP32 follows
the caller's Torch matmul settings; this fallback is not a direct-distance
accuracy guarantee.

Pass native INT8/UINT8 tensors with `quantizer=None` on supported Triton
CUDA backends to retain integer statistics within the documented bounds.
Converting byte data to floating point uses FP32 statistics and loses that
exact-total contract. Floating and calibrated inputs retain FP32 statistics,
whose accumulation can depend on order. With `Int8Quantizer`, distances and
centroids are decoded FP32 values. Comparing decoded FP32 distances can make
exact integer-code ties appear slightly unequal; this does not imply an error
in integer-code ranking.

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

## Explicit optional centered Cast

Optional L2 centering is explicit and defaults off:

```python
q = CastQuantizer(d, torch.bfloat16, center=True)
q.train(representative_training_sample)  # Prefer original FP32, before model use.
q.to(x.device)
model = Emeans(d, k, quantizer=q)       # Backend and other defaults are unchanged.
centroids, last_pass_labels = model.train(x)
objective = model.obj[-1]
```

The fit owns an FP32 sample mean, accumulated in FP64. There is no automatic
sampling or fitting; all distributed ranks must explicitly use the same centre
and ordered lifecycle. Keep the fit unchanged while models/readers/graphs use
it. Ordinary in-place mutation is rejected; the borrowed `q.shift` must not be
mutated, including through `.data`. Explicit refitting or moving invalidates
the previous model fit; call `begin` or `train` to start a new run afterwards.
Fit performs only metadata checks and the mean reduction, with no numerical
validation or host data read. Nonfinite data follows Torch semantics. Call
`q.validate(x)` explicitly after fitting to check finite centre, finite/range
hazards and finite decoded original FP32 values after target-format rounding
and add-back. This applies to original input and matching preencoded codes;
that opt-in validation may synchronize. Encode/decode and assignment
preparation use tensor arithmetic on the operand device without value scans.

`compute_codes(x)` always encodes original coordinates as `(x.float()-shift)`
rounded to the target format. **Model input of the target dtype means already
encoded codes**, as with existing casts. If original source rows already have
that dtype, call `compute_codes` explicitly or widen before passing them for
JIT encoding. Centering cannot recover information lost by earlier rounding.
Explicit initial centroids always stay original-coordinate FP32. Assignment
centroids use the same encoder as rows; decoding adds the centre to each row
before the existing FP32 statistics. Public sums, mean/EMA state, final centroids
and borrowed active-run getter retain original coordinates and their FP32
offset/accumulation limits. No late-add reconstruction or centered state is used.
Centering is rejected with `spherical=True`, including a fitted zero centre.

Revival keeps the existing Faiss multiplicative perturbation first. Only a
collapsed represented pair repairs coordinate zero: it tries decoded code
neighbours and at least one original-FP32 ULP, re-encodes both directions and
accepts a finite distinct code with finite original decode. This prevents a
tiny centred step disappearing in FP32 add-back. There is a representability
limit: at FP32 origin `2**40`, the closest original neighbours differ by
`+131072` or `-65536`, both beyond finite FP16 range (`65504`). No finite
nonzero FP16 code can describe another original-FP32 value there. The helper
retains the child in that case; it does not clamp or change coordinate frames.
The actual model's existing FP16 split-range guard rejects its much larger
Faiss perturbation at such an origin before repair. No universal split or
large-origin training guarantee is claimed.

Native/Triton FP16/BF16 and CuTe BF16 use their existing device/stack guards.
CuTe's host preparation performs subtraction **on CUDA**, then uses the existing
BF16 affine mirror kernel. This first implementation adds a temporary FP32
`K*d` tensor (`4*K*d` logical bytes) and one subtraction per mirror refresh;
unchanged cached assignments avoid it. The cache retains the fit tensor and
refreshes existing mirror storage; captured temporary storage follows the
existing graph pool lifetime. This is an explicit memory/work cost, not a speed
claim. Streaming, distribution and replay retain the common decoded-row
contract; their qualification results are tracked separately from CPU codec
tests. INT8/FP8 calibration policies and all default routing remain unchanged.


## Train API defaults

Resident train returns (centroids, last_pass_assignments), with capture on by
default and capture_assignments=False yielding None labels. Objective history
remains in obj. Labels precede closing means; no extra assignment search.
Outputs alias owned model results; copy before mutating. Capture retains8N
bytes plus staging/workspace. Caller loops retain only supplied-chunk labels.

Auto is the default assignment backend, selecting available compatible
implementations from actual device/effective input dtype without an implicit
precision conversion. Training fixes metadata for its run; standalone assign
resolves per call and reports assignment_recipe. Selected execution errors
propagate without retry. CPU uses native without optional GPU imports.

This auto resolver retains dtype, dependency and shape-based choices. Replacing
it with hardware-only selection is deferred together with missing CuTe FP16/FP32
and other support; the current behavior does not meet that intended policy.
No implicit precision conversion is introduced by the grouped release.
