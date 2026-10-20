# One, two, and three RHT rounds for L2 search

`bench_rht.py` compares `none`, `urr`, `rht1`, `rht2`, and `rht3` for both
`RaBitQ` and unbiased-scale `EDEN`, at 2, 3, and 4 stored bits per coordinate.
The fixed RHT modes use native `HadamardRotation(d, seed, rounds)` inside a
native `IndexPreTransform`. The default `HadamardRotation(d, seed)` still
means three rounds. This benchmark does not implement per-vector routing.

The default budgets are 2/3/4 bits. Use `--bits 1 2 3 4` to include the
one-bit operating point supported by both native quantizers.

`urr` is a Haar draw from the orthogonal group: Gaussian QR in float64 with
the diagonal signs of R corrected, then rounded to float32 and installed in
a native `LinearTransform`. It is distinct from the operational Faiss `RR`
factory transform. QR initialization time and dense matrix storage count in
the recorded preparation time and serialized size.

The benchmark reports reconstruction NMSE, set Recall@k, and full search
speed. These flat compressed indexes scan the database; results do not
establish performance of IVF, HNSW, or other search structures. More rounds
need not improve every dataset, quantizer, or recall target, and there is no
universal VQ or ANN improvement claim.

## Build and correctness checks

Build this checkout, including the new native constructor, rather than using
an unrelated installed wheel. Follow [INSTALL.md](../INSTALL.md) for platform
dependencies and optimized build targets. For example, a portable CPU build
on Linux is:

```sh
python -m pip install numpy threadpoolctl
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=ON \
  -DFAISS_OPT_LEVEL=generic -DBUILD_TESTING=OFF
cmake --build build --target faiss swigfaiss -j4
(cd build/faiss/python && python setup.py build)
export PYTHONPATH="$(ls -d "$PWD"/build/faiss/python/build/lib*/)"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
FAISS_RHT_NATIVE_TESTS=1 python -m unittest discover -s tests \
  -p test_bench_rht.py -v
```

Select a CPU-specific optimized target when measuring production performance
and retain that build's metadata with the results. Generic-build correctness
checks do not establish optimized-kernel performance. Native tests cover all
five modes, both quantizers, both dimension controls, serialization, EDEN's
decoder, and agreement between full multibit RaBitQ reconstruction and its
unquantized-query L2 scores. Without `FAISS_RHT_NATIVE_TESTS=1`, the remaining
helper tests need only NumPy and threadpoolctl.

## Small smoke and measurement runs

With no arguments the tool runs a small Gaussian smoke: `d=64`, 512 training
vectors, 1,000 base vectors, 64 queries, all modes and stored bit budgets,
five transform seeds, batches 1/32/256, one warmup pass and three measured
passes. Smoke timings are for validating the tool, not performance claims.
Use the designated measurement host for reported speed results and avoid
concurrent CPU workloads. The host guard checks the case-insensitive short
hostname before loading data or the native extension:

```sh
python benchs/bench_rht.py --require-host amumu \
  --synthetic student_t --d 256 --nt 10000 --nb 100000 --nq 1000 \
  --indexes RaBitQ EDEN --bits 2 3 4 \
  --modes none urr rht1 rht2 rht3 \
  --seeds 12345 12346 12347 12348 12349 \
  --batch-sizes 1 32 256 --k 10 --repeats 3 --threads 1 \
  --output results/student_t_native_k10.json
```

The `--threads` option controls Faiss OpenMP workers and BLAS/OpenMP runtimes
through threadpoolctl, including NumPy's dense QR. The CLI also sets the
thread environment before importing NumPy. Effective runtime threadpool
details are saved with the metadata; programmatic callers receive the same
runtime limits and their previous limits are restored afterward. Rerun with
`--k 100` for Recall@100, and use a distinct output path.
The script records the native extension path, Faiss version/compile options,
Git revision and working-tree status, hostname, CPU/platform, thread settings,
and command arguments. It checkpoints after each completed case and marks
the final file `status: complete` only after summaries are written.

Controlled synthetic choices are independent Gaussian samples, Student-t
with 3 degrees of freedom divided by sqrt(3), and `sparse32` (exactly 32
random active coordinates with Gaussian values, requiring d >= 32). They
share a fixed `--data-seed` independent of the transform seeds. They are
stress cases, not substitutes for public embedding/vector benchmarks.

## Real vectors and split provenance

Supply an NPZ with float arrays `xt`, `xb`, `xq`, or a directory of fvecs:

```sh
python benchs/bench_rht.py --require-host amumu \
  --npz /data/vectors.npz --nt 10000 --nb 100000 --nq 1000 \
  --check-disjoint --hash-files --output results/vectors_native.json

python benchs/bench_rht.py --require-host amumu \
  --data-dir /data/sift --train sift_learn.fvecs \
  --base sift_base.fvecs --queries sift_query.fvecs \
  --nt 10000 --nb 100000 --nq 1000 \
  --check-disjoint --hash-files --output results/sift_native.json
```

The requested counts are exact; undersized arrays fail instead of silently
shrinking the workload. All selected vectors are finite contiguous float32.
`--d` applies only to synthetic data; file dimensions are read from the data.
The JSON records the selected shapes, source paths, file sizes, and optional
SHA-256 hashes. It does not use supplied ground truth: exact `IndexFlatL2`
neighbors are computed for the actual selected base and query arrays in
their original coordinates.

If `xt` is absent from the NPZ, or `--train` is omitted for fvecs, the first
`nt` source base rows are reserved for training and removed from the searched
base. The next `nb` rows form the searched base, so the source needs at least
`nt + nb` rows. Half-open source row ranges are recorded. Separate supplied
training and base files do not by themselves prove disjointness. The default
report makes that limitation explicit; `--check-disjoint` additionally counts
selected base rows whose exact float32 values occur in selected training
rows, normalizing signed zero. This optional check uses extra memory and
time, does not reject overlap, and does not check query/base overlap or
near-duplicates.

Each native leaf learns its ordinary training mean from the same training
rows after the mode's transform. There is no per-mode training subset or
test-query-dependent calibration. Native EDEN and RaBitQ scores are
comparable approximate **squared-L2** scores, including their native clamp
at zero; the benchmark does not assert that clamped scores are exactly
unbiased. `--qb` is RaBitQ's query quantization budget, default 4; it is
recorded as not applicable for EDEN. Runs with different `qb` values are
different experiments and must not be pooled.

## Dimension and storage controls

RHT zero-pads to the next power of two. By default `none` and `urr` retain
the native input dimension. The `native-size-baselines` case label makes
this mismatch visible: bit budget per coordinate alone does not equalize
total storage. Run the same command again with `--pad-baselines` and a new
output filename to compare every mode at the padded dimension. In that
control, `none` appends zeros and URR rotates the zero-padded vector with a
Haar matrix at the padded dimension. For a power-of-two input dimension the
two controls coincide.

Every raw row includes input/output dimension, native leaf `code_size_bytes`
per vector (including stored scale/norm factors), total code storage, and
the full serialized native index size (including transforms and learned
centers). The flat leaves use implicit sequential IDs, so their separate
ID storage is zero. These measurements are preferable to inferring memory
from `bits * d` alone.

## Reconstruction diagnostics

Diagnostics use the first `--nmse-max-vectors` base rows, default 10,000;
zero means all selected base rows. Every mode uses the same row sample.
Let y be the transformed input vector, c the leaf's training mean, and
y_hat its reconstruction. The output contains:

* `nmse`: sum(||y-y_hat||²) / sum(||y-c||²), pooled over sampled vectors.
* `vnmse`: mean(||y-y_hat||² / ||y-c||²), over nonzero residual vectors.
* Error/norm sums, sample count, zero-residual count, and the number of
  zero-residual rows with nonzero reconstruction error.

Zero residuals contribute to pooled error/norm sums but are excluded from
the per-vector ratio. A wholly zero denominator produces pooled zero only
when the total error is also zero; otherwise it is JSON null. vNMSE is null
if every residual is zero. For square orthogonal transforms the error equals
original-space reconstruction error. With padding, these **transformed-domain**
errors can include components outside the input subspace; they can exceed
the error after projecting a reconstruction back into the original space.

EDEN uses its native complete reconstruction. Multibit RaBitQ requires
decoding both its sign bits and refinement bits, then applying the stored
IP factor to reconstruct the vector in its full search cross term. Its
native `sa_decode` exposes only the sign-bit baseline and would give a
misleading multibit diagnostic. The benchmark explicitly decodes the full
code and tests agreement with native scores at `qb=0`. Reconstruction NMSE
does not by itself determine ANN recall, especially with query quantization
and clamped approximate distance scores.

## Timing and paired comparisons

Construction (including QR), training, and add/encoding are timed separately.
Search measurements perform full warmup passes, then retain every full
search sample, its median, min/max, sample standard deviation, and QPS.
They include query transformation, native scanning, Python batching, output
allocation, and copying result IDs into the final array. A separate transform
diagnostic records repeated native transform passes with the same batch size;
it must not be subtracted from full search time because allocation/cache
effects can differ. Case execution order is reproducibly shuffled and saved.
Do not interpret tiny query workloads as stable speed measurements.

Set Recall@k is the fraction of distinct exact top-k IDs found in each
returned top-k set. Repeated prediction IDs and missing IDs receive no extra
credit. Raw rows retain one recall value per query. Summaries compare each
mode with `rht1` using identical family, bit budget, batch size, data, query
indices, and transform seed. Positive recall deltas favor the compared mode;
negative NMSE/vNMSE deltas favor it. Speedup is 1RHT median search time divided
by the compared mode's median, paired by seed.

Recall confidence intervals use a paired two-way percentile bootstrap:
resample transform seeds, resample one shared set of query indices across
those seeds, and retain mode pairing for every draw. This preserves the
shared query axis rather than pretending repeated evaluations of one query
are independent observations. NMSE/vNMSE delta intervals resample paired
seed-level metrics only. These intervals condition on the fixed training/base
dataset and, for reconstruction, the fixed vector sample. They do not
quantify dataset-to-dataset variability, and few transform seeds can yield
poor seed uncertainty estimates. Five seeds are a minimum suggestion; repeat
with more seeds and independent datasets before drawing broader conclusions.
