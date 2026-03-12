# Batched SIMD Threshold Distance Optimization

## Overview

This change adds early-abort distance computation to FAISS's brute-force
search paths. The key idea: process vector dimensions in fixed-size batches
using existing SIMD functions, checking a distance threshold at batch
boundaries. If the partial result already exceeds the threshold (the k-th
best distance found so far), we abort early and skip the remaining
dimensions.

This is a port of the optimization from the
[nmslib knn-opt branch](https://github.com/nmslib/nmslib), adapted to
FAISS's architecture.

## How It Works

### The Problem

In k-NN search, FAISS maintains a heap of the top-k nearest neighbors.
The heap top is the k-th best distance — the threshold for entering the
result set. For each candidate vector, FAISS computes the **full** distance
across all dimensions, then checks if it beats the threshold.

For high-dimensional vectors (d=1024+), most candidates don't beat the
threshold. Computing all dimensions for these candidates is wasted work.

### The Solution: Batch-Then-Check

```
for each batch of 16 dimensions:
    partial_distance += SIMD_compute(batch)
    if partial_distance > threshold:
        return early  // skip remaining dimensions
```

**Why batching instead of per-dimension checks:**
- Per-dimension threshold checks cause branch mispredictions (~15-20 cycle
  penalty per misprediction)
- Batching keeps SIMD loops branch-free
- The branch predictor learns the periodic check pattern
- Batch size 16 balances abort latency vs branch prediction cost

### Supported Metrics

| Metric | Function | Bound Type | Quality |
|--------|----------|------------|---------|
| L2 squared | `fvec_L2sqr_batched` | Monotonic (partial sum) | Tight |
| Inner product | `fvec_inner_product_batched` | Optimistic (sum + remaining) | Loose |
| Linf | `fvec_Linf_batched` | Monotonic (running max) | Tight |

**L2:** Partial sum of squared differences is a lower bound on the final
distance. If partial > threshold, the full distance will also exceed it.

**Inner product:** For normalized vectors, the optimistic bound assumes
remaining dimensions each contribute +1. If even this best case is below
threshold, the actual similarity cannot beat it.

**Linf:** The running maximum is monotonically non-decreasing. If the
current max exceeds threshold, the final max will too.

## Integration Points

### Free Functions (`faiss/utils/distances.h`)

Three new functions alongside existing distance functions:

```cpp
float fvec_L2sqr_batched(
    const float* x, const float* y, size_t d,
    size_t batch_size, float threshold);

float fvec_inner_product_batched(
    const float* x, const float* y, size_t d,
    size_t batch_size, float threshold);

float fvec_Linf_batched(
    const float* x, const float* y, size_t d,
    size_t batch_size, float threshold);
```

### DistanceComputer (`faiss/impl/DistanceComputer.h`)

Three new virtual methods on the base class:

```cpp
// Compute distance with early abort opportunity
virtual float operator()(idx_t i, float threshold);

// Does this implementation support threshold optimization?
virtual bool supports_threshold() const;

// Threshold that accepts all candidates (infinity for L2, -infinity for IP)
virtual float degenerate_threshold() const;
```

Default implementations fall back to standard (non-threshold) computation,
so existing code is unaffected.

### Exhaustive Search (`faiss/utils/distances.cpp`)

`exhaustive_L2sqr_seq()` and `exhaustive_inner_product_seq()` now use the
batched functions with the result handler's threshold. These are the
sequential brute-force scan paths used when:
- `nx < distance_compute_blas_threshold`, or
- An `IDSelector` is active

The BLAS path is unaffected (it computes distances in matrix blocks).

### FlatIndex Distance Computers (`faiss/IndexFlat.cpp`)

`FlatL2Dis` and `FlatIPDis` override the threshold methods to use the
batched free functions with batch_size=16.

## Where It Does NOT Apply

### HNSW and Graph-Based Search

HNSW uses computed distances for **two purposes**:
1. Result filtering (is this candidate in the top-k?)
2. Candidate ordering (`candidates.push(idx, dis)`)

Early abort returns a partial distance that corrupts candidate ordering,
changing the graph exploration path and reducing recall. The optimization
is fundamentally incompatible with graph-based search where distance
values drive traversal decisions.

### BLAS-Based Distance Computation

The BLAS path (`exhaustive_L2sqr_blas`) computes distances as matrix
multiplications. Threshold checking doesn't apply to block matrix ops.

## Performance Characteristics

The optimization is **data-dependent**. It works best when:
- Many candidates are far from the query (clustered data)
- Vectors are high-dimensional (d > 512)
- k is small relative to database size

It provides less benefit when:
- Distances are uniformly distributed (random Gaussian data)
- Vectors are low-dimensional
- Most candidates are close to the query

### Benchmark Results (Apple Silicon M-series, d=2048, k=100)

**Clustered data (10 clusters, query near one cluster):**

| Batch Size | Speedup |
|------------|---------|
| 4          | 0.93x   |
| 8          | 1.29x   |
| 16         | 1.39x   |
| 32         | 1.65x   |
| 64         | 1.44x   |

**Random Gaussian data (worst case):**

| Batch Size | Speedup |
|------------|---------|
| 16         | 0.50x   |
| 32         | 0.83x   |
| 64         | 0.93x   |

The overhead on random data comes from the batching loop and threshold
checks when early abort rarely triggers. Real-world data with cluster
structure sees meaningful speedups.

## Files Modified

| File | Change |
|------|--------|
| `faiss/utils/distances.h` | Declarations for 3 batched functions |
| `faiss/utils/distances.cpp` | Implementations + exhaustive search integration |
| `faiss/impl/DistanceComputer.h` | 3 new virtual methods on base class |
| `faiss/IndexFlat.cpp` | Threshold overrides in FlatL2Dis, FlatIPDis |
| `tests/test_distances_simd.cpp` | 9 new unit tests |
| `benchs/bench_batched_distances.cpp` | Benchmark tool |

## Running the Benchmark

```bash
# Build
c++ -std=c++17 -O2 -I. -Ibuild -Lbuild/faiss -lfaiss \
    -framework Accelerate -o build/bench_batched_distances \
    benchs/bench_batched_distances.cpp

# Run (defaults: 100k vectors, 2048 dims, k=100, 50 queries)
./build/bench_batched_distances

# Custom parameters
./build/bench_batched_distances <num_vectors> <dimensions> <k> <num_queries>
```

## Running the Tests

```bash
./build/tests/faiss_test --gtest_filter="TestFvec*"
```

All 18 distance tests should pass (9 original + 9 new batched tests).
