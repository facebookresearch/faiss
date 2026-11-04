# Fused BF16 query evaluation

Host: h100-217-143, four H100 GPUs, 2026-11-03.
The benchmark evaluates all 100,000 shipped DINO queries against 2,097,152
frozen source-vector centroids, sharded equally across the GPUs, at d=1024.
Source values are normalized with mean 128 and scale 64 in both modes.

## Results

| Assignment | Query batch | Evaluation time |
| --- | ---: | ---: |
| FP32 Triton, previous default path | 1024 | 72.883 s |
| Fused BF16 Triton | 1024 | 0.402 s median of three trials |
| Fused BF16 Triton | 8192 | 0.490 s median of three trials |
| Fused BF16 Triton | 32768 | 0.358 s median of three trials |

The default batch size remains 1024. At that size the new path is approximately
181 times faster in this benchmark. Three evaluations would take about 1.2 s
after warmup, compared with the previous full run's 219.2 s of evaluation.
That suggests approximately 3m38s less total runtime; the full clustering run
has not been repeated with this change.

Timing includes BF16 mirror and norm preparation, query reader/transfers,
fused assignment, cross-shard winner reductions, and FP32 distance recomputation.
It excludes centroid loading and the warmed assignment compilation. Peak
allocated GPU memory in this isolated benchmark was 7.00 GiB for BF16 and
4.00 GiB for FP32, including original centroid storage and preparation temporaries.

The optional native FP32 comparison took 2.839 s. It uses Torch matrix
operations rather than the FP32 Triton path used by the previous clustering
run, so it is recorded separately.

## Implementation and correctness

- `query_mse(..., precision="bf16")` prepares one BF16 centroid mirror per
  evaluation and uses the existing fused Triton assignment kernel.
- `assign_rows(..., return_score=True)` returns the unshifted ranking score,
  without the row-norm addition or zero clamp. This preserves cross-shard
  ordering and negative scores.
- Existing winner reductions select the global label with lowest-ID ties.
  Only the winning owner contributes its FP32 direct squared distance; SSE is
  accumulated and reduced in FP64.
- `--query-precision fp32` retains the previous evaluation path. Both demos
  default to BF16 and record the precision in their saved results.
- The two-GPU SSD self-test passed for FP32 and BF16 against independent
  dense oracles, including query tails, scaling, completed and finalized
  trainers, unchanged trainer/index state, empty queries, and BF16 ties across
  owners where the original FP32 positions differ.

All benchmark modes reported original-coordinate MSE 233056.43498 on these
frozen source centroids. Those normalized source values are exactly
representable in BF16. Trained centroids generally are not; BF16 can change
assignment near boundaries. This measurement establishes throughput and does
not establish identical quality for a trained codebook.

## Reproduction

From `benchs/e_means/ssd_read_bench` on the data host:

```sh
torchrun --standalone --nproc_per_node=4 bench_query_mse.py \
  --output results/query_mse_bf16_20261103.json
```

For the native comparison add `--fp32-backend native` and use a separate output.
Raw results: `query_mse_bf16_20261103.json` and
`query_mse_bf16_native_20261103.json`.
