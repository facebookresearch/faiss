# Saved-assignment closing

The SSD demo now captures final-epoch global labels in CPU int32 storage,
scattered into original source-row order. Its sequential closing pass supplies
these labels to `EmeansShardedCentroids.final_closing_mean(assignments=...)`.
Closing performs no nearest-centroid search: it routes each vector only to
its saved centroid owner, reuses the existing training sum table at training
precision, checks counts against final-epoch counts, and finalizes the means.
Empty clusters retain their final EMA positions. Routing chunks contain at
most 65,536 source rows per rank. CPU codes add 0.931 GiB per rank for 1B rows
across four ranks, outside the host-window budget.

This intentionally changes the SSD flag's semantics from fresh assignments
against the final centroid snapshot to fixed final-epoch assignments, matching
the reference. The library retains its existing fresh-assignment behavior
when `assignments` is omitted. `--no-final-closing-mean` uses accumulated
last-pass statistics without rereading SSD data.

## Correctness checks

- Two-GPU SSD demo self-test: permutation/source-order capture, duplicates,
  bvecs file boundaries, sequential reads, async batches, independently
  calculated saved-label means, and a guard forbidding assignment calls.
- New two-GPU NCCL contract test: deliberately shifted final centroids,
  saved-label means and MSE, empty receiving ranks, mismatched counts,
  and uncoded rows/empty exchanges after a partial final epoch.
- Four-GPU real training smoke run: 100,000 DINO rows, 4096 clusters,
  two epochs, global batch 65536, IVF64/nprobe4. All four ranks passed;
  saved-code closing took 1.205 s. Log: `faiss_saved_closing_smoke.log`.

## Full-scale closing-only timing

Host h100-217-143, 2026-11-03. `bench_closing.py` reads actual DINO bvecs
through the production SSD loader and calls the production closing method.
It constructs uniform synthetic saved codes and their exact counts outside
the timer, using a completed lifecycle without running training.

| Setting | Value |
| --- | --- |
| Rows | 1,000,000,000 |
| Clusters | 2,097,152 |
| GPUs | 4 |
| Global batch | 2,097,152 |
| Reader/extractor threads per rank | 4 / 4 |
| Source reads | O_DIRECT, one sequential corpus pass |
| Closing time, slowest rank | **76.27 s** |
| Peak allocated GPU memory per rank | approximately 12.77 GiB |

All ranks observed the expected saved-code counts for all 1B vectors and
produced finite centroids. Timed work includes host windows, SSD reads, CUDA
copies/transformation, routing, accumulation, fixed-code MSE, and finalization.
No search or evaluation is timed. Uniform synthetic code imbalance can
differ from a real trained codebook, so this is not a new end-to-end training
measurement. Raw report: `faiss_saved_closing_1b.json`.
