# Isolating Faiss shuffled loading costs

Measurements on h100-217-143, 2026-11-03. O_DIRECT throughout, no cache flush.
Production loader code was unchanged. The instrumented script reproduces
its planner and extraction expression around the actual `_scan_chunks` and
`read_views` functions. Four CPU workers, one reader and four extractors per
worker, one first-touched 50 GiB pageable output window per worker.

## Full billion-vector confirmation

All variants use the same requests, row order, source files, chunk boundaries,
and disjoint worker CPU affinity sets. Each issues 2,056,109,842,432 physical
read bytes across five windows. Baseline and reference-planner variants
deliver all 1B vectors; 320 dispersed payload checks passed per variant.
Read-only performs exactly the same reads but omits extraction/validation.

| Variant | Planning (s) | Scan pipeline (s) | Total (s) |
| --- | ---: | ---: | ---: |
| Faiss baseline | 41.4 | 108.5 | 149.4 |
| Reference planner, Faiss reader/extractor | 16.9 | 91.2 | 107.1 |
| Faiss planner, reads only | 42.2 | 50.6 | 92.2 |

Each column sums per-window maxima across ranks, so planning and scan
columns need not add exactly to total. Original uninstrumented Faiss was
146.4 s; the instrumented baseline is within 2.1% of that result.

### Findings

1. **Sorting is a confirmed cost.** The four full windows each spend about
   8.8 s in the Faiss sort planner versus 3.1–3.7 s in the reference mask and
   inverse planner. Total planning falls 24.5 s. The complete intervention
   saves 42.3 s (28.3%); scan times also improve, but this experiment does not
   establish why. Do not attribute the entire improvement to sorting time.

2. **Extraction limits the dense windows.** In the first four windows,
   read-only scanning takes 6.3–6.5 s per window. Baseline scanning with
   extraction takes 17.1–24.1 s. The NumPy gather `data[local]` accounts for
   approximately 70–82% of extraction-thread time in those windows, with
   scatter accounting for most of the rest. These are summed concurrent
   thread durations, not extra serial wall time. This is vectorized row
   indexing, not a Python loop per vector. It still creates a compact
   temporary of the selected rows before scattering to the output window.

3. **The final window is dominated by amplified I/O.** It delivers only
   40,284,800 rows per rank, but the combined tails of all four chunk groups
   require rereading the full 1.028 TB corpus. Read-only takes 24.9 s;
   baseline scanning takes 27.0 s. Planning/copy improvements cannot remove
   that reread. The first four windows combined also read about 1.028 TB.

4. **Storage has headroom.** The read-only scan moves 2.056 TB in 50.6 s,
   approximately 40.7 GB/s. Its total of 92.2 s still includes sorting. The
   dense-window slowdown is principally request planning and CPU extraction,
   rather than insufficient SSD bandwidth. The existing futures pipeline can
   sustain those read-only rates; scheduling overhead under extraction load
   was not independently isolated.

## First-window ablations

Each variant requests the same 52,428,800 rows per rank, drawing from the
same native bvecs chunks. Raw-layout variants read already-staged uint8
shards but keep these same row requests and chunk boundaries. This removes
the permutation difference present in the initial reference/Faiss comparison.

| Variant | Total, shared inherited affinity (s) | Total, disjoint worker cores (s) |
| --- | ---: | ---: |
| Baseline | 32.5 | 23.5 |
| Reference planner | 17.7 | 17.4 |
| Int32 Torch gather/scatter on bvecs views | 24.5 | 24.4 |
| Raw uint8 files, NumPy extraction | 28.7 | 27.3 |
| Raw uint8 files, int64 Torch extraction | 23.6 | Not measured |
| Reference planner plus int32 extraction | Not measured | 19.0 |
| Read-only, including Faiss planner | 15.2 | 15.2 |
| Repeated baseline | 27.3 | 28.1 |
| Repeated reference planner | Not measured | 19.0 |

The baseline varies substantially even with affinity constrained, chiefly in
gather time. Int32 extraction and raw layout therefore have no demonstrated
consistent advantage in these trials. The reference planner improvement
persists and was confirmed across the full billion rows. No claim is made
that a wider Torch copy or removal of bvecs headers alone fixes extraction.

## Recommended next changes

- Port the reference mask/inverse planner first, retaining its duplicate
  fallback and exact output-order semantics.
- Preserve the current borrowed bvecs views and buffer reuse.
- Investigate a compiled fused gather/scatter that copies each requested
  payload directly from the read buffer into its final output row, avoiding
  the selected-row temporary. Benchmark it before replacing NumPy.
- Treat rereads separately: each current group is approximately 62.5M rows,
  exceeding the 52.4M-row output window, so it needs multiple scans. Choosing
  groups small enough to fit entirely in a window and aligning fill boundaries
  with groups could reduce amplification. Separating the current tails alone
  would still reread each group. Changes must preserve training batch order or
  explicitly document a sampling change.

Raw measurements: `ssd_isolate_20261103.json`,
`ssd_isolate_bound_20261103.json`, and `ssd_isolate_full_20261103.json`.

## Production planner replacement

After the diagnostic runs, Faiss `SSDPrefetch.fill_window` was changed to use
the adapted reference planner. Its reader and extraction pipeline were
retained. Running the original `bench_faiss.py --mode shuffled` harness again
(shared inherited CPU affinity, matching the initial benchmark) gave:

- Read time: **92.98 s**, versus the original 146.44 s: **36.5% less time**,
  or 1.57 times the throughput.
- Window times: 16.17, 16.21, 15.94, 15.58, 29.08 s.
- Planning totals per rank: 10.83–11.49 s.
- Physical read bytes: 2,056,109,842,432, identical to the old loader.
- All 320 dispersed payload checks passed. The existing two-GPU self-test
  also passed unique and duplicate requests, empty requests, invalid bounds,
  file boundaries, asynchronous batches, and sequential/closing behavior.

This is a reader benchmark, not a new end-to-end training measurement, and
the timings are single runs. Planner memory includes an int32 inverse map
(1 GB per 250M-row shard), allocated after freeing the byte mask, plus
request-sized arrays. These allocations are outside the host-window budget.
See `ssd_read_faiss_reference_planner_1b.json` for the actual-loader report.
