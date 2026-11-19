# Optional IVF assignment

`EmeansShardedCentroidsIVF` keeps the centroid-owner lifecycle and uses the
faithful IVF-Flat candidate-search algorithm. It is an approximate assignment
engine: coarse routing probes `nprobe` lists, then the fine search (`ivf_fine`)
selects within those candidates. It does not replace limited probing with a
full-table assignment.

```python
from faiss.contrib.e_means import EmeansShardedCentroidsIVF

model = EmeansShardedCentroidsIVF(
    d, global_k, group=group, nlist=256, nprobe=4,
    coarse_niter=10, search_workspace_bytes=512 << 20)
model.begin(global_n, local_initial_centroids)
# Existing accumulate/update/end_epoch/finalize lifecycle.
```

The class is available from `faiss.contrib.e_means`,
`sharded_centroids.py`, and `sharded_centroids_ivf.py`. Package exports of the
two sharder classes are lazy, so importing the ordinary package does not require
Torch. Accessing either class requires Torch.

Caller-owned groups, CUDA centroid shards, equal rank contributions and the base
validation/lifecycle rules remain. Quantizers are unsupported. Spherical mode
is rejected before collectives because this score is not the spherical metric.
Coarse training preserves seeded sample membership, FP16 Triton assignment and
distributed reduction. Local lists are rebuilt on the first positive transfer
of each frozen logical batch and shared by later transfers. Global occupancy is
reduced in INT64. During training, each reader routes its local rows before
gathering them. Global winner reduction retains lowest-global-ID ties.

Fine assignment and split separation use BF16; sums/counts and SSE retain
original FP32 rows. With `unused_threshold=None`, revival uses removal costs
and Oja donor gains, with IVF nearest-neighbor protection across all owners. Removal costs use
the two best distinct centroids in the probed lists, so they are approximate
when probing is limited. A missing finite second candidate contributes zero
cost, as in the base removal-statistics implementation.

Each owner retains its removal EMA, batch removal sums and Oja axes. On CUDA,
Triton kernels handle owner accumulation and removal EMA updates even
with the default native assignment backend. The update normalizes axes and
clears batch sums in the same kernel, without a full FP64 vector temporary.
It also caches FP32 axis lengths for the next revival check. Unsupported
statistics layouts retain the ordinary Torch path. Splits invalidate the cache;
the next update refreshes it, or a check before another update recomputes it in
bounded chunks.
Selection sends three FP32 scalars per centroid only to rank 0: 192 MiB for
16M centroids, instead of replicating a 384 MiB FP64 table on every owner.
Rank 0 computes the median threshold on the GPU in FP64 units, sorts only
removal candidates by cost and global ID, and broadcasts their IDs for neighbor
search. Ordered nearest-neighbor protection runs on the GPU, retaining the
same forward greedy rule. Rank 0 selects the required donors using a kth-value
cutoff, then sorts only those donors by FP32 Oja gain, retaining lowest-global-ID
ties. Selected donor and target IDs are broadcast to the other owners. No full
cost or neighbor arrays are copied to the CPU for selection.
Protection reuses the frozen IVF index prepared for training, searches two
neighbors in the configured probes, excludes self, and reduces the best
remaining score and global ID across owners. Queries use updated centroid
positions; indexed vectors and list memberships precede the batch update.
Rank 0 routes these shared queries once and broadcasts INT32 probe IDs; every
owner searches its own shard with those probes.
The next training batch rebuilds the index. Protection
uses bounded query chunks and BF16 search scores; limited probing makes
protection approximate. A candidate with no other finite neighbor in the probes
protects no one, without an exhaustive fallback. Split children receive their donor's
halved removal EMA and copied axis. Both moved centroids retain EMA positions
when closing. Fractional epoch budgets use the base revival cutoff. Setting
`unused_threshold` explicitly selects mass-based revival (0 disables it).

Recipe additions include `index_factory`, `nprobe`, `fine_dtype`, `revival_protection="ivf"` and
`coarse_query_chunk`. The sharder uses up to 16,384 rows per coarse-routing
chunk, capped by the search workspace budget.

## Index options and mutable precision policy

```python
from faiss.contrib.e_means.ivf_flat import IndexIVFFlat

index = IndexIVFFlat(coarse_centroids, nprobe=4, query_chunk=2048)
index.add(candidates)
scores, ids = index.search_1nn(queries)
# Two distinct candidates, scores/IDs shaped (len(queries), 2):
scores, ids = index.search_2nn(queries)
```

The mutable attributes accept only these values:

| Attribute | Contract |
|---|---|
| `implem` | Integer 1–5. `<2` uses one full query operation; `>=2` chunks queries. `<3` selects native operands; `>=3` selects `dtype`. `>=4` stores only reduced candidates when reduced precision is active. |
| `dtype` | Exactly `"fp16"` or `"bf16"`. Native is selected with `implem < 3`, not another dtype spelling. |
| `lowp_scope` | Exactly `"all"` or `"fine"`. `"fine"` keeps coarse query helpers native while fine search uses the selected reduced precision. |

Unsupported values, Torch dtype objects and misspellings raise `ValueError`.
Defaults remain `implem=4`, `dtype="fp16"`, `lowp_scope="all"`.

Add membership and query routing are separate state:

- CUDA `add` keeps fixed FP16 Triton coarse membership. CPU `add` uses the
  effective coarse policy at that call. The resulting membership and list order
  stay fixed until the next `add`; later selector changes never relabel rows.
- Routing norms are cached by effective query policy. The next route after an
  effective policy change computes norms from the same represented coarse
  operands used by its dot product, then reapplies the saved local or global
  occupancy. This costs one bounded `nlist*d` norm pass and stores an `nlist`
  boolean occupancy copy; it adds no host value scan or synchronization.
- Fine candidate storage records the effective representation used by `add`.
  Changes that keep it compatible proceed: `1↔2`, `3↔4↔5`, and any
  `lowp_scope` change. Native↔reduced changes require `add` again, as do
  FP16↔BF16 changes while reduced precision is effective. Search raises a clear
  rebuild error instead of reinterpreting stored bytes. Changing back to the
  stored representation is compatible because those bytes were not modified.

`search_1nn` and `search_2nn` run the fine search of `ivf_fine` on the probed
lists. On CUDA, probes are selected from the coarse scores by an exact radix
select, for any `nlist` and `nprobe` (the lists of `torch.topk`, ordered by
score, lower list IDs first among equal scores); CPU routing uses
`torch.topk`.
`_exact_1nn` and `_exact_topk` construct coherent norms when none are supplied.
Explicitly supplied norms remain caller-authoritative and are used verbatim;
the helper does not reinterpret or silently replace them after a policy change.
Supplied probe IDs likewise retain their existing shape/device conversion rules.

## Precision and numerical limits

The fine search accumulates the represented operands in FP32: on CUDA in
16-wide contraction steps in order (the order of a BMM over list widths padded
to multiples of 16), on CPU as FP32 products of the decoded operands. Query norms are omitted from returned scores, so
owners searching the same row can compare scores. Candidate and routing norms
use the representation of their matching dot operands.

Bit-equal scores retain the lowest row ID. Non-finite scores, which only
overflowing or NaN inputs produce, count as missing. Limited probing and reduced operands remain approximate.
No normalization, clipping or input-value scan is added.

Caller code that supplies helper norms is responsible for matching them to its
operands. The index guarantees coherence only for automatically constructed
query/candidate norms and its fixed add membership operation.

## Ownership, aliasing and allocation

Contiguous FP32 coarse input may alias index storage and must remain fixed for
the index lifetime. No clone or external-mutation detector is added. Candidate
storage is rebuilt by `add`; selector changes never convert it in place. The
index owns its occupancy copy and derived norm caches. Calls on one index share
mutable state, so callers must order stream changes and must not search it
concurrently with mutation/rebuild.

Returned IDs are INT32 original row IDs. Prefixes/group offsets must fit signed
INT32; exact guards reject larger counts. Empty or no-local-hit search returns
`(+inf, -1)`. Stable sorting, lowest-ID merge and per-owner `ndis`/
`coarse_ndis` accounting remain.

The fine search allocates, per routed query/probe pair, two FP32 scores, two
INT64 IDs and the sort and filing entries (about 44 bytes), plus one item
table entry per 64 or 128 pairs. Candidate storage, query representations, coarse scores,
allocator state and base statistics are additional. CPU search uses FP32
decode temporaries for one list at a time.

Top-2 search computes each fine dot product once and merges both candidates
across probes. Ties use the lowest candidate ID; missing
results use `inf` and `-1`. Sharded training reduces the global first score and
ID, then performs one additional MIN reduction for the second score. Revival
adds two FP32 tables of shape `(local_k, d + 1)` and one FP32 axis table of
shape `(local_k, d)` per owner. FP8 retains its existing mass-based revival.

On CUDA, routed pairs are filed by list; each work item scans one list for up
to 64 (Triton) or 128 (CuTe) of its pairs, fusing the distance epilogue and
the running top-2, and a final kernel merges probes once per query. Each pair
owns one result slot, so no atomics touch results. The CuTe kernel runs on SM90
with BF16 operands when its optional packages import; the Triton kernel returns
the same bits elsewhere and serves FP16 and FP32 operands. CPU search runs the
same algorithm with PyTorch operations. Revival protection reuses the current training IVF
index across query chunks without another rebuild; it does not cache neighbor
IDs across centroid updates.

For CUDA training with the Triton backend, one owner-statistics kernel
accumulates FP32 sums/counts, SSE and removal/Oja statistics directly from the
gathered rows. This avoids materializing owned rows and residual tables.
FP32 atomic additions may change accumulation order; SSE squares remain FP32
and are summed in FP64. Deterministic mode, custom owner hooks, centered
quantizers and separate closing-mean buffers retain the Torch accumulation.

The public tests cover selectors, norm coherence, representation transitions,
helpers, ownership and bounded lifecycle cases. CPU results do not qualify CUDA
capture, every deployment geometry, quality or performance. Hardware
qualification is reported separately.
