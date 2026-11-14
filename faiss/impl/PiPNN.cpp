/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/PiPNN.h>

#include <omp.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <utility>
#include <vector>

#include <faiss/Index.h>
#include <faiss/IndexFlatCodes.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/FaissException.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/VisitedTable.h>
#include <faiss/impl/pipnn/HashPrune.h>
#include <faiss/impl/pipnn/Partition.h>
#include <faiss/impl/pipnn/kernels.h>
#include <faiss/utils/AlignedTable.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/ordered_key_value.h>
#include <faiss/utils/prefetch.h>
#include <faiss/utils/utils.h>

namespace faiss {

namespace pipnn {

namespace {

constexpr size_t kMedoidBatch = 1024;
// distances stay finite: at most 4x this in GEMM form
constexpr float kMaxSquaredNorm = FLT_MAX / 8;
constexpr size_t kNumLocks = size_t(1) << 16;

/// Reservoirs are allocated, and released after their final prune, in chunks
/// of this many points.
constexpr size_t kReservoirChunk = size_t(1) << 20;

/// Reservoirs of points [begin, begin + np) in one buffer: np * capacity
/// slots, then np counts, then np cached-farthest indices.
struct ReservoirChunk {
    std::unique_ptr<uint8_t[]> buf;
    size_t begin = 0;
    size_t np = 0;
};

/// Reservoirs of one build and the graph its final prune writes.
struct BuildState {
    size_t capacity;
    size_t chunk_size;
    std::vector<ReservoirChunk> chunks;
    // uninitialized; rows written by the final prune and link_copies
    std::unique_ptr<int32_t[]> graph;

    BuildState(size_t n_slots, size_t n, const PiPNN& p, size_t chunk)
            : capacity(size_t(p.reservoir_size)), chunk_size(chunk) {
        const size_t per_point = capacity * sizeof(Slot) + 2;
        for (size_t begin = 0; begin < n_slots; begin += chunk_size) {
            ReservoirChunk ch;
            ch.begin = begin;
            ch.np = std::min(chunk_size, n_slots - begin);
            ch.buf.reset(new uint8_t[ch.np * per_point]);
            chunks.push_back(std::move(ch));
        }
        graph.reset(new int32_t[n * size_t(p.R)]);
        // Touch every reservoir page now so an oversized build fails early;
        // small static blocks interleave the pages over NUMA nodes.
#pragma omp parallel for schedule(static, 8)
        for (int64_t i = 0; i < int64_t(n_slots); i++) {
            const ReservoirRef r = reservoir(size_t(i));
            std::memset(r.slots, 0, capacity * sizeof(Slot));
            *r.count = 0;
            *r.farthest = kNoFarthest;
        }
    }

    ReservoirRef reservoir(size_t i) {
        ReservoirChunk& ch = chunks[i / chunk_size];
        const size_t off = i - ch.begin;
        Slot* slots = reinterpret_cast<Slot*>(ch.buf.get());
        uint8_t* counts = reinterpret_cast<uint8_t*>(slots + ch.np * capacity);
        return ReservoirRef{
                slots + off * capacity,
                counts + off,
                counts + ch.np + off,
                int(capacity)};
    }
};

/// Keeps the graph alive as long as a MaybeOwnedVector views it.
struct GraphOwner : MaybeOwnedVectorOwner {
    std::unique_ptr<int32_t[]> data;
    explicit GraphOwner(std::unique_ptr<int32_t[]> d) : data(std::move(d)) {}
};

/// Rows with equal stored codes: the graph is built over the lowest, the
/// others (copies) are linked behind it. Empty if all differ.
struct Duplicates {
    /// ascending ids of the rows the graph is built over
    std::vector<int32_t> ids;
    /// slot_of[i]: slot of the lowest id with the code of row i
    std::vector<int32_t> slot_of;

    explicit Duplicates(const Index& storage) {
        const auto* flat = dynamic_cast<const IndexFlatCodes*>(&storage);
        if (!flat) {
            return;
        }
        const size_t n = size_t(storage.ntotal);
        std::vector<int64_t> rep(n);
        if (find_duplicate_codes(
                    n, flat->code_size, flat->codes.data(), rep.data()) == n) {
            return;
        }
        slot_of.resize(n);
        for (size_t i = 0; i < n; i++) {
            if (rep[i] == int64_t(i)) {
                slot_of[i] = int32_t(ids.size());
                ids.push_back(int32_t(i));
            } else {
                slot_of[i] = slot_of[rep[i]];
            }
        }
    }

    size_t n_slots(size_t n) const {
        return ids.empty() ? n : ids.size();
    }
    size_t slot(int32_t id) const {
        return slot_of.empty() ? size_t(id) : size_t(slot_of[id]);
    }
    int32_t id(size_t slot) const {
        return ids.empty() ? int32_t(slot) : ids[slot];
    }

    /// Returns next_copy and clears the graph rows of the copies; consumes ids.
    std::vector<int32_t> link_copies(int32_t* graph, size_t R) {
        std::vector<int32_t> next(slot_of.size(), -1);
        for (size_t i = 0; i < next.size(); i++) {
            int32_t& tail = ids[slot_of[i]];
            if (tail != int32_t(i)) {
                std::fill(graph + i * R, graph + (i + 1) * R, -1);
                next[tail] = int32_t(i);
                tail = int32_t(i);
            }
        }
        return next;
    }
};

} // namespace

int32_t medoid_and_validate(const Index& storage, float* norm_cv) {
    const size_t d = size_t(storage.d);
    const size_t n = size_t(storage.ntotal);
    FAISS_THROW_IF_MSG(n == 0, "PiPNN: the medoid of 0 vectors is undefined");
    FAISS_THROW_IF_MSG(
            n > size_t(std::numeric_limits<int32_t>::max()),
            "PiPNN: more than 2^31 - 1 vectors");
    const size_t nblocks = (n + kMedoidBlock - 1) / kMedoidBlock;
    std::vector<double> sums(nblocks * d);
    std::vector<float> mean(d);
    // (distance to the mean, id): pairs order ties by the lower id
    std::vector<std::pair<float, int32_t>> block_best(nblocks);
    std::vector<std::array<double, 3>> norm_sums(norm_cv ? nblocks : 0);

    // pass 0: validate and sum per block; pass 1: per-block argmin
    for (int pass = 0; pass < 2; pass++) {
        std::exception_ptr ex;
        std::atomic<bool> interrupt{false};
#pragma omp parallel
        {
            AlignedTable<float, 64> buf;
            int32_t ids[kMedoidBatch];
#pragma omp for schedule(dynamic)
            for (int64_t b = 0; b < int64_t(nblocks); b++) {
                if (interrupt.load(std::memory_order_relaxed)) {
                    continue;
                }
                try {
                    ensure_size(buf, kMedoidBatch * d);
                    const size_t i0 = size_t(b) * kMedoidBlock;
                    const size_t i1 = std::min(n, i0 + kMedoidBlock);
                    double* acc = sums.data() + size_t(b) * d;
                    std::pair<float, int32_t> best{
                            std::numeric_limits<float>::infinity(), -1};
                    for (size_t j0 = i0; j0 < i1; j0 += kMedoidBatch) {
                        const size_t k = std::min(kMedoidBatch, i1 - j0);
                        std::iota(ids, ids + k, int32_t(j0));
                        gather_rows(storage, ids, k, buf.get());
                        for (size_t r = 0; r < k; r++) {
                            const float* x = buf.get() + r * d;
                            if (pass == 0) {
                                // NaN and inf fail this comparison too
                                const float nrm = fvec_norm_L2sqr(x, d);
                                FAISS_THROW_IF_NOT_FMT(
                                        nrm <= kMaxSquaredNorm,
                                        "PiPNN: vector %zu has squared norm "
                                        "%g; need finite and <= FLT_MAX / 8",
                                        j0 + r,
                                        double(nrm));
                                if (norm_cv && nrm > 0) {
                                    norm_sums[b][0] += 1;
                                    norm_sums[b][1] += std::sqrt(double(nrm));
                                    norm_sums[b][2] += nrm;
                                }
                                for (size_t j = 0; j < d; j++) {
                                    acc[j] += double(x[j]);
                                }
                            } else {
                                best = std::min(
                                        best,
                                        {fvec_L2sqr(x, mean.data(), d),
                                         int32_t(j0 + r)});
                            }
                        }
                    }
                    if (pass == 1) {
                        block_best[b] = best;
                    }
                    // blocks are large: poll after every one
                    if (omp_get_thread_num() == 0 &&
                        InterruptCallback::is_interrupted()) {
                        FAISS_THROW_MSG("computation interrupted");
                    }
                } catch (...) {
                    omp_capture_exception(ex, [&] { interrupt = true; });
                }
            }
        }
        omp_rethrow_if_exception(ex);
        if (pass == 0) {
            for (size_t j = 0; j < d; j++) {
                double s = 0;
                for (size_t b = 0; b < nblocks; b++) {
                    s += sums[b * d + j];
                }
                mean[j] = float(s / double(n));
            }
        }
    }
    if (norm_cv) {
        double c = 0, s1 = 0, s2 = 0;
        for (const auto& [bc, b1, b2] : norm_sums) {
            c += bc;
            s1 += b1;
            s2 += b2;
        }
        if (c == 0) {
            *norm_cv = 0;
        } else {
            const double mean = s1 / c;
            *norm_cv = std::sqrt(std::max(s2 / c - mean * mean, 0.0)) / mean;
        }
    }
    return std::min_element(block_best.begin(), block_best.end())->second;
}

namespace {

PartitionParams partition_params(const PiPNN& p, MetricType metric) {
    return {.c_max = p.c_max,
            .c_min = p.c_min,
            .leader_cap = p.leader_cap,
            .leader_fraction = p.leader_fraction,
            .fanout = p.fanout,
            .metric = metric,
            .ip_partition_by_angle = p.ip_partition_by_angle,
            .wave_budget = p.partition_wave_budget};
}

/// Offers the nearest neighbours of every leaf point to the HashPrune
/// reservoirs.
void fill_reservoirs(
        BuildState& st,
        const Index& storage,
        const Duplicates& dup,
        const PiPNN& p,
        bool verbose) {
    const size_t n = dup.n_slots(size_t(storage.ntotal));
    const size_t d = size_t(storage.d);
    const MetricType metric = storage.metric_type;
    const bool is_ip = metric == METRIC_INNER_PRODUCT;
    // n <= c_max is a single leaf; there each point offers its
    // reservoir_size nearest instead of its leaf_k nearest.
    const size_t k =
            n <= size_t(p.c_max) ? size_t(p.reservoir_size) : size_t(p.leaf_k);

    const Hyperplanes hp(storage.d, p.hash_bits, p.seed);
    std::vector<std::mutex> locks(kNumLocks);

    struct LeafScratch {
        AlignedTable<float, 64> vecs, gram;
        std::vector<float> norms, nn_dist, sketches;
        std::vector<int32_t> nn_idx;
    };
    std::vector<LeafScratch> scratch(omp_get_max_threads());
    const LeafVisitor on_leaf = [&](const int32_t* ids, size_t s) {
        LeafScratch& sc = scratch[omp_get_thread_num()];
        ensure_size(sc.vecs, s * d);
        gather_rows(storage, ids, s, sc.vecs.get());
        sc.norms.resize(s);
        sc.sketches.resize(s * 16);
        for (size_t i = 0; i < s; i++) {
            const float* x = sc.vecs.get() + i * d;
            sc.norms[i] = fvec_norm_L2sqr(x, d);
            hp.sketch(x, sc.sketches.data() + i * 16);
        }
        ensure_size(sc.gram, s * s);
        sc.nn_idx.resize(s * k);
        sc.nn_dist.resize(s * k);
        leaf_knn(
                sc.vecs.get(),
                sc.norms.data(),
                s,
                d,
                metric,
                int(k),
                sc.gram.get(),
                sc.nn_idx.data(),
                sc.nn_dist.data());
        // Offers leaf point b to the reservoir of leaf point a. Keys are
        // dist - d(a, a), so near candidates sit near 0 where bf16 is finest.
        auto offer = [&](size_t a, size_t b, float dist) {
            const uint16_t hash = residual_hash(
                    sc.sketches.data() + a * 16,
                    sc.sketches.data() + b * 16,
                    p.hash_bits);
            const uint16_t key =
                    distance_key(is_ip ? dist + sc.norms[a] : dist);
            std::lock_guard<std::mutex> guard(
                    locks[uint32_t(ids[a]) & (kNumLocks - 1)]);
            st.reservoir(dup.slot(ids[a])).offer(ids[b], hash, key);
        };
        for (size_t i = 0; i < s; i++) {
            for (size_t t = 0; t < k; t++) {
                const int32_t j = sc.nn_idx[i * k + t];
                if (j < 0) {
                    break;
                }
                offer(i, size_t(j), sc.nn_dist[i * k + t]);
                offer(size_t(j), i, sc.nn_dist[i * k + t]);
            }
        }
    };

    const double t0 = getmillisecs();
    PartitionStats stats;
    partition_and_visit(
            storage,
            partition_params(p, metric),
            mix_seed(uint64_t(p.seed), 0),
            on_leaf,
            &stats,
            dup.ids);
    if (verbose) {
        printf("PiPNN: partition: %.3f s, %zu leaves, replication %.2f\n",
               (getmillisecs() - t0) / 1000,
               stats.n_leaves,
               double(stats.n_point_instances) / double(n));
    }
}

/// Prunes each reservoir into its graph row, freeing each chunk once pruned.
void final_prune(
        BuildState& st,
        const Index& storage,
        const Duplicates& dup,
        float alpha,
        int R,
        bool cosine) {
    const size_t d = size_t(storage.d);
    const MetricType metric = storage.metric_type;
    int32_t* graph = st.graph.get();
    for (ReservoirChunk& ch : st.chunks) {
        std::exception_ptr ex;
        std::atomic<bool> interrupt{false};
#pragma omp parallel
        {
            AlignedTable<float, 64> X;
            AlignedTable<float, 64> D;
            std::vector<float> norms;
            std::vector<int32_t> ids;
            std::vector<int> order;
            size_t n_done = 0;
#pragma omp for schedule(dynamic, 64)
            for (int64_t i = int64_t(ch.begin); i < int64_t(ch.begin + ch.np);
                 i++) {
                if (interrupt.load(std::memory_order_relaxed)) {
                    continue;
                }
                try {
                    const ReservoirRef r = st.reservoir(size_t(i));
                    const size_t l = *r.count;
                    const size_t m = l + 1;
                    ensure_size(X, m * d);
                    ensure_size(D, m * m);
                    norms.resize(m);
                    ids.resize(m);
                    order.resize(l);
                    ids[0] = dup.id(size_t(i));
                    for (size_t t = 0; t < l; t++) {
                        ids[1 + t] = r.slots[t].id;
                    }
                    gather_rows(storage, ids.data(), m, X.get());
                    if (cosine) {
                        fvec_renorm_L2(d, m, X.get());
                    }
                    for (size_t t = 0; t < m; t++) {
                        norms[t] = fvec_norm_L2sqr(X.get() + t * d, d);
                    }
                    pairwise_distances(
                            X.get(), norms.data(), m, d, metric, D.get());
                    robust_prune(
                            D.get(),
                            m,
                            ids.data() + 1,
                            metric,
                            alpha,
                            R,
                            order.data(),
                            graph + size_t(ids[0]) * size_t(R));
                    if (omp_get_thread_num() == 0 && n_done++ % 64 == 0 &&
                        InterruptCallback::is_interrupted()) {
                        FAISS_THROW_MSG("computation interrupted");
                    }
                } catch (...) {
                    omp_capture_exception(ex, [&] { interrupt = true; });
                }
            }
        }
        omp_rethrow_if_exception(ex);
        ch.buf.reset();
    }
}

} // namespace

void robust_prune(
        const float* D,
        size_t m,
        const int32_t* cand,
        MetricType metric,
        float alpha,
        int R,
        int* order,
        int32_t* out) {
    const size_t l = m - 1;
    std::iota(order, order + l, 0);
    std::sort(order, order + l, [&](int a, int b) {
        return CMin<float, int32_t>::cmp2(D[1 + a], D[1 + b], cand[a], cand[b]);
    });
    const bool is_ip = metric == METRIC_INNER_PRODUCT;
    int na = 0;
    for (size_t t = 0; t < l && na < R; t++) {
        const int c = order[t];
        const float dxc = D[1 + c];
        const bool kept = std::none_of(order, order + na, [&](int y) {
            const float dyc = D[size_t(1 + y) * m + 1 + c];
            return is_ip ? dyc < alpha * dxc : alpha * dyc < dxc;
        });
        if (kept) {
            order[na++] = c; // na <= t: never overwrites an unread entry
        }
    }
    for (int u = 0; u < na; u++) {
        out[u] = cand[order[u]];
    }
    std::fill(out + na, out + R, -1);
}

namespace detail {

void build_with_chunk(
        PiPNN& p,
        const Index& storage,
        size_t reservoir_chunk,
        bool verbose) {
    p.check_params();
    const size_t n = size_t(storage.ntotal);
    if (n == 0) {
        p.effective_alpha(storage.metric_type);
        p.reset();
        return;
    }

    const double t0 = getmillisecs();
    InterruptCallback::check();
    float norm_cv = 0;
    const int32_t ep = medoid_and_validate(
            storage,
            storage.metric_type == METRIC_INNER_PRODUCT ? &norm_cv : nullptr);
    const bool cosine = norm_cv > 0.05f;
    const float alpha = p.effective_alpha(storage.metric_type);
    if (verbose) {
        printf("PiPNN: medoid and validation of %zu vectors (d=%d): %.3f s\n",
               n,
               storage.d,
               (getmillisecs() - t0) / 1000);
    }

    Duplicates dup(storage);
    BuildState st(dup.n_slots(n), n, p, reservoir_chunk);
    fill_reservoirs(st, storage, dup, p, verbose);

    const double t1 = getmillisecs();
    final_prune(st, storage, dup, alpha, p.R, cosine);
    std::vector<int32_t> next_copy =
            dup.link_copies(st.graph.get(), size_t(p.R));
    InterruptCallback::check();
    if (verbose) {
        printf("PiPNN: final prune: %.3f s\n", (getmillisecs() - t1) / 1000);
    }

    // Publish: nothing below can throw after the members start changing.
    auto owner = std::make_shared<GraphOwner>(std::move(st.graph));
    p.graph = MaybeOwnedVector<int32_t>::create_view(
            owner->data.get(), n * size_t(p.R), owner);
    p.next_copy = std::move(next_copy);
    p.entry_point = ep;
    p.ntotal = idx_t(n);
}

} // namespace detail

namespace {

/// Up to cap entries in (dis, id) order; all before cursor are expanded.
struct SearchBeam {
    struct Entry {
        float dis;
        int32_t id;
        bool expanded;

        bool operator<(const Entry& o) const {
            return CMin<float, int32_t>::cmp2(dis, o.dis, id, o.id);
        }
    };

    std::vector<Entry> entries;
    size_t cap;
    size_t cursor = 0;

    explicit SearchBeam(size_t cap_in) : cap(cap_in) {
        entries.reserve(cap_in + 1);
    }

    void insert(float dis, int32_t id) {
        const Entry e{dis, id, false};
        if (entries.size() == cap && !(e < entries.back())) {
            return;
        }
        const auto it = std::lower_bound(entries.begin(), entries.end(), e);
        cursor = std::min(cursor, size_t(it - entries.begin()));
        entries.insert(it, e);
        if (entries.size() > cap) {
            entries.pop_back();
        }
    }

    /// The closest unexpanded entry after the cursor, or -1.
    int32_t peek_next() const {
        for (size_t c = cursor + 1; c < entries.size(); c++) {
            if (!entries[c].expanded) {
                return entries[c].id;
            }
        }
        return -1;
    }

    /// Marks the closest unexpanded entry expanded; returns its id, or -1.
    int32_t expand_next() {
        while (cursor < entries.size() && entries[cursor].expanded) {
            cursor++;
        }
        if (cursor == entries.size()) {
            return -1;
        }
        entries[cursor].expanded = true;
        return entries[cursor].id;
    }
};

template <class VT>
void beam_search(
        const PiPNN& pipnn,
        DistanceComputer& qdis,
        const uint8_t* codes,
        size_t code_size,
        size_t cap,
        size_t k,
        idx_t* I,
        float* D,
        VT& vt,
        const IDSelector* sel) {
    const uint32_t n = uint32_t(pipnn.ntotal);
    const size_t R = size_t(pipnn.R);
    const int32_t* graph = pipnn.graph.data();
    const int32_t* next_copy =
            pipnn.next_copy.size() == 0 ? nullptr : pipnn.next_copy.data();
    SearchBeam beam(cap);

    // id breaks ties, so the result does not depend on visit order
    auto offer = [&](int32_t id, float dis) {
        const bool better =
                CMax<float, idx_t>::cmp2(D[0], dis, I[0], idx_t(id));
        if (better && (sel == nullptr || sel->is_member(id))) {
            maxheap_replace_top(k, D, I, dis, idx_t(id));
        }
        return better;
    };
    // Copies skip the beam. Their ids increase, so the first one the heap
    // rejects ends the chain; checking that also stops a corrupt chain.
    auto visit = [&](int32_t id, float dis) {
        int32_t c = id;
        while (offer(c, dis) && next_copy && next_copy[c] > c &&
               uint32_t(next_copy[c]) < n) {
            c = next_copy[c];
        }
        beam.insert(dis, id);
    };

    vt.set(size_t(pipnn.entry_point));
    visit(pipnn.entry_point, qdis(pipnn.entry_point));

    // Prefetching more than 1 KiB per vector costs more than it saves.
    const size_t prefetch_bytes = std::min(code_size, size_t(1024));
    auto prefetch = [](const void* p, size_t nbytes) {
        const uintptr_t a = reinterpret_cast<uintptr_t>(p);
        for (uintptr_t b = a & ~uintptr_t(63); b < a + nbytes; b += 64) {
            prefetch_L1(reinterpret_cast<const void*>(b));
        }
    };
    auto prefetch_code = [&](int32_t id) {
        prefetch(codes + size_t(id) * code_size, prefetch_bytes);
    };
    // Vectors are prefetched about 4 KiB ahead of the distance computation,
    // at least one batch: prefetching all neighbors of a large-vector graph at
    // once evicts the ones about to be read when two threads share a core.
    const size_t ahead = std::max(
            size_t(4), size_t(4096) / std::max(prefetch_bytes, size_t(64)));
    std::vector<int32_t> todo(R);
    for (int32_t v0 = beam.expand_next(); v0 >= 0; v0 = beam.expand_next()) {
        // The closest unexpanded entry usually comes next.
        const int32_t next = beam.peek_next();
        if (next >= 0) {
            prefetch(graph + size_t(next) * R, R * sizeof(int32_t));
        }
        const int32_t* row = graph + size_t(v0) * R;
        // One unsigned compare stops at the -1 padding and at any id that is
        // out of range in a corrupt (e.g. mmapped) graph.
        size_t jmax = 0;
        while (jmax < R && uint32_t(row[jmax]) < n) {
            vt.prefetch(size_t(row[jmax]));
            jmax++;
        }
        size_t nt = 0;
        for (size_t j = 0; j < jmax; j++) {
            if (vt.set(size_t(row[j]))) {
                if (nt < ahead) {
                    prefetch_code(row[j]);
                }
                todo[nt++] = row[j];
            }
        }
        size_t i = 0;
        for (; i + 4 <= nt; i += 4) {
            for (size_t t = i + ahead; t < std::min(i + ahead + 4, nt); t++) {
                prefetch_code(todo[t]);
            }
            float d4[4];
            qdis.distances_batch_4(
                    todo[i],
                    todo[i + 1],
                    todo[i + 2],
                    todo[i + 3],
                    d4[0],
                    d4[1],
                    d4[2],
                    d4[3]);
            for (size_t t = 0; t < 4; t++) {
                visit(todo[i + t], d4[t]);
            }
        }
        for (; i < nt; i++) {
            visit(todo[i], qdis(todo[i]));
        }
    }
}

} // namespace

} // namespace pipnn

float PiPNN::effective_alpha(MetricType metric) const {
    FAISS_THROW_IF_NOT_MSG(
            metric == METRIC_L2 || metric == METRIC_INNER_PRODUCT,
            "PiPNN supports only METRIC_L2 and METRIC_INNER_PRODUCT");
    return alpha != 0 ? alpha : 1.2f;
}

void PiPNN::check_params() const {
    FAISS_THROW_IF_NOT_FMT(R >= 1, "PiPNN: R must be >= 1, got %d", R);
    FAISS_THROW_IF_NOT_FMT(
            leaf_k >= 1 && leaf_k <= c_max,
            "PiPNN: need 1 <= leaf_k <= c_max, got leaf_k=%d c_max=%d",
            leaf_k,
            c_max);
    FAISS_THROW_IF_NOT_FMT(
            hash_bits >= 1 && hash_bits <= 16,
            "PiPNN: hash_bits must be in [1, 16], got %d",
            hash_bits);
    FAISS_THROW_IF_NOT_FMT(
            reservoir_size >= 1 && reservoir_size <= 255,
            "PiPNN: reservoir_size must be in [1, 255], got %d",
            reservoir_size);
    FAISS_THROW_IF_NOT_FMT(
            R <= reservoir_size,
            "PiPNN: need R <= reservoir_size, got R=%d reservoir_size=%d",
            R,
            reservoir_size);
    for (int f : fanout) {
        FAISS_THROW_IF_NOT_FMT(
                f >= 1, "PiPNN: every fanout must be >= 1, got %d", f);
    }
    FAISS_THROW_IF_NOT_FMT(
            c_min >= 1 && c_min <= c_max,
            "PiPNN: need 1 <= c_min <= c_max, got c_min=%d c_max=%d",
            c_min,
            c_max);
    FAISS_THROW_IF_NOT_FMT(
            c_max <= pipnn::kMaxCMax,
            "PiPNN: c_max must be <= %d, got %d",
            pipnn::kMaxCMax,
            c_max);
    FAISS_THROW_IF_NOT_FMT(
            leader_cap >= 1 && leader_cap <= pipnn::kMaxLeaderCap,
            "PiPNN: leader_cap must be in [1, %d], got %d",
            pipnn::kMaxLeaderCap,
            leader_cap);
    FAISS_THROW_IF_NOT_FMT(
            leader_fraction >= 0 && leader_fraction <= 1,
            "PiPNN: leader_fraction must be in [0, 1], got %g",
            double(leader_fraction));
    FAISS_THROW_IF_NOT_FMT(
            alpha == 0 || (std::isfinite(alpha) && alpha >= 1),
            "PiPNN: alpha must be 0 (default) or finite and >= 1, got %g",
            double(alpha));
}

void PiPNN::build(const Index& storage, bool verbose) {
    pipnn::detail::build_with_chunk(
            *this, storage, pipnn::kReservoirChunk, verbose);
}

void PiPNN::search(
        DistanceComputer& qdis,
        int L,
        idx_t k,
        idx_t* I,
        float* D,
        VisitedTable& vt,
        const IDSelector* sel) const {
    FAISS_THROW_IF_NOT_MSG(k > 0, "PiPNN::search: k must be positive");
    maxheap_heapify(size_t(k), D, I);
    if (ntotal == 0) {
        return;
    }
    // At most ntotal distinct nodes enter the beam.
    const size_t cap = size_t(std::min(std::max(idx_t(L), k), ntotal));
    // The codes of Flat and SQ storage, to prefetch.
    DistanceComputer* base = &qdis;
    if (auto* neg = dynamic_cast<NegativeDistanceComputer*>(base)) {
        base = neg->basedis;
    }
    const auto* fc = dynamic_cast<const FlatCodesDistanceComputer*>(base);
    const uint8_t* codes = fc ? fc->codes : nullptr;
    const size_t code_size = fc ? fc->code_size : 0;
    auto run = [&](auto& vtc) {
        pipnn::beam_search(
                *this, qdis, codes, code_size, cap, size_t(k), I, D, vtc, sel);
    };
    if (auto* vtv = dynamic_cast<VisitedTableVector*>(&vt)) {
        run(*vtv);
    } else {
        run(dynamic_cast<VisitedTableSet&>(vt));
    }
    maxheap_reorder(size_t(k), D, I);
    vt.advance();
}

void PiPNN::reset() {
    // not graph.clear(): it asserts on the view build() publishes
    graph = MaybeOwnedVector<int32_t>();
    next_copy = MaybeOwnedVector<int32_t>();
    entry_point = -1;
    ntotal = 0;
}

} // namespace faiss
