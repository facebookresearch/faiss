/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexIVFRaBitQSSD.h>

#include <omp.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/RaBitQUtils.h>
#include <faiss/impl/fast_scan/fast_scan.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/impl/simd_dispatch.h>
#include <faiss/invlists/DirectMap.h>
#include <faiss/invlists/DiskIOBackend.h>
#include <faiss/invlists/RaBitQSSDInvertedLists.h>
#include <faiss/utils/AlignedTable.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/rabitq_simd.h>
#include <faiss/utils/utils.h>

namespace faiss {

/* Distance estimates. For a vector o in the list of centroid c, with
 * residual r = o - c and sign bits x_u[i] = (r[i] > 0), the RaBitQ estimator
 * of <q - c, r> uses the code vector xb[i] = x_u[i] * 2^ex_bits + ex[i] + cb
 * with cb = -(2^ex_bits - 1/2):
 *
 *     <q - c, r> ~ |r|^2 <xb, q - c> / <xb, r>
 *
 * The terms that depend on c are folded into per-vector factors, so that the
 * estimate is affine in <xb, q> and a query needs a single look-up table:
 *
 *     dist(q, o) ~ g_add + f_add_ex + f_rescale_ex * <xb, q>
 *
 * L2:  dist = |q - o|^2, g_add = |q - c|^2,
 *      f_add_ex = |r|^2 - f_rescale_ex * <xb, c>,
 *      f_rescale_ex = -2 |r|^2 / <xb, r>
 * IP:  dist = -<q, o>, g_add = -<q, c>,
 *      f_add_ex = -<r, c> - f_rescale_ex * <xb, c>,
 *      f_rescale_ex = -|r|^2 / <xb, r>
 *
 * The in-memory estimate uses the sign bits of dimensions [0, mem_dim) only:
 * with x1 = x_u - 1/2, f_add, f_rescale and <x1, q> replace the above, with
 * <x1, q> and <x1, c> summed over [0, mem_dim), <x1, r> over [0, d), and an
 * extra factor d / mem_dim in f_rescale. Its error is bounded by
 * f_error * |q - c|, with f_error = k * epsilon * |r| / sqrt(mem_dim) (k = 2
 * for L2, 1 for IP, epsilon = 1.9).
 */

IndexIVFRaBitQSSDStats indexIVFRaBitQSSD_stats;

void IndexIVFRaBitQSSDStats::reset() {
    n_pages_read = 0;
    n_reranked = 0;
}

void IndexIVFRaBitQSSDStats::add(const IndexIVFRaBitQSSDStats& other) {
    n_pages_read += other.n_pages_read;
    n_reranked += other.n_reranked;
}

namespace {

using Lists = RaBitQSSDInvertedLists;

constexpr size_t kBatchSize = Lists::kBatchSize;

/// constant of the RaBitQ error bound
constexpr double kConstEpsilon = 1.9;

/// vectors encoded at once when adding to a list
constexpr size_t kAddChunk = 4096;

/*********************************************************
 * Encoding
 *********************************************************/

/// Split the RaBitQuantizer code of one vector into its in-memory part (the
/// sign bits of [0, mem_dim) and kMemFactors factors) and its file record.
void split_code(
        const Lists& il,
        MetricType metric,
        const float* x,
        const float* centroid,
        const uint8_t* code,
        idx_t id,
        uint8_t* mem_code,
        float* mem_factors,
        uint8_t* record) {
    const size_t d = il.d;
    const size_t mem_dim = il.mem_dim;
    const size_t ex_bits = il.ex_bits;
    // RaBitQuantizer code layout for nb_bits > 1:
    // [sign bits][SignBitFactorsWithError][ex code][ExtraBitsFactors]
    const uint8_t* sign_bits = code;
    const uint8_t* ex_code =
            code + d / 8 + sizeof(rabitq_utils::SignBitFactorsWithError);
    const double cb = -(double(1 << ex_bits) - 0.5);

    double r_norm2 = 0, ip_r_c = 0;
    double ip_r_x1 = 0, ip_c_x1_mem = 0; // with the 1-bit code x_u - 1/2
    double ip_r_xb = 0, ip_c_xb = 0;     // with the full code xb
    for (size_t i = 0; i < d; i++) {
        const double ci = centroid[i];
        const double r = double(x[i]) - ci;
        const int bit = (sign_bits[i / 8] >> (i % 8)) & 1;
        r_norm2 += r * r;
        ip_r_c += r * ci;
        const double x1 = bit - 0.5;
        ip_r_x1 += r * x1;
        if (i < mem_dim) {
            ip_c_x1_mem += ci * x1;
        }
        const int ex = rabitq_utils::extract_code_inline(ex_code, i, ex_bits);
        const double xb = double((bit << ex_bits) + ex) + cb;
        ip_r_xb += r * xb;
        ip_c_xb += ci * xb;
    }
    const double k = metric == METRIC_L2 ? 2 : 1;
    const double f_base = metric == METRIC_L2 ? r_norm2 : -ip_r_c;

    // ip_r_x1 = sum |r[i]| / 2, and every term of ip_r_xb has the sign of
    // r[i], so they are 0 only for r = 0
    const double f_rescale =
            ip_r_x1 > 0 ? -k * r_norm2 * (double(d) / mem_dim) / ip_r_x1 : 0;
    mem_factors[0] = float(f_base - f_rescale * ip_c_x1_mem);
    mem_factors[1] = float(f_rescale);
    mem_factors[2] = float(k * kConstEpsilon * std::sqrt(r_norm2 / mem_dim));
    memcpy(mem_code, sign_bits, mem_dim / 8);

    // record: [sign bits mem_dim..d) [ex code] [f_add_ex f_rescale_ex] [id]
    uint8_t* p = record;
    memcpy(p, sign_bits + mem_dim / 8, (d - mem_dim) / 8);
    p += (d - mem_dim) / 8;
    memcpy(p, ex_code, d * ex_bits / 8);
    p += d * ex_bits / 8;
    const double f_rescale_ex = ip_r_xb != 0 ? -k * r_norm2 / ip_r_xb : 0;
    const float ex_factors[2] = {
            float(f_base - f_rescale_ex * ip_c_xb), float(f_rescale_ex)};
    memcpy(p, ex_factors, sizeof(ex_factors));
    p += sizeof(ex_factors);
    memcpy(p, &id, sizeof(id));
}

/// Add the vectors x[members[i]] to list list_no, by increasing distance to
/// the centroid, ties broken by id.
void add_to_list(
        const IndexIVFRaBitQSSD& index,
        Lists& il,
        idx_t list_no,
        const idx_t* members,
        size_t n,
        const float* x,
        const idx_t* xids) {
    const size_t d = index.d;
    std::vector<float> centroid(d);
    index.quantizer->reconstruct(list_no, centroid.data());

    std::vector<idx_t> ids(n);
    std::vector<double> key(n);
    for (size_t i = 0; i < n; i++) {
        const idx_t j = members[i];
        ids[i] = xids ? xids[j] : index.ntotal + j;
        const float* xj = x + j * d;
        double acc = 0;
        for (size_t t = 0; t < d; t++) {
            if (index.metric_type == METRIC_L2) {
                const double diff = double(xj[t]) - double(centroid[t]);
                acc += diff * diff;
            } else {
                acc -= double(xj[t]) * double(centroid[t]);
            }
        }
        key[i] = acc;
    }
    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        const bool a_nan = std::isnan(key[a]);
        const bool b_nan = std::isnan(key[b]);
        if (a_nan != b_nan) {
            return !a_nan;
        }
        if (key[a] < key[b]) {
            return true;
        }
        if (key[a] > key[b]) {
            return false;
        }
        return ids[a] < ids[b];
    });

    il.reserve(list_no, n);
    const size_t chunk = std::min(n, kAddChunk);
    const size_t code_size = index.rabitq.code_size;
    std::vector<float> xbuf(chunk * d);
    std::vector<uint8_t> codes(chunk * code_size);
    std::vector<uint8_t> mem_codes(chunk * il.mem_dim / 8);
    std::vector<float> mem_factors(chunk * Lists::kMemFactors);
    std::vector<uint8_t> records(chunk * il.record_size);

    for (size_t i0 = 0; i0 < n; i0 += chunk) {
        const size_t i1 = std::min(n, i0 + chunk);
        for (size_t i = i0; i < i1; i++) {
            memcpy(xbuf.data() + (i - i0) * d,
                   x + members[order[i]] * d,
                   sizeof(float) * d);
        }
        index.rabitq.compute_codes_core(
                xbuf.data(), codes.data(), i1 - i0, centroid.data());
        for (size_t i = i0; i < i1; i++) {
            const size_t j = i - i0;
            split_code(
                    il,
                    index.metric_type,
                    xbuf.data() + j * d,
                    centroid.data(),
                    codes.data() + j * code_size,
                    ids[order[i]],
                    mem_codes.data() + j * il.mem_dim / 8,
                    mem_factors.data() + j * Lists::kMemFactors,
                    records.data() + j * il.record_size);
        }
        il.append(
                list_no,
                i1 - i0,
                mem_codes.data(),
                mem_factors.data(),
                records.data());
    }
}

/*********************************************************
 * Search
 *********************************************************/

/// The k best results so far, sorted by increasing distance.
struct TopKBuffer {
    struct Entry {
        float distance;
        idx_t id;
    };
    std::vector<Entry> data; // capacity + 1 entries
    size_t size = 0;
    size_t capacity = 0;

    void reset(size_t k) {
        capacity = k;
        size = 0;
        data.resize(k + 1);
    }

    bool is_full() const {
        return size == capacity;
    }

    float top_dist() const {
        return is_full() ? data[size - 1].distance
                         : std::numeric_limits<float>::max();
    }

    void insert(idx_t id, float dist) {
        if (dist > top_dist()) {
            return;
        }
        // before the entries with the same distance
        const size_t lo = std::lower_bound(
                                  data.begin(),
                                  data.begin() + size,
                                  dist,
                                  [](const Entry& e, float v) {
                                      return e.distance < v;
                                  }) -
                data.begin();
        memmove(&data[lo + 1], &data[lo], (size - lo) * sizeof(Entry));
        data[lo] = Entry{dist, id};
        size += size < capacity;
    }
};

/// The pruning threshold: the alpha-quantile of the results, e.g. the 90th
/// distance for k = 100 and alpha = 0.9. With alpha >= 1, or until there are
/// k results, the k-th distance.
float pruning_threshold(const TopKBuffer& results, size_t topk, float alpha) {
    if (alpha >= 1.0f || !results.is_full()) {
        return results.top_dist();
    }
    long rank = std::lround(alpha * double(topk));
    rank = std::min(std::max(rank, 1L), long(topk));
    return results.data[rank - 1].distance;
}

/// Per query: the look-up table of the in-memory scan, and sums of the query
/// components.
struct QueryLUT {
    /// packed for FastScan; with 16-bit entries, the low bytes of the
    /// table, then the high bytes (as expected by accumulate_hacc)
    std::vector<uint8_t> table;
    /// <x_u, q> over [0, mem_dim) = delta * (sum of entries) + sum_vl
    float delta = 0;
    float sum_vl = 0;
    float k1xsumq = 0; ///< -1/2 sum of q over [0, mem_dim)
    float q_sum = 0;   ///< sum of q over [0, d)

    void build(const float* q, size_t mem_dim, size_t d, bool use_hacc) {
        const size_t nsq = mem_dim / 4;
        // entry c of table t: the sum of q over the dimensions 4t..4t+3
        // whose bit is set in c
        std::vector<float> lut_float(nsq * 16);
        for (size_t t = 0; t < nsq; t++) {
            rabitq_utils::write_subset_sum_lut(
                    lut_float.data() + t * 16,
                    0.0f,
                    q[4 * t],
                    q[4 * t + 1],
                    q[4 * t + 2],
                    q[4 * t + 3]);
        }
        const auto [vl_it, vr_it] =
                std::minmax_element(lut_float.begin(), lut_float.end());
        const float vl = *vl_it, vr = *vr_it;
        const uint32_t max_entry = use_hacc ? 65535 : 255;
        delta = (vr - vl) / max_entry;
        if (delta == 0) { // all entries equal: they quantize to 0
            delta = 1;
        }
        const float one_over_delta = 1.0f / delta;
        std::vector<uint8_t> lo(nsq * 16), hi(use_hacc ? nsq * 16 : 0);
        for (size_t i = 0; i < nsq * 16; i++) {
            const long e = std::lrint((lut_float[i] - vl) * one_over_delta);
            const uint32_t entry =
                    uint32_t(std::min(std::max(e, 0L), long(max_entry)));
            lo[i] = entry & 0xff;
            if (use_hacc) {
                hi[i] = entry >> 8;
            }
        }
        table.resize((use_hacc ? 2 : 1) * nsq * 16);
        pq4_pack_LUT(1, nsq, lo.data(), table.data());
        if (use_hacc) {
            pq4_pack_LUT(1, nsq, hi.data(), table.data() + nsq * 16);
        }
        sum_vl = vl * float(nsq);

        float q_sum_mem = 0;
        for (size_t i = 0; i < mem_dim; i++) {
            q_sum_mem += q[i];
        }
        k1xsumq = -0.5f * q_sum_mem;
        q_sum = q_sum_mem;
        for (size_t i = mem_dim; i < d; i++) {
            q_sum += q[i];
        }
    }
};

/// a vector to re-rank, in a page to read
struct Candidate {
    size_t offset;   ///< in its list
    int in_page_idx; ///< record number in the page
    float mem_low_dist;
    float mem_est_dist;
    float mem_ip_x0_qr; ///< <x_u, q> over [0, mem_dim)
};

/// a page to read, with its candidates
struct CandidatePage {
    float page_lower_bound = 0; ///< smallest lower bound of its candidates
    size_t probe_idx = 0;
    size_t page_no = 0; ///< in its list
    int req_slot = -1;
    uint16_t candidate_num = 0;
};

using SlotID = uint32_t;
constexpr SlotID kSlotMax = std::numeric_limits<SlotID>::max();

/// The pages waiting to be read, sorted by lower bound.
struct PendingPages {
    struct Entry {
        SlotID slot_id;
        float key;
    };
    std::vector<Entry> entries;
    size_t head = 0;

    void reset() {
        entries.clear();
        head = 0;
    }

    bool empty() const {
        return size() == 0;
    }

    size_t size() const {
        return entries.size() - head;
    }

    void insert(SlotID slot_id, float key) {
        compact_if_needed();
        const auto it = std::lower_bound(
                entries.begin() + head,
                entries.end(),
                key,
                [](const Entry& e, float value) { return e.key < value; });
        entries.insert(it, Entry{slot_id, key});
    }

    SlotID front_slot() const {
        return entries[head].slot_id;
    }

    void pop_front() {
        ++head;
        compact_if_needed();
    }

    /// remove the entries with a key above threshold, appending them to pruned
    void prune_greater(float threshold, std::vector<SlotID>& pruned) {
        const auto it = std::upper_bound(
                entries.begin() + head,
                entries.end(),
                threshold,
                [](float value, const Entry& e) { return value < e.key; });
        for (auto cur = it; cur != entries.end(); ++cur) {
            pruned.push_back(cur->slot_id);
        }
        entries.erase(it, entries.end());
        if (head == entries.size()) {
            entries.clear();
            head = 0;
            return;
        }
        compact_if_needed();
    }

    void compact_if_needed() {
        if (head >= 256 && head * 2 >= entries.size()) {
            entries.erase(entries.begin(), entries.begin() + head);
            head = 0;
        }
    }
};

/// Scan position in the list being scanned. A page can span several batches.
struct ListScanState {
    int64_t last_page = -1;
    SlotID cur_page_slot_id = kSlotMax;
    int64_t cur_inner_pid = -1;
    size_t cur_page = 0;
    int in_page_idx = 0;
    int remaining_in_page = 0;
    const uint8_t* mem_data = nullptr;
};

/// Per-thread search state. Each query is searched by
/// search_one().
template <SIMDLevel SL>
struct RaBitQSSDSearcher {
    static constexpr size_t kMaxReadsInFlight =
            IndexIVFRaBitQSSD::kMaxReadsInFlight;
    static constexpr size_t kInitialPageSlots = kMaxReadsInFlight + 1;
    static constexpr uint8_t kStateFree = 0;
    static constexpr uint8_t kStateReserved = 1;
    static constexpr uint8_t kStatePending = 2;
    static constexpr uint8_t kStateInflight = 3;
    static constexpr uint8_t kStateCompleted = 4;

    const IndexIVFRaBitQSSD& index;
    const Lists& il;
    const size_t d, mem_dim, ex_bits;
    const bool is_ip;        // metric of the index
    const bool coarse_is_ip; // metric of the coarse distances
    const bool use_hacc;
    const float drain_alpha;
    const size_t qd_cap;
    const IDSelector* sel;
    const bool store_pairs;
    const int fd;
    const float cb; // -(2^ex_bits - 1/2)
    const size_t suffix_bytes;

    // query
    const float* q = nullptr;
    size_t topk = 0;
    QueryLUT lut;
    std::vector<float> g_adds, g_errors; // per probe
    std::vector<idx_t> probe_lists;      // per probe
    TopKBuffer results;

    // estimates of one batch
    int32_t accu_arr[kBatchSize];
    uint16_t accu16[kBatchSize];
    float est_distance[kBatchSize];
    float low_distance[kBatchSize];
    float ip_x0_qr[kBatchSize];

    // pages to read, with their candidates
    size_t page_candidate_capacity;
    std::vector<CandidatePage> page_slots;
    std::vector<Candidate> candidate_storage;
    std::vector<SlotID> free_page_slots;

    // read requests, one buffer of page_size bytes each
    struct ReadRequest {
        size_t offset = 0;
        size_t len = 0;
        uint8_t* buf = nullptr;
    };
    AlignedTableTightAlloc<uint8_t, 4096> read_buffers;
    std::array<ReadRequest, kMaxReadsInFlight> requests;
    std::array<SlotID, kMaxReadsInFlight> req_slot_owner;
    std::vector<int> free_req_slots;
    std::vector<SlotID> completed_slots;

    // I/O queue
    PendingPages pending;
    std::vector<uint8_t> slot_state;
    std::vector<int> slot_req_slot;
    size_t inflight_count = 0;
    std::vector<SlotID> pruned_slots;
    std::array<void*, kMaxReadsInFlight> io_ud;
    std::array<long, kMaxReadsInFlight> io_res;
    std::unique_ptr<DiskIOBackend> io;

    // statistics
    size_t n_lists = 0, n_codes = 0, n_knn_inserts = 0;
    IndexIVFRaBitQSSDStats disk_stats;

    RaBitQSSDSearcher(
            const IndexIVFRaBitQSSD& index_in,
            const Lists& il_in,
            bool use_hacc_in,
            float drain_alpha_in,
            size_t qd_in,
            const IDSelector* sel_in,
            bool store_pairs_in)
            : index(index_in),
              il(il_in),
              d(il_in.d),
              mem_dim(il_in.mem_dim),
              ex_bits(il_in.ex_bits),
              is_ip(index_in.metric_type == METRIC_INNER_PRODUCT),
              coarse_is_ip(
                      index_in.quantizer->metric_type == METRIC_INNER_PRODUCT),
              use_hacc(use_hacc_in),
              drain_alpha(drain_alpha_in),
              qd_cap(qd_in),
              sel(sel_in),
              store_pairs(store_pairs_in),
              fd(il_in.file_descriptor()),
              cb(-(float(1 << il_in.ex_bits) - 0.5f)),
              suffix_bytes((il_in.d - il_in.mem_dim) / 8),
              page_candidate_capacity(il_in.records_per_page),
              read_buffers(kMaxReadsInFlight * il_in.page_size),
              io(il_in.acquire_io(int(kMaxReadsInFlight))) {
        reserve_page_slots(kInitialPageSlots);
        reset();
    }

    ~RaBitQSSDSearcher() {
        // after an exception, reads can still be in flight into
        // read_buffers: wait for them before the buffers are freed, and
        // give the context back for the next searches. If waiting fails, the
        // destructor of the context waits instead.
        if (io && io->wait_all()) {
            il.release_io(std::move(io));
        }
        io.reset();
    }

    /*********** page slots and read requests ***********/

    void reserve_page_slots(size_t target_slots) {
        if (page_slots.size() >= target_slots) {
            return;
        }
        const size_t old_size = page_slots.size();
        page_slots.resize(target_slots);
        candidate_storage.resize(target_slots * page_candidate_capacity);
        for (size_t i = target_slots; i > old_size; --i) {
            free_page_slots.push_back(SlotID(i - 1));
        }
        slot_state.resize(target_slots, kStateFree);
        slot_req_slot.resize(target_slots, -1);
    }

    void reset() {
        free_page_slots.clear();
        for (size_t i = page_slots.size(); i > 0; --i) {
            page_slots[i - 1] = CandidatePage();
            free_page_slots.push_back(SlotID(i - 1));
        }
        free_req_slots.clear();
        completed_slots.clear();
        for (int req_slot = int(kMaxReadsInFlight); req_slot > 0; --req_slot) {
            requests[req_slot - 1] = ReadRequest();
            req_slot_owner[req_slot - 1] = kSlotMax;
            free_req_slots.push_back(req_slot - 1);
        }
        pending.reset();
        inflight_count = 0;
        std::fill(slot_state.begin(), slot_state.end(), kStateFree);
        std::fill(slot_req_slot.begin(), slot_req_slot.end(), -1);
    }

    uint8_t* read_buffer(int req_slot) {
        return read_buffers.get() + size_t(req_slot) * il.page_size;
    }

    Candidate* page_candidates(SlotID slot_id) {
        return candidate_storage.data() +
                size_t(slot_id) * page_candidate_capacity;
    }

    /*********** pending pages and reads ***********/

    bool reads_done() const {
        return pending.empty() && inflight_count == 0;
    }

    SlotID acquire_page_slot() {
        if (free_page_slots.empty()) {
            const size_t old_size = page_slots.size();
            reserve_page_slots(
                    old_size + std::max(old_size, kInitialPageSlots));
        }
        const SlotID slot_id = free_page_slots.back();
        free_page_slots.pop_back();
        page_slots[slot_id] = CandidatePage();
        slot_state[slot_id] = kStateReserved;
        return slot_id;
    }

    void release_page_slot(SlotID slot_id) {
        page_slots[slot_id] = CandidatePage();
        slot_state[slot_id] = kStateFree;
        slot_req_slot[slot_id] = -1;
        free_page_slots.push_back(slot_id);
    }

    void prune_pending(float threshold) {
        if (pending.empty()) {
            return;
        }
        pruned_slots.clear();
        pending.prune_greater(threshold, pruned_slots);
        for (SlotID slot_id : pruned_slots) {
            release_page_slot(slot_id);
        }
    }

    void enqueue_page(SlotID slot_id) {
        slot_state[slot_id] = kStatePending;
        pending.insert(slot_id, page_slots[slot_id].page_lower_bound);
    }

    void prepare_read(SlotID slot_id, int req_slot) {
        CandidatePage& cand = page_slots[slot_id];
        requests[req_slot] = ReadRequest{
                il.page_offset(probe_lists[cand.probe_idx], cand.page_no),
                il.page_size,
                read_buffer(req_slot)};
        req_slot_owner[req_slot] = slot_id;
        cand.req_slot = req_slot;
        slot_req_slot[slot_id] = req_slot;
    }

    /// submit up to budget pending pages, smallest lower bound first, after
    /// dropping the pending pages whose lower bound exceeds threshold
    void submit_reads(size_t budget, float threshold) {
        if (budget == 0 || pending.empty()) {
            return;
        }
        prune_pending(threshold);
        int submitted = 0;
        while (budget > 0 && !pending.empty() && !free_req_slots.empty()) {
            const SlotID slot_id = pending.front_slot();
            pending.pop_front();
            const int req_slot = free_req_slots.back();
            free_req_slots.pop_back();
            prepare_read(slot_id, req_slot);
            ReadRequest& req = requests[req_slot];
            io->stage(fd, req.buf, req.len, req.offset, &req);
            slot_state[slot_id] = kStateInflight;
            inflight_count++;
            submitted++;
            disk_stats.n_pages_read++;
            --budget;
        }
        if (submitted > 0) {
            io->submit();
        }
    }

    /// collect the completed reads into completed_slots, without waiting
    void poll_reads() {
        completed_slots.clear();
        if (inflight_count == 0) {
            return;
        }
        const int ret =
                io->reap(io_ud.data(), io_res.data(), int(kMaxReadsInFlight));
        for (int i = 0; i < ret; ++i) {
            // re-ranking the stale bytes of a failed or short read would
            // silently return wrong results
            FAISS_THROW_IF_NOT_FMT(
                    io_res[i] >= 0,
                    "read of %s failed: %s",
                    il.filename.c_str(),
                    strerror(-io_res[i]));
            const ReadRequest* req = static_cast<const ReadRequest*>(io_ud[i]);
            FAISS_THROW_IF_NOT_FMT(
                    size_t(io_res[i]) == req->len,
                    "short read of %s at offset %zu: %ld of %zu bytes",
                    il.filename.c_str(),
                    req->offset,
                    io_res[i],
                    req->len);
            const int req_slot = int(req - requests.data());
            const SlotID slot_id = req_slot_owner[req_slot];
            if (slot_id == kSlotMax || slot_state[slot_id] != kStateInflight) {
                continue;
            }
            slot_state[slot_id] = kStateCompleted;
            inflight_count--;
            completed_slots.push_back(slot_id);
        }
    }

    void release_read(SlotID slot_id) {
        const int req_slot = slot_req_slot[slot_id];
        if (req_slot != -1) {
            requests[req_slot] = ReadRequest();
            req_slot_owner[req_slot] = kSlotMax;
            free_req_slots.push_back(req_slot);
        }
        release_page_slot(slot_id);
    }

    /*********** scan and re-rank ***********/

    /// FastScan over the in-memory bits of one batch of a list
    void estimate_batch(const uint8_t* batch, size_t probe_idx) {
        const size_t nsq = mem_dim / 4;
        if (use_hacc) {
            // Reuse the search's SIMD level, with compile-time fallback.
            dispatch_with_fallback<FAST_SCAN_HACC_SIMD_LEVELS, SL>(
                    [&]<SIMDLevel AccSL>() {
                        accumulate_hacc_impl<AccSL>(
                                1,
                                kBatchSize,
                                nsq,
                                batch,
                                lut.table.data(),
                                accu_arr);
                    });
        } else {
            // a segment of D dimensions sums D / 4 entries of at most 255
            // into a uint16, so (D / 4) * 255 must stay below 2^16
            constexpr size_t kNoHaccSegmentSq = 256;
            std::fill(accu_arr, accu_arr + kBatchSize, 0);
            dispatch_with_fallback<AVAILABLE_SIMD_LEVELS_BASE, SL>(
                    [&]<SIMDLevel AccSL>() {
                        for (size_t sq0 = 0; sq0 < nsq;
                             sq0 += kNoHaccSegmentSq) {
                            const size_t nsq_s =
                                    std::min(kNoHaccSegmentSq, nsq - sq0);
                            accumulate_to_mem_impl<AccSL>(
                                    1,
                                    kBatchSize,
                                    int(nsq_s),
                                    batch + 16 * sq0,
                                    lut.table.data() + 16 * sq0,
                                    accu16);
                            for (size_t i = 0; i < kBatchSize; ++i) {
                                accu_arr[i] += accu16[i];
                            }
                        }
                    });
        }

        const float* f_add = reinterpret_cast<const float*>(
                batch + kBatchSize * mem_dim / 8);
        const float* f_rescale = f_add + kBatchSize;
        const float* f_error = f_rescale + kBatchSize;
        const float g_add = g_adds[probe_idx];
        const float g_error = g_errors[probe_idx];
        for (size_t i = 0; i < kBatchSize; ++i) {
            ip_x0_qr[i] = lut.delta * float(accu_arr[i]) + lut.sum_vl;
            est_distance[i] = f_add[i] + g_add +
                    f_rescale[i] * (ip_x0_qr[i] + lut.k1xsumq);
            low_distance[i] = est_distance[i] - f_error[i] * g_error;
        }
    }

    /// scan batch iter of a list of n vectors: the vectors whose lower bound
    /// does not exceed threshold become candidates of their page, and each page
    /// is queued once the scan has moved past it
    void scan_batch(
            size_t n,
            size_t probe_idx,
            float threshold,
            ListScanState& state,
            size_t iter) {
        const size_t batch_size = std::min(kBatchSize, n - iter * kBatchSize);
        estimate_batch(state.mem_data, probe_idx);

        for (size_t idx = 0; idx < batch_size; ++idx) {
            ++state.cur_inner_pid;
            if (state.remaining_in_page == 0) {
                state.cur_page += 1;
                state.remaining_in_page = int(il.records_per_page);
                state.in_page_idx = 0;
            }
            const size_t cur_page_id = state.cur_page;
            const int cur_in_page_idx = state.in_page_idx;
            --state.remaining_in_page;
            ++state.in_page_idx;

            if (low_distance[idx] > threshold) {
                continue;
            }

            if (int64_t(cur_page_id) != state.last_page) {
                if (state.last_page != -1) {
                    enqueue_page(state.cur_page_slot_id);
                }
                state.last_page = cur_page_id;
                state.cur_page_slot_id = acquire_page_slot();
                CandidatePage& page_cand = page_slots[state.cur_page_slot_id];
                page_cand.probe_idx = probe_idx;
                page_cand.page_no = cur_page_id;
                page_cand.candidate_num = 0;
                page_cand.page_lower_bound = low_distance[idx];
            }

            CandidatePage& page_cand = page_slots[state.cur_page_slot_id];
            Candidate& cand = page_candidates(
                    state.cur_page_slot_id)[page_cand.candidate_num++];
            cand.offset = size_t(state.cur_inner_pid);
            cand.in_page_idx = cur_in_page_idx;
            cand.mem_low_dist = low_distance[idx];
            cand.mem_est_dist = est_distance[idx];
            cand.mem_ip_x0_qr = ip_x0_qr[idx];
            page_cand.page_lower_bound =
                    std::min(page_cand.page_lower_bound, low_distance[idx]);
        }
        state.mem_data += il.batch_bytes;
    }

    /// queue the last page of a list
    void finish_list(ListScanState& state) {
        if (state.last_page != -1) {
            enqueue_page(state.cur_page_slot_id);
        }
    }

    /// re-rank the candidates of a page whose read completed
    void rerank_page(SlotID slot_id) {
        const CandidatePage& page_cand = page_slots[slot_id];
        const uint8_t* data = requests[slot_req_slot[slot_id]].buf;
        const size_t probe_idx = page_cand.probe_idx;
        const idx_t list_no = probe_lists[probe_idx];
        const float g_add = g_adds[probe_idx];
        const Candidate* candidates = page_candidates(slot_id);

        float threshold = results.top_dist();
        for (size_t cidx = 0; cidx < page_cand.candidate_num; ++cidx) {
            const Candidate& cand = candidates[cidx];
            if (cand.mem_low_dist > threshold) {
                continue;
            }
            const uint8_t* rec = data + cand.in_page_idx * il.record_size;
            idx_t id;
            memcpy(&id, rec + il.record_size - sizeof(idx_t), sizeof(id));
            if (sel && !sel->is_member(id)) {
                continue;
            }

            float sign_ip = cand.mem_ip_x0_qr;
            if (d > mem_dim) {
                sign_ip += rabitq::sign_inner_product<SL>(
                        rec, q + mem_dim, d - mem_dim);
            }
            const float ip =
                    rabitq::multibit::compute_inner_product_with_sign_ip<SL>(
                            rec + suffix_bytes,
                            q,
                            d,
                            ex_bits,
                            cb,
                            sign_ip,
                            lut.q_sum);
            float ex_factors[2]; // f_add_ex, f_rescale_ex
            memcpy(ex_factors,
                   rec + suffix_bytes + d * ex_bits / 8,
                   sizeof(ex_factors));
            const float full_dist = ex_factors[0] + g_add + ex_factors[1] * ip;
            disk_stats.n_reranked++;

            if (full_dist > threshold) {
                continue;
            }
            const idx_t label =
                    store_pairs ? idx_t(lo_build(list_no, cand.offset)) : id;
            results.insert(label, full_dist);
            n_knn_inserts++;
            threshold = results.top_dist();
        }
        release_read(slot_id);
    }

    /// re-rank the pages read since the last call
    void rerank_read_pages() {
        poll_reads();
        for (SlotID slot_id : completed_slots) {
            rerank_page(slot_id);
        }
    }

    /// <q, c> and |q - c|^2 from the coarse distance of list_no
    void set_list_terms(
            size_t probe_idx,
            idx_t list_no,
            float coarse_dis,
            float q_norm2) {
        float qc_dist2, qc_ip = 0;
        if (!is_ip && !coarse_is_ip) {
            qc_dist2 = coarse_dis;
        } else {
            const float c_norm2 = index.centroid_norms[list_no];
            if (coarse_is_ip) {
                qc_ip = coarse_dis;
                qc_dist2 = q_norm2 + c_norm2 - 2 * coarse_dis;
            } else {
                qc_dist2 = coarse_dis;
                qc_ip = 0.5f * (q_norm2 + c_norm2 - coarse_dis);
            }
        }
        qc_dist2 = std::max(qc_dist2, 0.0f);
        g_adds[probe_idx] = is_ip ? -qc_ip : qc_dist2;
        g_errors[probe_idx] = std::sqrt(qc_dist2);
    }

    /// search one query
    void search_one(
            const float* query,
            idx_t k,
            size_t nprobe,
            const idx_t* assign,
            const float* coarse_dis,
            float* distances,
            idx_t* labels) {
        reset();
        q = query;
        topk = k;
        results.reset(k);
        lut.build(q, mem_dim, d, use_hacc);
        const float q_norm2 =
                (is_ip || coarse_is_ip) ? fvec_norm_L2sqr(q, d) : 0.0f;
        g_adds.resize(nprobe);
        g_errors.resize(nprobe);
        probe_lists.assign(assign, assign + nprobe);

        for (size_t i = 0; i < nprobe; ++i) {
            const idx_t list_no = assign[i];
            if (list_no < 0) {
                continue;
            }
            FAISS_THROW_IF_NOT_FMT(
                    list_no < idx_t(index.nlist),
                    "invalid list number %" PRId64,
                    list_no);
            const size_t n = il.list_size(list_no);
            if (n == 0) {
                continue;
            }
            n_lists++;
            n_codes += n;
            set_list_terms(i, list_no, coarse_dis[i], q_norm2);

            // for each batch: scan it, re-rank the pages read so far and
            // submit the best pending pages. The search over the lists does
            // not prune beyond the k-th distance (alpha = 1).
            const size_t n_iter = (n + kBatchSize - 1) / kBatchSize;
            ListScanState state;
            state.remaining_in_page = int(il.records_per_page);
            state.mem_data = il.mem_data[list_no].get();
            for (size_t iter = 0; iter < n_iter; ++iter) {
                scan_batch(
                        n,
                        i,
                        pruning_threshold(results, topk, 1.0f),
                        state,
                        iter);
                rerank_read_pages();
                if (inflight_count < qd_cap) {
                    submit_reads(
                            qd_cap - inflight_count,
                            pruning_threshold(results, topk, 1.0f));
                }
            }
            finish_list(state);
        }

        // drain: read the remaining pages, pruning with the
        // drain_alpha-quantile of the results
        while (!reads_done()) {
            rerank_read_pages();
            if (inflight_count < qd_cap) {
                submit_reads(
                        qd_cap - inflight_count,
                        pruning_threshold(results, topk, drain_alpha));
            }
        }

        for (size_t j = 0; j < size_t(k); j++) {
            if (j < results.size) {
                labels[j] = results.data[j].id;
                // the search uses -<q, o> for the inner product
                distances[j] = is_ip ? -results.data[j].distance
                                     : results.data[j].distance;
            } else {
                labels[j] = -1;
                distances[j] = is_ip ? std::numeric_limits<float>::lowest()
                                     : std::numeric_limits<float>::max();
            }
        }
    }
};

/// settings of one search, from the index and the search parameters
struct SearchSettings {
    size_t nprobe = 0;
    bool use_hacc = true;
    float drain_alpha = 1.0f;
    size_t qd = 0;
    const IDSelector* sel = nullptr;
};

SearchSettings get_search_settings(
        const IndexIVFRaBitQSSD& index,
        const IVFSearchParameters* params,
        bool store_pairs) {
    FAISS_THROW_IF_NOT_MSG(index.quantizer, "IVF quantizer must not be null");
    FAISS_THROW_IF_NOT_MSG(index.is_trained, "IVF index is not trained");
    SearchSettings s;
    s.nprobe = std::min(index.nlist, params ? params->nprobe : index.nprobe);
    FAISS_THROW_IF_NOT(s.nprobe > 0);
    FAISS_THROW_IF_NOT_MSG(
            (params ? params->max_codes : index.max_codes) == 0,
            "IndexIVFRaBitQSSD: max_codes not supported");
    s.sel = params ? params->sel : nullptr;
    FAISS_THROW_IF_NOT_MSG(
            !(s.sel && store_pairs),
            "selector and store_pairs cannot be combined");
    s.use_hacc = index.use_hacc;
    s.drain_alpha = index.drain_alpha;
    s.qd = index.qd_per_thread;
    if (auto p = dynamic_cast<const IVFRaBitQSSDSearchParameters*>(params)) {
        s.use_hacc = p->use_hacc;
        s.drain_alpha = p->drain_alpha;
        s.qd = p->qd_per_thread;
    }
    FAISS_THROW_IF_NOT_FMT(
            s.drain_alpha >= 0 && s.drain_alpha <= 1,
            "drain_alpha=%g must be in [0, 1]",
            s.drain_alpha);
    FAISS_THROW_IF_NOT_FMT(
            s.qd >= 1 && s.qd <= IndexIVFRaBitQSSD::kMaxReadsInFlight,
            "qd_per_thread=%zu must be in [1, %zu]",
            s.qd,
            IndexIVFRaBitQSSD::kMaxReadsInFlight);
    FAISS_THROW_IF_NOT_MSG(
            !(index.parallel_mode & index.PARALLEL_MODE_NO_HEAP_INIT),
            "IndexIVFRaBitQSSD: PARALLEL_MODE_NO_HEAP_INIT not supported");
    const MetricType coarse_metric = index.quantizer->metric_type;
    FAISS_THROW_IF_NOT_MSG(
            coarse_metric == METRIC_L2 || coarse_metric == METRIC_INNER_PRODUCT,
            "IndexIVFRaBitQSSD: the coarse quantizer must use METRIC_L2 or "
            "METRIC_INNER_PRODUCT");
    // the norms are needed to convert the coarse distances, except for L2
    // with an L2 quantizer
    FAISS_THROW_IF_NOT_MSG(
            (index.metric_type == METRIC_L2 && coarse_metric == METRIC_L2) ||
                    index.centroid_norms.size() == index.nlist ||
                    index.ntotal == 0,
            "IndexIVFRaBitQSSD: centroid norms missing, call "
            "update_centroid_norms()");
    return s;
}

/** Search n queries. Without assign and centroid_dis, the thread that
 * searches a query first runs the coarse quantizer on it, so that the coarse
 * search of one query overlaps the reads of the others. */
template <SIMDLevel SL>
void search_impl(
        const IndexIVFRaBitQSSD& index,
        const Lists& il,
        idx_t n,
        const float* x,
        idx_t k,
        const idx_t* assign,
        const float* centroid_dis,
        const SearchParameters* quantizer_params,
        float* distances,
        idx_t* labels,
        bool store_pairs,
        const SearchSettings& s,
        bool do_parallel,
        IndexIVFStats* ivf_stats) {
    if (n == 0) {
        return;
    }
    const size_t nprobe = s.nprobe;
    size_t nlistv = 0, ndis = 0, nheap = 0, npages = 0, nreranked = 0;
    double quantization_ms = 0, search_ms = 0;
    std::exception_ptr ex;
    std::atomic<bool> interrupt{false};

    auto capture = [&]() {
#pragma omp critical(faiss_ivf_rabitq_ssd_search)
        {
            if (!ex) {
                ex = std::current_exception();
            }
        }
        interrupt = true;
    };

#pragma omp parallel if (do_parallel)                         \
        reduction(+ : nlistv, ndis, nheap, npages, nreranked) \
        reduction(+ : quantization_ms, search_ms)
    {
        // every thread must reach the omp for, even if its setup fails
        std::unique_ptr<RaBitQSSDSearcher<SL>> searcher;
        std::vector<idx_t> q_assign;
        std::vector<float> q_dis;
        try {
            searcher = std::make_unique<RaBitQSSDSearcher<SL>>(
                    index,
                    il,
                    s.use_hacc,
                    s.drain_alpha,
                    s.qd,
                    s.sel,
                    store_pairs);
            if (!assign) {
                q_assign.resize(nprobe);
                q_dis.resize(nprobe);
            }
        } catch (...) {
            capture();
        }

#pragma omp for schedule(dynamic, 1)
        for (idx_t i = 0; i < n; i++) {
            if (interrupt) {
                continue;
            }
            try {
                const float* xi = x + i * index.d;
                if (assign) {
                    searcher->search_one(
                            xi,
                            k,
                            nprobe,
                            assign + i * nprobe,
                            centroid_dis + i * nprobe,
                            distances + i * k,
                            labels + i * k);
                } else {
                    const double t0 = getmillisecs();
                    index.quantizer->search(
                            1,
                            xi,
                            nprobe,
                            q_dis.data(),
                            q_assign.data(),
                            quantizer_params);
                    const double t1 = getmillisecs();
                    searcher->search_one(
                            xi,
                            k,
                            nprobe,
                            q_assign.data(),
                            q_dis.data(),
                            distances + i * k,
                            labels + i * k);
                    quantization_ms += t1 - t0;
                    search_ms += getmillisecs() - t0;
                }
            } catch (...) {
                capture();
            }
        }
        if (searcher) {
            nlistv += searcher->n_lists;
            ndis += searcher->n_codes;
            nheap += searcher->n_knn_inserts;
            npages += searcher->disk_stats.n_pages_read;
            nreranked += searcher->disk_stats.n_reranked;
        }
    }
    if (ex) {
        std::rethrow_exception(ex);
    }

    if (ivf_stats) {
        ivf_stats->nq += n;
        ivf_stats->nlist += nlistv;
        ivf_stats->ndis += ndis;
        ivf_stats->nheap_updates += nheap;
        ivf_stats->quantization_time += quantization_ms;
        ivf_stats->search_time += search_ms;
    }
    detail::atomic_fetch_add_relaxed(
            indexIVFRaBitQSSD_stats.n_pages_read, npages);
    detail::atomic_fetch_add_relaxed(
            indexIVFRaBitQSSD_stats.n_reranked, nreranked);
}

} // namespace

/*********************************************************
 * IndexIVFRaBitQSSD
 *********************************************************/

IndexIVFRaBitQSSD::IndexIVFRaBitQSSD(
        Index* quantizer_in,
        size_t d_in,
        size_t nlist_in,
        size_t mem_dim_in,
        uint8_t nb_bits,
        const char* filename,
        MetricType metric)
        : IndexIVF(quantizer_in, d_in, nlist_in, 0, metric, false),
          rabitq(d_in, metric, nb_bits),
          mem_dim(mem_dim_in) {
    FAISS_THROW_IF_NOT_FMT(
            d_in > 0 && d_in % 64 == 0 && d_in < (size_t(1) << 31),
            "d=%zu must be a positive multiple of 64",
            d_in);
    FAISS_THROW_IF_NOT_FMT(
            mem_dim >= 64 && mem_dim <= d_in && mem_dim % 64 == 0,
            "mem_dim=%zu must be a multiple of 64 in [64, d=%zu]",
            mem_dim,
            d_in);
    FAISS_THROW_IF_NOT_MSG(
            metric == METRIC_L2 || metric == METRIC_INNER_PRODUCT,
            "IndexIVFRaBitQSSD supports METRIC_L2 and "
            "METRIC_INNER_PRODUCT only");
    // with nb_bits = 1, either the file holds no code, or the in-memory
    // estimate has no extra bits to be refined with
    FAISS_THROW_IF_NOT_FMT(
            nb_bits >= 2 && nb_bits <= 9,
            "IndexIVFRaBitQSSD: nb_bits=%d must be in [2, 9]",
            int(nb_bits));
    code_size = rabitq.code_size;
    by_residual = true;
    if (filename) {
        replace_invlists(
                new RaBitQSSDInvertedLists(
                        nlist, d, mem_dim, rabitq.nb_bits - 1, filename),
                true);
    }
}

IndexIVFRaBitQSSD::IndexIVFRaBitQSSD() {
    by_residual = true;
}

RaBitQSSDInvertedLists* IndexIVFRaBitQSSD::get_ssd_invlists() const {
    auto* il = dynamic_cast<RaBitQSSDInvertedLists*>(invlists);
    FAISS_THROW_IF_NOT_MSG(
            il, "IndexIVFRaBitQSSD needs RaBitQSSDInvertedLists");
    FAISS_THROW_IF_NOT_MSG(
            il->d == size_t(d) && il->mem_dim == mem_dim &&
                    il->ex_bits + 1 == rabitq.nb_bits,
            "IndexIVFRaBitQSSD: the inverted lists do not match the index");
    return il;
}

void IndexIVFRaBitQSSD::reset() {
    if (invlists) {
        IndexIVF::reset();
    } else {
        direct_map.clear();
        ntotal = 0;
    }
}

void IndexIVFRaBitQSSD::update_centroid_norms() {
    std::vector<float> norms(nlist);
    const size_t bs = 4096;
    std::vector<float> centroids(std::min(nlist, bs) * d);
    for (size_t i0 = 0; i0 < nlist; i0 += bs) {
        const size_t i1 = std::min(nlist, i0 + bs);
        quantizer->reconstruct_n(i0, i1 - i0, centroids.data());
        fvec_norms_L2sqr(norms.data() + i0, centroids.data(), d, i1 - i0);
    }
    centroid_norms = std::move(norms);
}

void IndexIVFRaBitQSSD::add_core(
        idx_t n,
        const float* x,
        const idx_t* xids,
        const idx_t* coarse_idx,
        void* /*inverted_list_context*/) {
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT_MSG(
            direct_map.no(), "IndexIVFRaBitQSSD does not support direct maps");
    RaBitQSSDInvertedLists* il = get_ssd_invlists();
    if (centroid_norms.size() != nlist) {
        update_centroid_norms();
    }

    // vectors of each list, in input order
    std::vector<size_t> list_begin(nlist + 1, 0);
    for (idx_t i = 0; i < n; i++) {
        const idx_t list_no = coarse_idx[i];
        if (list_no >= 0) {
            FAISS_THROW_IF_NOT_FMT(
                    list_no < idx_t(nlist),
                    "invalid list number %" PRId64,
                    list_no);
            list_begin[list_no + 1]++;
        }
    }
    std::vector<idx_t> lists;
    for (size_t l = 0; l < nlist; l++) {
        if (list_begin[l + 1] > 0) {
            lists.push_back(l);
        }
        list_begin[l + 1] += list_begin[l];
    }
    std::vector<idx_t> members(list_begin[nlist]);
    {
        std::vector<size_t> pos(list_begin.begin(), list_begin.end() - 1);
        for (idx_t i = 0; i < n; i++) {
            if (coarse_idx[i] >= 0) {
                members[pos[coarse_idx[i]]++] = i;
            }
        }
    }

    std::exception_ptr ex;
    std::atomic<bool> interrupt{false};
#pragma omp parallel for schedule(dynamic)
    for (idx_t j = 0; j < idx_t(lists.size()); j++) {
        if (interrupt) {
            continue;
        }
        const idx_t list_no = lists[j];
        try {
            add_to_list(
                    *this,
                    *il,
                    list_no,
                    members.data() + list_begin[list_no],
                    list_begin[list_no + 1] - list_begin[list_no],
                    x,
                    xids);
        } catch (...) {
#pragma omp critical(faiss_ivf_rabitq_ssd_add)
            {
                if (!ex) {
                    ex = std::current_exception();
                }
            }
            interrupt = true;
        }
    }
    if (ex) {
        std::rethrow_exception(ex);
    }
    ntotal += n;
}

void IndexIVFRaBitQSSD::encode_vectors(
        idx_t,
        const float*,
        const idx_t*,
        uint8_t*,
        bool) const {
    FAISS_THROW_MSG("IndexIVFRaBitQSSD: encode_vectors not supported");
}

void IndexIVFRaBitQSSD::search_preassigned(
        idx_t n,
        const float* x,
        idx_t k,
        const idx_t* assign,
        const float* centroid_dis,
        float* distances,
        idx_t* labels,
        bool store_pairs,
        const IVFSearchParameters* params,
        IndexIVFStats* stats) const {
    FAISS_THROW_IF_NOT(k > 0);
    const SearchSettings s = get_search_settings(*this, params, store_pairs);
    const RaBitQSSDInvertedLists* il = get_ssd_invlists();

    // queries are processed in parallel, unless the caller already does it
    // (parallel_mode 0, as in IndexIVF::search)
    const bool do_parallel = omp_get_max_threads() >= 2 && n > 1 &&
            (parallel_mode & ~PARALLEL_MODE_NO_HEAP_INIT) != 0;

    with_simd_level_with_vpopcnt([&]<SIMDLevel SL>() {
        search_impl<SL>(
                *this,
                *il,
                n,
                x,
                k,
                assign,
                centroid_dis,
                nullptr,
                distances,
                labels,
                store_pairs,
                s,
                do_parallel,
                stats);
    });
}

void IndexIVFRaBitQSSD::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params_in) const {
    FAISS_THROW_IF_NOT(k > 0);
    const IVFSearchParameters* params = nullptr;
    if (params_in) {
        params = dynamic_cast<const IVFSearchParameters*>(params_in);
        FAISS_THROW_IF_NOT_MSG(params, "IndexIVF params have incorrect type");
    }
    const SearchSettings s = get_search_settings(*this, params, false);
    const RaBitQSSDInvertedLists* il = get_ssd_invlists();

    // IndexIVF::search runs the coarse quantizer on all the queries before
    // searching the lists, which leaves the SSD idle meanwhile. Here each
    // thread runs it on its own query, while the other threads read.
    const bool do_parallel = omp_get_max_threads() >= 2 && n > 1;

    with_simd_level_with_vpopcnt([&]<SIMDLevel SL>() {
        search_impl<SL>(
                *this,
                *il,
                n,
                x,
                k,
                nullptr,
                nullptr,
                params ? params->quantizer_params : nullptr,
                distances,
                labels,
                false,
                s,
                do_parallel,
                &indexIVF_stats);
    });
}

void IndexIVFRaBitQSSD::range_search_preassigned(
        idx_t,
        const float*,
        float,
        const idx_t*,
        const float*,
        RangeSearchResult*,
        bool,
        const IVFSearchParameters*,
        IndexIVFStats*) const {
    FAISS_THROW_MSG("IndexIVFRaBitQSSD: range search not supported");
}

void IndexIVFRaBitQSSD::reconstruct(idx_t, float*) const {
    // IndexIVF::reconstruct would first ask for a direct map
    FAISS_THROW_MSG("IndexIVFRaBitQSSD: reconstruct not supported");
}

void IndexIVFRaBitQSSD::reconstruct_from_offset(int64_t, int64_t, float*)
        const {
    FAISS_THROW_MSG("IndexIVFRaBitQSSD: reconstruct not supported");
}

size_t IndexIVFRaBitQSSD::remove_ids(const IDSelector&) {
    FAISS_THROW_MSG("IndexIVFRaBitQSSD: remove_ids not supported");
}

void IndexIVFRaBitQSSD::update_vectors(int, const idx_t*, const float*) {
    FAISS_THROW_MSG("IndexIVFRaBitQSSD: update_vectors not supported");
}

void IndexIVFRaBitQSSD::merge_from(Index&, idx_t) {
    FAISS_THROW_MSG("IndexIVFRaBitQSSD: merge_from not supported");
}

} // namespace faiss
