/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/DeterministicNNDescent.h>

#include <omp.h>
#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <limits>
#include <memory>
#include <numeric>
#include <typeinfo>

#include <atomic>

#include <faiss/IndexFlat.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/nndescent/gram.h>
#include <faiss/impl/pipnn/kernels.h>
#include <faiss/impl/simd_dispatch.h>
#include <faiss/utils/AlignedTable.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/fp16.h>
#include <faiss/utils/prefetch.h>
#include <faiss/utils/random.h>
#include <faiss/utils/utils.h>

extern "C" {

#ifndef FINTEGER
#define FINTEGER long
#endif

int sgemm_(
        const char* transa,
        const char* transb,
        FINTEGER* m,
        FINTEGER* n,
        FINTEGER* k,
        const float* alpha,
        const float* a,
        FINTEGER* lda,
        const float* b,
        FINTEGER* ldb,
        float* beta,
        float* c,
        FINTEGER* ldc);

int ssyrk_(
        const char* uplo,
        const char* trans,
        FINTEGER* n,
        FINTEGER* k,
        float* alpha,
        float* a,
        FINTEGER* lda,
        float* beta,
        float* c,
        FINTEGER* ldc);
}

namespace faiss {

namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();

template <class Id>
struct DisIdT {
    float dis;
    Id id;
};
using DisId = DisIdT<int32_t>;
using DisId64 = DisIdT<int64_t>;

template <class Id>
inline bool dis_id_less_t(const DisIdT<Id>& a, const DisIdT<Id>& b) {
    return a.dis < b.dis || (a.dis == b.dis && a.id < b.id);
}

inline bool dis_id_less(const DisId& a, const DisId& b) {
    return dis_id_less_t(a, b);
}

/// a hash of x: the first output of SplitMix64 seeded with x
uint64_t mix64(uint64_t x) {
    return SplitMix64RandomGenerator(int64_t(x)).next();
}

/// open-addressing map from ids to 0, 1, 2, ... in insertion order;
/// cleared in O(size)
struct IdMap {
    std::vector<int32_t> keys; // -1: empty
    std::vector<int32_t> vals;
    std::vector<uint32_t> used;
    uint32_t mask = 0;

    IdMap() {
        rehash(1024);
    }

    void rehash(size_t size) {
        std::vector<int32_t> ok = std::move(keys), ov = std::move(vals);
        std::vector<uint32_t> ou = std::move(used);
        keys.assign(size, -1);
        vals.assign(size, 0);
        used.clear();
        mask = uint32_t(size - 1);
        for (uint32_t h : ou) {
            insert_new(ok[h], ov[h]);
        }
    }

    void clear() {
        for (uint32_t h : used) {
            keys[h] = -1;
        }
        used.clear();
    }

    uint32_t slot(int32_t id) const {
        // high bits of a multiplicative hash: ids that share low bits spread
        uint32_t h = uint32_t(
                             (uint64_t(uint32_t(id)) * 0x9e3779b97f4a7c15ULL) >>
                             32) &
                mask;
        while (keys[h] >= 0 && keys[h] != id) {
            h = (h + 1) & mask;
        }
        return h;
    }

    void insert_new(int32_t id, int32_t val) {
        const uint32_t h = slot(id);
        keys[h] = id;
        vals[h] = val;
        used.push_back(h);
    }

    /// value of id, inserting size() if absent; sets is_new
    int32_t find_or_insert(int32_t id, bool& is_new) {
        uint32_t h = slot(id);
        if (keys[h] >= 0) {
            is_new = false;
            return vals[h];
        }
        is_new = true;
        const int32_t v = int32_t(used.size());
        keys[h] = id;
        vals[h] = v;
        used.push_back(h);
        if (2 * used.size() > keys.size()) {
            rehash(2 * keys.size());
        }
        return v;
    }
};

/** Per-thread read access to the float vectors of the storage. IndexFlat is
 * read in place and IndexScalarQuantizer is decoded directly:
 * Index::reconstruct of IndexScalarQuantizer opens a nested OpenMP region and
 * allocates for each call. The pointer that get() returns is valid until the
 * next get(). */
struct VectorReader {
    const Index& storage;
    const size_t d;
    const float* flat = nullptr;
    const IndexScalarQuantizer* sq = nullptr;
    std::unique_ptr<ScalarQuantizer::SQuantizer> squant;
    std::vector<float> buf;

    explicit VectorReader(const Index& s) : storage(s), d(s.d), buf(s.d) {
        // exact types: subclasses such as IndexFlatPanorama change the layout
        const std::type_info& t = typeid(s);
        if (t == typeid(IndexFlat) || t == typeid(IndexFlatL2) ||
            t == typeid(IndexFlatIP)) {
            flat = static_cast<const IndexFlat&>(s).get_xb();
        } else if (auto* q = dynamic_cast<const IndexScalarQuantizer*>(&s)) {
            // throws for a quantizer without a decoder, such as QT_0bit
            sq = q;
            squant.reset(q->sq.select_quantizer());
        }
    }

    const float* get(idx_t i) {
        if (flat) {
            return flat + i * d;
        }
        decode(i, buf.data());
        return buf.data();
    }

    void decode(idx_t i, float* out) {
        if (flat) {
            std::copy_n(flat + i * d, d, out);
        } else if (squant) {
            squant->decode_vector(sq->codes.data() + i * sq->code_size, out);
        } else {
            storage.reconstruct(i, out);
        }
    }

    void prefetch(idx_t i) const {
        const uint8_t* p;
        size_t nbytes;
        if (flat) {
            p = reinterpret_cast<const uint8_t*>(flat + i * d);
            nbytes = d * sizeof(float);
        } else if (squant) {
            p = sq->codes.data() + i * sq->code_size;
            nbytes = sq->code_size;
        } else {
            return;
        }
        for (size_t o = 0; o < std::min(nbytes, size_t(4096)); o += 64) {
            prefetch_L1(p + o);
        }
    }
};

/// smaller-is-better distances: L2, or negated inner products
struct Metric {
    bool ip;

    /// one distance; NaN (overflow) becomes +inf and -0 becomes +0, as in
    /// products_to_distances()
    float one(const float* x, const float* y, size_t d) const {
        const float v = ip ? -fvec_inner_product(x, y, d) : fvec_L2sqr(x, y, d);
        return std::isnan(v) ? kInf : v + 0.0f;
    }
};

/** Rows [0, r) of the upper triangle of the Gram matrix of the m rows of x
 * into out (row-major, m columns): the entries i < r, j > i, by ssyrk
 * (new x new) and sgemm (new x old); the other entries are not set. */
void gram_upper_rows_blas(
        const float* x,
        size_t r,
        size_t m,
        size_t d,
        float* out) {
    FINTEGER ri = FINTEGER(r), di = FINTEGER(d), mi = FINTEGER(m);
    float alpha = 1, beta = 0;
    // column-major lower triangle = row-major upper triangle (j >= i)
    ssyrk_("L",
           "T",
           &ri,
           &di,
           &alpha,
           const_cast<float*>(x),
           &di,
           &beta,
           out,
           &mi);
    if (m > r) {
        FINTEGER oi = FINTEGER(m - r);
        sgemm_("T",
               "N",
               &oi,
               &ri,
               &di,
               &alpha,
               x + r * d,
               &di,
               x,
               &di,
               &beta,
               out + r,
               &mi);
    }
}

/// levels with the nndescent::gram_upper_rows kernel: the higher AVX-512
/// levels fall back to AVX512, and ARM_SVE to ARM_NEON. The others use BLAS.
constexpr int kGramLevels =
        AVAILABLE_SIMD_LEVELS_AVX2_NEON | (1 << int(SIMDLevel::AVX512));

using GramKernel = void (*)(
        const float* const* rows,
        size_t r,
        size_t m,
        size_t d,
        float* out);

/// the kernel of the SIMD level, or null for BLAS. The kernel reads the rows
/// through pointers, in place for flat storage, without the gather copy and
/// BLAS. BLAS called from an OpenMP region can open a nested region, so the
/// kernel is also the safer path.
GramKernel gram_kernel() {
    GramKernel kernel = nullptr;
    with_selected_simd_levels<kGramLevels>([&]<SIMDLevel SL>() {
        if constexpr (SL != SIMDLevel::NONE) {
            kernel = &nndescent::gram_upper_rows<SL>;
        }
    });
    return kernel;
}

/** Turns the inner products of gram_upper_rows_*() into distances, in place:
 * negated for IP; for L2 from the squared norms, clamped at 0. Scaling by -1
 * or -2 is exact. A NaN (overflow) becomes +inf as in
 * pipnn::gram_to_distance, and -0 becomes +0 so that a value has one bit
 * pattern. A pair can get a different value in another join; the rounds do
 * not depend on it (see drop_old_ids()). */
void products_to_distances(
        bool ip,
        const float* norms,
        size_t r,
        size_t m,
        float* out) {
    for (size_t i = 0; i < r; i++) {
        float* oi = out + i * m;
        for (size_t j = i + 1; j < m; j++) {
            float v = ip ? -oi[j] : norms[i] + norms[j] - 2 * oi[j];
            if (std::isnan(v)) {
                v = kInf;
            } else if (!ip) {
                v = std::max(0.0f, v);
            }
            oi[j] = v + 0.0f; // -0 + 0 = +0
        }
    }
}

/// keeps the k smallest of cand by (dis, id), in any order
void keep_best(std::vector<DisId>& cand, size_t k) {
    if (cand.size() > k) {
        std::nth_element(
                cand.begin(), cand.begin() + k, cand.end(), dis_id_less);
        cand.resize(k);
    }
}

struct Segment {
    size_t begin, end;
    uint64_t hash;
};

/// maximum points in a cluster tree leaf
constexpr size_t kLeafSize = 512;

/** Splits the points of perms[t] by random cluster tree t, for all trees at
 * once, and returns the leaves of each tree. Each level reads each vector
 * once for all trees. The pivots of a segment come from its hash, so the
 * trees do not depend on the thread count. */
std::vector<std::vector<Segment>> cluster_trees(
        const Index& storage,
        const Metric& metric,
        std::vector<std::vector<int32_t>>& perms,
        uint64_t seed) {
    const int T = int(perms.size());
    const size_t n = storage.ntotal;
    const size_t d = storage.d;
    std::vector<std::vector<Segment>> work(T), leaves(T);
    for (int t = 0; t < T; t++) {
        const Segment root = {0, n, mix64(seed + t)};
        (n <= kLeafSize ? leaves[t] : work[t]).push_back(root);
    }
    // segment of each vector in the current level (-1: in a leaf), and side
    std::vector<int32_t> segv(size_t(T) * n);
    std::vector<uint8_t> sidev(size_t(T) * n);
    std::vector<std::vector<float>> w(T), c(T);
    while (std::any_of(work.begin(), work.end(), [](const auto& wt) {
        return !wt.empty();
    })) {
        for (int t = 0; t < T; t++) {
            // side test: nearer to pivot a than to pivot b, as one dot
            // product: <x, b - a> <= (|b|^2 - |a|^2) / 2 for L2, <= 0 for IP
            const std::vector<int32_t>& perm = perms[t];
            const size_t ns = work[t].size();
            w[t].resize(ns * d);
            c[t].resize(ns);
            VectorReader vr(storage);
            std::vector<float> pa(d), pb(d);
            for (size_t s = 0; s < ns; s++) {
                const Segment& sg = work[t][s];
                const size_t size = sg.end - sg.begin;
                const uint64_t r = mix64(sg.hash);
                const size_t a = r % size;
                const size_t b = (a + 1 + mix64(r) % (size - 1)) % size;
                vr.decode(perm[sg.begin + a], pa.data());
                vr.decode(perm[sg.begin + b], pb.data());
                for (size_t k = 0; k < d; k++) {
                    w[t][s * d + k] = pb[k] - pa[k];
                }
                c[t][s] = metric.ip ? 0
                                    : (fvec_norm_L2sqr(pb.data(), d) -
                                       fvec_norm_L2sqr(pa.data(), d)) /
                                2;
            }
            int32_t* sv = segv.data() + size_t(t) * n;
            std::fill(sv, sv + n, -1);
#pragma omp parallel for schedule(dynamic)
            for (int64_t s = 0; s < int64_t(ns); s++) {
                for (size_t i = work[t][s].begin; i < work[t][s].end; i++) {
                    sv[perm[i]] = int32_t(s);
                }
            }
        }
#pragma omp parallel
        {
            VectorReader vr(storage);
#pragma omp for schedule(static)
            for (int64_t v = 0; v < int64_t(n); v++) {
                const float* x = nullptr;
                for (int t = 0; t < T; t++) {
                    const int32_t q = segv[size_t(t) * n + v];
                    if (q < 0) {
                        continue;
                    }
                    if (!x) {
                        x = vr.get(v);
                    }
                    sidev[size_t(t) * n + v] =
                            fvec_inner_product(x, w[t].data() + q * d, d) <=
                                    c[t][q]
                            ? 0
                            : 1;
                }
            }
        }
        for (int t = 0; t < T; t++) {
            std::vector<int32_t>& perm = perms[t];
            const uint8_t* side = sidev.data() + size_t(t) * n;
            const size_t ns = work[t].size();
            std::vector<Segment> children(2 * ns);
#pragma omp parallel for schedule(dynamic)
            for (int64_t s = 0; s < int64_t(ns); s++) {
                const Segment sg = work[t][s];
                const auto first = perm.begin() + sg.begin;
                size_t mid = sg.begin +
                        (std::stable_partition(
                                 first,
                                 perm.begin() + sg.end,
                                 [&](int32_t v) { return side[v] == 0; }) -
                         first);
                if (mid == sg.begin || mid == sg.end) {
                    // copies of one vector: split by position
                    mid = sg.begin + (sg.end - sg.begin) / 2;
                }
                children[2 * s] = {sg.begin, mid, mix64(sg.hash ^ 1)};
                children[2 * s + 1] = {mid, sg.end, mix64(sg.hash ^ 2)};
            }
            std::vector<Segment> next;
            for (const Segment& cs : children) {
                (cs.end - cs.begin <= kLeafSize ? leaves[t] : next)
                        .push_back(cs);
            }
            work[t].swap(next);
        }
    }
    return leaves;
}

} // namespace

namespace {

/// whether id is in the K slots of row; no early exit, so that the loop
/// vectorizes
template <class Id>
inline bool row_contains(const Id* row, int K, Id id) {
    int hit = 0;
    for (int i = 0; i < K; i++) {
        hit |= row[i] == id;
    }
    return hit != 0;
}

/** Merges the candidates c[0..m) into the sorted row of K slots. The
 * candidates are sorted by (distance, id) and have distinct ids. The row
 * keeps the K best entries by (distance, id), with the smallest distance of
 * each id. An entry from a candidate is new. The result does not depend on
 * the order of the candidates. buf holds K entries and K flags. */
template <class Id>
void merge_into_row(
        int K,
        Id* ids,
        float* dis,
        uint8_t* is_new,
        DisIdT<Id>* c,
        size_t m,
        std::vector<DisIdT<Id>>& buf,
        std::vector<uint8_t>& buf_new) {
    // a candidate with an id of the row replaces the entry if it is better,
    // else it is dropped
    size_t mk = 0;
    for (size_t j = 0; j < m; j++) {
        if (row_contains(ids, K, c[j].id)) {
            int pos = 0;
            while (ids[pos] != c[j].id) {
                pos++;
            }
            if (!dis_id_less_t<Id>(c[j], {dis[pos], ids[pos]})) {
                continue;
            }
            ids[pos] = -2; // removed
        }
        c[mk++] = c[j];
    }
    if (mk == 0) {
        return;
    }
    buf.resize(K);
    buf_new.resize(K);
    int i = 0, k = 0;
    size_t j = 0;
    while (k < K) {
        while (i < K && ids[i] == -2) {
            i++;
        }
        const bool has_row = i < K && ids[i] >= 0;
        const bool has_cand = j < mk;
        if (!has_row && !has_cand) {
            break;
        }
        if (has_cand &&
            (!has_row || dis_id_less_t<Id>(c[j], {dis[i], ids[i]}))) {
            buf[k] = c[j++];
            buf_new[k++] = 1;
        } else {
            buf[k] = {dis[i], ids[i]};
            buf_new[k++] = is_new[i++];
        }
    }
    for (int t = 0; t < k; t++) {
        ids[t] = buf[t].id;
        dis[t] = buf[t].dis;
        is_new[t] = buf_new[t];
    }
    for (int t = k; t < K; t++) {
        ids[t] = -1;
        dis[t] = kInf;
        is_new[t] = 0;
    }
}

/** Keeps at the front of cand the candidates that can enter the row: by
 * (distance, id), better than the K-th entry of a full row (its distance is
 * kth), and with an id that is not in the old row orow. Returns the number
 * kept. An id of the old row keeps
 * its old entry. If the id could leave the row and come back, the row would
 * depend on the order of the offers when one pair gets two distances. */
template <class Id>
size_t drop_old_ids(
        const Id* orow,
        float kth,
        int K,
        std::vector<DisIdT<Id>>& cand) {
    const bool full = orow[K - 1] >= 0;
    const DisIdT<Id> worst = {kth, orow[K - 1]};
    size_t m = 0;
    for (const DisIdT<Id>& c : cand) {
        if ((full && !dis_id_less_t(c, worst)) || row_contains(orow, K, c.id)) {
            continue;
        }
        cand[m++] = c;
    }
    return m;
}

struct LeafScratch {
    AlignedTable<float, 64> xl, dis;
    std::vector<float> norms;
    std::vector<const float*> rows;
};

/// all distances between the m points lp of a leaf, into ls.dis (m x m)
void leaf_distances(
        const Metric& metric,
        GramKernel kernel,
        const int32_t* lp,
        size_t m,
        VectorReader& vr,
        LeafScratch& ls) {
    const size_t d = vr.d;
    // the kernel reads flat storage in place; BLAS reads one matrix
    const bool in_place = kernel && vr.flat;
    if (!in_place) {
        pipnn::ensure_size(ls.xl, m * d);
    }
    ls.rows.resize(m);
    ls.norms.resize(m);
    for (size_t p = 0; p < m; p++) {
        if (in_place) {
            ls.rows[p] = vr.flat + size_t(lp[p]) * d;
        } else {
            vr.decode(lp[p], ls.xl.data() + p * d);
            ls.rows[p] = ls.xl.data() + p * d;
        }
        ls.norms[p] = fvec_norm_L2sqr(ls.rows[p], d);
    }
    pipnn::ensure_size(ls.dis, m * m);
    float* leaf_dis = ls.dis.data();
    if (kernel) {
        kernel(ls.rows.data(), m, m, d, leaf_dis);
        products_to_distances(metric.ip, ls.norms.data(), m, m, leaf_dis);
        for (size_t p = 0; p < m; p++) {
            for (size_t q = p + 1; q < m; q++) {
                leaf_dis[q * m + p] = leaf_dis[p * m + q];
            }
        }
        return;
    }
    pipnn::pairwise_distances(
            ls.xl.data(),
            ls.norms.data(),
            m,
            d,
            metric.ip ? METRIC_INNER_PRODUCT : METRIC_L2,
            leaf_dis);
    for (size_t e = 0; e < m * m; e++) {
        leaf_dis[e] += 0.0f; // -0 + 0 = +0, as in products_to_distances
    }
}

} // namespace

void DeterministicNNDescent::init(const Index& storage) {
    check_params();
    FAISS_THROW_IF_NOT_MSG(
            storage.metric_type == METRIC_L2 ||
                    storage.metric_type == METRIC_INNER_PRODUCT,
            "DeterministicNNDescent: only METRIC_L2 and METRIC_INNER_PRODUCT");
    FAISS_THROW_IF_NOT_MSG(
            storage.d > 0, "DeterministicNNDescent: d must be > 0");
    FAISS_THROW_IF_NOT_MSG(
            storage.ntotal < (idx_t(1) << 31),
            "DeterministicNNDescent: at most 2^31 - 1 vectors");
    n = storage.ntotal;
    d = storage.d;
    metric_type = storage.metric_type;
    const size_t nK = size_t(n) * K;
    const Metric metric{storage.metric_type == METRIC_INNER_PRODUCT};
    if (n > 0) {
        // a storage that cannot reconstruct throws here, not inside OpenMP
        VectorReader(storage).get(0);
    }
    ids.assign(nK, -1);
    dis.assign(nK, kInf);
    is_new.assign(nK, 1);
    order.resize(n);
    std::iota(order.begin(), order.end(), 0);
    ndis = 0;
    if (n <= 1) {
        return;
    }
    const double t0 = getmillisecs();

    // merges the candidates into the row of v, as round() does
    auto add_to_row = [&](idx_t v,
                          std::vector<DisId>& cand,
                          std::vector<DisId>& buf,
                          std::vector<uint8_t>& buf_new) {
        const size_t o = size_t(v) * K;
        const size_t m = drop_old_ids(ids.data() + o, dis[o + K - 1], K, cand);
        if (m == 0) {
            return;
        }
        std::sort(cand.begin(), cand.begin() + m, dis_id_less);
        merge_into_row(
                K,
                ids.data() + o,
                dis.data() + o,
                is_new.data() + o,
                cand.data(),
                m,
                buf,
                buf_new);
    };

    // 2K random ids, then the next ids, so that a row of n - 1 >= K fills
    auto random_ids = [&](idx_t v, std::vector<int32_t>& c) {
        const int want = int(std::min<idx_t>(K, n - 1));
        uint64_t r = mix64(uint64_t(seed) ^ mix64(v) ^ 0x5bd1e995);
        for (int j = 0; j < 2 * want; j++) {
            r = mix64(r);
            c.push_back(int32_t(r % n));
        }
        for (int j = 1; j <= want; j++) {
            c.push_back(int32_t((v + j) % n));
        }
    };

    size_t nd_total = 0;
    if (ntrees > 0) {
        std::vector<std::vector<int32_t>> perms(
                ntrees, std::vector<int32_t>(n));
        for (auto& perm : perms) {
            std::iota(perm.begin(), perm.end(), 0);
        }
        const std::vector<std::vector<Segment>> all_leaves =
                cluster_trees(storage, metric, perms, uint64_t(seed));
        order = perms[0];
        const GramKernel kernel = gram_kernel();
        for (int t = 0; t < ntrees; t++) {
            const std::vector<int32_t>& perm = perms[t];
            const std::vector<Segment>& leaves = all_leaves[t];
#pragma omp parallel reduction(+ : nd_total)
            {
                VectorReader vr(storage);
                LeafScratch ls;
                std::vector<DisId> cand, buf;
                std::vector<uint8_t> buf_new;
#pragma omp for schedule(dynamic)
                for (int64_t l = 0; l < int64_t(leaves.size()); l++) {
                    const int32_t* lp = perm.data() + leaves[l].begin;
                    const size_t m = leaves[l].end - leaves[l].begin;
                    leaf_distances(metric, kernel, lp, m, vr, ls);
                    const auto& leaf_dis = ls.dis;
                    nd_total += m * (m - 1) / 2;
                    for (size_t p = 0; p < m; p++) {
                        const float* ld = leaf_dis.data() + p * m;
                        // what cannot beat the K-th of the row cannot enter
                        const size_t ro = size_t(lp[p]) * K + K - 1;
                        const float rw = ids[ro] >= 0 ? dis[ro] : kInf;
                        cand.clear();
                        for (size_t q = 0; q < m; q++) {
                            if (q != p && ld[q] <= rw) {
                                cand.push_back({ld[q], lp[q]});
                            }
                        }
                        keep_best(cand, K);
                        add_to_row(lp[p], cand, buf, buf_new);
                    }
                }
            }
        }
    }
    // rows that the leaves left short, and all rows with ntrees = 0, get
    // random neighbors
#pragma omp parallel reduction(+ : nd_total)
    {
        VectorReader rv(storage), rc(storage);
        IdMap seen;
        std::vector<int32_t> c;
        std::vector<DisId> cand, buf;
        std::vector<uint8_t> buf_new;
#pragma omp for schedule(dynamic, 256)
        for (idx_t v = 0; v < n; v++) {
            const int want = int(std::min<idx_t>(K, n - 1));
            if (ids[size_t(v) * K + want - 1] >= 0) {
                continue;
            }
            c.clear();
            random_ids(v, c);
            // the ids of c that are not v and not in the row
            seen.clear();
            bool is_new_id;
            seen.find_or_insert(int32_t(v), is_new_id);
            for (int j = 0; j < K && ids[size_t(v) * K + j] >= 0; j++) {
                seen.find_or_insert(ids[size_t(v) * K + j], is_new_id);
            }
            const float* xv = rv.get(v);
            cand.clear();
            for (int32_t id : c) {
                seen.find_or_insert(id, is_new_id);
                if (is_new_id) {
                    cand.push_back({metric.one(xv, rc.get(id), d), id});
                }
            }
            nd_total += cand.size();
            keep_best(cand, K);
            add_to_row(v, cand, buf, buf_new);
        }
    }
    // every entry is new for the first round, as in NNDescentShard::set_rows()
#pragma omp parallel for schedule(static)
    for (int64_t e = 0; e < int64_t(nK); e++) {
        is_new[e] = ids[e] >= 0;
    }
    ndis = nd_total;
    if (verbose) {
        printf("DeterministicNNDescent: init n=%" PRId64
               " K=%d in %.1f s, %.0f distances per node\n",
               n,
               K,
               (getmillisecs() - t0) / 1000,
               double(ndis) / n);
    }
}

namespace {

struct SpinLock {
    std::atomic_flag& flag;
    explicit SpinLock(std::atomic_flag& f) : flag(f) {
        while (flag.test_and_set(std::memory_order_acquire)) {
        }
    }
    SpinLock(const SpinLock&) = delete;
    SpinLock& operator=(const SpinLock&) = delete;
    ~SpinLock() {
        flag.clear(std::memory_order_release);
    }
};

/// merge_into_row() of the sorted c[0..m) under the lock of the row; lowers
/// the K-th distance worst of the row
template <class Id>
void merge_into_row_locked(
        int K,
        std::atomic_flag& lock,
        std::atomic<float>& worst,
        Id* ids,
        float* dis,
        uint8_t* is_new,
        DisIdT<Id>* c,
        size_t m,
        std::vector<DisIdT<Id>>& buf,
        std::vector<uint8_t>& buf_new) {
    SpinLock guard(lock);
    merge_into_row(K, ids, dis, is_new, c, m, buf, buf_new);
    if (ids[K - 1] >= 0) {
        worst.store(dis[K - 1], std::memory_order_relaxed);
    }
}

/** The flags of the next rows at the start of a round: the local joins use
 * the first S new entries of each row, and the other new entries stay new
 * for the next round. */
void unsampled_new(
        const std::vector<uint8_t>& is_new,
        idx_t n,
        int K,
        int S,
        std::vector<uint8_t>& next_new) {
    next_new.assign(is_new.size(), 0);
#pragma omp parallel for schedule(static)
    for (idx_t v = 0; v < n; v++) {
        int taken = 0;
        for (int j = 0; j < K; j++) {
            const size_t e = size_t(v) * K + j;
            if (is_new[e] && taken++ >= S) {
                next_new[e] = 1;
            }
        }
    }
}

/// number of rows with a new entry
idx_t count_rows_with_new(const std::vector<uint8_t>& is_new, idx_t n, int K) {
    idx_t changed = 0;
#pragma omp parallel for reduction(+ : changed)
    for (idx_t v = 0; v < n; v++) {
        const auto row = is_new.begin() + size_t(v) * K;
        changed += std::find(row, row + K, 1) != row + K;
    }
    return changed;
}

/// per-thread scratch of local_join()
template <class Id>
struct JoinScratch {
    IdMap seen;
    std::vector<int32_t> nbr; ///< the new members, then the old ones
    std::vector<std::pair<int32_t, uint8_t>> lst; ///< (member, is new)
    AlignedTable<float, 64> xs, out;
    std::vector<const float*> rows;
    std::vector<float> nrm, thr;
    std::vector<std::vector<DisIdT<Id>>> bk;
    std::vector<DisIdT<Id>> mbuf;
    std::vector<uint8_t> mbuf_new;
};

/** The local join of one neighborhood, for DeterministicNNDescent::round()
 * and NNDescentShard::join_impl(). The members are storage indices: the
 * out-list out_l (K slots) and the in-list in_l (Rr slots), -1 padded, with
 * their new flags. Up to S new members of each list are new. For the pairs
 * (new, new) and (new, old), each distance that can beat the K-th distance
 * nworst[] of a member goes to flush(member, cand, buf, buf_new) as
 * (distance, gid(other member)).
 *
 * row(s, buf) returns the vector of member s, or decodes it into buf (d
 * floats) and returns buf. prefetch(s) is called two members ahead. The
 * products come from kernel, or from BLAS if it is null. Adds the distance
 * computations to ndis. */
template <class Id, class RowFn, class PrefetchFn, class GidFn, class FlushFn>
void local_join(
        const int32_t* out_l,
        const uint8_t* out_f,
        int K,
        const int32_t* in_l,
        const uint8_t* in_f,
        int Rr,
        int S,
        GramKernel kernel,
        bool ip,
        size_t d,
        const float* norms,
        const std::vector<std::atomic<float>>& nworst,
        JoinScratch<Id>& js,
        size_t& ndis,
        RowFn row,
        PrefetchFn prefetch,
        GidFn gid,
        FlushFn flush) {
    js.seen.clear();
    js.nbr.clear();
    js.lst.clear();
    // an id is new if one of its edges is new
    auto collect = [&](const int32_t* l, const uint8_t* f, int len) {
        int taken = 0;
        for (int i = 0; i < len && l[i] >= 0; i++) {
            bool fresh;
            const bool want_new = f[i] && taken < S;
            const int32_t k = js.seen.find_or_insert(l[i], fresh);
            if (fresh) {
                js.lst.push_back({l[i], uint8_t(want_new)});
                taken += want_new;
            } else if (want_new && !js.lst[k].second) {
                js.lst[k].second = 1;
                taken++;
            }
        }
    };
    collect(out_l, out_f, K);
    collect(in_l, in_f, Rr);
    int r = 0;
    for (auto& p : js.lst) {
        r += p.second;
    }
    if (r == 0) {
        return;
    }
    for (auto& p : js.lst) {
        if (p.second) {
            js.nbr.push_back(p.first);
        }
    }
    for (auto& p : js.lst) {
        if (!p.second) {
            js.nbr.push_back(p.first);
        }
    }
    const size_t m = js.nbr.size();
    pipnn::ensure_size(js.out, r * m);
    pipnn::ensure_size(js.xs, m * d);
    js.nrm.resize(m);
    js.thr.resize(m);
    js.rows.resize(m);
    if (js.bk.size() < m) {
        js.bk.resize(m);
    }
    for (size_t i = 0; i < m; i++) {
        if (i + 2 < m) {
            prefetch(js.nbr[i + 2]);
        }
        float* buf = js.xs.data() + i * d;
        const float* x = row(js.nbr[i], buf);
        if (!kernel && x != buf) {
            std::copy_n(x, d, buf); // BLAS reads one matrix
        }
        js.rows[i] = kernel ? x : buf;
        js.nrm[i] = norms[js.nbr[i]];
        // nworst starts at the K-th distance of the old row and only goes
        // down
        js.thr[i] = nworst[js.nbr[i]].load(std::memory_order_relaxed);
        js.bk[i].clear();
    }
    if (kernel) {
        kernel(js.rows.data(), r, m, d, js.out.data());
    } else {
        gram_upper_rows_blas(js.xs.data(), r, m, d, js.out.data());
    }
    products_to_distances(ip, js.nrm.data(), r, m, js.out.data());
    for (int i = 0; i < r; i++) {
        const float* oi = js.out.data() + size_t(i) * m;
        ndis += m - i - 1;
        for (size_t j = i + 1; j < m; j++) {
            const float v = oi[j];
            if (v <= js.thr[i]) {
                js.bk[i].push_back({v, gid(js.nbr[j])});
            }
            if (v <= js.thr[j]) {
                js.bk[j].push_back({v, gid(js.nbr[i])});
            }
        }
    }
    for (size_t i = 0; i < m; i++) {
        if (!js.bk[i].empty()) {
            flush(js.nbr[i], js.bk[i], js.mbuf, js.mbuf_new);
        }
    }
}

} // namespace

idx_t DeterministicNNDescent::round(const Index& storage) {
    FAISS_THROW_IF_NOT_MSG(
            storage.ntotal == n && ids.size() == size_t(n) * K &&
                    order.size() == size_t(n),
            "DeterministicNNDescent: init() before round()");
    check_params();
    FAISS_THROW_IF_NOT_MSG(
            size_t(storage.d) == d && storage.metric_type == metric_type,
            "DeterministicNNDescent: round() needs the storage of init()");
    if (n > 0) {
        // a storage that cannot reconstruct throws here, not inside OpenMP
        VectorReader(storage).get(0);
    }
    const int Rr = nreverse();
    const int S = nsample();
    const Metric metric{storage.metric_type == METRIC_INNER_PRODUCT};
    const double t0 = getmillisecs();

    // reverse neighbors: the Rr nearest of each node, by (distance, source).
    // The edges are counted and placed in parallel; each list is sorted after.
    // rpos[t + 1] counts the edges to t, then rpos[t] is the next free slot of
    // list t; after the placement, list t is [rpos[t - 1], rpos[t]).
    std::vector<std::atomic<int64_t>> rpos(n + 1);
    for (idx_t i = 0; i <= n; i++) {
        rpos[i].store(0, std::memory_order_relaxed);
    }
#pragma omp parallel for schedule(static)
    for (idx_t s = 0; s < n; s++) {
        for (int j = 0; j < K; j++) {
            const int32_t t = ids[size_t(s) * K + j];
            if (t >= 0) {
                rpos[t + 1].fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    for (idx_t i = 0; i < n; i++) {
        rpos[i + 1].store(
                rpos[i + 1].load(std::memory_order_relaxed) +
                        rpos[i].load(std::memory_order_relaxed),
                std::memory_order_relaxed);
    }
    const int64_t nedges = rpos[n].load(std::memory_order_relaxed);
    std::vector<DisId> redges(nedges);
    std::vector<uint8_t> rnew(nedges);
#pragma omp parallel for schedule(static)
    for (idx_t s = 0; s < n; s++) {
        for (int j = 0; j < K; j++) {
            const size_t e = size_t(s) * K + j;
            const int32_t t = ids[e];
            if (t >= 0) {
                const int64_t p =
                        rpos[t].fetch_add(1, std::memory_order_relaxed);
                redges[p] = {dis[e], int32_t(s)};
                rnew[p] = is_new[e];
            }
        }
    }
    std::vector<int32_t> rev(size_t(n) * Rr, -1);
    std::vector<uint8_t> rev_new(size_t(n) * Rr, 0);
#pragma omp parallel
    {
        // a source is in a list at most once, so the pairs compare as their
        // (distance, source)
        std::vector<std::pair<std::pair<float, int32_t>, uint8_t>> sel;
#pragma omp for schedule(dynamic, 1024)
        for (idx_t t = 0; t < n; t++) {
            const int64_t b =
                    t == 0 ? 0 : rpos[t - 1].load(std::memory_order_relaxed);
            const int64_t e = rpos[t].load(std::memory_order_relaxed);
            sel.resize(e - b);
            for (int64_t p = b; p < e; p++) {
                sel[p - b] = {{redges[p].dis, redges[p].id}, rnew[p]};
            }
            const size_t m = std::min<size_t>(sel.size(), Rr);
            std::partial_sort(sel.begin(), sel.begin() + m, sel.end());
            for (size_t j = 0; j < m; j++) {
                rev[size_t(t) * Rr + j] = sel[j].first.second;
                rev_new[size_t(t) * Rr + j] = sel[j].second;
            }
        }
    }
    redges = std::vector<DisId>();
    rnew = std::vector<uint8_t>();

    // the next rows start as the current rows; offers come from the local
    // joins under a lock per row
    std::vector<int32_t> nids = ids;
    std::vector<float> ndis_ = dis;
    std::vector<uint8_t> nnew;
    unsampled_new(is_new, n, K, S, nnew);
    std::vector<std::atomic_flag> locks(n);
    // K-th distance of each next row, read without the lock: it can only go
    // down, so an offer above it cannot enter the final row
    std::vector<std::atomic<float>> nworst(n);
    for (idx_t v = 0; v < n; v++) {
        const size_t e = size_t(v) * K + K - 1;
        nworst[v].store(ids[e] >= 0 ? dis[e] : kInf, std::memory_order_relaxed);
    }

    // squared norms, once per round
    std::vector<float> gnorms(n);
#pragma omp parallel
    {
        VectorReader vr(storage);
#pragma omp for schedule(static)
        for (idx_t v = 0; v < n; v++) {
            gnorms[v] = fvec_norm_L2sqr(vr.get(v), d);
        }
    }

    const GramKernel kernel = gram_kernel();
    size_t nd_total = 0;
#pragma omp parallel reduction(+ : nd_total)
    {
        VectorReader vr(storage);
        JoinScratch<int32_t> js;
        auto row = [&](int32_t s, float* buf) -> const float* {
            if (vr.flat) {
                return vr.flat + size_t(s) * d;
            }
            vr.decode(s, buf);
            return buf;
        };
        auto prefetch = [&](int32_t s) { vr.prefetch(s); };
        auto gid = [](int32_t s) { return s; };
        auto flush = [&](int32_t a,
                         std::vector<DisId>& cand,
                         std::vector<DisId>& buf,
                         std::vector<uint8_t>& buf_new) {
            const size_t o = size_t(a) * K;
            const size_t m =
                    drop_old_ids(ids.data() + o, dis[o + K - 1], K, cand);
            if (m == 0) {
                return;
            }
            std::sort(cand.begin(), cand.begin() + m, dis_id_less);
            merge_into_row_locked(
                    K,
                    locks[a],
                    nworst[a],
                    nids.data() + o,
                    ndis_.data() + o,
                    nnew.data() + o,
                    cand.data(),
                    m,
                    buf,
                    buf_new);
        };
#pragma omp for schedule(dynamic, 64)
        for (idx_t pos = 0; pos < n; pos++) {
            const idx_t u = order[pos];
            local_join(
                    ids.data() + size_t(u) * K,
                    is_new.data() + size_t(u) * K,
                    K,
                    rev.data() + size_t(u) * Rr,
                    rev_new.data() + size_t(u) * Rr,
                    Rr,
                    S,
                    kernel,
                    metric.ip,
                    d,
                    gnorms.data(),
                    nworst,
                    js,
                    nd_total,
                    row,
                    prefetch,
                    gid,
                    flush);
        }
    }
    const idx_t changed = count_rows_with_new(nnew, n, K);
    ids.swap(nids);
    dis.swap(ndis_);
    is_new.swap(nnew);
    ndis = nd_total;
    if (verbose) {
        printf("DeterministicNNDescent: round in %.1f s, %" PRId64
               " nodes changed, %.0f distances per node\n",
               (getmillisecs() - t0) / 1000,
               changed,
               double(ndis) / n);
    }
    return changed;
}

void DeterministicNNDescent::check_params() const {
    FAISS_THROW_IF_NOT_FMT(
            K >= 1 && K <= 4096,
            "DeterministicNNDescent: K must be in [1, 4096], got %d",
            K);
    FAISS_THROW_IF_NOT_FMT(
            R >= 0 && R <= 4096,
            "DeterministicNNDescent: R must be in [0, 4096], got %d",
            R);
    FAISS_THROW_IF_NOT_FMT(
            sample >= 0,
            "DeterministicNNDescent: sample must be >= 0, got %d",
            sample);
    FAISS_THROW_IF_NOT_FMT(
            ntrees >= 0,
            "DeterministicNNDescent: ntrees must be >= 0, got %d",
            ntrees);
}

void DeterministicNNDescent::build(const Index& storage) {
    init(storage);
    for (int r = 0; r < max_rounds; r++) {
        if (round(storage) < delta * n) {
            break;
        }
    }
}

/*************************************************************
 * NNDescentShard
 *************************************************************/

namespace {

/// owner process of graph id v
inline int owner_of(const std::vector<int64_t>& owner_begin, int64_t v) {
    return int(std::upper_bound(owner_begin.begin(), owner_begin.end(), v) -
               owner_begin.begin()) -
            1;
}

/// throws msg if one of the n ids is not in [lo, hi), or -1 with allow_none.
/// An id out of range would index past the end of per-owner arrays.
void check_graph_ids(
        const int64_t* v,
        size_t n,
        int64_t lo,
        int64_t hi,
        bool allow_none,
        const char* msg) {
    int bad = 0;
#pragma omp parallel for schedule(static) reduction(| : bad)
    for (int64_t e = 0; e < int64_t(n); e++) {
        const bool ok = (allow_none && v[e] == -1) || (v[e] >= lo && v[e] < hi);
        bad |= ok ? 0 : 1;
    }
    if (bad) {
        FAISS_THROW_MSG(msg);
    }
}

/// counts of a list grouped by owner, from the owner of each item
void group_counts(
        const std::vector<int64_t>& owner_begin,
        const std::vector<int64_t>& sorted_ids,
        std::vector<int64_t>& counts) {
    const int world = int(owner_begin.size()) - 1;
    counts.assign(world, 0);
    for (int p = 0; p < world; p++) {
        counts[p] =
                std::lower_bound(
                        sorted_ids.begin(),
                        sorted_ids.end(),
                        owner_begin[p + 1]) -
                std::lower_bound(
                        sorted_ids.begin(), sorted_ids.end(), owner_begin[p]);
    }
}

} // namespace

void NNDescentShard::set_rows(
        int K_in,
        MetricType metric_type_in,
        size_t d_in,
        int rank_in,
        int world,
        const int64_t* owner_begin_in,
        const float* x_in,
        const int64_t* ids_in,
        const float* dis_in) {
    FAISS_THROW_IF_NOT_FMT(
            K_in >= 1 && K_in <= 4096,
            "NNDescentShard: K must be in [1, 4096], got %d",
            K_in);
    FAISS_THROW_IF_NOT_MSG(
            metric_type_in == METRIC_L2 ||
                    metric_type_in == METRIC_INNER_PRODUCT,
            "NNDescentShard: only METRIC_L2 and METRIC_INNER_PRODUCT");
    FAISS_THROW_IF_NOT_MSG(d_in > 0, "NNDescentShard: d must be > 0");
    FAISS_THROW_IF_NOT(world >= 1 && rank_in >= 0 && rank_in < world);
    for (int p = 0; p < world; p++) {
        FAISS_THROW_IF_NOT_MSG(
                owner_begin_in[p] <= owner_begin_in[p + 1],
                "NNDescentShard: owner_begin must be non-decreasing");
    }
    const size_t nK =
            size_t(owner_begin_in[rank_in + 1] - owner_begin_in[rank_in]) *
            K_in;
    check_graph_ids(
            ids_in,
            nK,
            owner_begin_in[0],
            owner_begin_in[world],
            true,
            "NNDescentShard: row id out of range");
    K = K_in;
    metric_type = metric_type_in;
    d = d_in;
    rank = rank_in;
    owner_begin.assign(owner_begin_in, owner_begin_in + world + 1);
    x = x_in;
    ids.assign(ids_in, ids_in + nK);
    dis.assign(dis_in, dis_in + nK);
    is_new.resize(nK);
    for (size_t e = 0; e < nK; e++) {
        is_new[e] = ids[e] >= 0;
    }
    remote_ids.clear();
    reverse_ids.clear();
    reverse_new.clear();
}

idx_t NNDescentShard::nhome() const {
    return owner_begin.empty() ? 0 : owner_begin[rank + 1] - owner_begin[rank];
}

void NNDescentShard::reverse_edges(
        std::vector<int64_t>& counts,
        std::vector<int64_t>& target,
        std::vector<int64_t>& source,
        std::vector<float>& edge_dis,
        std::vector<uint8_t>& edge_new) const {
    const int world = int(owner_begin.size()) - 1;
    const idx_t nh = nhome();
    const size_t nK = size_t(nh) * K;
    std::vector<int32_t> owner(nK, -1);
#pragma omp parallel for schedule(static)
    for (int64_t e = 0; e < int64_t(nK); e++) {
        if (ids[e] >= 0) {
            owner[e] = owner_of(owner_begin, ids[e]);
        }
    }
    const int nt = omp_get_max_threads();
    // pos[c * world + p]: the edges of row chunk c to owner p, then the
    // position of the next one
    std::vector<int64_t> pos(size_t(nt) * world, 0);
#pragma omp parallel for schedule(static)
    for (int c = 0; c < nt; c++) {
        int64_t* mine = pos.data() + size_t(c) * world;
        for (size_t e = nK * c / nt; e < nK * (c + 1) / nt; e++) {
            if (owner[e] >= 0) {
                mine[owner[e]]++;
            }
        }
    }
    counts.assign(world, 0);
    int64_t total = 0;
    for (int p = 0; p < world; p++) {
        for (int c = 0; c < nt; c++) {
            const int64_t m = pos[size_t(c) * world + p];
            pos[size_t(c) * world + p] = total;
            total += m;
            counts[p] += m;
        }
    }
    target.resize(total);
    source.resize(total);
    edge_dis.resize(total);
    edge_new.resize(total);
#pragma omp parallel for schedule(static)
    for (int c = 0; c < nt; c++) {
        int64_t* cur = pos.data() + size_t(c) * world;
        for (size_t e = nK * c / nt; e < nK * (c + 1) / nt; e++) {
            if (owner[e] < 0) {
                continue;
            }
            const int64_t q = cur[owner[e]]++;
            target[q] = ids[e];
            source[q] = owner_begin[rank] + int64_t(e / K);
            edge_dis[q] = dis[e];
            edge_new[q] = is_new[e];
        }
    }
}

namespace {

/// Groups the n items by their home row target[e] - hb, and calls
/// row_fn(t, begin, end) on the items [begin, end) of each home row t. The
/// items of a row keep their input order. make(e) builds item e; the
/// function sets its field row. Threads process disjoint ranges of rows.
template <class Item, class Make, class RowFn>
void group_by_home_row(
        size_t n,
        const int64_t* target,
        int64_t hb,
        idx_t nh,
        const char* msg,
        Make make,
        RowFn row_fn) {
    const int nt = omp_get_max_threads();
    // bucket b holds the rows [b * rows_per, (b + 1) * rows_per)
    const idx_t nchunks = 16 * idx_t(nt);
    const idx_t rows_per = std::max<idx_t>(1, (nh + nchunks - 1) / nchunks);
    const idx_t nb = (nh + rows_per - 1) / rows_per;
    // pos[c * nb + b]: the items of input chunk c in bucket b, then the
    // position of the next one
    std::vector<size_t> pos(size_t(nt) * nb, 0);
    std::vector<uint8_t> bad(nt, 0);
#pragma omp parallel for schedule(static)
    for (int c = 0; c < nt; c++) {
        size_t* mine = pos.data() + size_t(c) * nb;
        for (size_t e = n * c / nt; e < n * (c + 1) / nt; e++) {
            const int64_t r = target[e] - hb;
            if (r < 0 || r >= nh) {
                bad[c] = 1;
                break;
            }
            mine[r / rows_per]++;
        }
    }
    FAISS_THROW_IF_NOT_FMT(
            std::find(bad.begin(), bad.end(), 1) == bad.end(), "%s", msg);
    std::vector<size_t> bucket_begin(nb + 1);
    size_t acc = 0;
    for (idx_t b = 0; b < nb; b++) {
        bucket_begin[b] = acc;
        for (int c = 0; c < nt; c++) {
            const size_t m = pos[size_t(c) * nb + b];
            pos[size_t(c) * nb + b] = acc;
            acc += m;
        }
    }
    bucket_begin[nb] = acc;
    std::vector<Item> items(n);
#pragma omp parallel for schedule(static)
    for (int c = 0; c < nt; c++) {
        size_t* cur = pos.data() + size_t(c) * nb;
        for (size_t e = n * c / nt; e < n * (c + 1) / nt; e++) {
            const idx_t r = target[e] - hb;
            Item& it = items[cur[r / rows_per]++];
            it = make(e);
            it.row = uint32_t(r);
        }
    }
#pragma omp parallel
    {
        std::vector<Item> tmp;
        std::vector<size_t> off;
#pragma omp for schedule(dynamic)
        for (idx_t b = 0; b < nb; b++) {
            const idx_t r0 = b * rows_per;
            const idx_t r1 = std::min(nh, r0 + rows_per);
            const Item* in = items.data() + bucket_begin[b];
            const size_t m = bucket_begin[b + 1] - bucket_begin[b];
            off.assign(r1 - r0 + 1, 0);
            for (size_t i = 0; i < m; i++) {
                off[in[i].row - r0 + 1]++;
            }
            for (idx_t t = r0; t < r1; t++) {
                off[t - r0 + 1] += off[t - r0];
            }
            tmp.resize(m);
            for (size_t i = 0; i < m; i++) {
                tmp[off[in[i].row - r0]++] = in[i];
            }
            for (idx_t t = r1; t > r0; t--) {
                off[t - r0] = off[t - r0 - 1];
            }
            off[0] = 0;
            for (idx_t t = r0; t < r1; t++) {
                row_fn(t,
                       tmp.data() + off[t - r0],
                       tmp.data() + off[t - r0 + 1]);
            }
        }
    }
}

} // namespace

void NNDescentShard::set_reverse_edges(
        size_t n,
        const int64_t* target,
        const int64_t* source,
        const float* edge_dis,
        const uint8_t* edge_new) {
    const idx_t nh = nhome();
    const int64_t hb = owner_begin[rank];
    const int Rr = nreverse();
    struct InEdge {
        float dis;
        uint32_t row;
        int64_t id;
        uint8_t is_new;
    };
    check_graph_ids(
            source,
            n,
            owner_begin.front(),
            owner_begin.back(),
            false,
            "NNDescentShard: reverse edge source out of range");
    reverse_ids.assign(size_t(nh) * Rr, -1);
    reverse_new.assign(size_t(nh) * Rr, 0);
    auto less = [](const InEdge& a, const InEdge& b) {
        return a.dis < b.dis || (a.dis == b.dis && a.id < b.id);
    };
    group_by_home_row<InEdge>(
            n,
            target,
            hb,
            nh,
            "NNDescentShard: reverse edge to an id of another process",
            [&](size_t e) {
                return InEdge{edge_dis[e], 0, source[e], edge_new[e]};
            },
            [&](idx_t t, InEdge* b, InEdge* e) {
                const size_t m = std::min<size_t>(e - b, Rr);
                std::partial_sort(b, b + m, e, less);
                for (size_t j = 0; j < m; j++) {
                    reverse_ids[size_t(t) * Rr + j] = b[j].id;
                    reverse_new[size_t(t) * Rr + j] = b[j].is_new;
                }
            });

    select_rows(0, nh);
}

void NNDescentShard::select_rows(idx_t begin, idx_t end) {
    FAISS_THROW_IF_NOT_MSG(
            begin >= 0 && begin <= end && end <= nhome(),
            "NNDescentShard: invalid row range");
    FAISS_THROW_IF_NOT_MSG(
            reverse_ids.size() == size_t(nhome()) * nreverse(),
            "NNDescentShard: set_reverse_edges() before select_rows()");
    selected_begin = begin;
    selected_end = end;
    collect_remote_ids(true, begin, end);
}

bool NNDescentShard::has_new_member(idx_t t) const {
    const int Rr = nreverse();
    for (int j = 0; j < K && ids[size_t(t) * K + j] >= 0; j++) {
        if (is_new[size_t(t) * K + j]) {
            return true;
        }
    }
    for (int j = 0; j < Rr && reverse_ids[size_t(t) * Rr + j] >= 0; j++) {
        if (reverse_new[size_t(t) * Rr + j]) {
            return true;
        }
    }
    return false;
}

// the ids are bucketed by owner, then sorted and unique in each bucket
void NNDescentShard::collect_remote_ids(
        bool with_reverse,
        idx_t begin,
        idx_t end) {
    const int Rr = with_reverse ? nreverse() : 0;
    const idx_t nh = nhome();
    const int64_t hb = owner_begin[rank];
    const int world = int(owner_begin.size()) - 1;
    const int nt = omp_get_max_threads();
    std::vector<std::vector<std::vector<int64_t>>> buckets(
            nt, std::vector<std::vector<int64_t>>(world));
#pragma omp parallel
    {
        auto& mine = buckets[omp_get_thread_num()];
        auto add = [&](int64_t v) {
            if (v >= 0 && (v < hb || v >= hb + nh)) {
                mine[owner_of(owner_begin, v)].push_back(v);
            }
        };
#pragma omp for schedule(static)
        for (idx_t t = begin; t < end; t++) {
            if (with_reverse && !has_new_member(t)) {
                continue;
            }
            for (int j = 0; j < K; j++) {
                add(ids[size_t(t) * K + j]);
            }
            for (int j = 0; j < Rr; j++) {
                add(reverse_ids[size_t(t) * Rr + j]);
            }
        }
    }
    std::vector<std::vector<int64_t>> per_owner(world);
#pragma omp parallel for schedule(dynamic)
    for (int p = 0; p < world; p++) {
        auto& v = per_owner[p];
        for (int t = 0; t < nt; t++) {
            v.insert(v.end(), buckets[t][p].begin(), buckets[t][p].end());
            buckets[t][p] = std::vector<int64_t>();
        }
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
    }
    remote_ids.clear();
    for (auto& v : per_owner) {
        remote_ids.insert(remote_ids.end(), v.begin(), v.end());
    }
}

void NNDescentShard::set_remote_ids_of_rows() {
    collect_remote_ids(false, 0, nhome());
}

void NNDescentShard::original_ids(
        const int64_t* home_original,
        const int64_t* remote_original,
        int64_t* out) const {
    const idx_t nh = nhome();
    const int64_t hb = owner_begin[rank];
    int missing = 0;
#pragma omp parallel for schedule(static) reduction(| : missing)
    for (int64_t e = 0; e < int64_t(size_t(nh) * K); e++) {
        const int64_t v = ids[e];
        if (v < 0) {
            out[e] = -1;
        } else if (v >= hb && v < hb + nh) {
            out[e] = home_original[v - hb];
        } else {
            const auto it =
                    std::lower_bound(remote_ids.begin(), remote_ids.end(), v);
            if (it != remote_ids.end() && *it == v) {
                out[e] = remote_original[it - remote_ids.begin()];
            } else {
                out[e] = -1;
                missing = 1;
            }
        }
    }
    // throw outside the OpenMP region, where the caller can catch it
    FAISS_THROW_IF_MSG(
            missing,
            "NNDescentShard: row id not in remote_ids; call set_remote_ids_of_rows()");
}

void NNDescentShard::remote_counts(std::vector<int64_t>& counts) const {
    group_counts(owner_begin, remote_ids, counts);
}

void NNDescentShard::join(
        const float* remote_x,
        const float* remote_kth,
        std::vector<int64_t>& counts,
        std::vector<int64_t>& target,
        std::vector<int64_t>& candidate,
        std::vector<float>& offer_dis) {
    RemoteVectors rv;
    rv.x = remote_x;
    join_impl(rv, remote_kth, counts, target, candidate, offer_dis);
}

void NNDescentShard::join_fp16(
        const uint16_t* remote_x,
        const float* remote_kth,
        std::vector<int64_t>& counts,
        std::vector<int64_t>& target,
        std::vector<int64_t>& candidate,
        std::vector<float>& offer_dis) {
    RemoteVectors rv;
    rv.x16 = remote_x;
    join_impl(rv, remote_kth, counts, target, candidate, offer_dis);
}

void NNDescentShard::join_sq8(
        const uint8_t* remote_codes,
        const float* vmin,
        const float* vdiff,
        const float* remote_kth,
        std::vector<int64_t>& counts,
        std::vector<int64_t>& target,
        std::vector<int64_t>& candidate,
        std::vector<float>& offer_dis) {
    RemoteVectors rv;
    rv.codes = remote_codes;
    rv.vmin = vmin;
    rv.vdiff = vdiff;
    join_impl(rv, remote_kth, counts, target, candidate, offer_dis);
}

void NNDescentShard::join_impl(
        const RemoteVectors& rv,
        const float* remote_kth,
        std::vector<int64_t>& counts,
        std::vector<int64_t>& target,
        std::vector<int64_t>& candidate,
        std::vector<float>& offer_dis) {
    const idx_t nh = nhome();
    const size_t nr = remote_ids.size();
    const int64_t hb = owner_begin[rank];
    const int Rr = nreverse();
    const int S = nsample();
    const bool ip = metric_type == METRIC_INNER_PRODUCT;
    FAISS_THROW_IF_NOT_MSG(
            reverse_ids.size() == size_t(nh) * Rr,
            "NNDescentShard: set_reverse_edges() before join()");
    FAISS_THROW_IF_NOT_MSG(
            size_t(nh) + nr < (size_t(1) << 31),
            "NNDescentShard: at most 2^31 - 1 home and remote ids");

    // storage index of a graph id: home ids first, then the remote ids
    auto sidx = [&](int64_t v) -> int32_t {
        if (v >= hb && v < hb + nh) {
            return int32_t(v - hb);
        }
        const auto it =
                std::lower_bound(remote_ids.begin(), remote_ids.end(), v);
        return it != remote_ids.end() && *it == v
                ? int32_t(nh + (it - remote_ids.begin()))
                : -1;
    };
    auto gid = [&](int32_t s) -> int64_t {
        return s < nh ? hb + s : remote_ids[s - nh];
    };
    // the vector of storage index s; a coded remote vector is decoded into buf
    auto row = [&](int32_t s, float* buf) -> const float* {
        if (s < nh) {
            return x + size_t(s) * d;
        }
        const size_t o = size_t(s - nh) * d;
        if (rv.x) {
            return rv.x + o;
        }
        if (rv.x16) {
            for (size_t j = 0; j < d; j++) {
                buf[j] = decode_fp16(rv.x16[o + j]);
            }
            return buf;
        }
        // two loops: a fused multiply-add would round differently from the
        // encoder's float32 decode
        for (size_t j = 0; j < d; j++) {
            buf[j] = (rv.codes[o + j] + 0.5f) / 255.f * rv.vdiff[j];
        }
        for (size_t j = 0; j < d; j++) {
            buf[j] += rv.vmin[j];
        }
        return buf;
    };
    // remote_ids has only the members of the neighborhoods with a new member;
    // the other rows stay -1 and their local joins are empty
    std::vector<int32_t> out_s(size_t(nh) * K, -1), in_s(size_t(nh) * Rr, -1);
    std::vector<float> norms(nh + nr);
    int missing = 0;
#pragma omp parallel for schedule(static) reduction(| : missing)
    for (idx_t t = selected_begin; t < selected_end; t++) {
        if (!has_new_member(t)) {
            continue;
        }
        for (int j = 0; j < K; j++) {
            const int64_t v = ids[size_t(t) * K + j];
            if (v >= 0) {
                out_s[size_t(t) * K + j] = sidx(v);
                missing |= out_s[size_t(t) * K + j] < 0;
            }
        }
        for (int j = 0; j < Rr; j++) {
            const int64_t v = reverse_ids[size_t(t) * Rr + j];
            if (v >= 0) {
                in_s[size_t(t) * Rr + j] = sidx(v);
                missing |= in_s[size_t(t) * Rr + j] < 0;
            }
        }
    }
    FAISS_THROW_IF_MSG(
            missing,
            "NNDescentShard: a neighborhood id is not in remote_ids; "
            "call select_rows() before join()");
#pragma omp parallel
    {
        std::vector<float> buf(d);
#pragma omp for schedule(static)
        for (int64_t s = 0; s < int64_t(nh + nr); s++) {
            norms[s] = fvec_norm_L2sqr(row(int32_t(s), buf.data()), d);
        }
    }

    // next rows, as in round(), at the first join() of the round
    if (next_ids.size() != ids.size()) {
        next_ids = ids;
        next_dis = dis;
        unsampled_new(is_new, nh, K, S, next_new);
        ndis = 0;
    }
    std::vector<std::atomic_flag> locks(nh + nr);
    std::vector<std::atomic<float>> nworst(nh + nr);
    for (idx_t v = 0; v < nh; v++) {
        const size_t e = size_t(v) * K + K - 1;
        nworst[v].store(
                next_ids[e] >= 0 ? next_dis[e] : kInf,
                std::memory_order_relaxed);
    }
    for (size_t i = 0; i < nr; i++) {
        nworst[nh + i].store(remote_kth[i], std::memory_order_relaxed);
    }
    // offers to remote rows: the K best by (distance, candidate), one per id
    std::vector<std::vector<DisId64>> hbuf(nr);

    auto flush = [&](int32_t a,
                     std::vector<DisId64>& cand,
                     std::vector<DisId64>& buf,
                     std::vector<uint8_t>& buf_new) {
        size_t m = 0;
        if (a < nh) {
            const size_t o = size_t(a) * K;
            m = drop_old_ids(ids.data() + o, dis[o + K - 1], K, cand);
        } else {
            // the owner drops the ids of its row in add_offers()
            const float kth = remote_kth[a - nh];
            for (const DisId64& c : cand) {
                if (c.dis <= kth) {
                    cand[m++] = c;
                }
            }
        }
        if (m == 0) {
            return;
        }
        std::sort(cand.begin(), cand.begin() + m, dis_id_less_t<int64_t>);
        if (a < nh) {
            const size_t o = size_t(a) * K;
            merge_into_row_locked(
                    K,
                    locks[a],
                    nworst[a],
                    next_ids.data() + o,
                    next_dis.data() + o,
                    next_new.data() + o,
                    cand.data(),
                    m,
                    buf,
                    buf_new);
            return;
        }
        // the K best distinct candidates by their smallest distance, in any
        // order of the offers
        SpinLock lock(locks[a]);
        auto& hb_ = hbuf[a - nh];
        for (size_t j = 0; j < m; j++) {
            const DisId64 c = cand[j];
            auto it =
                    std::find_if(hb_.begin(), hb_.end(), [&](const DisId64& h) {
                        return h.id == c.id;
                    });
            if (it != hb_.end()) {
                if (dis_id_less_t(c, *it)) {
                    *it = c;
                }
            } else if (hb_.size() < size_t(K)) {
                hb_.push_back(c);
            } else {
                auto w = std::max_element(
                        hb_.begin(), hb_.end(), dis_id_less_t<int64_t>);
                if (dis_id_less_t(c, *w)) {
                    *w = c;
                }
            }
        }
        if (hb_.size() == size_t(K)) {
            const auto w = std::max_element(
                    hb_.begin(), hb_.end(), dis_id_less_t<int64_t>);
            if (w->dis < nworst[a].load(std::memory_order_relaxed)) {
                nworst[a].store(w->dis, std::memory_order_relaxed);
            }
        }
    };

    const GramKernel kernel = gram_kernel();
    size_t nd_total = 0;
#pragma omp parallel reduction(+ : nd_total)
    {
        JoinScratch<int64_t> js;
        auto prefetch = [](int32_t) {};
#pragma omp for schedule(dynamic, 64)
        for (idx_t u = selected_begin; u < selected_end; u++) {
            local_join(
                    out_s.data() + size_t(u) * K,
                    is_new.data() + size_t(u) * K,
                    K,
                    in_s.data() + size_t(u) * Rr,
                    reverse_new.data() + size_t(u) * Rr,
                    Rr,
                    S,
                    kernel,
                    ip,
                    d,
                    norms.data(),
                    nworst,
                    js,
                    nd_total,
                    row,
                    prefetch,
                    gid,
                    flush);
        }
    }
    ndis += nd_total;

    // offers to remote rows, grouped by owner because remote_ids is sorted
    size_t total = 0;
    std::vector<size_t> start(nr + 1, 0);
    for (size_t i = 0; i < nr; i++) {
        start[i + 1] = start[i] + hbuf[i].size();
    }
    total = start[nr];
    target.resize(total);
    candidate.resize(total);
    offer_dis.resize(total);
#pragma omp parallel for schedule(dynamic, 1024)
    for (int64_t i = 0; i < int64_t(nr); i++) {
        auto& b = hbuf[i];
        std::sort(b.begin(), b.end(), dis_id_less_t<int64_t>);
        for (size_t j = 0; j < b.size(); j++) {
            target[start[i] + j] = remote_ids[i];
            candidate[start[i] + j] = b[j].id;
            offer_dis[start[i] + j] = b[j].dis;
        }
    }
    group_counts(owner_begin, target, counts);
}

void NNDescentShard::add_offers(
        size_t n,
        const int64_t* target,
        const int64_t* candidate,
        const float* offer_dis) {
    const idx_t nh = nhome();
    const int64_t hb = owner_begin[rank];
    FAISS_THROW_IF_NOT_MSG(
            next_ids.size() == ids.size(),
            "NNDescentShard: join() before add_offers()");
    check_graph_ids(
            candidate,
            n,
            owner_begin.front(),
            owner_begin.back(),
            false,
            "NNDescentShard: offer candidate out of range");
    struct Offer {
        float dis;
        uint32_t row;
        int64_t id;
    };
    std::vector<std::vector<DisId64>> cand(omp_get_max_threads());
    std::vector<std::vector<DisId64>> mbuf(cand.size());
    std::vector<std::vector<uint8_t>> mbuf_new(cand.size());
    group_by_home_row<Offer>(
            n,
            target,
            hb,
            nh,
            "NNDescentShard: offer to an id of another process",
            [&](size_t e) { return Offer{offer_dis[e], 0, candidate[e]}; },
            [&](idx_t t, Offer* b, Offer* e) {
                const int th = omp_get_thread_num();
                const size_t o = size_t(t) * K;
                // one entry per candidate, its smallest distance
                std::sort(b, e, [](const Offer& p, const Offer& q) {
                    return p.id < q.id || (p.id == q.id && p.dis < q.dis);
                });
                auto& c = cand[th];
                c.clear();
                for (auto it = b; it != e; ++it) {
                    if (it == b || it->id != it[-1].id) {
                        c.push_back({it->dis, it->id});
                    }
                }
                const size_t m =
                        drop_old_ids(ids.data() + o, dis[o + K - 1], K, c);
                if (m == 0) {
                    return;
                }
                std::sort(c.begin(), c.begin() + m, dis_id_less_t<int64_t>);
                merge_into_row(
                        K,
                        next_ids.data() + o,
                        next_dis.data() + o,
                        next_new.data() + o,
                        c.data(),
                        m,
                        mbuf[th],
                        mbuf_new[th]);
            });
}

idx_t NNDescentShard::end_round() {
    const idx_t nh = nhome();
    FAISS_THROW_IF_NOT_MSG(
            next_ids.size() == ids.size(),
            "NNDescentShard: join() before end_round()");
    const idx_t changed = count_rows_with_new(next_new, nh, K);
    ids.swap(next_ids);
    dis.swap(next_dis);
    is_new.swap(next_new);
    next_ids.clear();
    next_dis.clear();
    next_new.clear();
    reverse_ids.clear();
    reverse_new.clear();
    remote_ids.clear();
    return changed;
}

} // namespace faiss
