/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// IndexScalarQuantizer::use_fp16_gemm search path: packs the QT_fp16 codes
// once and scores blocks of queries against blocks of codes with the
// kernels declared in sq-fp16-gemm.h.

#include <faiss/IndexScalarQuantizer.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/fp16.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace faiss {

namespace fp16_gemm {

struct PackedCodes {
    int d = 0;
    idx_t ntotal = 0;
    MetricType metric = METRIC_INNER_PRODUCT;
    size_t bs_y = 0;
    const uint8_t* source_codes = nullptr;
    std::vector<uint16_t> rows;
    std::vector<uint16_t> packed;
    std::vector<size_t> offset;
    std::vector<float> norms;
};

} // namespace fp16_gemm

namespace {

size_t round_up(size_t value, size_t block) {
    FAISS_THROW_IF_NOT_MSG(
            value <= std::numeric_limits<size_t>::max() - (block - 1),
            "fp16 gemm block size overflow");
    return (value + block - 1) / block * block;
}

size_t input_size(idx_t n, int d) {
    FAISS_THROW_IF_NOT_MSG(
            n >= 0 && d > 0, "invalid fp16 gemm input count or dimension");
    FAISS_THROW_IF_NOT_MSG(
            static_cast<uint64_t>(n) <=
                    std::numeric_limits<size_t>::max() / static_cast<size_t>(d),
            "fp16 gemm input size overflow");
    return static_cast<size_t>(n) * static_cast<size_t>(d);
}

// Below this many multiply-adds (n * ntotal * d) a small batch is not split
// across threads: thread start-up would dominate.
constexpr size_t kMinSplitWork = size_t{1} << 22;
// Caps the per-range top-k buffers used when splitting by centroid block
// (number of (distance, id) entries summed over all ranges).
constexpr size_t kMaxSplitHeapEntries = size_t{1} << 22;
// Smallest group of query rows a search is split into: one full AMX block.
constexpr size_t kFP16TileRows = fp16_gemm::kFP16BlockM;

struct ScopedSingleThread {
    int saved = -1;
    explicit ScopedSingleThread([[maybe_unused]] bool on) {
#ifdef _OPENMP
        if (on) {
            saved = omp_get_max_threads();
            omp_set_num_threads(1);
        }
#endif
    }
    ~ScopedSingleThread() {
#ifdef _OPENMP
        if (saved > 0) {
            omp_set_num_threads(saved);
        }
#endif
    }
};

// Ties keep the result with the smaller id at the top of the heap, so the
// smaller id is the one kept when trimming to k.
struct IPHeap : CMin<float, idx_t> {
    static bool cmp2(float a, float b, idx_t ai, idx_t bi) {
        return a < b || (a == b && ai > bi);
    }
    static float neutral() {
        return -std::numeric_limits<float>::infinity();
    }
};
struct L2Heap : CMax<float, idx_t> {
    static float neutral() {
        return std::numeric_limits<float>::infinity();
    }
};

void sort_selected(
        idx_t n,
        idx_t k,
        float* distances,
        idx_t* labels,
        bool is_l2) {
    const size_t kk = static_cast<size_t>(k);
    std::vector<std::pair<float, idx_t>> entries(kk);
    for (idx_t i = 0; i < n; ++i) {
        float* ds = distances + static_cast<size_t>(i) * kk;
        idx_t* ids = labels + static_cast<size_t>(i) * kk;
        size_t valid = 0;
        for (size_t j = 0; j < kk; ++j) {
            if (ids[j] >= 0) {
                entries[valid++] = {ds[j], ids[j]};
            }
        }
        std::sort(
                entries.begin(),
                entries.begin() + valid,
                [is_l2](const auto& a, const auto& b) {
                    if (a.first != b.first) {
                        return is_l2 ? a.first < b.first : a.first > b.first;
                    }
                    return a.second < b.second;
                });
        for (size_t j = 0; j < valid; ++j) {
            ds[j] = entries[j].first;
            ids[j] = entries[j].second;
        }
        for (size_t j = valid; j < kk; ++j) {
            ds[j] = is_l2 ? std::numeric_limits<float>::infinity()
                          : -std::numeric_limits<float>::infinity();
            ids[j] = -1;
        }
    }
}

void check_format(const IndexScalarQuantizer& index) {
    const int d = index.d;
    FAISS_THROW_IF_NOT_MSG(
            d > 0 && index.sq.d == static_cast<size_t>(d),
            "fp16 gemm index dimension mismatch");
    FAISS_THROW_IF_NOT_MSG(
            index.sq.qtype == ScalarQuantizer::QT_fp16 &&
                    index.code_size == static_cast<size_t>(d) * 2 &&
                    index.sq.code_size == index.code_size,
            "use_fp16_gemm requires a QT_fp16 IndexScalarQuantizer");
    FAISS_THROW_IF_NOT_MSG(
            index.metric_type == METRIC_L2 ||
                    index.metric_type == METRIC_INNER_PRODUCT,
            "use_fp16_gemm only supports inner product and L2");
    FAISS_THROW_IF_NOT_MSG(
            index.ntotal >= 0 &&
                    index.codes.size() ==
                            static_cast<size_t>(index.ntotal) * index.code_size,
            "fp16 gemm index code count mismatch");
}

std::shared_ptr<const fp16_gemm::PackedCodes> build_pack(
        const IndexScalarQuantizer& index) {
    const int d = index.d;
    const idx_t ntotal = index.ntotal;
    const MetricType metric_type = index.metric_type;
    const size_t bs_y = index.fp16_gemm_bs_y;
    const auto& codes = index.codes;
    fp16_gemm::validate_codes(codes.data(), static_cast<size_t>(ntotal) * d);
    auto cache = std::make_shared<fp16_gemm::PackedCodes>();
    cache->d = d;
    cache->ntotal = ntotal;
    cache->metric = metric_type;
    cache->bs_y = std::max(round_up(bs_y, 32), size_t{32});
    cache->source_codes = codes.data();
    cache->rows.resize(static_cast<size_t>(ntotal) * d);
    if (codes.size()) {
        std::memcpy(cache->rows.data(), codes.data(), codes.size());
    }
    const size_t ny = static_cast<size_t>(ntotal);
    const size_t nblocks = (ny + cache->bs_y - 1) / cache->bs_y;
    cache->offset.assign(nblocks + 1, 0);
    for (size_t block = 0; block < nblocks; ++block) {
        const size_t nj = std::min(cache->bs_y, ny - block * cache->bs_y);
        cache->offset[block + 1] = cache->offset[block] +
                fp16_gemm::packed_b_size(round_up(nj, 32), d);
    }
    cache->packed.assign(cache->offset.back(), 0);
    for (size_t block = 0; block < nblocks; ++block) {
        const size_t j0 = block * cache->bs_y;
        const size_t nj = std::min(cache->bs_y, ny - j0);
        if (d >= 32) {
            fp16_gemm::pack_centroids(
                    cache->rows.data() + j0 * d,
                    nj,
                    d,
                    round_up(nj, 32),
                    d,
                    cache->packed.data() + cache->offset[block]);
        }
    }
    if (metric_type == METRIC_L2) {
        cache->norms.assign(ny, 0);
        for (size_t j = 0; j < ny; ++j) {
            float sum = 0;
            for (int kk = 0; kk < d; ++kk) {
                const float value = decode_fp16(cache->rows[j * d + kk]);
                sum += value * value;
            }
            cache->norms[j] = sum;
        }
    }
    return cache;
}

template <class C>
void knn_gemm(
        const IndexScalarQuantizer& index,
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const fp16_gemm::PackedCodes& cache) {
    const int d = index.d;
    const idx_t ntotal = index.ntotal;
    const size_t bs_x = index.fp16_gemm_bs_x;
    const fp16_gemm::GemmKernel kernel = index.fp16_gemm_kernel;
    const MetricType metric_type = index.metric_type;
    const size_t dd = static_cast<size_t>(d);
    const size_t ny = static_cast<size_t>(ntotal);
    const size_t kk_size = static_cast<size_t>(k);
    const size_t bsx = std::max(round_up(bs_x, 32), size_t{32});
    const size_t bsy = cache.bs_y;
    const fp16_gemm::GemmFn gemm = fp16_gemm::get_gemm_fn(kernel);
    // Input is only padded to what the kernel requires: AMX needs a
    // multiple of 16 rows, FMA and scalar process rows one at a time.
    const size_t row_multiple = fp16_gemm::gemm_row_multiple(kernel);
    const bool is_l2 = metric_type == METRIC_L2;

    // Encodes queries [i0, i0 + nm) into abuf (zero-padded to mpad rows)
    // and, for L2, their squared norms into xnorm.
    auto encode_block = [&](idx_t i0,
                            size_t nm,
                            std::vector<uint16_t>& abuf,
                            std::vector<float>& xnorm) {
        const size_t mpad = round_up(nm, row_multiple);
        if (abuf.size() < mpad * dd) {
            abuf.resize(mpad * dd);
        }
        fp16_gemm::encode_rows(
                x + static_cast<size_t>(i0) * dd, abuf.data(), nm, dd);
        fp16_gemm::validate_codes(
                reinterpret_cast<const uint8_t*>(abuf.data()), nm * dd);
        std::fill(
                abuf.begin() + nm * dd, abuf.begin() + mpad * dd, uint16_t{0});
        if (is_l2) {
            if (xnorm.size() < nm) {
                xnorm.resize(nm);
            }
            for (size_t i = 0; i < nm; ++i) {
                float sum = 0;
                for (size_t kk = 0; kk < dd; ++kk) {
                    const float v = decode_fp16(abuf[i * dd + kk]);
                    sum += v * v;
                }
                xnorm[i] = sum;
            }
        }
        return mpad;
    };

    // Scores the nm encoded queries against centroids [j0, j0 + nj) and
    // leaves the nm x nj distance matrix packed (row stride nj) in ipblk.
    auto score_block = [&](const std::vector<uint16_t>& abuf,
                           size_t nm,
                           size_t mpad,
                           const std::vector<float>& xnorm,
                           size_t j0,
                           std::vector<float>& ipblk) {
        const size_t nj = std::min(bsy, ny - j0);
        if (ipblk.size() < mpad * bsy) {
            ipblk.resize(mpad * bsy);
        }
        const uint16_t* packed = cache.packed.empty()
                ? nullptr
                : cache.packed.data() + cache.offset[j0 / bsy];
        gemm(abuf.data(),
             mpad,
             dd,
             packed,
             cache.rows.data() + j0 * dd,
             dd,
             round_up(nj, 32),
             nj,
             dd,
             ipblk.data(),
             bsy);
        if (is_l2) {
            for (size_t i = 0; i < nm; ++i) {
                for (size_t j = 0; j < nj; ++j) {
                    ipblk[i * bsy + j] = std::max(
                            0.f,
                            xnorm[i] + cache.norms[j0 + j] -
                                    2.f * ipblk[i * bsy + j]);
                }
            }
        }
        if (nj != bsy) {
            for (size_t i = 1; i < nm; ++i) {
                std::memmove(
                        ipblk.data() + i * nj,
                        ipblk.data() + i * bsy,
                        nj * sizeof(float));
            }
        }
        return nj;
    };

    const size_t n_query_blocks = (static_cast<size_t>(n) + bsx - 1) / bsx;
    const size_t n_centroid_blocks = (ny + bsy - 1) / bsy;
    int max_threads = 1;
#ifdef _OPENMP
    if (!omp_in_parallel()) {
        max_threads = omp_get_max_threads();
    }
#endif
    std::exception_ptr error;
    std::atomic<bool> failed{false};
    auto record_error = [&]() {
#pragma omp critical(fp16_search_exception)
        {
            if (!error) {
                error = std::current_exception();
            }
        }
        failed.store(true, std::memory_order_relaxed);
    };

    // With fewer query blocks than threads, parallelizing over bs_x query
    // blocks alone leaves most threads idle (a batch of up to bs_x queries
    // would run on a single thread). Instead, cut the batch into tiles of
    // (group of rows) x (range of centroid blocks). Rows are only split
    // down to 32 (one full AMX block); centroids are split into as few
    // ranges as needed to keep every thread busy, because each extra range
    // keeps its own top-k, and a top-k that has seen fewer centroids admits
    // many more candidates (heap insertions, not the gemm, then dominate).
    // Single queries keep the single-threaded path, like
    // IndexScalarQuantizer::search does, so that callers already running
    // one search per thread are unaffected.
    const size_t nn = static_cast<size_t>(n);
    const size_t threads = static_cast<size_t>(max_threads);
    size_t group_rows = kFP16TileRows;
    size_t n_groups = (nn + group_rows - 1) / group_rows;
    size_t n_ranges = 1;
    if (n_groups >= threads) {
        group_rows = round_up((nn + threads - 1) / threads, kFP16TileRows);
        n_groups = (nn + group_rows - 1) / group_rows;
    } else {
        n_ranges = std::min(
                n_centroid_blocks, (threads + n_groups - 1) / n_groups);
        // Caps the per-range top-k buffers (n_ranges * n * k entries).
        n_ranges = std::min(
                n_ranges,
                std::max(size_t{1}, kMaxSplitHeapEntries / (nn * kk_size)));
    }
    const size_t n_tiles = n_groups * n_ranges;
    const bool split = n > 1 && threads > 1 && n_tiles > n_query_blocks &&
            nn * ny * dd >= kMinSplitWork;

    if (split) {
        std::vector<float> local_dis;
        std::vector<idx_t> local_ids;
        if (n_ranges > 1) {
            local_dis.resize(n_ranges * nn * kk_size);
            local_ids.resize(n_ranges * nn * kk_size);
        }
        using NativeTop1C = std::conditional_t<
                C::is_max,
                CMax<float, idx_t>,
                CMin<float, idx_t>>;
#pragma omp parallel num_threads(static_cast<int>(std::min(threads, n_tiles)))
        {
            static thread_local std::vector<uint16_t> abuf;
            static thread_local std::vector<float> ipblk;
            static thread_local std::vector<float> xnorm;
            static thread_local std::vector<std::pair<float, idx_t>> merged;
#pragma omp for schedule(dynamic, 1)
            for (size_t tile = 0; tile < n_tiles; ++tile) {
                if (failed.load(std::memory_order_relaxed)) {
                    continue;
                }
                try {
                    const size_t g = tile / n_ranges;
                    const size_t r = tile % n_ranges;
                    const size_t i0 = g * group_rows;
                    const size_t i1 = std::min(i0 + group_rows, nn);
                    const size_t nm = i1 - i0;
                    const size_t cb0 = r * n_centroid_blocks / n_ranges;
                    const size_t cb1 = (r + 1) * n_centroid_blocks / n_ranges;
                    const size_t mpad = encode_block(
                            static_cast<idx_t>(i0), nm, abuf, xnorm);
                    if (n_ranges == 1) {
                        // Whole centroid range: write the final result
                        // directly, exactly like the per-block path below.
                        Top1BlockResultHandler<NativeTop1C> top1(
                                n, distances, labels);
                        HeapBlockResultHandler<C> topk(n, distances, labels, k);
                        if (k == 1) {
                            top1.begin_multiple(i0, i1);
                        } else {
                            topk.begin_multiple(i0, i1);
                        }
                        for (size_t cb = cb0; cb < cb1; ++cb) {
                            const size_t j0 = cb * bsy;
                            const size_t nj = score_block(
                                    abuf, nm, mpad, xnorm, j0, ipblk);
                            if (k == 1) {
                                top1.add_results(j0, j0 + nj, ipblk.data());
                            } else {
                                topk.add_results(j0, j0 + nj, ipblk.data());
                            }
                        }
                        if (k == 1) {
                            top1.end_multiple();
                        } else {
                            topk.end_multiple();
                        }
                    } else {
                        HeapBlockResultHandler<C> topk(
                                n,
                                local_dis.data() + r * nn * kk_size,
                                local_ids.data() + r * nn * kk_size,
                                k);
                        topk.begin_multiple(i0, i1);
                        for (size_t cb = cb0; cb < cb1; ++cb) {
                            const size_t j0 = cb * bsy;
                            const size_t nj = score_block(
                                    abuf, nm, mpad, xnorm, j0, ipblk);
                            topk.add_results(j0, j0 + nj, ipblk.data());
                        }
                    }
                } catch (...) {
                    record_error();
                }
            }
            if (n_ranges > 1) {
                // Each range kept the exact top-k of its own centroids under
                // the (distance, smaller id first) order, so the top-k of
                // their union under that same order is exactly what a
                // single pass over all centroids would have kept.
#pragma omp for schedule(static)
                for (size_t i = 0; i < nn; ++i) {
                    if (failed.load(std::memory_order_relaxed)) {
                        continue;
                    }
                    try {
                        merged.resize(n_ranges * kk_size);
                        size_t count = 0;
                        for (size_t r = 0; r < n_ranges; ++r) {
                            const size_t base = (r * nn + i) * kk_size;
                            for (size_t j = 0; j < kk_size; ++j) {
                                if (local_ids[base + j] >= 0) {
                                    merged[count++] = {
                                            local_dis[base + j],
                                            local_ids[base + j]};
                                }
                            }
                        }
                        const size_t keep = std::min(count, kk_size);
                        std::partial_sort(
                                merged.begin(),
                                merged.begin() + keep,
                                merged.begin() + count,
                                [is_l2](const auto& a, const auto& b) {
                                    if (a.first != b.first) {
                                        return is_l2 ? a.first < b.first
                                                     : a.first > b.first;
                                    }
                                    return a.second < b.second;
                                });
                        float* ds = distances + i * kk_size;
                        idx_t* ids = labels + i * kk_size;
                        for (size_t j = 0; j < keep; ++j) {
                            ds[j] = merged[j].first;
                            ids[j] = merged[j].second;
                        }
                        for (size_t j = keep; j < kk_size; ++j) {
                            ds[j] = C::neutral();
                            ids[j] = -1;
                        }
                    } catch (...) {
                        record_error();
                    }
                }
            }
        }
        if (error) {
            std::rethrow_exception(error);
        }
        return;
    }

    const ScopedSingleThread thread_guard(n_query_blocks <= 1);
#pragma omp parallel
    {
        static thread_local std::vector<uint16_t> abuf;
        static thread_local std::vector<float> ipblk;
        static thread_local std::vector<float> xnorm;
#pragma omp for schedule(dynamic, 1)
        for (idx_t i0 = 0; i0 < n; i0 += static_cast<idx_t>(bsx)) {
            if (failed.load(std::memory_order_relaxed)) {
                continue;
            }
            try {
                const idx_t i1 = std::min<idx_t>(i0 + bsx, n);
                const size_t nm = static_cast<size_t>(i1 - i0);
                const size_t mpad = encode_block(i0, nm, abuf, xnorm);
                using NativeTop1C = std::conditional_t<
                        C::is_max,
                        CMax<float, idx_t>,
                        CMin<float, idx_t>>;
                Top1BlockResultHandler<NativeTop1C> top1(n, distances, labels);
                HeapBlockResultHandler<C> topk(n, distances, labels, k);
                if (k == 1) {
                    top1.begin_multiple(i0, i1);
                } else {
                    topk.begin_multiple(i0, i1);
                }
                for (size_t j0 = 0; j0 < ny; j0 += bsy) {
                    const size_t nj =
                            score_block(abuf, nm, mpad, xnorm, j0, ipblk);
                    if (k == 1) {
                        top1.add_results(j0, j0 + nj, ipblk.data());
                    } else {
                        topk.add_results(j0, j0 + nj, ipblk.data());
                    }
                }
                if (k == 1) {
                    top1.end_multiple();
                } else {
                    topk.end_multiple();
                }
            } catch (...) {
                record_error();
            }
        }
    }
    if (error) {
        std::rethrow_exception(error);
    }
}

void rescore(
        const IndexScalarQuantizer& index,
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) {
    const int d = index.d;
    const ScalarQuantizer& sq = index.sq;
    const MetricType metric_type = index.metric_type;
    const auto& codes = index.codes;
    const size_t code_size = index.code_size;
    const ScopedSingleThread thread_guard(n <= 1);
#pragma omp parallel
    {
        std::unique_ptr<ScalarQuantizer::SQDistanceComputer> dc(
                sq.get_distance_computer(metric_type));
        std::vector<float> rounded(static_cast<size_t>(d));
#pragma omp for schedule(static)
        for (idx_t i = 0; i < n; ++i) {
            fp16_gemm::round_values(
                    x + static_cast<size_t>(i) * d, rounded.data(), d);
            dc->set_query(rounded.data());
            for (idx_t j = 0; j < k; ++j) {
                const size_t offset = static_cast<size_t>(i) * k + j;
                if (labels[offset] >= 0) {
                    distances[offset] = dc->query_to_code(
                            codes.data() +
                            static_cast<size_t>(labels[offset]) * code_size);
                }
            }
        }
    }
}

} // namespace

void IndexScalarQuantizer::invalidate_fp16_gemm_cache() {
    std::lock_guard<std::mutex> lock(fp16_gemm_state_.mutex);
    fp16_gemm_state_.packed.reset();
}

fp16_gemm::GemmKernel IndexScalarQuantizer::fp16_gemm_effective_kernel() const {
    return fp16_gemm::resolve_kernel(fp16_gemm_kernel);
}

size_t IndexScalarQuantizer::fp16_gemm_packed_bytes() const {
    std::lock_guard<std::mutex> lock(fp16_gemm_state_.mutex);
    const auto& cache = fp16_gemm_state_.packed;
    return cache ? (cache->rows.size() + cache->packed.size()) * 2 +
                    cache->norms.size() * sizeof(float)
                 : 0;
}

size_t IndexScalarQuantizer::fp16_gemm_nb_queries() const {
    return fp16_gemm_state_.nb_queries.load(std::memory_order_relaxed);
}

void IndexScalarQuantizer::search_fp16_gemm(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    check_format(*this);
    const size_t count = input_size(n, d);
    FAISS_THROW_IF_NOT(k > 0 && is_trained);
    fp16_gemm::resolve_kernel(fp16_gemm_kernel);
    if (n == 0) {
        return;
    }
    FAISS_THROW_IF_NOT_MSG(
            !(params && params->sel),
            "fp16 gemm search does not support an ID selector");
    const bool is_l2 = metric_type == METRIC_L2;
    if (ntotal == 0) {
        // Still reject queries that do not fit in fp16.
        // Not std::vector: it would zero the whole buffer single-threaded
        // first; let each thread's chunk write it on first use instead.
        std::unique_ptr<float[]> rounded(new float[count]);
        fp16_gemm::round_values_parallel(x, rounded.get(), count);
        const size_t nk = static_cast<size_t>(n) * static_cast<size_t>(k);
        std::fill(
                distances,
                distances + nk,
                is_l2 ? std::numeric_limits<float>::infinity()
                      : -std::numeric_limits<float>::infinity());
        std::fill(labels, labels + nk, idx_t(-1));
        return;
    }
    std::shared_ptr<const fp16_gemm::PackedCodes> cache;
    {
        const size_t bsy = std::max(round_up(fp16_gemm_bs_y, 32), size_t{32});
        std::lock_guard<std::mutex> lock(fp16_gemm_state_.mutex);
        auto& packed = fp16_gemm_state_.packed;
        if (!packed || packed->bs_y != bsy || packed->d != d ||
            packed->ntotal != ntotal || packed->metric != metric_type ||
            packed->source_codes != codes.data()) {
            packed = build_pack(*this);
        }
        cache = packed;
    }
    if (metric_type == METRIC_INNER_PRODUCT) {
        knn_gemm<IPHeap>(*this, n, x, k, distances, labels, *cache);
    } else {
        knn_gemm<L2Heap>(*this, n, x, k, distances, labels, *cache);
    }
    fp16_gemm_state_.nb_queries.fetch_add(
            static_cast<size_t>(n), std::memory_order_relaxed);
    if (fp16_gemm_rescore) {
        rescore(*this, n, x, k, distances, labels);
    }
    sort_selected(n, k, distances, labels, is_l2);
}

} // namespace faiss
