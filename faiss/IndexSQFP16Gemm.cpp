/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexSQFP16Gemm.h>

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

} // namespace

IndexSQFP16Gemm::IndexSQFP16Gemm(int dim, MetricType metric)
        : IndexScalarQuantizer(dim, ScalarQuantizer::QT_fp16, metric) {
    check_format();
}

IndexSQFP16Gemm::IndexSQFP16Gemm()
        : IndexScalarQuantizer(0, ScalarQuantizer::QT_fp16) {}

void IndexSQFP16Gemm::check_format() const {
    FAISS_THROW_IF_NOT_MSG(
            d > 0 && sq.d == static_cast<size_t>(d),
            "fp16 gemm index dimension mismatch");
    FAISS_THROW_IF_NOT_MSG(
            sq.qtype == ScalarQuantizer::QT_fp16 &&
                    code_size == static_cast<size_t>(d) * 2 &&
                    sq.code_size == code_size,
            "fp16 gemm index format or code size mismatch");
    FAISS_THROW_IF_NOT_MSG(
            metric_type == METRIC_L2 || metric_type == METRIC_INNER_PRODUCT,
            "fp16 gemm index only supports inner product and L2");
    FAISS_THROW_IF_NOT_MSG(
            ntotal >= 0 &&
                    codes.size() == static_cast<size_t>(ntotal) * code_size,
            "fp16 gemm index code count mismatch");
}

void IndexSQFP16Gemm::train(idx_t n, const float* x) {
    check_format();
    std::vector<float> rounded(input_size(n, d));
    fp16_gemm::round_values(x, rounded.data(), rounded.size());
    IndexScalarQuantizer::train(n, rounded.data());
}

void IndexSQFP16Gemm::sa_encode(idx_t n, const float* x, uint8_t* bytes) const {
    check_format();
    const size_t count = input_size(n, d);
    std::vector<uint16_t> values(count);
    fp16_gemm::encode_array(x, values.data(), count);
    fp16_gemm::validate_codes(
            reinterpret_cast<const uint8_t*>(values.data()), count);
    if (count) {
        std::memcpy(bytes, values.data(), count * sizeof(uint16_t));
    }
}

void IndexSQFP16Gemm::add(idx_t n, const float* x) {
    check_format();
    const size_t count = input_size(n, d);
    // Encode before touching the index so that invalid input never leaves
    // behind a half-finished add.
    std::vector<uint16_t> values(count);
    fp16_gemm::encode_array(x, values.data(), count);
    fp16_gemm::validate_codes(
            reinterpret_cast<const uint8_t*>(values.data()), count);
    IndexScalarQuantizer::add_sa_codes(
            n, reinterpret_cast<const uint8_t*>(values.data()), nullptr);
    repack();
}

void IndexSQFP16Gemm::reset() {
    IndexScalarQuantizer::reset();
    invalidate_pack();
}

void IndexSQFP16Gemm::add_sa_codes(
        idx_t n,
        const uint8_t* values,
        const idx_t* xids) {
    check_format();
    fp16_gemm::validate_codes(values, input_size(n, d));
    IndexScalarQuantizer::add_sa_codes(n, values, xids);
    repack();
}

size_t IndexSQFP16Gemm::remove_ids(const IDSelector& selector) {
    check_format();
    const size_t removed = IndexScalarQuantizer::remove_ids(selector);
    repack();
    return removed;
}

void IndexSQFP16Gemm::merge_from(Index& other, idx_t add_id) {
    check_format();
    const auto* sq_other = dynamic_cast<const IndexSQFP16Gemm*>(&other);
    FAISS_THROW_IF_NOT_MSG(
            sq_other && sq_other->metric_type == metric_type,
            "fp16 gemm merge index type or metric mismatch");
    sq_other->check_format();
    fp16_gemm::validate_codes(
            sq_other->codes.data(), static_cast<size_t>(sq_other->ntotal) * d);
    IndexScalarQuantizer::merge_from(other, add_id);
    repack();
}

void IndexSQFP16Gemm::invalidate_pack() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    cache_.reset();
}

void IndexSQFP16Gemm::repack() {
    check_format();
    std::lock_guard<std::mutex> lock(cache_mutex_);
    cache_ = ntotal && use_gemm ? build_pack() : nullptr;
}

fp16_gemm::GemmKernel IndexSQFP16Gemm::effective_kernel() const {
    return use_gemm ? fp16_gemm::resolve_kernel(kernel)
                    : fp16_gemm::GemmKernel::SCALAR;
}

size_t IndexSQFP16Gemm::packed_bytes() const {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    return cache_ ? (cache_->rows.size() + cache_->packed.size()) * 2 +
                    cache_->norms.size() * sizeof(float)
                  : 0;
}

std::shared_ptr<const IndexSQFP16Gemm::PackedCache> IndexSQFP16Gemm::
        ensure_packed() const {
    check_format();
    const size_t bsy = std::max(round_up(bs_y, 32), size_t{32});
    std::lock_guard<std::mutex> lock(cache_mutex_);
    if (!cache_ || cache_->bs_y != bsy || cache_->d != d ||
        cache_->ntotal != ntotal || cache_->metric != metric_type ||
        cache_->source_codes != codes.data()) {
        cache_ = build_pack();
    }
    return cache_;
}

std::shared_ptr<const IndexSQFP16Gemm::PackedCache> IndexSQFP16Gemm::
        build_pack() const {
    fp16_gemm::validate_codes(codes.data(), static_cast<size_t>(ntotal) * d);
    auto cache = std::make_shared<PackedCache>();
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
void IndexSQFP16Gemm::knn_gemm(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const PackedCache& cache) const {
    const size_t dd = static_cast<size_t>(d);
    const size_t ny = static_cast<size_t>(ntotal);
    const size_t bsx = std::max(round_up(bs_x, 32), size_t{32});
    const size_t bsy = cache.bs_y;
    const fp16_gemm::GemmFn gemm = fp16_gemm::get_gemm_fn(kernel);
    // Input is only padded to what the kernel requires: AMX needs a
    // multiple of 16 rows, FMA and scalar process rows one at a time.
    const size_t row_multiple = fp16_gemm::gemm_row_multiple(kernel);
    const bool is_l2 = metric_type == METRIC_L2;
    const ScopedSingleThread thread_guard(n <= static_cast<idx_t>(bsx));
    std::exception_ptr error;
    std::atomic<bool> failed{false};
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
                if (abuf.size() < bsx * dd) {
                    abuf.resize(bsx * dd);
                }
                if (ipblk.size() < bsx * bsy) {
                    ipblk.resize(bsx * bsy);
                }
                if (is_l2 && xnorm.size() < bsx) {
                    xnorm.resize(bsx);
                }
                const idx_t i1 = std::min<idx_t>(i0 + bsx, n);
                const size_t nm = static_cast<size_t>(i1 - i0);
                const size_t mpad = round_up(nm, row_multiple);
                fp16_gemm::encode_rows(
                        x + static_cast<size_t>(i0) * dd, abuf.data(), nm, dd);
                fp16_gemm::validate_codes(
                        reinterpret_cast<const uint8_t*>(abuf.data()), nm * dd);
                std::fill(
                        abuf.begin() + nm * dd,
                        abuf.begin() + mpad * dd,
                        uint16_t{0});
                if (is_l2) {
                    for (size_t i = 0; i < nm; ++i) {
                        float sum = 0;
                        for (size_t kk = 0; kk < dd; ++kk) {
                            const float v = decode_fp16(abuf[i * dd + kk]);
                            sum += v * v;
                        }
                        xnorm[i] = sum;
                    }
                }
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
                    const size_t nj = std::min(bsy, ny - j0);
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
#pragma omp critical(fp16_search_exception)
                {
                    if (!error) {
                        error = std::current_exception();
                    }
                }
                failed.store(true, std::memory_order_relaxed);
            }
        }
    }
    if (error) {
        std::rethrow_exception(error);
    }
}

void IndexSQFP16Gemm::rescore(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const {
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

void IndexSQFP16Gemm::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    check_format();
    input_size(n, d);
    FAISS_THROW_IF_NOT(k > 0 && is_trained);
    if (use_gemm) {
        fp16_gemm::resolve_kernel(kernel);
    }
    if (n == 0) {
        return;
    }
    FAISS_THROW_IF_NOT_MSG(
            !(use_gemm && params && params->sel),
            "fp16 gemm search does not support an ID selector");
    if (!use_gemm || ntotal == 0) {
        // Not std::vector: it would zero the whole buffer single-threaded
        // first (about 1s for a million queries); allocate directly and
        // let each thread's chunk write it on first use instead.
        const size_t count = input_size(n, d);
        std::unique_ptr<float[]> rounded(new float[count]);
        fp16_gemm::round_values_parallel(x, rounded.get(), count);
        IndexScalarQuantizer::search(
                n, rounded.get(), k, distances, labels, params);
    } else {
        const auto cache = ensure_packed();
        if (metric_type == METRIC_INNER_PRODUCT) {
            knn_gemm<IPHeap>(n, x, k, distances, labels, *cache);
        } else {
            knn_gemm<L2Heap>(n, x, k, distances, labels, *cache);
        }
        nb_gemm_queries.fetch_add(
                static_cast<size_t>(n), std::memory_order_relaxed);
        if (rescore_native) {
            rescore(n, x, k, distances, labels);
        }
    }
    sort_selected(n, k, distances, labels, metric_type == METRIC_L2);
}

} // namespace faiss
