/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexPiPNN.h>

#include <omp.h>
#include <algorithm>
#include <atomic>
#include <exception>
#include <memory>
#include <string>
#include <typeinfo>

#include <faiss/IndexFlat.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/FaissException.h>
#include <faiss/impl/VisitedTable.h>

namespace faiss {

IndexPiPNN::IndexPiPNN() = default;

IndexPiPNN::IndexPiPNN(Index* storage_in, int R)
        : pipnn(R), storage(storage_in) {
    FAISS_THROW_IF_NOT_MSG(storage, "IndexPiPNN has no storage");
    check_storage_supported(*storage);
    d = storage->d;
    metric_type = storage->metric_type;
    is_trained = storage->is_trained;
    ntotal = storage->ntotal;
}

IndexPiPNN::~IndexPiPNN() {
    if (own_fields) {
        delete storage;
    }
}

void IndexPiPNN::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(storage, "IndexPiPNN has no storage");
    storage->train(n, x);
    is_trained = storage->is_trained;
}

void IndexPiPNN::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(storage, "IndexPiPNN has no storage");
    FAISS_THROW_IF_MSG(
            is_built(), "IndexPiPNN does not support incremental addition");
    storage->add(n, x);
    ntotal = storage->ntotal;
    if (!defer_build && ntotal > 0) {
        build();
    }
}

void IndexPiPNN::build() {
    FAISS_THROW_IF_NOT_MSG(storage, "IndexPiPNN has no storage");
    check_storage_supported(*storage);
    FAISS_THROW_IF_NOT_MSG(
            storage->ntotal > 0, "IndexPiPNN: cannot build over 0 vectors");
    pipnn.build(*storage, verbose);
    ntotal = storage->ntotal;
}

void IndexPiPNN::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT_MSG(storage, "IndexPiPNN has no storage");
    FAISS_THROW_IF_NOT_MSG(
            storage->is_trained,
            "IndexPiPNN: storage must be trained before search()");
    FAISS_THROW_IF_NOT_MSG(
            pipnn.ntotal == ntotal && storage->ntotal == ntotal,
            "IndexPiPNN: the index is not built over its storage; call "
            "build()");
    const auto* pipnn_params =
            dynamic_cast<const SearchParametersPiPNN*>(params);
    const int search_L = pipnn_params && pipnn_params->search_L != 0
            ? pipnn_params->search_L
            : pipnn.search_L;
    const IDSelector* sel = params ? params->sel : nullptr;
    const bool negate = metric_type == METRIC_INNER_PRODUCT;

    const idx_t check_period =
            InterruptCallback::get_period_hint(
                    size_t(d) * size_t(std::max(idx_t(search_L), k))) *
            omp_get_max_threads();

    for (idx_t i0 = 0; i0 < n; i0 += check_period) {
        const idx_t i1 = std::min(i0 + check_period, n);
        std::exception_ptr ex;
        std::atomic<bool> interrupt{false};

#pragma omp parallel if (i1 - i0 > 1)
        {
            VisitedTable* vt = nullptr;
            std::unique_ptr<DistanceComputer> dis;
            try {
                vt = &VisitedTable::get_reusable(ntotal, use_visited_hashset);
                dis.reset(
                        negate ? new NegativeDistanceComputer(
                                         storage->get_distance_computer())
                               : storage->get_distance_computer());
            } catch (...) {
                omp_capture_exception(ex, [&] { interrupt = true; });
            }

#pragma omp for schedule(guided)
            for (idx_t i = i0; i < i1; i++) {
                if (interrupt.load(std::memory_order_relaxed)) {
                    continue;
                }
                try {
                    dis->set_query(x + i * d);
                    pipnn.search(
                            *dis,
                            search_L,
                            k,
                            labels + i * k,
                            distances + i * k,
                            *vt,
                            sel);
                } catch (...) {
                    omp_capture_exception(ex, [&] { interrupt = true; });
                }
            }
        }
        omp_rethrow_if_exception(ex);
        InterruptCallback::check();
    }

    if (negate) {
        for (idx_t i = 0; i < n * k; i++) {
            distances[i] = -distances[i];
        }
    }
}

void IndexPiPNN::reconstruct(idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT_MSG(storage, "IndexPiPNN has no storage");
    storage->reconstruct(key, recons);
}

void IndexPiPNN::reset() {
    FAISS_THROW_IF_NOT_MSG(storage, "IndexPiPNN has no storage");
    storage->reset();
    pipnn.reset();
    ntotal = 0;
}

void IndexPiPNN::check_storage_supported(const Index& storage) {
    const std::type_info& t = typeid(storage);
    bool ok = t == typeid(IndexFlat) || t == typeid(IndexFlatL2) ||
            t == typeid(IndexFlatIP);
    if (t == typeid(IndexScalarQuantizer)) {
        const auto qt =
                static_cast<const IndexScalarQuantizer&>(storage).sq.qtype;
        ok = qt != ScalarQuantizer::QT_0bit &&
                !ScalarQuantizer::TurboQuantRefine::is_turboq_full(qt);
    }
    ok = ok &&
            (storage.metric_type == METRIC_L2 ||
             storage.metric_type == METRIC_INNER_PRODUCT);
    FAISS_THROW_IF_NOT_FMT(
            ok,
            "IndexPiPNN: unsupported storage %s; use IndexFlat or "
            "IndexScalarQuantizer (not QT_0bit or QT_*bit_tq), L2 or IP",
            demangle_cpp_symbol(t.name()).c_str());
}

IndexPiPNNFlat::IndexPiPNNFlat() = default;

IndexPiPNNFlat::IndexPiPNNFlat(int d_in, int R, MetricType metric)
        : IndexPiPNN(new IndexFlat(d_in, metric), R) {
    own_fields = true;
}

IndexPiPNNSQ::IndexPiPNNSQ() = default;

IndexPiPNNSQ::IndexPiPNNSQ(
        int d_in,
        ScalarQuantizer::QuantizerType qtype,
        int R,
        MetricType metric)
        : IndexPiPNN(new IndexScalarQuantizer(d_in, qtype, metric), R) {
    own_fields = true;
}

} // namespace faiss
