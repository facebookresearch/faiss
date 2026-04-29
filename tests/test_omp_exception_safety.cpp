/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <atomic>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <faiss/Index2Layer.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexFlatCodes.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexNNDescent.h>
#include <faiss/IndexNSG.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissException.h>
#include <faiss/invlists/InvertedLists.h>

using namespace faiss;

namespace {

// InvertedLists wrapper that delegates to an underlying InvertedLists but
// throws from get_codes() after a configurable number of successful calls.
// This injects exceptions inside scan_one_list / scan_list_func, which
// execute inside OMP worksharing constructs.
struct ThrowingInvertedLists : InvertedLists {
    const InvertedLists* delegate;
    mutable std::atomic<int> calls_until_throw;

    ThrowingInvertedLists(const InvertedLists* delegate_in, int throw_after)
            : InvertedLists(delegate_in->nlist, delegate_in->code_size),
              delegate(delegate_in),
              calls_until_throw(throw_after) {}

    size_t list_size(size_t list_no) const override {
        return delegate->list_size(list_no);
    }

    const uint8_t* get_codes(size_t list_no) const override {
        if (calls_until_throw.fetch_sub(1) <= 0) {
            FAISS_THROW_MSG("injected get_codes failure");
        }
        return delegate->get_codes(list_no);
    }

    const idx_t* get_ids(size_t list_no) const override {
        return delegate->get_ids(list_no);
    }

    size_t add_entries(size_t, size_t, const idx_t*, const uint8_t*) override {
        FAISS_THROW_MSG("unexpected call");
    }

    void update_entries(size_t, size_t, size_t, const idx_t*, const uint8_t*)
            override {
        FAISS_THROW_MSG("unexpected call");
    }

    void resize(size_t, size_t) override {
        FAISS_THROW_MSG("unexpected call");
    }
};

// InterruptCallback that always signals an interrupt.
struct AlwaysInterrupt : InterruptCallback {
    bool want_interrupt() override {
        return true;
    }
};

// Build a trained IndexIVFFlat with data added, using enough vectors and
// lists to exercise multi-threaded parallel modes.
struct IVFFixture {
    static constexpr int d = 8;
    static constexpr int nb = 2000;
    static constexpr int nlist = 16;
    static constexpr int nq = 64;

    IndexFlatL2 quantizer{d};
    IndexIVFFlat index{&quantizer, d, nlist};
    std::vector<float> xq;

    IVFFixture() {
        index.own_fields = false;
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist;

        std::vector<float> train(nlist * 40 * d);
        for (auto& v : train) {
            v = dist(rng);
        }
        index.train(train.size() / d, train.data());

        std::vector<float> xb(nb * d);
        for (auto& v : xb) {
            v = dist(rng);
        }
        index.add(nb, xb.data());

        xq.resize(nq * d);
        for (auto& v : xq) {
            v = dist(rng);
        }
    }

    // Swap in a ThrowingInvertedLists that delegates to the real one.
    // Returns the throwing wrapper. The real inverted lists is kept alive
    // (disowned from the index) so the delegate pointer stays valid.
    std::unique_ptr<ThrowingInvertedLists> install_throwing(int throw_after) {
        auto* real = index.invlists;
        auto throwing =
                std::make_unique<ThrowingInvertedLists>(real, throw_after);
        // Disown so replace_invlists doesn't delete the real lists.
        index.own_invlists = false;
        index.replace_invlists(throwing.get(), false);
        return throwing;
    }

    // Restore the original inverted lists (the delegate from the wrapper).
    void restore(const ThrowingInvertedLists& throwing) {
        index.replace_invlists(
                const_cast<InvertedLists*>(throwing.delegate), true);
    }
};

} // namespace

// ---------------------------------------------------------------------------
// search_preassigned: exception in scan_one_list, pmode 0
// ---------------------------------------------------------------------------
TEST(OMPExceptionSafety, search_preassigned_pmode0) {
    IVFFixture f;
    f.index.parallel_mode = 0;
    f.index.nprobe = 4;

    auto throwing = f.install_throwing(2);

    idx_t k = 4;
    std::vector<float> dis(f.nq * k);
    std::vector<idx_t> labels(f.nq * k);

    EXPECT_THROW(
            f.index.search(f.nq, f.xq.data(), k, dis.data(), labels.data()),
            FaissException);

    f.restore(*throwing);
}

// ---------------------------------------------------------------------------
// search_preassigned: exception in scan_one_list, pmode 1
// ---------------------------------------------------------------------------
TEST(OMPExceptionSafety, search_preassigned_pmode1) {
    IVFFixture f;
    f.index.parallel_mode = 1;
    f.index.nprobe = 8;

    auto throwing = f.install_throwing(2);

    idx_t k = 4;
    std::vector<float> dis(f.nq * k);
    std::vector<idx_t> labels(f.nq * k);

    EXPECT_THROW(
            f.index.search(f.nq, f.xq.data(), k, dis.data(), labels.data()),
            FaissException);

    f.restore(*throwing);
}

// ---------------------------------------------------------------------------
// search_preassigned: exception in scan_one_list, pmode 2
// ---------------------------------------------------------------------------
TEST(OMPExceptionSafety, search_preassigned_pmode2) {
    IVFFixture f;
    f.index.parallel_mode = 2;
    f.index.nprobe = 8;

    auto throwing = f.install_throwing(2);

    idx_t k = 4;
    std::vector<float> dis(f.nq * k);
    std::vector<idx_t> labels(f.nq * k);

    EXPECT_THROW(
            f.index.search(f.nq, f.xq.data(), k, dis.data(), labels.data()),
            FaissException);

    f.restore(*throwing);
}

// ---------------------------------------------------------------------------
// search_preassigned: InterruptCallback::check() throws inside pmode 0 loop
// ---------------------------------------------------------------------------
TEST(OMPExceptionSafety, search_interrupt_pmode0) {
    IVFFixture f;
    f.index.parallel_mode = 0;
    f.index.nprobe = 4;

    idx_t k = 4;
    std::vector<float> dis(f.nq * k);
    std::vector<idx_t> labels(f.nq * k);

    InterruptCallback::instance.reset(new AlwaysInterrupt());

    EXPECT_THROW(
            f.index.search(f.nq, f.xq.data(), k, dis.data(), labels.data()),
            FaissException);

    InterruptCallback::clear_instance();
}

// ---------------------------------------------------------------------------
// range_search_preassigned: exception in scan_list_func, pmode 0
// ---------------------------------------------------------------------------
TEST(OMPExceptionSafety, range_search_pmode0) {
    IVFFixture f;
    f.index.parallel_mode = 0;
    f.index.nprobe = 4;

    auto throwing = f.install_throwing(2);
    RangeSearchResult result(f.nq);

    EXPECT_THROW(
            f.index.range_search(
                    f.nq,
                    f.xq.data(),
                    std::numeric_limits<float>::max(),
                    &result),
            FaissException);

    f.restore(*throwing);
}

// ---------------------------------------------------------------------------
// range_search_preassigned: exception in scan_list_func, pmode 1
// ---------------------------------------------------------------------------
TEST(OMPExceptionSafety, range_search_pmode1) {
    IVFFixture f;
    f.index.parallel_mode = 1;
    f.index.nprobe = 8;

    auto throwing = f.install_throwing(2);
    RangeSearchResult result(f.nq);

    EXPECT_THROW(
            f.index.range_search(
                    f.nq,
                    f.xq.data(),
                    std::numeric_limits<float>::max(),
                    &result),
            FaissException);

    f.restore(*throwing);
}

// ---------------------------------------------------------------------------
// range_search_preassigned: exception in scan_list_func, pmode 2
// ---------------------------------------------------------------------------
TEST(OMPExceptionSafety, range_search_pmode2) {
    IVFFixture f;
    f.index.parallel_mode = 2;
    f.index.nprobe = 8;

    auto throwing = f.install_throwing(2);
    RangeSearchResult result(f.nq);

    EXPECT_THROW(
            f.index.range_search(
                    f.nq,
                    f.xq.data(),
                    std::numeric_limits<float>::max(),
                    &result),
            FaissException);

    f.restore(*throwing);
}

// ---------------------------------------------------------------------------
// search (top-level OMP slicing): exception propagates from sub_search_func
// ---------------------------------------------------------------------------
TEST(OMPExceptionSafety, search_top_level_omp_slicing) {
    IVFFixture f;
    f.index.parallel_mode = 0;
    f.index.nprobe = 4;

    // Allow a few successful calls so some slices start, then fail.
    auto throwing = f.install_throwing(5);

    idx_t k = 4;
    std::vector<float> dis(f.nq * k);
    std::vector<idx_t> labels(f.nq * k);

    EXPECT_THROW(
            f.index.search(f.nq, f.xq.data(), k, dis.data(), labels.data()),
            FaissException);

    f.restore(*throwing);
}

// Minimal IndexFlatCodes subclass whose sa_decode always throws, to
// verify that exceptions thrown inside the OpenMP parallel region in
// IndexFlatCodes::search are propagated to the caller.
struct ThrowingIndex : IndexFlatCodes {
    explicit ThrowingIndex(int d)
            : IndexFlatCodes(sizeof(float) * d, d, METRIC_L2) {
        ntotal = 1;
        is_trained = true;
        codes.resize(code_size, 0);
    }

    void sa_decode(idx_t /*n*/, const uint8_t* /*codes*/, float* /*x*/)
            const override {
        throw std::runtime_error("corrupt index");
    }
};

// ---------------------------------------------------------------------------
// IndexFlatCodes::search: exception in OMP worker propagates to caller
// ---------------------------------------------------------------------------
TEST(OMPExceptionSafety, flatcodes_search) {
    ThrowingIndex index(4);

    std::vector<float> xq(4, 0.0f);
    std::vector<float> distances(1);
    std::vector<idx_t> labels(1);

    EXPECT_THROW(
            index.search(1, xq.data(), 1, distances.data(), labels.data()),
            std::runtime_error);
}

// ---------------------------------------------------------------------------
// IndexNNDescent::search: exception in OMP worker propagates to caller.
// Constructing with has_built=false triggers a FaissException inside the
// worksharing loop body.
// ---------------------------------------------------------------------------
TEST(OMPExceptionSafety, nndescent_search) {
    int d = 4;
    auto storage = std::make_unique<IndexFlatL2>(d);
    std::vector<float> xb(d, 1.0f);
    storage->add(1, xb.data());

    IndexNNDescent index(storage.get(), 4);
    index.ntotal = 1;
    // has_built defaults to false, so nndescent.search() will throw.

    std::vector<float> xq(d, 0.0f);
    std::vector<float> distances(1);
    std::vector<idx_t> labels(1);

    EXPECT_THROW(
            index.search(1, xq.data(), 1, distances.data(), labels.data()),
            FaissException);
}

// ---------------------------------------------------------------------------
// IndexNSG::search: exception in OMP worker propagates to caller
// ---------------------------------------------------------------------------
TEST(OMPExceptionSafety, nsg_search) {
    int d = 4;
    auto storage = std::make_unique<IndexFlatL2>(d);
    std::vector<float> xb(d, 1.0f);
    storage->add(1, xb.data());

    IndexNSG index(storage.get(), 4);
    index.ntotal = 1;
    // nsg graph is not built, so nsg.search() will throw.

    std::vector<float> xq(d, 0.0f);
    std::vector<float> distances(1);
    std::vector<idx_t> labels(1);

    EXPECT_THROW(
            index.search(1, xq.data(), 1, distances.data(), labels.data()),
            FaissException);
}

// ---------------------------------------------------------------------------
// IndexHNSW OMP exception safety tests.
//
// There are 3 OMP regions wrapped with exception capture in IndexHNSW.cpp:
//   1. hnsw_search()             — primary search path
//   2. IndexHNSW::search_level_0 — level-0 search
//   3. IndexHNSW2Level::search   — mixed IVFPQ+HNSW path
//
// Each region has two throw locations:
//   (a) Init phase  — DistanceComputer/VisitedTable construction
//   (b) Loop body   — set_query, graph traversal, etc.
// ---------------------------------------------------------------------------

// DistanceComputer that throws from set_query(). Exercises loop-body
// try/catch after successful init-phase construction.
struct ThrowingDC : DistanceComputer {
    void set_query(const float*) override {
        throw std::runtime_error("corrupt storage");
    }
    float operator()(idx_t) override {
        return 0;
    }
    float symmetric_dis(idx_t, idx_t) override {
        return 0;
    }
};

// Storage whose get_distance_computer() returns a ThrowingDC.
// Exercises loop-body throws (DC construction succeeds, set_query throws).
struct LoopThrowingStorage : IndexFlatL2 {
    using IndexFlatL2::IndexFlatL2;
    DistanceComputer* get_distance_computer() const override {
        return new ThrowingDC();
    }
};

// Storage whose get_distance_computer() itself throws.
// Exercises init-phase throws (DC construction fails).
struct InitThrowingStorage : IndexFlatL2 {
    using IndexFlatL2::IndexFlatL2;
    DistanceComputer* get_distance_computer() const override {
        throw std::runtime_error("corrupt storage init");
    }
};

// Helper: build a valid IndexHNSWFlat with data, then swap storage.
struct HNSWFixture {
    static constexpr int d = 4;
    static constexpr int nb = 10;
    static constexpr int M = 16;

    IndexHNSWFlat index{d, M};
    std::vector<float> xb;
    std::vector<float> xq;

    HNSWFixture() : xb(d * nb), xq(d) {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist;
        for (auto& v : xb) {
            v = dist(rng);
        }
        index.add(nb, xb.data());
        for (auto& v : xq) {
            v = dist(rng);
        }
    }

    void swap_storage(Index* new_storage) {
        index.own_fields = false;
        index.storage = new_storage;
    }

    void restore_storage(Index* old_storage) {
        index.storage = old_storage;
        index.own_fields = true;
    }
};

// ---------------------------------------------------------------------------
// hnsw_search() init-phase throw: storage_distance_computer() throws
// ---------------------------------------------------------------------------
TEST(OMPExceptionSafety, hnsw_search_init) {
    HNSWFixture f;
    auto throwing = std::make_unique<InitThrowingStorage>(f.d);
    throwing->add(f.nb, f.xb.data());
    auto* old = f.index.storage;
    f.swap_storage(throwing.get());

    std::vector<float> distances(1);
    std::vector<idx_t> labels(1);
    EXPECT_THROW(
            f.index.search(1, f.xq.data(), 1, distances.data(), labels.data()),
            std::runtime_error);

    f.restore_storage(old);
}

// ---------------------------------------------------------------------------
// hnsw_search() loop-body throw: set_query() throws
// ---------------------------------------------------------------------------
TEST(OMPExceptionSafety, hnsw_search_loop) {
    HNSWFixture f;
    auto throwing = std::make_unique<LoopThrowingStorage>(f.d);
    throwing->add(f.nb, f.xb.data());
    auto* old = f.index.storage;
    f.swap_storage(throwing.get());

    std::vector<float> distances(1);
    std::vector<idx_t> labels(1);
    EXPECT_THROW(
            f.index.search(1, f.xq.data(), 1, distances.data(), labels.data()),
            std::runtime_error);

    f.restore_storage(old);
}

// ---------------------------------------------------------------------------
// search_level_0() init-phase throw: storage_distance_computer() throws
// ---------------------------------------------------------------------------
TEST(OMPExceptionSafety, hnsw_search_level_0_init) {
    HNSWFixture f;
    auto throwing = std::make_unique<InitThrowingStorage>(f.d);
    throwing->add(f.nb, f.xb.data());
    auto* old = f.index.storage;
    f.swap_storage(throwing.get());

    idx_t k = 1;
    int nprobe = 1;
    std::vector<HNSW::storage_idx_t> nearest(nprobe, 0);
    std::vector<float> nearest_d(nprobe, 0.0f);
    std::vector<float> distances(k);
    std::vector<idx_t> labels(k);

    EXPECT_THROW(
            f.index.search_level_0(
                    1,
                    f.xq.data(),
                    k,
                    nearest.data(),
                    nearest_d.data(),
                    distances.data(),
                    labels.data(),
                    nprobe),
            std::runtime_error);

    f.restore_storage(old);
}

// ---------------------------------------------------------------------------
// search_level_0() loop-body throw: set_query() throws
// ---------------------------------------------------------------------------
TEST(OMPExceptionSafety, hnsw_search_level_0_loop) {
    HNSWFixture f;
    auto throwing = std::make_unique<LoopThrowingStorage>(f.d);
    throwing->add(f.nb, f.xb.data());
    auto* old = f.index.storage;
    f.swap_storage(throwing.get());

    idx_t k = 1;
    int nprobe = 1;
    std::vector<HNSW::storage_idx_t> nearest(nprobe, 0);
    std::vector<float> nearest_d(nprobe, 0.0f);
    std::vector<float> distances(k);
    std::vector<idx_t> labels(k);

    EXPECT_THROW(
            f.index.search_level_0(
                    1,
                    f.xq.data(),
                    k,
                    nearest.data(),
                    nearest_d.data(),
                    distances.data(),
                    labels.data(),
                    nprobe),
            std::runtime_error);

    f.restore_storage(old);
}

// IndexIVFPQ subclass with a counted throw mechanism. The first
// `throw_after` calls to get_distance_computer() delegate to the
// real implementation; subsequent calls throw (init) or return
// ThrowingDC (loop). This lets search_preassigned succeed while
// the HNSW2Level OMP region's storage_distance_computer() fails.
struct CountedThrowIVFPQ : IndexIVFPQ {
    mutable std::atomic<int> calls_remaining;
    bool throw_from_init;

    CountedThrowIVFPQ(
            Index* quantizer,
            size_t d,
            size_t nlist,
            size_t M,
            size_t nbits,
            MetricType metric,
            int throw_after,
            bool init_throw)
            : IndexIVFPQ(quantizer, d, nlist, M, nbits, metric),
              calls_remaining(throw_after),
              throw_from_init(init_throw) {}

    DistanceComputer* get_distance_computer() const override {
        if (calls_remaining.fetch_sub(1) <= 0) {
            if (throw_from_init) {
                throw std::runtime_error("IVFPQ init throw");
            }
            return new ThrowingDC();
        }
        return IndexIVFPQ::get_distance_computer();
    }
};

struct HNSW2LevelFixture {
    static constexpr int d = 8;
    static constexpr int nlist = 4;
    static constexpr int m_pq = 4;
    static constexpr int nb = 512;
    static constexpr int M = 16;

    IndexFlatL2 quantizer{d};
    IndexHNSW2Level index{&quantizer, nlist, m_pq, M};
    std::vector<float> xb;
    std::vector<float> xq;

    HNSW2LevelFixture() : xb(nb * d), xq(d) {
        index.own_fields = false;
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist;
        for (auto& v : xb) {
            v = dist(rng);
        }
        index.train(nb, xb.data());
        index.add(nb, xb.data());
        // Convert Index2Layer storage to IndexIVFPQ for the mixed
        // search path. After this, storage is an IndexIVFPQ owned
        // by the HNSW2Level index.
        index.flip_to_ivf();
        index.own_fields = true;
        for (auto& v : xq) {
            v = dist(rng);
        }
    }
};

// ---------------------------------------------------------------------------
// IndexHNSW2Level::search() mixed path init-phase throw:
// storage_distance_computer() throws during OMP init, after
// search_preassigned has already succeeded.
// ---------------------------------------------------------------------------
TEST(OMPExceptionSafety, hnsw_2level_mixed_init) {
    HNSW2LevelFixture f;

    // The replacement IVFPQ shares the real index's trained state
    // (codebooks, inverted lists). We copy the real IVFPQ's invlists
    // and PQ so search_preassigned succeeds, then the counted throw
    // fires during the HNSW2Level OMP init phase.
    auto* real_ivfpq = dynamic_cast<IndexIVFPQ*>(f.index.storage);
    ASSERT_NE(real_ivfpq, nullptr);

    CountedThrowIVFPQ throwing(
            &f.quantizer,
            f.d,
            f.nlist,
            f.m_pq,
            8,
            METRIC_L2,
            /*throw_after=*/100,
            /*init_throw=*/true);
    throwing.own_fields = false;
    throwing.is_trained = true;
    throwing.ntotal = real_ivfpq->ntotal;
    throwing.pq = real_ivfpq->pq;
    throwing.code_size = real_ivfpq->code_size;
    throwing.nprobe = real_ivfpq->nprobe;
    throwing.replace_invlists(real_ivfpq->invlists, false);

    auto* old = f.index.storage;
    f.index.storage = &throwing;
    f.index.own_fields = false;

    std::vector<float> distances(1);
    std::vector<idx_t> labels(1);
    EXPECT_THROW(
            f.index.search(1, f.xq.data(), 1, distances.data(), labels.data()),
            std::exception);

    f.index.storage = old;
    f.index.own_fields = true;
}

// ---------------------------------------------------------------------------
// IndexHNSW2Level::search() mixed path loop-body throw:
// set_query() throws inside OMP for loop, after search_preassigned
// and OMP init have succeeded.
// ---------------------------------------------------------------------------
TEST(OMPExceptionSafety, hnsw_2level_mixed_loop) {
    HNSW2LevelFixture f;

    auto* real_ivfpq = dynamic_cast<IndexIVFPQ*>(f.index.storage);
    ASSERT_NE(real_ivfpq, nullptr);

    CountedThrowIVFPQ throwing(
            &f.quantizer,
            f.d,
            f.nlist,
            f.m_pq,
            8,
            METRIC_L2,
            /*throw_after=*/100,
            /*init_throw=*/false);
    throwing.own_fields = false;
    throwing.is_trained = true;
    throwing.ntotal = real_ivfpq->ntotal;
    throwing.pq = real_ivfpq->pq;
    throwing.code_size = real_ivfpq->code_size;
    throwing.nprobe = real_ivfpq->nprobe;
    throwing.replace_invlists(real_ivfpq->invlists, false);

    auto* old = f.index.storage;
    f.index.storage = &throwing;
    f.index.own_fields = false;

    std::vector<float> distances(1);
    std::vector<idx_t> labels(1);
    EXPECT_THROW(
            f.index.search(1, f.xq.data(), 1, distances.data(), labels.data()),
            std::exception);

    f.index.storage = old;
    f.index.own_fields = true;
}
