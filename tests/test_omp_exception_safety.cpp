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

#include <faiss/IndexFlat.h>
#include <faiss/IndexFlatCodes.h>
#include <faiss/IndexIVFFlat.h>
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
