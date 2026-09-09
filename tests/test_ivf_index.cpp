/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <omp.h>
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <set>

#include <gtest/gtest.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ResultHandler.h>

namespace {

// stores all ivf lists, used to verify the context
// object is passed to the iterator
class TestContext {
   public:
    TestContext() {}

    void save_code(size_t list_no, const uint8_t* code, size_t code_size) {
        list_nos.emplace(id, list_no);
        codes.emplace(id, std::vector<uint8_t>(code_size));
        for (size_t i = 0; i < code_size; i++) {
            codes[id][i] = code[i];
        }
        id++;
    }

    // id to codes map
    std::unordered_map<faiss::idx_t, std::vector<uint8_t>> codes;
    // id to list_no map
    std::unordered_map<faiss::idx_t, size_t> list_nos;
    faiss::idx_t id = 0;
    std::set<size_t> lists_probed;
};

// the iterator that iterates over the codes stored in context object
class TestInvertedListIterator : public faiss::InvertedListsIterator {
   public:
    TestInvertedListIterator(size_t list_no_in, TestContext* context_in)
            : list_no{list_no_in}, context{context_in} {
        it = context->codes.cbegin();
        seek_next();
    }
    ~TestInvertedListIterator() override {}

    // move the cursor to the first valid entry
    void seek_next() {
        while (it != context->codes.cend() &&
               context->list_nos[it->first] != list_no) {
            it++;
        }
    }

    virtual bool is_available() const override {
        return it != context->codes.cend();
    }

    virtual void next() override {
        it++;
        seek_next();
    }

    virtual std::pair<faiss::idx_t, const uint8_t*> get_id_and_codes()
            override {
        if (it == context->codes.cend()) {
            FAISS_THROW_MSG("invalid state");
        }
        return std::make_pair(it->first, it->second.data());
    }

   private:
    size_t list_no;
    TestContext* context;
    decltype(context->codes.cbegin()) it;
};

class TestInvertedLists : public faiss::InvertedLists {
   public:
    TestInvertedLists(size_t nlist_in, size_t code_size_in)
            : faiss::InvertedLists(nlist_in, code_size_in) {
        use_iterator = true;
    }

    ~TestInvertedLists() override {}
    size_t list_size(size_t /*list_no*/) const override {
        FAISS_THROW_MSG("unexpected call");
    }

    faiss::InvertedListsIterator* get_iterator(size_t list_no, void* context)
            const override {
        auto testContext = (TestContext*)context;
        testContext->lists_probed.insert(list_no);
        return new TestInvertedListIterator(list_no, testContext);
    }

    const uint8_t* get_codes(size_t /* list_no */) const override {
        FAISS_THROW_MSG("unexpected call");
    }

    const faiss::idx_t* get_ids(size_t /* list_no */) const override {
        FAISS_THROW_MSG("unexpected call");
    }

    // store the codes in context object
    size_t add_entry(
            size_t list_no,
            faiss::idx_t /*theid*/,
            const uint8_t* code,
            void* context) override {
        auto testContext = (TestContext*)context;
        testContext->save_code(list_no, code, code_size);
        return 0;
    }

    size_t add_entries(
            size_t /*list_no*/,
            size_t /*n_entry*/,
            const faiss::idx_t* /*ids*/,
            const uint8_t* /*code*/) override {
        FAISS_THROW_MSG("unexpected call");
    }

    void update_entries(
            size_t /*list_no*/,
            size_t /*offset*/,
            size_t /*n_entry*/,
            const faiss::idx_t* /*ids*/,
            const uint8_t* /*code*/) override {
        FAISS_THROW_MSG("unexpected call");
    }

    void resize(size_t /*list_no*/, size_t /*new_size*/) override {
        FAISS_THROW_MSG("unexpected call");
    }
};
} // namespace

TEST(IVF, list_context) {
    // this test verifies that the context object is passed
    // to the InvertedListsIterator and InvertedLists::add_entry.
    // the test InvertedLists and InvertedListsIterator reads/writes
    // to the test context object.
    // the test verifies the context object is modified as expected.

    constexpr int d = 32;      // dimension
    constexpr int nb = 100000; // database size
    constexpr int nlist = 100;

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    // disable parallism, or we need to make Context object
    // thread-safe
    omp_set_num_threads(1);

    faiss::IndexFlatL2 quantizer(d); // the other index
    faiss::IndexIVFFlat index(&quantizer, d, nlist);
    TestInvertedLists inverted_lists(nlist, index.code_size);
    index.replace_invlists(&inverted_lists);
    {
        // training
        constexpr size_t nt = 1500; // nb of training vectors
        std::vector<float> trainvecs(nt * d);
        for (size_t i = 0; i < nt * d; i++) {
            trainvecs[i] = distrib(rng);
        }
        index.verbose = true;
        index.train(nt, trainvecs.data());
    }
    TestContext context;
    std::vector<float> query_vector;
    constexpr faiss::idx_t query_vector_id = 100;
    {
        // populating the database
        std::vector<float> database(nb * d);
        for (size_t i = 0; i < nb * d; i++) {
            database[i] = distrib(rng);
            // populate the query vector
            if (i >= query_vector_id * d && i < query_vector_id * d + d) {
                query_vector.push_back(database[i]);
            }
        }
        std::vector<faiss::idx_t> coarse_idx(nb);
        index.quantizer->assign(nb, database.data(), coarse_idx.data());
        // pass dummy ids, the actual ids are assigned in TextContext object
        std::vector<faiss::idx_t> xids(nb, 42);
        index.add_core(
                nb, database.data(), xids.data(), coarse_idx.data(), &context);

        // check the context object get updated
        EXPECT_EQ(nb, context.id) << "should have added all ids";
        EXPECT_EQ(nb, context.codes.size())
                << "should have correct number of codes";
        EXPECT_EQ(nb, context.list_nos.size())
                << "should have correct number of list numbers";
    }
    {
        constexpr size_t num_vecs = 5; // number of vectors
        std::vector<float> vecs(num_vecs * d);
        for (size_t i = 0; i < num_vecs * d; i++) {
            vecs[i] = distrib(rng);
        }
        const size_t codeSize = index.sa_code_size();
        std::vector<uint8_t> encodedData(num_vecs * codeSize);
        index.sa_encode(num_vecs, vecs.data(), encodedData.data());
        std::vector<float> decodedVecs(num_vecs * d);
        index.sa_decode(num_vecs, encodedData.data(), decodedVecs.data());
        EXPECT_EQ(vecs, decodedVecs)
                << "decoded vectors should be the same as the original vectors that were encoded";
    }
    {
        constexpr faiss::idx_t k = 100;
        constexpr size_t nprobe = 10;
        std::vector<float> distances(k);
        std::vector<faiss::idx_t> labels(k);
        faiss::SearchParametersIVF params;
        params.inverted_list_context = &context;
        params.nprobe = nprobe;
        index.search(
                1,
                query_vector.data(),
                k,
                distances.data(),
                labels.data(),
                &params);
        EXPECT_EQ(nprobe, context.lists_probed.size())
                << "should probe nprobe lists";

        // check the result contains the query vector, the probablity of
        // this fail should be low
        auto query_vector_listno = context.list_nos[query_vector_id];
        auto& lists_probed = context.lists_probed;
        EXPECT_TRUE(
                std::find(
                        lists_probed.cbegin(),
                        lists_probed.cend(),
                        query_vector_listno) != lists_probed.cend())
                << "should probe the list of the query vector";
        EXPECT_TRUE(
                std::find(labels.cbegin(), labels.cend(), query_vector_id) !=
                labels.cend())
                << "should return the query vector";
    }
}

TEST(IVF, jaccard_search_returns_most_similar_vector) {
    constexpr int d = 3;
    constexpr int nb = 3;
    const float xb[nb * d] = {
            1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f};

    faiss::IndexFlatL2 quantizer(d);
    quantizer.add(1, xb);
    faiss::IndexIVFFlat index(&quantizer, d, 1, faiss::METRIC_Jaccard);
    index.add(nb, xb);

    for (int parallel_mode = 0; parallel_mode < 4; parallel_mode++) {
        index.parallel_mode = parallel_mode;
        float distance;
        faiss::idx_t label;
        index.search(1, xb, 1, &distance, &label);

        EXPECT_EQ(label, 0) << "parallel mode " << parallel_mode;
        EXPECT_FLOAT_EQ(distance, 1.0f) << "parallel mode " << parallel_mode;
    }
}

// Test: search_preassigned with out-of-range keys throws a catchable
// FaissException instead of calling std::terminate from an uncaught
// exception inside the OpenMP parallel region.
TEST(IVF, search_preassigned_out_of_range_key) {
    int d = 4;
    int nlist = 2;
    faiss::IndexFlatL2 quantizer(d);
    faiss::IndexIVFFlat idx(&quantizer, d, nlist);
    idx.own_fields = false;

    // Train and add some vectors so the index is usable.
    std::vector<float> train_data(nlist * d, 0.0f);
    for (int i = 0; i < nlist * d; i++) {
        train_data[i] = static_cast<float>(i);
    }
    idx.train(nlist, train_data.data());
    idx.add(nlist, train_data.data());

    // Query vector.
    std::vector<float> xq(d, 1.0f);
    std::vector<float> distances(1);
    std::vector<faiss::idx_t> labels(1);

    // Pass a key >= nlist to search_preassigned.
    faiss::idx_t bad_key = nlist; // out of range
    float coarse_dis = 0.0f;

    EXPECT_THROW(
            idx.search_preassigned(
                    1,
                    xq.data(),
                    1,
                    &bad_key,
                    &coarse_dis,
                    distances.data(),
                    labels.data(),
                    false),
            faiss::FaissException);
}

// Test: range_search_preassigned with out-of-range keys throws a catchable
// FaissException instead of calling std::terminate from an uncaught
// exception inside the OpenMP parallel region.
TEST(IVF, range_search_preassigned_out_of_range_key) {
    int d = 4;
    int nlist = 2;
    faiss::IndexFlatL2 quantizer(d);
    faiss::IndexIVFFlat idx(&quantizer, d, nlist);
    idx.own_fields = false;

    std::vector<float> train_data(nlist * d, 0.0f);
    for (int i = 0; i < nlist * d; i++) {
        train_data[i] = static_cast<float>(i);
    }
    idx.train(nlist, train_data.data());
    idx.add(nlist, train_data.data());

    std::vector<float> xq(d, 1.0f);
    faiss::RangeSearchResult result(1);

    faiss::idx_t bad_key = nlist; // out of range
    float coarse_dis = 0.0f;

    EXPECT_THROW(
            idx.range_search_preassigned(
                    1,
                    xq.data(),
                    std::numeric_limits<float>::max(),
                    &bad_key,
                    &coarse_dis,
                    &result,
                    false),
            faiss::FaissException);
}

// Minimal ResultHandler that just collects results presented to it.
struct CollectResultHandler : faiss::ResultHandler {
    bool add_result(float, faiss::idx_t) override {
        return false;
    }
};

// Test: search1 with a quantizer that returns out-of-range keys throws
// FaissException.
TEST(IVF, search1_out_of_range_key) {
    int d = 4;
    int nlist = 2;
    faiss::IndexFlatL2 quantizer(d);
    faiss::IndexIVFFlat idx(&quantizer, d, nlist);
    idx.own_fields = false;

    // Train and add vectors so the index is usable.
    std::vector<float> train_data(nlist * d, 0.0f);
    for (int i = 0; i < nlist * d; i++) {
        train_data[i] = static_cast<float>(i);
    }
    idx.train(nlist, train_data.data());
    idx.add(nlist, train_data.data());

    // Corrupt the quantizer by adding an extra centroid far away, so it
    // can return key == nlist (out of range) for a query near that point.
    std::vector<float> extra_centroid(d, 1e6f);
    quantizer.add(1, extra_centroid.data());
    // Now quantizer has nlist+1 centroids, but idx.nlist is still nlist.

    // Query near the extra centroid so quantizer returns the bad key.
    std::vector<float> xq(d, 1e6f);
    CollectResultHandler handler;
    handler.threshold = std::numeric_limits<float>::max();

    EXPECT_THROW(idx.search1(xq.data(), handler), faiss::FaissException);
}

// Iterator that enables search callbacks and tracks invocations.
class CallbackTrackingIterator : public TestInvertedListIterator {
   public:
    CallbackTrackingIterator(
            size_t list_no,
            TestContext* context,
            size_t& distance_count,
            size_t& heap_count)
            : TestInvertedListIterator(list_no, context),
              distance_count_{distance_count},
              heap_count_{heap_count} {
        has_search_callbacks_ = true;
    }

    void on_distance_computed(faiss::idx_t id, float distance) override {
        EXPECT_GE(id, 0) << "vector ID should be non-negative";
        EXPECT_GE(distance, 0.0f) << "L2 distance should be non-negative";
        distance_count_++;
    }

    void on_heap_changed(faiss::idx_t new_id, faiss::idx_t evicted_id)
            override {
        EXPECT_GE(new_id, 0) << "new heap entry ID should be non-negative";
        (void)evicted_id; // may be -1 when heap not yet full
        heap_count_++;
    }

   private:
    size_t& distance_count_;
    size_t& heap_count_;
};

// InvertedLists that uses CallbackTrackingIterator.
class CallbackTrackingInvertedLists : public TestInvertedLists {
   public:
    CallbackTrackingInvertedLists(
            size_t nlist_in,
            size_t code_size_in,
            size_t& distance_count,
            size_t& heap_count)
            : TestInvertedLists(nlist_in, code_size_in),
              distance_count_{distance_count},
              heap_count_{heap_count} {}

    faiss::InvertedListsIterator* get_iterator(size_t list_no, void* context)
            const override {
        auto testContext = (TestContext*)context;
        testContext->lists_probed.insert(list_no);
        return new CallbackTrackingIterator(
                list_no, testContext, distance_count_, heap_count_);
    }

   private:
    size_t& distance_count_;
    size_t& heap_count_;
};

// Test: on_distance_computed and on_heap_changed fire during search
// when has_search_callbacks_ is true.
TEST(IVF, search_callbacks) {
    constexpr int d = 8;
    constexpr int nb = 200;
    constexpr int nlist = 4;

    std::mt19937 rng(42);
    std::uniform_real_distribution<> distrib;

    omp_set_num_threads(1);

    faiss::IndexFlatL2 quantizer(d);
    faiss::IndexIVFFlat index(&quantizer, d, nlist);

    size_t distance_count = 0;
    size_t heap_count = 0;
    CallbackTrackingInvertedLists invlists(
            nlist, index.code_size, distance_count, heap_count);
    index.replace_invlists(&invlists);

    // Train
    constexpr size_t nt = 100;
    std::vector<float> trainvecs(nt * d);
    for (size_t i = 0; i < nt * d; i++) {
        trainvecs[i] = distrib(rng);
    }
    index.train(nt, trainvecs.data());

    // Populate via context
    TestContext context;
    std::vector<float> database(nb * d);
    for (size_t i = 0; i < nb * d; i++) {
        database[i] = distrib(rng);
    }
    std::vector<faiss::idx_t> coarse_idx(nb);
    index.quantizer->assign(nb, database.data(), coarse_idx.data());
    std::vector<faiss::idx_t> xids(nb, 42);
    index.add_core(
            nb, database.data(), xids.data(), coarse_idx.data(), &context);

    // Search
    constexpr faiss::idx_t k = 5;
    constexpr size_t nprobe = 2;
    std::vector<float> query(d);
    for (int i = 0; i < d; i++) {
        query[i] = distrib(rng);
    }
    std::vector<float> distances(k);
    std::vector<faiss::idx_t> labels(k);
    faiss::SearchParametersIVF params;
    params.inverted_list_context = &context;
    params.nprobe = nprobe;

    index.search(1, query.data(), k, distances.data(), labels.data(), &params);

    EXPECT_GT(distance_count, 0)
            << "on_distance_computed should fire for scored vectors";
    EXPECT_GT(heap_count, 0)
            << "on_heap_changed should fire when vectors enter the heap";
    EXPECT_GE(distance_count, heap_count)
            << "not every distance computation leads to a heap change";
}

namespace {

struct SetupCountingScanner : faiss::InvertedListScanner {
    std::unique_ptr<faiss::InvertedListScanner> inner;
    std::atomic<size_t>& setup_count;
    std::atomic<size_t>& range_codes;

    SetupCountingScanner(
            faiss::InvertedListScanner* inner_in,
            std::atomic<size_t>& setup_count_in,
            std::atomic<size_t>& range_codes_in)
            : inner(inner_in),
              setup_count(setup_count_in),
              range_codes(range_codes_in) {}

    void set_query(const float* query) override {
        inner->set_query(query);
    }

    void set_list(faiss::idx_t list, float coarse_distance) override {
        setup_count.fetch_add(1, std::memory_order_relaxed);
        inner->set_list(list, coarse_distance);
    }

    void scan_codes_range(
            size_t n,
            const uint8_t* codes,
            const faiss::idx_t* ids,
            float radius,
            faiss::RangeQueryResult& result) const override {
        range_codes.fetch_add(n, std::memory_order_relaxed);
        inner->scan_codes_range(n, codes, ids, radius, result);
    }

    float distance_to_code(const uint8_t* code) const override {
        return inner->distance_to_code(code);
    }

    size_t scan_codes(
            size_t n,
            const uint8_t* codes,
            const faiss::idx_t* ids,
            faiss::ResultHandler& handler) const override {
        return inner->scan_codes(n, codes, ids, handler);
    }
};

struct SetupCountingIVF : faiss::IndexIVFFlat {
    using faiss::IndexIVFFlat::IndexIVFFlat;
    mutable std::atomic<size_t> setup_count{0};
    mutable std::atomic<size_t> range_codes{0};

    faiss::InvertedListScanner* get_InvertedListScanner(
            bool store_pairs,
            const faiss::IDSelector* selector,
            const faiss::IVFSearchParameters* params) const override {
        return new SetupCountingScanner(
                faiss::IndexIVFFlat::get_InvertedListScanner(
                        store_pairs, selector, params),
                setup_count,
                range_codes);
    }
};

} // namespace

TEST(IVF, sorted_range_bounds_range_scan) {
    omp_set_num_threads(2);
    faiss::IndexFlatL2 quantizer(1);
    const float centroids[] = {0, 10};
    quantizer.add(2, centroids);
    SetupCountingIVF index(&quantizer, 1, 2);
    index.is_trained = true;
    const float database[] = {0, 1, 10, 11};
    index.add(4, database);
    const float query = 0;
    faiss::IDSelectorRange selector(0, 2, true);
    faiss::SearchParametersIVF params;
    params.nprobe = 2;
    params.sel = &selector;
    for (int mode : {0, 1, 2}) {
        index.parallel_mode = mode;
        for (const auto& bounds : std::vector<std::pair<int, int>>{
                     {-1, 5},
                     {0, 2},
                     {2, 4},
                     {1, 3},
                     {1, 2},
                     {2, 2},
                     {0, 0},
                     {4, 5}}) {
            selector.imin = bounds.first;
            selector.imax = bounds.second;
            index.setup_count = 0;
            index.range_codes = 0;
            faiss::RangeSearchResult results(1);
            index.range_search(1, &query, 1000, &results, &params);
            std::set<faiss::idx_t> expected;
            std::set<faiss::idx_t> lists;
            for (faiss::idx_t id = 0; id < 4; id++) {
                if (selector.is_member(id)) {
                    expected.insert(id);
                    lists.insert(id / 2);
                }
            }
            EXPECT_EQ(index.setup_count.load(), lists.size());
            EXPECT_EQ(index.range_codes.load(), expected.size());
            ASSERT_EQ(results.lims[1], expected.size());
            std::set<faiss::idx_t> labels;
            for (size_t i = 0; i < results.lims[1]; i++) {
                const auto id = results.labels[i];
                ASSERT_GE(id, 0);
                ASSERT_LT(id, 4);
                labels.insert(id);
                EXPECT_FLOAT_EQ(
                        results.distances[i], database[id] * database[id]);
            }
            EXPECT_EQ(labels, expected);
        }
    }
    index.parallel_mode = 0;
    faiss::IDSelectorRange empty(4, 5, true);
    params.sel = &empty;
    index.setup_count = 0;
    index.range_codes = 0;
    faiss::indexIVF_stats.reset();
    faiss::RangeSearchResult no_results(1);
    index.range_search(1, &query, 1000, &no_results, &params);
    EXPECT_EQ(no_results.lims[1], 0);
    EXPECT_EQ(index.setup_count.load(), 0);
    EXPECT_EQ(index.range_codes.load(), 0);
    EXPECT_EQ(faiss::indexIVF_stats.nlist, 2);

    // A subclass can expose context membership unrelated to its range fields.
    // Preserve that dispatch instead of bypassing it with sorted bounds.
    struct EvenContext : faiss::IDSelectorWithContext {
        bool is_member(faiss::idx_t id) const override {
            return id % 2 == 0;
        }
        bool is_member_with_context(
                faiss::idx_t id,
                const faiss::IDScanContext&) const override {
            return id % 2 == 0;
        }
    };
    struct CustomRange : faiss::IDSelectorRange, EvenContext {
        CustomRange() : faiss::IDSelectorRange(0, 2, true) {}
    } custom;
    params.sel = static_cast<faiss::IDSelectorRange*>(&custom);
    index.setup_count = 0;
    index.range_codes = 0;
    faiss::RangeSearchResult custom_results(1);
    index.range_search(1, &query, 1000, &custom_results, &params);
    EXPECT_EQ(index.setup_count.load(), 2);
    EXPECT_EQ(index.range_codes.load(), 4);
    ASSERT_EQ(custom_results.lims[1], 2);
    std::set<faiss::idx_t> labels(
            custom_results.labels, custom_results.labels + 2);
    EXPECT_EQ(labels, (std::set<faiss::idx_t>{0, 2}));
}

TEST(IVF, sorted_range_preserves_iterator_and_pair_paths) {
    omp_set_num_threads(1);
    faiss::IndexFlatL2 quantizer(1);
    const float centroids[] = {0, 10};
    quantizer.add(2, centroids);
    const float database[] = {0, 1, 10, 11};
    const float query = 0;
    faiss::IDSelectorRange selector(1, 3, true);
    faiss::SearchParametersIVF params;
    params.nprobe = 2;
    params.sel = &selector;

    faiss::IndexIVFFlat array_index(&quantizer, 1, 2);
    array_index.is_trained = true;
    array_index.add(4, database);
    const faiss::idx_t keys[] = {0, 1};
    const float coarse_distances[] = {0, 100};
    faiss::RangeSearchResult pairs(1);
    array_index.range_search_preassigned(
            1, &query, 1000, keys, coarse_distances, &pairs, true, &params);
    // With store_pairs, the native selector sees encoded pair labels rather
    // than database IDs. Only list 0, offset 1 lies inside [1, 3).
    ASSERT_EQ(pairs.lims[1], 1);
    EXPECT_EQ(pairs.labels[0], 1);

    faiss::IndexIVFFlat iterator_index(&quantizer, 1, 2);
    TestInvertedLists inverted_lists(2, iterator_index.code_size);
    iterator_index.replace_invlists(&inverted_lists);
    iterator_index.is_trained = true;
    iterator_index.ntotal = 4;
    TestContext context;
    for (size_t i = 0; i < 4; i++) {
        context.save_code(
                i / 2,
                reinterpret_cast<const uint8_t*>(database + i),
                sizeof(float));
    }
    params.inverted_list_context = &context;
    // Sorted-range dispatch must preserve the generic iterator path.
    faiss::RangeSearchResult results(1), reference(1);
    iterator_index.range_search(1, &query, 1000, &results, &params);
    selector.assume_sorted = false;
    iterator_index.range_search(1, &query, 1000, &reference, &params);
    ASSERT_EQ(results.lims[1], reference.lims[1]);
    std::map<faiss::idx_t, float> actual, expected;
    for (size_t i = 0; i < results.lims[1]; i++) {
        actual[results.labels[i]] = results.distances[i];
        expected[reference.labels[i]] = reference.distances[i];
    }
    EXPECT_EQ(actual, expected);
}
