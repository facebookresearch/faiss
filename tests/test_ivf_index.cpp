/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <omp.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <random>
#include <set>

#include <gtest/gtest.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/ClusteringHelpers.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/invlists/InvertedLists.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/fp16.h>

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
    // assume_sorted picks the bounded-section shortcut on an array-backed
    // list. An iterable list has no section to bound, so it must keep the
    // selector and filter per entry instead.
    for (bool assume_sorted : {false, true}) {
        // Iterator scans must honor the same selector contract as array-backed
        // scans. Keep only the upper half of the IDs and verify both KNN and
        // range search.
        SCOPED_TRACE(testing::Message() << "assume_sorted=" << assume_sorted);
        constexpr faiss::idx_t k = 100;
        constexpr size_t nprobe = 10;
        faiss::IDSelectorRange selector(nb / 2, nb, assume_sorted);
        faiss::SearchParametersIVF params;
        params.inverted_list_context = &context;
        params.nprobe = nprobe;
        params.sel = &selector;

        context.lists_probed.clear();
        std::vector<float> distances(k);
        std::vector<faiss::idx_t> labels(k);
        index.search(
                1,
                query_vector.data(),
                k,
                distances.data(),
                labels.data(),
                &params);
        const size_t rejected_knn = std::count_if(
                labels.cbegin(), labels.cend(), [&](faiss::idx_t id) {
                    return id != -1 && !selector.is_member(id);
                });
        EXPECT_EQ(0, rejected_knn);

        context.lists_probed.clear();
        faiss::RangeSearchResult result(1);
        index.range_search(
                1,
                query_vector.data(),
                std::numeric_limits<float>::max(),
                &result,
                &params);
        const size_t rejected_range = std::count_if(
                result.labels,
                result.labels + result.lims[1],
                [&](faiss::idx_t id) { return !selector.is_member(id); });
        EXPECT_EQ(0, rejected_range);
    }
}

TEST(IVF, sorted_range_selector_rejects_store_pairs) {
    constexpr size_t d = 4;
    constexpr size_t nb = 100;
    constexpr faiss::idx_t k = 50;

    faiss::IndexFlatL2 quantizer(d);
    const std::vector<float> centroid(d, 0.0f);
    quantizer.add(1, centroid.data());
    faiss::IndexIVFFlat index(&quantizer, d, 1);
    index.is_trained = true;

    std::vector<float> database(nb * d);
    std::vector<faiss::idx_t> ids(nb);
    for (size_t i = 0; i < nb; ++i) {
        ids[i] = static_cast<faiss::idx_t>(1000 + i);
        for (size_t j = 0; j < d; ++j) {
            database[i * d + j] = static_cast<float>(i + j);
        }
    }
    index.add_with_ids(nb, database.data(), ids.data());

    faiss::IDSelectorRange selector(1020, 1040, true);
    faiss::SearchParametersIVF params;
    params.nprobe = 1;
    params.sel = &selector;
    const faiss::idx_t key = 0;
    const float coarse_distance = 0.0f;
    std::vector<float> distances(k);
    std::vector<faiss::idx_t> labels(k);
    EXPECT_THROW(
            index.search_preassigned(
                    1,
                    centroid.data(),
                    k,
                    &key,
                    &coarse_distance,
                    distances.data(),
                    labels.data(),
                    true,
                    &params),
            faiss::FaissException);
}

namespace {

// Counts the code fetches that a scan performs, so a test can show that a list
// the range filter empties never reaches its codes.
struct CountingInvertedLists : faiss::ArrayInvertedLists {
    mutable size_t code_fetches = 0;

    CountingInvertedLists(size_t nlist_in, size_t code_size_in)
            : faiss::ArrayInvertedLists(nlist_in, code_size_in) {}

    const uint8_t* get_codes(size_t list_no) const override {
        code_fetches++;
        return faiss::ArrayInvertedLists::get_codes(list_no);
    }
};

} // namespace

TEST(IVF, sorted_range_selector_skips_setup_for_filtered_lists) {
    constexpr size_t d = 8;
    constexpr size_t nb = 1000;
    constexpr size_t nlist = 16;
    constexpr faiss::idx_t k = 10;
    constexpr faiss::idx_t range_end = 20;

    faiss::IndexFlatL2 quantizer(d);
    faiss::IndexIVFFlat index(&quantizer, d, nlist);

    std::mt19937 rng(12345);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> database(nb * d);
    for (float& value : database) {
        value = normal(rng);
    }
    std::vector<faiss::idx_t> ids(nb);
    for (size_t i = 0; i < nb; ++i) {
        ids[i] = static_cast<faiss::idx_t>(i); // ascending, so each list sorts
    }
    index.train(nb, database.data());

    auto* counting = new CountingInvertedLists(nlist, index.code_size);
    index.replace_invlists(counting, /*own=*/true);
    index.add_with_ids(nb, database.data(), ids.data());

    // Lists that hold no id below range_end cannot contribute a result, so the
    // scan must return before it fetches their codes.
    size_t lists_in_range = 0;
    size_t lists_non_empty = 0;
    for (size_t list_no = 0; list_no < nlist; ++list_no) {
        const size_t list_size = counting->list_size(list_no);
        if (list_size == 0) {
            continue;
        }
        lists_non_empty++;
        const faiss::idx_t* list_ids = counting->ids[list_no].data();
        if (list_ids[0] < range_end) {
            lists_in_range++;
        }
    }
    ASSERT_GT(lists_in_range, 0u);
    ASSERT_LT(lists_in_range, lists_non_empty);

    faiss::IDSelectorRange selector(0, range_end, /*assume_sorted=*/true);
    faiss::SearchParametersIVF params;
    params.nprobe = nlist;
    params.sel = &selector;

    std::vector<float> distances(k);
    std::vector<faiss::idx_t> labels(k);
    counting->code_fetches = 0;
    index.search(
            1, database.data(), k, distances.data(), labels.data(), &params);
    EXPECT_EQ(lists_in_range, counting->code_fetches);

    // Every returned id honours the selector.
    for (faiss::idx_t label : labels) {
        if (label != -1) {
            EXPECT_TRUE(selector.is_member(label));
        }
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

class LimitedEncoderIndex : public faiss::IndexIVFScalarQuantizer {
   public:
    LimitedEncoderIndex(faiss::Index* quantizer, int d, int nlist)
            : faiss::IndexIVFScalarQuantizer(
                      quantizer,
                      d,
                      nlist,
                      faiss::ScalarQuantizer::QT_8bit) {}

    faiss::idx_t train_encoder_num_vectors() const override {
        return 7;
    }

    void train_encoder(
            faiss::idx_t n,
            const float* x,
            const faiss::idx_t* assign) override {
        encoder_input.assign(x, x + n * d);
        encoder_assignments.assign(assign, assign + n);
    }

    std::vector<float> encoder_input;
    std::vector<faiss::idx_t> encoder_assignments;
};

class TrackingFp16Codec : public faiss::IndexScalarQuantizer {
   public:
    explicit TrackingFp16Codec(int d)
            : faiss::IndexScalarQuantizer(
                      d,
                      faiss::ScalarQuantizer::QuantizerType::QT_fp16) {}

    void sa_decode(faiss::idx_t n, const uint8_t* bytes, float* x)
            const override {
        decode_calls.fetch_add(1, std::memory_order_relaxed);
        max_decode_rows = std::max(max_decode_rows, static_cast<size_t>(n));
        faiss::IndexScalarQuantizer::sa_decode(n, bytes, x);
    }

    mutable std::atomic<size_t> decode_calls{0};
    mutable size_t max_decode_rows = 0;
};

std::vector<uint16_t> encode_fp16(const std::vector<float>& values) {
    std::vector<uint16_t> encoded(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        encoded[i] = faiss::encode_fp16(values[i]);
    }
    return encoded;
}

std::vector<float> decode_fp16(const std::vector<uint16_t>& values) {
    std::vector<float> decoded(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        decoded[i] = faiss::decode_fp16(values[i]);
    }
    return decoded;
}

} // namespace

TEST(IVF, train_float16_matches_float32_on_rounded_input) {
    constexpr int d = 4;
    constexpr int n = 64;
    constexpr int nlist = 4;

    std::vector<float> input(n * d);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>((i * 17) % 101) / 13.0f;
    }
    auto encoded = encode_fp16(input);
    auto rounded = decode_fp16(encoded);

    faiss::IndexFlatL2 float_quantizer(d);
    faiss::IndexIVFFlat float_index(&float_quantizer, d, nlist);
    float_index.cp.seed = 1234;
    float_index.cp.niter = 4;
    float_index.cp.min_points_per_centroid = 1;
    float_index.train(n, rounded.data());

    faiss::IndexFlatL2 half_quantizer(d);
    faiss::IndexIVFFlat half_index(&half_quantizer, d, nlist);
    half_index.cp.seed = 1234;
    half_index.cp.niter = 4;
    half_index.cp.min_points_per_centroid = 1;
    half_index.train_ex(n, encoded.data(), faiss::NumericType::Float16);

    ASSERT_TRUE(half_index.is_trained);
    ASSERT_EQ(half_quantizer.ntotal, nlist);
    std::vector<float> float_centroids(nlist * d);
    std::vector<float> half_centroids(nlist * d);
    float_quantizer.reconstruct_n(0, nlist, float_centroids.data());
    half_quantizer.reconstruct_n(0, nlist, half_centroids.data());
    EXPECT_EQ(half_centroids, float_centroids);
}

TEST(IVF, train_float16_with_super_kmeans_matches_float32) {
    constexpr int d = 32;
    constexpr int n = 96;
    constexpr int nlist = 4;

    std::vector<float> input(n * d);
    for (int i = 0; i < n; ++i) {
        const float center = static_cast<float>(i % nlist) * 8.0f;
        for (int j = 0; j < d; ++j) {
            input[static_cast<size_t>(i) * d + j] = center +
                    static_cast<float>((i * 17 + j * 13) % 31) / 100.0f;
        }
    }
    auto encoded = encode_fp16(input);
    auto rounded = decode_fp16(encoded);

    faiss::IndexFlatL2 float_quantizer(d);
    faiss::IndexIVFFlat float_index(&float_quantizer, d, nlist);
    float_index.cp.seed = 1234;
    float_index.cp.niter = 3;
    float_index.cp.min_points_per_centroid = 1;
    float_index.cp.use_super_kmeans = true;
    float_index.train(n, rounded.data());

    faiss::IndexFlatL2 half_quantizer(d);
    faiss::IndexIVFFlat half_index(&half_quantizer, d, nlist);
    half_index.cp = float_index.cp;
    half_index.cp.decode_block_size = n;
    half_index.train_ex(n, encoded.data(), faiss::NumericType::Float16);

    std::vector<float> float_centroids(nlist * d);
    std::vector<float> half_centroids(nlist * d);
    float_quantizer.reconstruct_n(0, nlist, float_centroids.data());
    half_quantizer.reconstruct_n(0, nlist, half_centroids.data());
    EXPECT_EQ(half_centroids, float_centroids);

    TrackingFp16Codec codec(d);
    faiss::IndexFlatL2 encoded_quantizer(d);
    faiss::IndexIVFFlat encoded_index(&encoded_quantizer, d, nlist);
    encoded_index.cp = float_index.cp;
    EXPECT_THROW(
            encoded_index.train_encoded(
                    n,
                    reinterpret_cast<const uint8_t*>(encoded.data()),
                    &codec),
            faiss::FaissException);

    faiss::IndexFlatIP ip_quantizer(d);
    faiss::IndexIVFFlat ip_index(
            &ip_quantizer, d, nlist, faiss::METRIC_INNER_PRODUCT);
    ip_index.cp = float_index.cp;
    EXPECT_THROW(
            ip_index.train_ex(n, encoded.data(), faiss::NumericType::Float16),
            faiss::FaissException);
}

TEST(IVF, encoded_training_rejects_non_finite_values) {
    constexpr int d = 2;
    constexpr int n = 4;
    constexpr int nlist = 2;
    auto encoded = encode_fp16(
            {0.0f,
             1.0f,
             2.0f,
             3.0f,
             std::numeric_limits<float>::infinity(),
             5.0f,
             6.0f,
             7.0f});

    faiss::IndexFlatL2 quantizer(d);
    faiss::IndexIVFFlat index(&quantizer, d, nlist);
    EXPECT_THROW(
            index.train_ex(n, encoded.data(), faiss::NumericType::Float16),
            faiss::FaissException);
}

TEST(IVF, encoded_training_decodes_in_bounded_batches) {
    constexpr int d = 4;
    constexpr int n = 40;
    constexpr int nlist = 2;
    constexpr size_t decode_block_size = 5;

    std::vector<float> input(n * d);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>((i * 11) % 97) / 17.0f;
    }
    auto encoded = encode_fp16(input);
    TrackingFp16Codec codec(d);
    faiss::IndexFlatL2 quantizer(d);
    faiss::IndexIVFFlat index(&quantizer, d, nlist);
    index.cp.decode_block_size = decode_block_size;
    index.cp.niter = 2;
    index.cp.min_points_per_centroid = 1;
    index.train_encoded(
            n, reinterpret_cast<const uint8_t*>(encoded.data()), &codec);

    EXPECT_TRUE(index.is_trained);
    EXPECT_GT(codec.max_decode_rows, 0);
    EXPECT_LE(codec.max_decode_rows, decode_block_size);
}

TEST(IVF, encoded_training_skips_nan_scan) {
    constexpr int d = 2;
    constexpr int n = 2;
    auto encoded = encode_fp16({0.0f, 1.0f, 2.0f, 3.0f});
    TrackingFp16Codec codec(d);
    faiss::Clustering clustering(d, n);
    faiss::IndexFlatL2 index(d);

    clustering.train_encoded(
            n, reinterpret_cast<const uint8_t*>(encoded.data()), &codec, index);

    EXPECT_EQ(codec.decode_calls.load(std::memory_order_relaxed), 1);
}

TEST(IVF, train_float16_trains_scalar_quantizer_encoder) {
    constexpr int d = 4;
    constexpr int n = 128;
    constexpr int nlist = 4;

    std::vector<float> input(n * d);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>((i * 19) % 113) / 23.0f;
    }
    auto encoded = encode_fp16(input);

    faiss::IndexFlatL2 quantizer(d);
    faiss::IndexIVFScalarQuantizer index(
            &quantizer,
            d,
            nlist,
            faiss::ScalarQuantizer::QuantizerType::QT_8bit);
    index.cp.seed = 1234;
    index.cp.niter = 4;
    index.cp.min_points_per_centroid = 1;
    index.train_ex(n, encoded.data(), faiss::NumericType::Float16);

    EXPECT_TRUE(index.is_trained);
    EXPECT_EQ(quantizer.ntotal, nlist);
    EXPECT_FALSE(index.sq.trained.empty());
}

TEST(IVF, encoded_training_validates_codec) {
    constexpr int d = 4;
    constexpr int n = 4;
    constexpr int nlist = 2;
    std::vector<uint16_t> encoded(n * (d + 1), faiss::encode_fp16(1.0f));

    faiss::IndexFlatL2 quantizer(d);
    std::vector<float> centroids(nlist * d, 0.0f);
    quantizer.add(nlist, centroids.data());
    faiss::IndexIVFFlat index(&quantizer, d, nlist);
    TrackingFp16Codec wrong_dimension_codec(d + 1);

    EXPECT_THROW(
            index.train_encoded(
                    n,
                    reinterpret_cast<const uint8_t*>(encoded.data()),
                    &wrong_dimension_codec),
            faiss::FaissException);
    EXPECT_THROW(
            index.train_encoded(
                    n,
                    reinterpret_cast<const uint8_t*>(encoded.data()),
                    nullptr),
            faiss::FaissException);
}

TEST(IVF, encoded_encoder_subsampling_matches_float32) {
    constexpr int d = 4;
    constexpr int n = 40;
    constexpr int nlist = 2;

    std::vector<float> input(n * d);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>((i * 23) % 127) / 29.0f;
    }
    auto encoded = encode_fp16(input);
    auto rounded = decode_fp16(encoded);

    faiss::IndexFlatL2 float_quantizer(d);
    LimitedEncoderIndex float_index(&float_quantizer, d, nlist);
    float_index.cp.seed = 1234;
    float_index.cp.niter = 2;
    float_index.cp.min_points_per_centroid = 1;
    float_index.train(n, rounded.data());

    faiss::IndexFlatL2 half_quantizer(d);
    LimitedEncoderIndex half_index(&half_quantizer, d, nlist);
    half_index.cp.seed = 1234;
    half_index.cp.niter = 2;
    half_index.cp.min_points_per_centroid = 1;
    half_index.train_ex(n, encoded.data(), faiss::NumericType::Float16);

    EXPECT_EQ(half_index.encoder_input, float_index.encoder_input);
    EXPECT_EQ(half_index.encoder_assignments, float_index.encoder_assignments);
}

namespace {

class TrackingFlatL2 : public faiss::IndexFlatL2 {
   public:
    explicit TrackingFlatL2(faiss::idx_t d) : faiss::IndexFlatL2(d) {}

    void search(
            faiss::idx_t n,
            const float* x,
            faiss::idx_t k,
            float* distances,
            faiss::idx_t* labels,
            const faiss::SearchParameters* params = nullptr) const override {
        max_search_rows = std::max(max_search_rows, static_cast<size_t>(n));
        faiss::IndexFlatL2::search(n, x, k, distances, labels, params);
    }

    void search_ex(
            faiss::idx_t n,
            const void* x,
            faiss::NumericType numeric_type,
            faiss::idx_t k,
            float* distances,
            faiss::idx_t* labels,
            const faiss::SearchParameters* params = nullptr) const override {
        max_search_rows = std::max(max_search_rows, static_cast<size_t>(n));
        if (numeric_type == faiss::NumericType::Float16) {
            fp16_search_calls++;
        }
        faiss::IndexFlatL2::search_ex(
                n, x, numeric_type, k, distances, labels, params);
    }

    mutable size_t max_search_rows = 0;
    mutable size_t fp16_search_calls = 0;
};

class InvalidAssignmentIndex : public faiss::IndexFlatL2 {
   public:
    explicit InvalidAssignmentIndex(faiss::idx_t d) : faiss::IndexFlatL2(d) {}

    void search_ex(
            faiss::idx_t n,
            const void* /*x*/,
            faiss::NumericType /*numeric_type*/,
            faiss::idx_t k,
            float* distances,
            faiss::idx_t* labels,
            const faiss::SearchParameters* /*params*/ =
                    nullptr) const override {
        const size_t result_size =
                static_cast<size_t>(n) * static_cast<size_t>(k);
        std::fill_n(distances, result_size, 0.0f);
        std::fill_n(labels, result_size, faiss::idx_t{-1});
    }
};

std::vector<float> make_training_data(size_t n, size_t d) {
    std::vector<float> x(n * d);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<float>((i * 29) % 131) / 17.0f - 3.0f;
    }
    return x;
}

faiss::ClusteringParameters small_clustering_params(int niter) {
    faiss::ClusteringParameters cp;
    cp.niter = niter;
    cp.seed = 1234;
    cp.min_points_per_centroid = 1;
    return cp;
}

} // namespace

TEST(Clustering, train_ex_float16_matches_float32_on_rounded_input) {
    constexpr size_t n = 200;
    constexpr size_t k = 5;
    // 16 exercises only the SIMD kernels, 13 and 37 also the scalar tails
    for (int d : {13, 16, 37}) {
        auto encoded = encode_fp16(make_training_data(n, d));
        auto rounded = decode_fp16(encoded);
        auto cp = small_clustering_params(6);

        faiss::Clustering full(d, k, cp);
        faiss::IndexFlatL2 full_index(d);
        full.train(n, rounded.data(), full_index);

        faiss::Clustering half(d, k, cp);
        faiss::IndexFlatL2 half_index(d);
        half.train_ex(
                n, encoded.data(), faiss::NumericType::Float16, half_index);

        EXPECT_EQ(half.centroids, full.centroids) << "d=" << d;
        ASSERT_EQ(half.iteration_stats.size(), full.iteration_stats.size());
        for (size_t i = 0; i < full.iteration_stats.size(); ++i) {
            EXPECT_EQ(half.iteration_stats[i].obj, full.iteration_stats[i].obj);
        }
        EXPECT_EQ(half_index.ntotal, static_cast<faiss::idx_t>(k));
    }
}

TEST(Clustering, train_ex_float16_weighted_matches_float32) {
    constexpr size_t n = 150;
    constexpr size_t d = 20;
    constexpr size_t k = 4;
    auto encoded = encode_fp16(make_training_data(n, d));
    auto rounded = decode_fp16(encoded);
    std::vector<float> weights(n);
    for (size_t i = 0; i < n; ++i) {
        weights[i] = 0.25f + static_cast<float>(i % 7);
    }
    auto cp = small_clustering_params(1);

    faiss::Clustering full(d, k, cp);
    faiss::IndexFlatL2 full_index(d);
    full.train(n, rounded.data(), full_index, weights.data());

    faiss::Clustering half(d, k, cp);
    faiss::IndexFlatL2 half_index(d);
    half.train_ex(
            n,
            encoded.data(),
            faiss::NumericType::Float16,
            half_index,
            weights.data());

    ASSERT_EQ(half.centroids.size(), full.centroids.size());
    for (size_t i = 0; i < full.centroids.size(); ++i) {
        EXPECT_NEAR(half.centroids[i], full.centroids[i], 1e-4);
    }
}

TEST(Clustering, train_ex_float16_subsampling_initializers_match_float32) {
    constexpr size_t n = 100;
    constexpr size_t d = 12;
    constexpr size_t k = 4;
    auto encoded = encode_fp16(make_training_data(n, d));
    auto rounded = decode_fp16(encoded);

    for (auto method :
         {faiss::ClusteringInitMethod::KMEANS_PLUS_PLUS,
          faiss::ClusteringInitMethod::AFK_MC2}) {
        auto cp = small_clustering_params(4);
        cp.max_points_per_centroid = 8;
        cp.init_method = method;
        cp.afkmc2_chain_length = 7;

        faiss::Clustering full(d, k, cp);
        faiss::IndexFlatL2 full_index(d);
        full.train(n, rounded.data(), full_index);

        faiss::Clustering half(d, k, cp);
        faiss::IndexFlatL2 half_index(d);
        half.train_ex(
                n, encoded.data(), faiss::NumericType::Float16, half_index);

        EXPECT_EQ(half.centroids, full.centroids)
                << "method=" << static_cast<int>(method);
    }
}

TEST(Clustering, float16_initializers_match_with_existing_and_early_return) {
    constexpr size_t n = 24;
    constexpr size_t d = 5;
    constexpr size_t k = 3;
    auto encoded = encode_fp16(make_training_data(n, d));
    auto rounded = decode_fp16(encoded);
    const std::vector<float> existing = {-4.0f, -2.0f, 0.0f, 2.0f, 4.0f};

    for (auto method :
         {faiss::ClusteringInitMethod::KMEANS_PLUS_PLUS,
          faiss::ClusteringInitMethod::AFK_MC2}) {
        faiss::ClusteringInitialization initializer(d, k);
        initializer.method = method;
        initializer.seed = 4321;
        initializer.afkmc2_chain_length = 7;
        std::vector<float> full(k * d);
        std::vector<float> half(k * d);

        initializer.init_centroids(
                n, rounded.data(), full.data(), 1, existing.data());
        faiss::detail::init_centroids_fp16(
                initializer,
                n,
                encoded.data(),
                half.data(),
                1,
                existing.data());
        EXPECT_EQ(half, full) << "existing method=" << static_cast<int>(method);

        faiss::ClusteringInitialization single(d, 1);
        single.method = method;
        single.seed = 4321;
        std::vector<float> full_single(d);
        std::vector<float> half_single(d);
        single.init_centroids(n, rounded.data(), full_single.data());
        faiss::detail::init_centroids_fp16(
                single, n, encoded.data(), half_single.data());
        EXPECT_EQ(half_single, full_single)
                << "early return method=" << static_cast<int>(method);
    }
}

TEST(Clustering, train_ex_float16_assigns_in_bounded_batches) {
    constexpr size_t n = 50;
    constexpr size_t d = 8;
    constexpr size_t k = 3;
    auto encoded = encode_fp16(make_training_data(n, d));
    auto cp = small_clustering_params(3);
    cp.decode_block_size = 7;

    faiss::Clustering clus(d, k, cp);
    TrackingFlatL2 index(d);
    clus.train_ex(n, encoded.data(), faiss::NumericType::Float16, index);

    EXPECT_GT(index.max_search_rows, 0);
    EXPECT_LE(index.max_search_rows, cp.decode_block_size);
    // fp16 blocks reach the index through search_ex, not widened beforehand
    EXPECT_GT(index.fp16_search_calls, 0);
}

TEST(Clustering, train_ex_invalid_assignment_throws) {
    constexpr int n = 8;
    constexpr int d = 3;
    constexpr int k = 2;
    auto encoded = encode_fp16(make_training_data(n, d));
    faiss::Clustering clustering(d, k, small_clustering_params(1));
    InvalidAssignmentIndex index(d);

    EXPECT_THROW(
            clustering.train_ex(
                    n, encoded.data(), faiss::NumericType::Float16, index),
            faiss::FaissException);
}

TEST(Index, search_ex_float16_default_matches_float32) {
    constexpr int d = 12;
    constexpr int nb = 300;
    constexpr int nq = 40;
    constexpr int k = 5;
    auto base = decode_fp16(encode_fp16(make_training_data(nb, d)));
    std::vector<float> queries_in(nq * d);
    for (size_t i = 0; i < queries_in.size(); ++i) {
        queries_in[i] = static_cast<float>((i * 37) % 113) / 19.0f - 3.0f;
    }
    auto queries = encode_fp16(queries_in);
    auto rounded = decode_fp16(queries);

    // IndexIVFFlat does not override search_ex: exercises the default
    faiss::IndexFlatL2 quantizer(d);
    faiss::IndexIVFFlat index(&quantizer, d, 4);
    index.cp = small_clustering_params(3);
    index.train(nb, base.data());
    index.add(nb, base.data());
    index.nprobe = 2;

    std::vector<float> float_distances(nq * k), half_distances(nq * k);
    std::vector<faiss::idx_t> float_labels(nq * k), half_labels(nq * k);
    index.search(
            nq, rounded.data(), k, float_distances.data(), float_labels.data());
    index.search_ex(
            nq,
            queries.data(),
            faiss::NumericType::Float16,
            k,
            half_distances.data(),
            half_labels.data());

    EXPECT_EQ(half_labels, float_labels);
    EXPECT_EQ(half_distances, float_distances);
    EXPECT_THROW(
            index.search_ex(
                    nq,
                    queries.data(),
                    faiss::NumericType::UInt8,
                    k,
                    half_distances.data(),
                    half_labels.data()),
            faiss::FaissException);
}

TEST(Clustering, train_ex_float16_rejects_non_finite_values) {
    constexpr size_t n = 8;
    constexpr size_t d = 3;
    constexpr size_t k = 2;
    const uint16_t max_finite = 0x7bff; // 65504
    for (uint16_t bad :
         {uint16_t(0x7e00), uint16_t(0x7c00), uint16_t(0xfc00)}) {
        auto encoded = encode_fp16(make_training_data(n, d));
        encoded[0] = max_finite;
        encoded[n * d - 1] = bad;
        faiss::Clustering clus(d, k, small_clustering_params(2));
        faiss::IndexFlatL2 index(d);
        EXPECT_THROW(
                clus.train_ex(
                        n, encoded.data(), faiss::NumericType::Float16, index),
                faiss::FaissException)
                << "bits=" << bad;
    }

    auto encoded = encode_fp16(make_training_data(n, d));
    encoded[0] = max_finite;
    faiss::Clustering clus(d, k, small_clustering_params(2));
    faiss::IndexFlatL2 index(d);
    EXPECT_NO_THROW(clus.train_ex(
            n, encoded.data(), faiss::NumericType::Float16, index));
}

TEST(Clustering, train_ex_rejects_unsupported_numeric_type) {
    constexpr size_t n = 8;
    constexpr size_t d = 4;
    std::vector<uint8_t> x(n * d, 1);
    faiss::Clustering clus(d, 2, small_clustering_params(2));
    faiss::IndexFlatL2 index(d);
    EXPECT_THROW(
            clus.train_ex(n, x.data(), faiss::NumericType::UInt8, index),
            faiss::FaissException);
}

TEST(IVF, train_float16_with_flat_assigner_matches_float32) {
    constexpr int d = 10;
    constexpr int n = 120;
    constexpr int nlist = 4;
    auto encoded = encode_fp16(make_training_data(n, d));
    auto rounded = decode_fp16(encoded);

    faiss::IndexFlatL2 float_quantizer(d);
    faiss::IndexIVFFlat float_index(&float_quantizer, d, nlist);
    float_index.quantizer_trains_alone = 2;
    float_index.cp = small_clustering_params(4);
    float_index.train(n, rounded.data());

    faiss::IndexFlatL2 half_quantizer(d);
    faiss::IndexIVFFlat half_index(&half_quantizer, d, nlist);
    half_index.quantizer_trains_alone = 2;
    half_index.cp = small_clustering_params(4);
    half_index.train_ex(n, encoded.data(), faiss::NumericType::Float16);

    ASSERT_TRUE(half_index.is_trained);
    std::vector<float> float_centroids(nlist * d);
    std::vector<float> half_centroids(nlist * d);
    float_quantizer.reconstruct_n(0, nlist, float_centroids.data());
    half_quantizer.reconstruct_n(0, nlist, half_centroids.data());
    EXPECT_EQ(half_centroids, float_centroids);
}

TEST(IVF, train_float16_clustering_index_sees_bounded_batches) {
    constexpr int d = 8;
    constexpr int n = 60;
    constexpr int nlist = 3;
    auto encoded = encode_fp16(make_training_data(n, d));

    faiss::IndexFlatL2 quantizer(d);
    TrackingFlatL2 clustering_index(d);
    faiss::IndexIVFScalarQuantizer index(
            &quantizer, d, nlist, faiss::ScalarQuantizer::QT_8bit);
    index.clustering_index = &clustering_index;
    index.cp = small_clustering_params(3);
    index.cp.decode_block_size = 5;
    index.train_ex(n, encoded.data(), faiss::NumericType::Float16);

    EXPECT_TRUE(index.is_trained);
    EXPECT_EQ(quantizer.ntotal, nlist);
    EXPECT_GT(clustering_index.max_search_rows, 0);
    EXPECT_LE(clustering_index.max_search_rows, index.cp.decode_block_size);
}

TEST(IVF, train_float16_rejected_by_dedup) {
    constexpr int d = 4;
    constexpr int n = 16;
    auto encoded = encode_fp16(make_training_data(n, d));
    faiss::IndexFlatL2 quantizer(d);
    faiss::IndexIVFFlatDedup index(&quantizer, d, 2);
    EXPECT_THROW(
            index.train_ex(n, encoded.data(), faiss::NumericType::Float16),
            faiss::FaissException);
}

TEST(IVF, train_ex_float32_preserves_deduplication) {
    constexpr int d = 1;
    constexpr int n = 4;
    constexpr int nlist = 1;
    std::vector<float> input{0.0f, 0.0f, 0.0f, 10.0f};

    faiss::IndexFlatL2 direct_quantizer(d);
    faiss::IndexIVFFlatDedup direct(&direct_quantizer, d, nlist);
    direct.cp = small_clustering_params(1);
    direct.train(n, input.data());

    faiss::IndexFlatL2 train_ex_quantizer(d);
    faiss::IndexIVFFlatDedup train_ex(&train_ex_quantizer, d, nlist);
    train_ex.cp = small_clustering_params(1);
    train_ex.train_ex(n, input.data(), faiss::NumericType::Float32);

    float direct_centroid;
    float train_ex_centroid;
    direct_quantizer.reconstruct(0, &direct_centroid);
    train_ex_quantizer.reconstruct(0, &train_ex_centroid);
    EXPECT_FLOAT_EQ(train_ex_centroid, direct_centroid);
    EXPECT_FLOAT_EQ(train_ex_centroid, 5.0f);
}

namespace {

/// forces the BLAS kernels with small query tiles, restored on scope exit
struct ScopedBlasTiles {
    int threshold = faiss::distance_compute_blas_threshold;
    int query_bs = faiss::distance_compute_blas_query_bs;

    explicit ScopedBlasTiles(int query_bs_in) {
        faiss::distance_compute_blas_threshold = 0;
        faiss::distance_compute_blas_query_bs = query_bs_in;
    }

    ~ScopedBlasTiles() {
        faiss::distance_compute_blas_threshold = threshold;
        faiss::distance_compute_blas_query_bs = query_bs;
    }
};

std::vector<float> make_queries(size_t n, size_t d) {
    std::vector<float> x(n * d);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<float>((i * 37) % 113) / 19.0f - 3.0f;
    }
    return x;
}

} // namespace

TEST(IndexFlat, search_ex_float16_native_matches_float32) {
    constexpr size_t nb = 500;
    constexpr size_t nq = 300;
    ScopedBlasTiles tiles(64); // several query tiles per search
    for (faiss::MetricType metric :
         {faiss::METRIC_L2, faiss::METRIC_INNER_PRODUCT}) {
        for (size_t d : {37, 64}) {
            for (size_t k : {1, 5}) {
                auto base = make_training_data(nb, d);
                auto queries = encode_fp16(make_queries(nq, d));
                auto rounded = decode_fp16(queries);
                faiss::IndexFlat index(d, metric);
                index.add(nb, base.data());

                std::vector<float> fd(nq * k), hd(nq * k);
                std::vector<faiss::idx_t> fi(nq * k), hi(nq * k);
                index.search(nq, rounded.data(), k, fd.data(), fi.data());
                index.search_ex(
                        nq,
                        queries.data(),
                        faiss::NumericType::Float16,
                        k,
                        hd.data(),
                        hi.data());
                EXPECT_EQ(hi, fi)
                        << "metric=" << metric << " d=" << d << " k=" << k;
                for (size_t i = 0; i < fd.size(); ++i) {
                    EXPECT_NEAR(
                            hd[i],
                            fd[i],
                            1e-5f * std::max(1.0f, std::fabs(fd[i])))
                            << "metric=" << metric << " d=" << d << " k=" << k
                            << " result=" << i;
                }
            }
        }
    }
}

TEST(IndexFlat, search_ex_float16_with_selector_matches_float32) {
    constexpr size_t d = 48;
    constexpr size_t nb = 400;
    constexpr size_t nq = 50;
    constexpr size_t k = 3;
    ScopedBlasTiles tiles(16);
    auto base = make_training_data(nb, d);
    auto queries = encode_fp16(make_queries(nq, d));
    auto rounded = decode_fp16(queries);
    faiss::IndexFlatL2 index(d);
    index.add(nb, base.data());

    faiss::IDSelectorRange sel(100, 250);
    faiss::SearchParameters params;
    params.sel = &sel;
    std::vector<float> fd(nq * k), hd(nq * k);
    std::vector<faiss::idx_t> fi(nq * k), hi(nq * k);
    index.search(nq, rounded.data(), k, fd.data(), fi.data(), &params);
    index.search_ex(
            nq,
            queries.data(),
            faiss::NumericType::Float16,
            k,
            hd.data(),
            hi.data(),
            &params);
    EXPECT_EQ(hi, fi);
    EXPECT_EQ(hd, fd);
    for (auto id : hi) {
        EXPECT_TRUE(id >= 100 && id < 250);
    }
}

TEST(IndexFlat, search_ex_float16_keeps_flat1d_search) {
    constexpr size_t nb = 100;
    constexpr size_t nq = 20;
    constexpr size_t k = 4;
    auto base = make_training_data(nb, 1);
    auto queries = encode_fp16(make_queries(nq, 1));
    auto rounded = decode_fp16(queries);
    faiss::IndexFlat1D index;
    index.add(nb, base.data());

    std::vector<float> fd(nq * k), hd(nq * k);
    std::vector<faiss::idx_t> fi(nq * k), hi(nq * k);
    index.search(nq, rounded.data(), k, fd.data(), fi.data());
    index.search_ex(
            nq,
            queries.data(),
            faiss::NumericType::Float16,
            k,
            hd.data(),
            hi.data());
    EXPECT_EQ(hi, fi);
    EXPECT_EQ(hd, fd); // L1 distances from IndexFlat1D::search
}

TEST(Clustering, train_ex_float16_native_flat_assignment_matches_float32) {
    constexpr size_t n = 3000;
    constexpr size_t d = 48;
    constexpr size_t k = 8;
    ScopedBlasTiles tiles(256);
    auto encoded = encode_fp16(make_training_data(n, d));
    auto rounded = decode_fp16(encoded);
    auto cp = small_clustering_params(5);

    faiss::Clustering full(d, k, cp);
    faiss::IndexFlatL2 full_index(d);
    full.train(n, rounded.data(), full_index);

    faiss::Clustering half(d, k, cp);
    faiss::IndexFlatL2 half_index(d);
    half.train_ex(n, encoded.data(), faiss::NumericType::Float16, half_index);

    EXPECT_EQ(half.centroids, full.centroids);
    ASSERT_EQ(half.iteration_stats.size(), full.iteration_stats.size());
    for (size_t i = 0; i < full.iteration_stats.size(); ++i) {
        EXPECT_EQ(half.iteration_stats[i].obj, full.iteration_stats[i].obj);
    }
}
