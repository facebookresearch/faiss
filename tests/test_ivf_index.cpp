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
