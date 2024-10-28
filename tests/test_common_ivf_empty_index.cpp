/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <vector>

#include <faiss/IndexIVF.h>
#include <faiss/clone_index.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/index_factory.h>
#include <faiss/invlists/InvertedLists.h>
#include <faiss/utils/random.h>

/* This demonstrates how to query several independent IVF indexes with a trained
 *index in common. This avoids to duplicate the coarse quantizer and metadata
 *in memory.
 **/

namespace {

int d = 64;

} // namespace

std::vector<float> get_random_vectors(size_t n, int seed) {
    std::vector<float> x(n * d);
    faiss::rand_smooth_vectors(n, d, x.data(), seed);
    seed++;
    return x;
}

/** InvetedLists implementation that dispatches the search to an InvertedList
 * object that is passed in at query time */

struct DispatchingInvertedLists : faiss::ReadOnlyInvertedLists {
    DispatchingInvertedLists(size_t nlist, size_t code_size)
            : faiss::ReadOnlyInvertedLists(nlist, code_size) {
        use_iterator = true;
    }

    faiss::InvertedListsIterator* get_iterator(
            size_t list_no,
            void* inverted_list_context = nullptr) const override {
        assert(inverted_list_context);
        auto il =
                static_cast<const faiss::InvertedLists*>(inverted_list_context);
        return il->get_iterator(list_no);
    }

    using idx_t = faiss::idx_t;

    size_t list_size(size_t list_no) const override {
        FAISS_THROW_MSG("use iterator interface");
    }
    const uint8_t* get_codes(size_t list_no) const override {
        FAISS_THROW_MSG("use iterator interface");
    }
    const idx_t* get_ids(size_t list_no) const override {
        FAISS_THROW_MSG("use iterator interface");
    }
};

TEST(COMMON, test_common_trained_index) {
    int N = 3;    // number of independent indexes
    int nt = 500; // training vectors
    int nb = 200; // nb database vectors per index
    int nq = 10;  // nb queries performed on each index
    int k = 4;    // restults requested per query

    // construct and build an "empty index": a trained index that does not
    // itself hold any data
    std::unique_ptr<faiss::IndexIVF> empty_index(dynamic_cast<faiss::IndexIVF*>(
            faiss::index_factory(d, "IVF32,PQ8np")));
    auto xt = get_random_vectors(nt, 123);
    empty_index->train(nt, xt.data());
    empty_index->nprobe = 4;

    // reference run: build one index for each set of db / queries and record
    // results
    std::vector<std::vector<faiss::idx_t>> ref_I(N);

    for (int i = 0; i < N; i++) {
        // clone the empty index
        std::unique_ptr<faiss::Index> index(
                faiss::clone_index(empty_index.get()));
        auto xb = get_random_vectors(nb, 1234 + i);
        auto xq = get_random_vectors(nq, 12345 + i);
        // add vectors and perform a search
        index->add(nb, xb.data());
        std::vector<float> D(k * nq);
        std::vector<faiss::idx_t> I(k * nq);
        index->search(nq, xq.data(), k, D.data(), I.data());
        // record result as reference
        ref_I[i] = I;
    }

    // build a set of inverted lists for each independent index
    std::vector<faiss::ArrayInvertedLists> sub_invlists;

    for (int i = 0; i < N; i++) {
        // swap in other inverted lists
        sub_invlists.emplace_back(empty_index->nlist, empty_index->code_size);
        faiss::InvertedLists* invlists = &sub_invlists.back();

        // replace_invlists swaps in a new InvertedLists for an existing index
        empty_index->replace_invlists(invlists, false);
        empty_index->reset(); // reset id counter to 0
        // populate inverted lists
        auto xb = get_random_vectors(nb, 1234 + i);
        empty_index->add(nb, xb.data());
    }

    // perform search dispatching to the sub-invlists. At search time, we don't
    // use replace_invlists because that would wreak havoc in a multithreaded
    // context
    DispatchingInvertedLists di(empty_index->nlist, empty_index->code_size);
    empty_index->replace_invlists(&di, false);

    std::vector<std::vector<faiss::idx_t>> new_I(N);

    // run searches in the independent indexes but with a common empty_index
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        auto xq = get_random_vectors(nq, 12345 + i);
        std::vector<float> D(k * nq);
        std::vector<faiss::idx_t> I(k * nq);

        // here we set to what sub-index the queries should be directed
        faiss::SearchParametersIVF params;
        params.nprobe = empty_index->nprobe;
        params.inverted_list_context = &sub_invlists[i];

        empty_index->search(nq, xq.data(), k, D.data(), I.data(), &params);
        new_I[i] = I;
    }

    // compare with reference reslt
    for (int i = 0; i < N; i++) {
        ASSERT_EQ(ref_I[i], new_I[i]);
    }
}
