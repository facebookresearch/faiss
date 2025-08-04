/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/IndexPreTransform.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/FaissAssert.h>

namespace faiss {

/*********************************************
 * IndexPreTransform
 *********************************************/

IndexPreTransform::IndexPreTransform() : index(nullptr), own_fields(false) {}

IndexPreTransform::IndexPreTransform(Index* index)
        : Index(index->d, index->metric_type), index(index), own_fields(false) {
    is_trained = index->is_trained;
    ntotal = index->ntotal;
}

IndexPreTransform::IndexPreTransform(VectorTransform* ltrans, Index* index)
        : Index(index->d, index->metric_type), index(index), own_fields(false) {
    is_trained = index->is_trained;
    ntotal = index->ntotal;
    prepend_transform(ltrans);
}

void IndexPreTransform::prepend_transform(VectorTransform* ltrans) {
    FAISS_THROW_IF_NOT(ltrans->d_out == d);
    is_trained = is_trained && ltrans->is_trained;
    chain.insert(chain.begin(), ltrans);
    d = ltrans->d_in;
}

IndexPreTransform::~IndexPreTransform() {
    if (own_fields) {
        for (int i = 0; i < chain.size(); i++) {
            delete chain[i];
        }
        delete index;
    }
}

void IndexPreTransform::train(idx_t n, const float* x) {
    int last_untrained = 0;
    if (!index->is_trained) {
        last_untrained = chain.size();
    } else {
        for (int i = chain.size() - 1; i >= 0; i--) {
            if (!chain[i]->is_trained) {
                last_untrained = i;
                break;
            }
        }
    }
    const float* prev_x = x;
    std::unique_ptr<const float[]> del;

    if (verbose) {
        printf("IndexPreTransform::train: training chain 0 to %d\n",
               last_untrained);
    }

    for (int i = 0; i <= last_untrained; i++) {
        if (i < chain.size()) {
            VectorTransform* ltrans = chain[i];
            if (!ltrans->is_trained) {
                if (verbose) {
                    printf("   Training chain component %d/%zd\n",
                           i,
                           chain.size());
                    if (OPQMatrix* opqm = dynamic_cast<OPQMatrix*>(ltrans)) {
                        opqm->verbose = true;
                    }
                }
                ltrans->train(n, prev_x);
            }
        } else {
            if (verbose) {
                printf("   Training sub-index\n");
            }
            index->train(n, prev_x);
        }
        if (i == last_untrained) {
            break;
        }
        if (verbose) {
            printf("   Applying transform %d/%zd\n", i, chain.size());
        }

        float* xt = chain[i]->apply(n, prev_x);

        if (prev_x != x) {
            del.reset();
        }

        prev_x = xt;
        del.reset(xt);
    }

    is_trained = true;
}

void IndexPreTransform::train(
        idx_t n,
        const void* x,
        NumericType numeric_type) {
    Index::train(n, x, numeric_type);
}

const float* IndexPreTransform::apply_chain(idx_t n, const float* x) const {
    const float* prev_x = x;
    std::unique_ptr<const float[]> del;

    for (int i = 0; i < chain.size(); i++) {
        float* xt = chain[i]->apply(n, prev_x);
        std::unique_ptr<const float[]> del2(xt);
        del2.swap(del);
        prev_x = xt;
    }
    del.release();
    return prev_x;
}

void IndexPreTransform::reverse_chain(idx_t n, const float* xt, float* x)
        const {
    const float* next_x = xt;
    std::unique_ptr<const float[]> del;

    for (int i = chain.size() - 1; i >= 0; i--) {
        float* prev_x = (i == 0) ? x : new float[n * chain[i]->d_in];
        std::unique_ptr<const float[]> del2((prev_x == x) ? nullptr : prev_x);
        chain[i]->reverse_transform(n, next_x, prev_x);
        del2.swap(del);
        next_x = prev_x;
    }
}

void IndexPreTransform::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(is_trained);
    TransformedVectors tv(x, apply_chain(n, x));
    index->add(n, tv.x);
    ntotal = index->ntotal;
}

void IndexPreTransform::add(idx_t n, const void* x, NumericType numeric_type) {
    Index::add(n, x, numeric_type);
}

void IndexPreTransform::add_with_ids(
        idx_t n,
        const float* x,
        const idx_t* xids) {
    FAISS_THROW_IF_NOT(is_trained);
    TransformedVectors tv(x, apply_chain(n, x));
    index->add_with_ids(n, tv.x, xids);
    ntotal = index->ntotal;
}

namespace {

const SearchParameters* extract_index_search_params(
        const SearchParameters* params_in) {
    auto params = dynamic_cast<const SearchParametersPreTransform*>(params_in);
    return params ? params->index_params : params_in;
}

} // namespace

void IndexPreTransform::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(is_trained);
    const float* xt = apply_chain(n, x);
    std::unique_ptr<const float[]> del(xt == x ? nullptr : xt);
    index->search(
            n, xt, k, distances, labels, extract_index_search_params(params));
}

void IndexPreTransform::search(
        idx_t n,
        const void* x,
        NumericType numeric_type,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    Index::search(n, x, numeric_type, k, distances, labels, params);
}

void IndexPreTransform::range_search(
        idx_t n,
        const float* x,
        float radius,
        RangeSearchResult* result,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(is_trained);
    TransformedVectors tv(x, apply_chain(n, x));
    index->range_search(
            n, tv.x, radius, result, extract_index_search_params(params));
}

void IndexPreTransform::reset() {
    index->reset();
    ntotal = 0;
}

size_t IndexPreTransform::remove_ids(const IDSelector& sel) {
    size_t nremove = index->remove_ids(sel);
    ntotal = index->ntotal;
    return nremove;
}

void IndexPreTransform::reconstruct(idx_t key, float* recons) const {
    float* x = chain.empty() ? recons : new float[index->d];
    std::unique_ptr<float[]> del(recons == x ? nullptr : x);
    // Initial reconstruction
    index->reconstruct(key, x);

    // Revert transformations from last to first
    reverse_chain(1, x, recons);
}

void IndexPreTransform::reconstruct_n(idx_t i0, idx_t ni, float* recons) const {
    float* x = chain.empty() ? recons : new float[ni * index->d];
    std::unique_ptr<float[]> del(recons == x ? nullptr : x);
    // Initial reconstruction
    index->reconstruct_n(i0, ni, x);

    // Revert transformations from last to first
    reverse_chain(ni, x, recons);
}

void IndexPreTransform::search_and_reconstruct(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        float* recons,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(is_trained);

    TransformedVectors trans(x, apply_chain(n, x));

    float* recons_temp = chain.empty() ? recons : new float[n * k * index->d];
    std::unique_ptr<float[]> del2(
            (recons_temp == recons) ? nullptr : recons_temp);
    index->search_and_reconstruct(
            n,
            trans.x,
            k,
            distances,
            labels,
            recons_temp,
            extract_index_search_params(params));

    // Revert transformations from last to first
    reverse_chain(n * k, recons_temp, recons);
}

size_t IndexPreTransform::sa_code_size() const {
    return index->sa_code_size();
}

void IndexPreTransform::sa_encode(idx_t n, const float* x, uint8_t* bytes)
        const {
    TransformedVectors tv(x, apply_chain(n, x));
    index->sa_encode(n, tv.x, bytes);
}

void IndexPreTransform::sa_decode(idx_t n, const uint8_t* bytes, float* x)
        const {
    if (chain.empty()) {
        index->sa_decode(n, bytes, x);
    } else {
        std::unique_ptr<float[]> x1(new float[index->d * n]);
        index->sa_decode(n, bytes, x1.get());
        // Revert transformations from last to first
        reverse_chain(n, x1.get(), x);
    }
}

void IndexPreTransform::merge_from(Index& otherIndex, idx_t add_id) {
    check_compatible_for_merge(otherIndex);
    auto other = static_cast<const IndexPreTransform*>(&otherIndex);
    index->merge_from(*other->index, add_id);
    ntotal = index->ntotal;
}

void IndexPreTransform::check_compatible_for_merge(
        const Index& otherIndex) const {
    auto other = dynamic_cast<const IndexPreTransform*>(&otherIndex);
    FAISS_THROW_IF_NOT(other);
    FAISS_THROW_IF_NOT(chain.size() == other->chain.size());
    for (int i = 0; i < chain.size(); i++) {
        chain[i]->check_identical(*other->chain[i]);
    }
    index->check_compatible_for_merge(*other->index);
}

namespace {

struct PreTransformDistanceComputer : DistanceComputer {
    const IndexPreTransform* index;
    std::unique_ptr<DistanceComputer> sub_dc;
    std::unique_ptr<const float[]> query;

    explicit PreTransformDistanceComputer(const IndexPreTransform* index)
            : index(index), sub_dc(index->index->get_distance_computer()) {}

    void set_query(const float* x) override {
        const float* xt = index->apply_chain(1, x);
        if (xt == x) {
            sub_dc->set_query(x);
        } else {
            query.reset(xt);
            sub_dc->set_query(xt);
        }
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return sub_dc->symmetric_dis(i, j);
    }

    float operator()(idx_t i) override {
        return (*sub_dc)(i);
    }
};

} // anonymous namespace

DistanceComputer* IndexPreTransform::get_distance_computer() const {
    if (chain.empty()) {
        return index->get_distance_computer();
    } else {
        return new PreTransformDistanceComputer(this);
    }
}

} // namespace faiss
