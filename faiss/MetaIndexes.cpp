/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/MetaIndexes.h>

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <limits>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/WorkerThread.h>
#include <faiss/utils/random.h>
#include <faiss/utils/utils.h>

namespace faiss {

/*****************************************************
 * IndexSplitVectors implementation
 *******************************************************/

IndexSplitVectors::IndexSplitVectors(idx_t d, bool threaded)
        : Index(d), own_fields(false), threaded(threaded), sum_d(0) {}

void IndexSplitVectors::add_sub_index(Index* index) {
    sub_indexes.push_back(index);
    sync_with_sub_indexes();
}

void IndexSplitVectors::sync_with_sub_indexes() {
    if (sub_indexes.empty())
        return;
    Index* index0 = sub_indexes[0];
    sum_d = index0->d;
    metric_type = index0->metric_type;
    is_trained = index0->is_trained;
    ntotal = index0->ntotal;
    for (int i = 1; i < sub_indexes.size(); i++) {
        Index* index = sub_indexes[i];
        FAISS_THROW_IF_NOT(metric_type == index->metric_type);
        FAISS_THROW_IF_NOT(ntotal == index->ntotal);
        sum_d += index->d;
    }
}

void IndexSplitVectors::add(idx_t /*n*/, const float* /*x*/) {
    FAISS_THROW_MSG("not implemented");
}

void IndexSplitVectors::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(
            !params, "search params not supported for this index");
    FAISS_THROW_IF_NOT_MSG(k == 1, "search implemented only for k=1");
    FAISS_THROW_IF_NOT_MSG(
            sum_d == d, "not enough indexes compared to # dimensions");

    int64_t nshard = sub_indexes.size();

    std::unique_ptr<float[]> all_distances(new float[nshard * k * n]);
    std::unique_ptr<idx_t[]> all_labels(new idx_t[nshard * k * n]);

    auto query_func =
            [n, x, k, distances, labels, &all_distances, &all_labels, this](
                    int no) {
                const IndexSplitVectors* index = this;
                float* distances1 =
                        no == 0 ? distances : all_distances.get() + no * k * n;
                idx_t* labels1 =
                        no == 0 ? labels : all_labels.get() + no * k * n;
                if (index->verbose)
                    printf("begin query shard %d on %" PRId64 " points\n",
                           no,
                           n);
                const Index* sub_index = index->sub_indexes[no];
                int64_t sub_d = sub_index->d, d = index->d;
                idx_t ofs = 0;
                for (int i = 0; i < no; i++)
                    ofs += index->sub_indexes[i]->d;

                std::unique_ptr<float[]> sub_x(new float[sub_d * n]);
                for (idx_t i = 0; i < n; i++)
                    memcpy(sub_x.get() + i * sub_d,
                           x + ofs + i * d,
                           sub_d * sizeof(float));
                sub_index->search(n, sub_x.get(), k, distances1, labels1);
                if (index->verbose)
                    printf("end query shard %d\n", no);
            };

    if (!threaded) {
        for (int i = 0; i < nshard; i++) {
            query_func(i);
        }
    } else {
        std::vector<std::unique_ptr<WorkerThread>> threads;
        std::vector<std::future<bool>> v;

        for (int i = 0; i < nshard; i++) {
            threads.emplace_back(new WorkerThread());
            WorkerThread* wt = threads.back().get();
            v.emplace_back(wt->add([i, query_func]() { query_func(i); }));
        }

        // Blocking wait for completion
        for (auto& func : v) {
            func.get();
        }
    }

    int64_t factor = 1;
    for (int i = 0; i < nshard; i++) {
        if (i > 0) { // results of 0 are already in the table
            const float* distances_i = all_distances.get() + i * k * n;
            const idx_t* labels_i = all_labels.get() + i * k * n;
            for (int64_t j = 0; j < n; j++) {
                if (labels[j] >= 0 && labels_i[j] >= 0) {
                    labels[j] += labels_i[j] * factor;
                    distances[j] += distances_i[j];
                } else {
                    labels[j] = -1;
                    distances[j] = std::numeric_limits<float>::quiet_NaN();
                }
            }
        }
        factor *= sub_indexes[i]->ntotal;
    }
}

void IndexSplitVectors::train(idx_t /*n*/, const float* /*x*/) {
    FAISS_THROW_MSG("not implemented");
}

void IndexSplitVectors::reset() {
    FAISS_THROW_MSG("not implemented");
}

IndexSplitVectors::~IndexSplitVectors() {
    if (own_fields) {
        for (int s = 0; s < sub_indexes.size(); s++)
            delete sub_indexes[s];
    }
}

/********************************************************
 * IndexRandom implementation
 */

IndexRandom::IndexRandom(
        idx_t d,
        idx_t ntotal,
        int64_t seed,
        MetricType metric_type)
        : Index(d, metric_type), seed(seed) {
    this->ntotal = ntotal;
    is_trained = true;
}

void IndexRandom::add(idx_t n, const float*) {
    ntotal += n;
}

void IndexRandom::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(
            !params, "search params not supported for this index");
    FAISS_THROW_IF_NOT(k <= ntotal);
#pragma omp parallel for if (n > 1000)
    for (idx_t i = 0; i < n; i++) {
        RandomGenerator rng(
                seed + ivec_checksum(d, (const int32_t*)(x + i * d)));
        idx_t* I = labels + i * k;
        float* D = distances + i * k;
        // assumes k << ntotal
        if (k < 100 * ntotal) {
            std::unordered_set<idx_t> map;
            for (int j = 0; j < k; j++) {
                idx_t ii;
                for (;;) {
                    // yes I know it's not strictly uniform...
                    ii = rng.rand_int64() % ntotal;
                    if (map.count(ii) == 0) {
                        break;
                    }
                }
                I[j] = ii;
                map.insert(ii);
            }
        } else {
            std::vector<idx_t> perm(ntotal);
            for (idx_t j = 0; j < ntotal; j++) {
                perm[j] = j;
            }
            for (int j = 0; j < k; j++) {
                std::swap(perm[j], perm[rng.rand_int(ntotal)]);
                I[j] = perm[j];
            }
        }
        float dprev = 0;
        for (int j = 0; j < k; j++) {
            float step = rng.rand_float();
            if (is_similarity_metric(metric_type)) {
                step = -step;
            }
            dprev += step;
            D[j] = dprev;
        }
    }
}

void IndexRandom::reconstruct(idx_t key, float* recons) const {
    RandomGenerator rng(seed + 123332 + key);
    for (size_t i = 0; i < d; i++) {
        recons[i] = rng.rand_float();
    }
}

void IndexRandom::reset() {
    ntotal = 0;
}

IndexRandom::~IndexRandom() = default;

} // namespace faiss
