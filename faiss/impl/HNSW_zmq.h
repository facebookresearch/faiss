#pragma once

#include <cassert>
#include <cstddef>
#include <memory>
#include <vector>

#include <faiss/Index.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>
namespace faiss {
struct ZmqDistanceComputer : DistanceComputer {
    size_t d;
    const int ZMQ_PORT = 5555;
    MetricType metric_type;
    float metric_arg;
    const Index* storage;
    std::unique_ptr<DistanceComputer> storage_dc_orig;
    std::unique_ptr<DistanceComputer> storage_dc_search;
    std::vector<float> query;

    std::vector<float> last_fetched_zmq_vector;

    const float* get_query() override {
        return query.data();
    }
    mutable size_t fetch_count = 0;
    ZmqDistanceComputer(size_t dim, MetricType mt, float marg = 0)
            : d(dim), metric_type(mt), metric_arg(marg) {
        FAISS_THROW_IF_NOT_MSG(d > 0, "Dimension must be positive");
        query.resize(d);
        last_fetched_zmq_vector.resize(d); // Preallocate
        reset_fetch_count();               // Initialize count
        printf("ZmqDistanceComputer initialized: d=%zu, metric=%d\n",
               d,
               (int)mt);
    }
    ZmqDistanceComputer(const Index* storage_ref)
            : d(storage_ref->d),
              metric_type(storage_ref->metric_type),
              metric_arg(storage_ref->metric_arg),
              storage(storage_ref),
              storage_dc_orig(storage_ref->get_distance_computer()),
              storage_dc_search(
                      is_similarity_metric(storage_ref->metric_type)
                              ? new NegativeDistanceComputer(
                                        storage_ref->get_distance_computer())
                              : storage_ref->get_distance_computer()) {
        query.resize(d);
        last_fetched_zmq_vector.resize(d); // Preallocate
        FAISS_THROW_IF_NOT_MSG(
                storage != nullptr,
                "Storage cannot be null for ZmqDistanceComputer");
        FAISS_THROW_IF_NOT(storage_dc_orig && storage_dc_search);
        reset_fetch_count(); // Initialize count
    }

    size_t get_fetch_count() const override {
        return fetch_count;
    }

    void reset_fetch_count() override {
        fetch_count = 0;
    }

    float operator()(idx_t i) override {
        const float* vec_zmq = get_vector_zmq(i);
        if (!vec_zmq)
            return (metric_type == METRIC_INNER_PRODUCT)
                    ? -std::numeric_limits<float>::max()
                    : std::numeric_limits<float>::max();
        if (is_similarity_metric(metric_type)) {
            return -fvec_inner_product(query.data(), vec_zmq, d);
        } else {
            return fvec_L2sqr(query.data(), vec_zmq, d);
        }
    }
    float symmetric_dis(idx_t i, idx_t j) override {
        const float* vec_i_zmq = get_vector_zmq(i);
        const float* vec_j_zmq = get_vector_zmq(j);
        if (!vec_i_zmq || !vec_j_zmq)
            return std::numeric_limits<float>::max();
        return fvec_L2sqr(vec_i_zmq, vec_j_zmq, d);
    }

    float distance_func(const float* vec_zmq) {
        if (!vec_zmq) // Handle case where ZMQ fetch failed
            return (metric_type == METRIC_INNER_PRODUCT)
                    ? -std::numeric_limits<float>::max()
                    : std::numeric_limits<float>::max();

        switch (metric_type) {
            case METRIC_INNER_PRODUCT:
                return -fvec_inner_product(
                        query.data(), vec_zmq, d); // Return negative IP
            case METRIC_L2:
                return fvec_L2sqr(
                        query.data(), vec_zmq, d); // Return L2 squared
            default:
                return std::numeric_limits<float>::max();
        }
    }

    void set_query(const float* x) override {
        reset_fetch_count();
        memcpy(query.data(), x, d * sizeof(float));
    }
    ~ZmqDistanceComputer() override = default;

    void distances_batch_4(
            idx_t id0,
            idx_t id1,
            idx_t id2,
            idx_t id3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) override {
        std::vector<idx_t> batch_ids = {id0, id1, id2, id3};
        std::vector<float> batch_distances;

        this->distances_batch(batch_ids, batch_distances);

        dis0 = batch_distances[0];
        dis1 = batch_distances[1];
        dis2 = batch_distances[2];
        dis3 = batch_distances[3];
    }

    const float* get_vector_zmq(idx_t id);
    void distances_batch(
            const std::vector<idx_t>& ids,
            std::vector<float>& distances_out) override;
};

} // namespace faiss