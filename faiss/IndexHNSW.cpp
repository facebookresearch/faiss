/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexHNSW.h>

#include <omp.h>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <limits>
#include <memory>
#include <queue>
#include <random>

#include <cstdint>

#include <faiss/Index2Layer.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>
#include <faiss/utils/sorting.h>

#include <msgpack.hpp>
#include <zmq.h>
#include <sstream> // For msgpack stringstream buffer

namespace faiss {

using MinimaxHeap = HNSW::MinimaxHeap;
using storage_idx_t = HNSW::storage_idx_t;
using NodeDistFarther = HNSW::NodeDistFarther;

HNSWStats hnsw_stats;
// --- MessagePack Data Structures (Define simple structs for serialization) ---
struct EmbeddingRequestMsgpack {
    std::vector<uint32_t> node_ids;
    MSGPACK_DEFINE_ARRAY(node_ids); // Use array format [ [ids] ]
};

struct EmbeddingResponseMsgpack {
    // Store dimensions as separate fields for clarity with msgpack map
    // Or keep as vector [batch_size, dim] if using array format
    // Let's use array format for simplicity matching MSGPACK_DEFINE_ARRAY
    std::vector<uint32_t> dimensions; // [batch_size, embedding_dim]
    // Store flat embedding data as raw bytes or vector<float>
    // Using vector<float> is easier to handle with msgpack-c directly
    std::vector<float>
            embeddings_data; // Flattened [batch_size * embedding_dim]
    // Optional: Add missing_ids if needed
    // std::vector<uint32_t> missing_ids;

    MSGPACK_DEFINE_ARRAY(dimensions, embeddings_data); // [ [dims], [data] ]
};

// --- ZMQ Fetch Function (Using MessagePack) ---
bool fetch_embeddings_zmq(
        const std::vector<uint32_t>& node_ids,
        std::vector<std::vector<float>>& out_embeddings,
        int zmq_port = 5555) // Default port kept
{
    // 0) 构造请求
    EmbeddingRequestMsgpack req_msgpack;
    req_msgpack.node_ids = node_ids;

    // 序列化请求
    std::stringstream buffer;
    try {
        msgpack::pack(buffer, req_msgpack);
    } catch (const std::exception& e) {
        std::cerr << "MessagePack pack failed: " << e.what() << std::endl;
        return false;
    }
    std::string req_str = buffer.str();

    // 1) 创建上下文和套接字 (Same as before)
    void* context = zmq_ctx_new();
    if (!context) {
        fprintf(stderr,
                "[fetch_zmq] zmq_ctx_new failed: %s\n",
                zmq_strerror(zmq_errno()));
        return false;
    }
    void* socket = zmq_socket(context, ZMQ_REQ);
    if (!socket) {
        fprintf(stderr,
                "[fetch_zmq] zmq_socket failed: %s\n",
                zmq_strerror(zmq_errno()));
        zmq_ctx_destroy(context);
        return false;
    }
    int timeout = 30000;
    zmq_setsockopt(socket, ZMQ_RCVTIMEO, &timeout, sizeof(timeout));
    zmq_setsockopt(socket, ZMQ_SNDTIMEO, &timeout, sizeof(timeout));
    std::string endpoint = "tcp://127.0.0.1:" + std::to_string(zmq_port);
    if (zmq_connect(socket, endpoint.c_str()) != 0) {
        fprintf(stderr,
                "[fetch_zmq] zmq_connect failed: %s\n",
                zmq_strerror(zmq_errno()));
        zmq_close(socket);
        zmq_ctx_destroy(context);
        return false;
    }

    // 4) 发送序列化请求
    if (zmq_send(socket, req_str.data(), req_str.size(), 0) < 0) { /*...*/
        fprintf(stderr,
                "[fetch_zmq] zmq_msg_recv failed: %s\n",
                zmq_strerror(zmq_errno()));
        zmq_close(socket);
        zmq_ctx_destroy(context);
        return false;
    }

    // 5) 等待接收服务端返回
    zmq_msg_t response;
    zmq_msg_init(&response);
    if (zmq_msg_recv(&response, socket, 0) < 0) { /*...*/
        fprintf(stderr,
                "[fetch_zmq] zmq_msg_recv failed: %s\n",
                zmq_strerror(zmq_errno()));
        zmq_msg_close(&response);
        zmq_close(socket);
        zmq_ctx_destroy(context);
        return false;
    }

    // 6) 解析响应
    EmbeddingResponseMsgpack resp_msgpack;
    const char* resp_data = static_cast<const char*>(zmq_msg_data(&response));
    size_t resp_size = zmq_msg_size(&response);
    printf("[fetch_zmq] Raw response bytes (first %d): ",
           (int)std::min((size_t)64, resp_size));

    msgpack::object_handle oh = msgpack::unpack(resp_data, resp_size);
    msgpack::object obj = oh.get();
    obj.convert(resp_msgpack); // Convert msgpack object to our struct
    for (size_t k = 0; k < std::min((size_t)64, resp_size); ++k)
        printf("%02x ", (unsigned char)resp_data[k]);
    printf("\n");

    // --- Print parsed values BEFORE NaN check ---
    printf("[fetch_zmq] Parsed response. Dimensions: %d x %d. Data floats: %zu\n",
           resp_msgpack.dimensions.empty() ? 0 : resp_msgpack.dimensions[0],
           resp_msgpack.dimensions.size() < 2 ? 0 : resp_msgpack.dimensions[1],
           resp_msgpack.embeddings_data.size());
    printf("[fetch_zmq] Parsed embeddings_data (first %d floats): ",
           (int)std::min((size_t)10, resp_msgpack.embeddings_data.size()));
    bool parse_contains_nan = false;
    for (size_t k = 0;
         k < std::min((size_t)10, resp_msgpack.embeddings_data.size());
         ++k) {
        printf("%.6f ", resp_msgpack.embeddings_data[k]);
        if (std::isnan(resp_msgpack.embeddings_data[k]))
            parse_contains_nan = true;
    }
    printf("%s\n",
           parse_contains_nan ? "!!! CONTAINS NaN AFTER PARSE !!!"
                              : "(Checked first 10 for NaN)");

    // 7) 验证维度
    if (resp_msgpack.dimensions.size() != 2) {
        std::cerr << "Server response has invalid dimensions size: "
                  << resp_msgpack.dimensions.size() << std::endl;
        zmq_msg_close(&response);
        zmq_close(socket);
        zmq_ctx_destroy(context);
        return false;
    }
    int batch_size = resp_msgpack.dimensions[0];
    int embedding_dim = resp_msgpack.dimensions[1];

    // Handle empty response
    if (batch_size == 0) {
        out_embeddings.clear();
        zmq_msg_close(&response);
        zmq_close(socket);
        zmq_ctx_destroy(context);
        return true; // Successful communication, no data returned
    }

    // 验证数据大小
    size_t expected_floats = (size_t)batch_size * embedding_dim;
    if (resp_msgpack.embeddings_data.size() != expected_floats) {
        std::cerr << "Embedding data size mismatch: Got "
                  << resp_msgpack.embeddings_data.size() << " floats, expected "
                  << expected_floats << " (" << batch_size << "x"
                  << embedding_dim << ")" << std::endl;
        zmq_msg_close(&response);
        zmq_close(socket);
        zmq_ctx_destroy(context);
        return false;
    }

    bool received_nan = false;
    for (float val : resp_msgpack.embeddings_data) {
        if (std::isnan(val)) {
            received_nan = true;
            break;
        }
    }
    if (received_nan) {
        fprintf(stderr,
                "!!! [fetch_zmq] ERROR: Final check confirms NaN values in parsed embeddings_data! First requested ID: %u !!!\n",
                node_ids.empty() ? 0 : node_ids[0]);
        return false; // Decide whether to fail here
    } else {
        printf("[fetch_zmq] Final check confirms embeddings data appears clean (no NaNs checked).\n"); // Can be verbose
    }

    // 8) 组装返回
    out_embeddings.clear();
    out_embeddings.resize(batch_size);
    const float* flat_data_ptr = resp_msgpack.embeddings_data.data();
    for (int i = 0; i < batch_size; i++) {
        out_embeddings[i].assign(
                flat_data_ptr + (size_t)i * embedding_dim,
                flat_data_ptr + ((size_t)i + 1) * embedding_dim);
    }

    // 9) 释放资源
    zmq_msg_close(&response);
    zmq_close(socket);
    zmq_ctx_destroy(context);

    return true;
}

struct ZmqDistanceComputer : DistanceComputer {
    size_t d;
    const int ZMQ_PORT = 5555;
    MetricType metric_type;
    float metric_arg;
    const Index* storage;
    std::unique_ptr<DistanceComputer> storage_dc_orig;
    std::unique_ptr<DistanceComputer> storage_dc_search;
    std::vector<float> query;

    // --- REMOVED CACHE ---
    // thread_local static std::unordered_map<idx_t, std::vector<float>>
    // zmq_embedding_cache;

    // --- ADD Member variable to hold the *last* fetched ZMQ vector ---
    // This is NOT thread-safe if the same DistanceComputer instance is shared
    // across threads, but Faiss search typically creates one DC per thread per
    // query. This is needed because we need to return a stable pointer.
    std::vector<float> last_fetched_zmq_vector;

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
    }

    // --- MODIFIED get_vector_zmq: No Cache, stores result in member ---
    const float* get_vector_zmq(idx_t id) {
        printf("DEBUG get_vector_zmq: Fetching ID %ld via ZMQ (no cache)...\n",
               (long)id); // Log fetch attempt

        std::vector<uint32_t> ids_to_fetch = {(uint32_t)id};
        std::vector<std::vector<float>>
                fetched_embeddings; // fetch_embeddings_zmq expects this
                                    // structure

        if (!fetch_embeddings_zmq(ids_to_fetch, fetched_embeddings, ZMQ_PORT)) {
            fprintf(stderr,
                    "!!! ERROR get_vector_zmq: fetch_embeddings_zmq call failed for ID %ld !!!\n",
                    (long)id);
            // Fill member with NaN to indicate failure?
            std::fill(
                    last_fetched_zmq_vector.begin(),
                    last_fetched_zmq_vector.end(),
                    std::numeric_limits<float>::quiet_NaN());
            return nullptr; // Indicate failure upstream
        }
        if (fetched_embeddings.empty() || fetched_embeddings[0].size() != d) {
            fprintf(stderr,
                    "!!! ERROR get_vector_zmq: fetch_embeddings_zmq returned incorrect data for ID %ld !!!\n",
                    (long)id);
            std::fill(
                    last_fetched_zmq_vector.begin(),
                    last_fetched_zmq_vector.end(),
                    std::numeric_limits<float>::quiet_NaN());
            return nullptr;
        }

        // --- Copy fetched data to member variable ---
        // fetched_embeddings[0] contains the vector data
        FAISS_ASSERT(fetched_embeddings[0].size() == d);
        memcpy(last_fetched_zmq_vector.data(),
               fetched_embeddings[0].data(),
               d * sizeof(float));

        // --- Log values RIGHT BEFORE returning pointer ---
        const float* return_ptr = last_fetched_zmq_vector.data();
        bool has_nan_before_return = false;
        printf("DEBUG get_vector_zmq: Fetched ID %ld. Values BEFORE return (ptr %p) [0..%d]: ",
               (long)id,
               (void*)return_ptr,
               (int)std::min((size_t)4, d - 1));
        for (size_t k = 0; k < std::min((size_t)5, d); ++k) {
            printf("%.6f ", return_ptr[k]);
            if (std::isnan(return_ptr[k]) || std::isinf(return_ptr[k]))
                has_nan_before_return = true;
        }
        printf("%s\n",
               has_nan_before_return ? "!!! HAS NaN/Inf BEFORE RETURN !!!"
                                     : "(OK Before Return)");
        // -------------------------------------------

        return return_ptr; // Return pointer to member data
    }

    // --- MODIFIED compare_and_log: Add check immediately after getting pointer
    // ---
    void compare_and_log(idx_t id, const char* context) {
        // Get pointer (which now triggers fetch and stores in
        // last_fetched_zmq_vector)
        const float* vec_zmq = get_vector_zmq(id);

        // --- ADD CHECK IMMEDIATELY AFTER get_vector_zmq returns ---
        printf("DEBUG_COMPARE [%s] ID %ld: Got vec_zmq ptr %p. Checking for NaN/Inf...\n",
               context,
               (long)id,
               (void*)vec_zmq);
        bool zmq_has_nan_inf = false;
        if (vec_zmq) { // Check if pointer is valid first
            for (size_t k = 0; k < d; ++k) {
                if (std::isnan(vec_zmq[k]) || std::isinf(vec_zmq[k])) {
                    zmq_has_nan_inf = true;
                    break;
                }
            }
            if (zmq_has_nan_inf) {
                printf("!!! ZMQ VEC HAS NaN/Inf AFTER get_vector_zmq !!! First values: ");
                for (size_t k = 0; k < std::min((size_t)5, d); ++k)
                    printf("%.6f ", vec_zmq[k]);
                printf("\n");
            } else {
                // Optional: print confirmation values look ok here
                printf("ZMQ Vec OK After Get. First values: ");
                for (size_t k = 0; k < std::min((size_t)5, d); ++k)
                    printf("%.6f ", vec_zmq[k]);
                printf("\n");
            }
        } else {
            printf("ZMQ Vec ptr is NULL after get_vector_zmq.\n");
        }
        // --- END IMMEDIATE CHECK ---

        // Fetch ground truth (unchanged)
        std::vector<float> vec_storage_data(d);
        const float* vec_storage = nullptr;
        bool reconstruct_ok = false;
        // ... (rest of reconstruct logic unchanged) ...
        if (id >= 0 && id < storage->ntotal) {
            try {
                storage->reconstruct(id, vec_storage_data.data());
                vec_storage = vec_storage_data.data();
                reconstruct_ok = true;
            } catch (...) {
                printf("DEBUG_COMPARE [%s] ID %ld: Reconstruct failed.\n",
                       context,
                       (long)id);
            }
        } else {
            printf("DEBUG_COMPARE [%s] ID %ld: Out of bounds.\n",
                   context,
                   (long)id);
        }

        // Original comparison log header
        printf("DEBUG_COMPARE [%s] ID %ld: ", context, (long)id);

        // Report fetch status (unchanged)
        if (!vec_zmq && !vec_storage) {
            printf("Both ZMQ and Storage fetch failed/skipped.\n");
            return;
        }
        if (!vec_zmq) {
            printf("ZMQ fetch failed, Storage fetch %s.\n",
                   reconstruct_ok ? "OK" : "Failed/Skipped");
            return;
        }
        if (!vec_storage) {
            printf("ZMQ fetch OK, Storage fetch failed/skipped.\n");
            return;
        }

        // Now report the NaN status we checked earlier
        if (zmq_has_nan_inf)
            printf("!!! ZMQ VEC HAS NaN/Inf !!! ");
        // (Optionally add storage NaN check here too)

        // Proceed with comparison only if ZMQ vec is valid (and optionally
        // storage vec)
        if (!zmq_has_nan_inf /* && !storage_has_nan_inf */) {
            // Compare Vectors
            float diff_sq = fvec_L2sqr(vec_zmq, vec_storage, d);
            const float tol_vec = 1e-5;
            if (std::isnan(diff_sq)) {
                printf("!!! VEC L2sqr is NaN !!! ");
            } // Check L2 result too
            else if (diff_sq > tol_vec) {
                printf("!!! VEC MISMATCH !!! Diff^2=%.6f ", diff_sq);
            } else {
                printf("Vec OK (D2=%.6f), ", diff_sq);
            }

            // Compare Distances
            float dist_storage = (*storage_dc_search)(id);
            float dist_zmq = 0;
            if (is_similarity_metric(metric_type)) {
                dist_zmq = -fvec_inner_product(query.data(), vec_zmq, d);
            } else {
                dist_zmq = fvec_L2sqr(query.data(), vec_zmq, d);
            }
            float dist_diff = std::abs(dist_storage - dist_zmq);
            const float tol_dist = 1e-3;
            if (dist_diff > tol_dist ||
                std::isnan(dist_storage) != std::isnan(dist_zmq)) {
                printf("!!! DIST MISMATCH !!! Sto=%.6f, Zmq=%.6f, Diff=%.6f\n",
                       dist_storage,
                       dist_zmq,
                       dist_diff);
            } else {
                printf("Dist OK (Sto=%.6f, Zmq=%.6f)\n",
                       dist_storage,
                       dist_zmq);
            }
        } else {
            printf("Comparison skipped due to invalid/NaN/Inf vectors.\n");
        }
    }

    // --- Other DistanceComputer methods unchanged ---
    float operator()(idx_t i) override {
        compare_and_log(i, "operator()");
        // --- RETURN STORAGE DISTANCE (for recall stability during debug) ---
        // return (*storage_dc_search)(i);
        // --- TO MAKE ZMQ FUNCTIONAL (after debugging): ---
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
        // --- End ZMQ Functional ---
    }
    float symmetric_dis(idx_t i, idx_t j) override {
        // printf("DEBUG_COMPARE [symmetric_dis] ID %ld vs %ld - Using storage
        // value.\n", (long)i, (long)j);

        // --- RETURN STORAGE DISTANCE (for recall stability during debug) ---
        // return storage_dc_search->symmetric_dis(i, j);
        // --- TO MAKE ZMQ FUNCTIONAL (after debugging): ---
        const float* vec_i_zmq = get_vector_zmq(i);
        const float* vec_j_zmq = get_vector_zmq(j);
        if (!vec_i_zmq || !vec_j_zmq)
            return std::numeric_limits<float>::max();
        // Calculate appropriate symmetric distance (e.g., L2) based on ZMQ
        // vectors
        return fvec_L2sqr(vec_i_zmq, vec_j_zmq, d);
        // --- End ZMQ Functional ---
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
                // Return max distance for unsupported metrics
                return std::numeric_limits<float>::max();
        }
    }

    void distances_batch_4(
            idx_t id0,
            idx_t id1,
            idx_t id2,
            idx_t id3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) override {
        compare_and_log(id0, "batch4_0");
        compare_and_log(id1, "batch4_1");
        compare_and_log(id2, "batch4_2");
        compare_and_log(id3, "batch4_3");
        // --- RETURN STORAGE DISTANCES (for recall stability during debug) ---
        //  storage_dc_search->distances_batch_4(id0, id1, id2, id3, dis0, dis1,
        //  dis2, dis3);
        // --- TO MAKE ZMQ FUNCTIONAL (after debugging): ---
        const float* vec0 = get_vector_zmq(id0);
        dis0 = distance_func(vec0); // Need updated distance_func
        const float* vec1 = get_vector_zmq(id1);
        dis1 = distance_func(vec1);
        const float* vec2 = get_vector_zmq(id2);
        dis2 = distance_func(vec2);
        const float* vec3 = get_vector_zmq(id3);
        dis3 = distance_func(vec3);
        // --- End ZMQ Functional ---
    }
    void set_query(const float* x) override {
        memcpy(query.data(), x, d * sizeof(float));
        storage_dc_orig->set_query(x);
        storage_dc_search->set_query(x); /* No cache to clear */
    }
    ~ZmqDistanceComputer() override = default;
};

/**************************************************************
 * add / search blocks of descriptors
 **************************************************************/

namespace {

DistanceComputer* storage_distance_computer(const Index* storage) {
    if (is_similarity_metric(storage->metric_type)) {
        return new NegativeDistanceComputer(storage->get_distance_computer());
    } else {
        return storage->get_distance_computer();
    }
}

void hnsw_add_vertices(
        IndexHNSW& index_hnsw,
        size_t n0,
        size_t n,
        const float* x,
        bool verbose,
        bool preset_levels = false) {
    size_t d = index_hnsw.d;
    HNSW& hnsw = index_hnsw.hnsw;
    size_t ntotal = n0 + n;
    double t0 = getmillisecs();
    if (verbose) {
        printf("hnsw_add_vertices: adding %zd elements on top of %zd "
               "(preset_levels=%d)\n",
               n,
               n0,
               int(preset_levels));
    }

    if (n == 0) {
        return;
    }

    int max_level = hnsw.prepare_level_tab(n, preset_levels);

    if (verbose) {
        printf("  max_level = %d\n", max_level);
    }

    std::vector<omp_lock_t> locks(ntotal);
    for (int i = 0; i < ntotal; i++)
        omp_init_lock(&locks[i]);

    // add vectors from highest to lowest level
    std::vector<int> hist;
    std::vector<int> order(n);

    { // make buckets with vectors of the same level

        // build histogram
        for (int i = 0; i < n; i++) {
            storage_idx_t pt_id = i + n0;
            int pt_level = hnsw.levels[pt_id] - 1;
            while (pt_level >= hist.size())
                hist.push_back(0);
            hist[pt_level]++;
        }

        // accumulate
        std::vector<int> offsets(hist.size() + 1, 0);
        for (int i = 0; i < hist.size() - 1; i++) {
            offsets[i + 1] = offsets[i] + hist[i];
        }

        // bucket sort
        for (int i = 0; i < n; i++) {
            storage_idx_t pt_id = i + n0;
            int pt_level = hnsw.levels[pt_id] - 1;
            order[offsets[pt_level]++] = pt_id;
        }
    }

    idx_t check_period = InterruptCallback::get_period_hint(
            max_level * index_hnsw.d * hnsw.efConstruction);

    { // perform add
        RandomGenerator rng2(789);

        int i1 = n;

        for (int pt_level = hist.size() - 1;
             pt_level >= int(!index_hnsw.init_level0);
             pt_level--) {
            int i0 = i1 - hist[pt_level];

            if (verbose) {
                printf("Adding %d elements at level %d\n", i1 - i0, pt_level);
            }

            // random permutation to get rid of dataset order bias
            for (int j = i0; j < i1; j++)
                std::swap(order[j], order[j + rng2.rand_int(i1 - j)]);

            bool interrupt = false;

#pragma omp parallel if (i1 > i0 + 100)
            {
                VisitedTable vt(ntotal);

                std::unique_ptr<DistanceComputer> dis(
                        storage_distance_computer(index_hnsw.storage));
                int prev_display =
                        verbose && omp_get_thread_num() == 0 ? 0 : -1;
                size_t counter = 0;

                // here we should do schedule(dynamic) but this segfaults for
                // some versions of LLVM. The performance impact should not be
                // too large when (i1 - i0) / num_threads >> 1
#pragma omp for schedule(static)
                for (int i = i0; i < i1; i++) {
                    storage_idx_t pt_id = order[i];
                    dis->set_query(x + (pt_id - n0) * d);

                    // cannot break
                    if (interrupt) {
                        continue;
                    }

                    hnsw.add_with_locks(
                            *dis,
                            pt_level,
                            pt_id,
                            locks,
                            vt,
                            index_hnsw.keep_max_size_level0 && (pt_level == 0));

                    if (prev_display >= 0 && i - i0 > prev_display + 10000) {
                        prev_display = i - i0;
                        printf("  %d / %d\r", i - i0, i1 - i0);
                        fflush(stdout);
                    }
                    if (counter % check_period == 0) {
                        if (InterruptCallback::is_interrupted()) {
                            interrupt = true;
                        }
                    }
                    counter++;
                }
            }
            if (interrupt) {
                FAISS_THROW_MSG("computation interrupted");
            }
            i1 = i0;
        }
        if (index_hnsw.init_level0) {
            FAISS_ASSERT(i1 == 0);
        } else {
            FAISS_ASSERT((i1 - hist[0]) == 0);
        }
    }
    if (verbose) {
        printf("Done in %.3f ms\n", getmillisecs() - t0);
    }

    for (int i = 0; i < ntotal; i++) {
        omp_destroy_lock(&locks[i]);
    }
}

} // namespace

/**************************************************************
 * IndexHNSW implementation
 **************************************************************/

IndexHNSW::IndexHNSW(int d, int M, MetricType metric, int M0)
        : Index(d, metric), hnsw(M, M0) {}

IndexHNSW::IndexHNSW(Index* storage, int M, int M0)
        : Index(storage->d, storage->metric_type),
          hnsw(M, M0),
          storage(storage) {
    metric_arg = storage->metric_arg;
}

IndexHNSW::~IndexHNSW() {
    if (own_fields) {
        delete storage;
    }
}

void IndexHNSW::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexHNSWFlat (or variants) instead of IndexHNSW directly");
    // hnsw structure does not require training
    storage->train(n, x);
    is_trained = true;
}

namespace {

template <class BlockResultHandler>
void hnsw_search(
        const IndexHNSW* index,
        idx_t n,
        const float* x,
        BlockResultHandler& bres,
        const SearchParameters* params) {
    FAISS_THROW_IF_NOT_MSG(
            index->storage,
            "No storage index, please use IndexHNSWFlat (or variants) "
            "instead of IndexHNSW directly");
    const HNSW& hnsw = index->hnsw;

    int efSearch = hnsw.efSearch;
    if (params) {
        if (const SearchParametersHNSW* hnsw_params =
                    dynamic_cast<const SearchParametersHNSW*>(params)) {
            efSearch = hnsw_params->efSearch;
        }
    }
    size_t n1 = 0, n2 = 0, ndis = 0, nhops = 0;

    idx_t check_period = InterruptCallback::get_period_hint(
            hnsw.max_level * index->d * efSearch);

    for (idx_t i0 = 0; i0 < n; i0 += check_period) {
        idx_t i1 = std::min(i0 + check_period, n);

#pragma omp parallel if (i1 - i0 > 1)
        {
            VisitedTable vt(index->ntotal);
            typename BlockResultHandler::SingleResultHandler res(bres);

            // std::unique_ptr<DistanceComputer> dis(
            //         storage_distance_computer(index->storage));
            std::unique_ptr<DistanceComputer> dis(
                    index->get_distance_computer());

#pragma omp for reduction(+ : n1, n2, ndis, nhops) schedule(guided)
            for (idx_t i = i0; i < i1; i++) {
                res.begin(i);
                dis->set_query(x + i * index->d);

                HNSWStats stats = hnsw.search(*dis, res, vt, params);
                n1 += stats.n1;
                n2 += stats.n2;
                ndis += stats.ndis;
                nhops += stats.nhops;
                res.end();
            }
        }
        InterruptCallback::check();
    }

    hnsw_stats.combine({n1, n2, ndis, nhops});
}

} // anonymous namespace

void IndexHNSW::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(k > 0);

    using RH = HeapBlockResultHandler<HNSW::C>;
    RH bres(n, distances, labels, k);

    hnsw_search(this, n, x, bres, params);

    if (is_similarity_metric(this->metric_type)) {
        // we need to revert the negated distances
        for (size_t i = 0; i < k * n; i++) {
            distances[i] = -distances[i];
        }
    }
}

void IndexHNSW::range_search(
        idx_t n,
        const float* x,
        float radius,
        RangeSearchResult* result,
        const SearchParameters* params) const {
    using RH = RangeSearchBlockResultHandler<HNSW::C>;
    RH bres(result, is_similarity_metric(metric_type) ? -radius : radius);

    hnsw_search(this, n, x, bres, params);

    if (is_similarity_metric(this->metric_type)) {
        // we need to revert the negated distances
        for (size_t i = 0; i < result->lims[result->nq]; i++) {
            result->distances[i] = -result->distances[i];
        }
    }
}

void IndexHNSW::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexHNSWFlat (or variants) instead of IndexHNSW directly");
    FAISS_THROW_IF_NOT(is_trained);
    int n0 = ntotal;
    storage->add(n, x);
    ntotal = storage->ntotal;

    hnsw_add_vertices(*this, n0, n, x, verbose, hnsw.levels.size() == ntotal);
}

void IndexHNSW::reset() {
    hnsw.reset();
    storage->reset();
    ntotal = 0;
}

void IndexHNSW::reconstruct(idx_t key, float* recons) const {
    // Create temporary fetcher
    ZmqDistanceComputer fetcher(storage);
    const float* vec = fetcher.get_vector_zmq(key);
    if (vec) {
        memcpy(recons, vec, d * sizeof(float));
    } else {
        // Let potential subsequent crash happen, or fill with NaN
        fprintf(stderr,
                "IndexHNSW::reconstruct: Failed to fetch vector for key %ld\n",
                (long)key);
        std::fill_n(recons, d, std::numeric_limits<float>::quiet_NaN());
    }

    // storage->reconstruct(key, recons);
}

/**************************************************************
 * This section of functions were used during the development of HNSW support.
 * They may be useful in the future but are dormant for now, and thus are not
 * unit tested at the moment.
 * shrink_level_0_neighbors
 * search_level_0
 * init_level_0_from_knngraph
 * init_level_0_from_entry_points
 * reorder_links
 * link_singletons
 **************************************************************/
void IndexHNSW::shrink_level_0_neighbors(int new_size) {
#pragma omp parallel
    {
        std::unique_ptr<DistanceComputer> dis(
                storage_distance_computer(storage));

#pragma omp for
        for (idx_t i = 0; i < ntotal; i++) {
            size_t begin, end;
            hnsw.neighbor_range(i, 0, &begin, &end);

            std::priority_queue<NodeDistFarther> initial_list;

            for (size_t j = begin; j < end; j++) {
                int v1 = hnsw.neighbors[j];
                if (v1 < 0)
                    break;
                initial_list.emplace(dis->symmetric_dis(i, v1), v1);

                // initial_list.emplace(qdis(v1), v1);
            }

            std::vector<NodeDistFarther> shrunk_list;
            HNSW::shrink_neighbor_list(
                    *dis, initial_list, shrunk_list, new_size);

            for (size_t j = begin; j < end; j++) {
                if (j - begin < shrunk_list.size())
                    hnsw.neighbors[j] = shrunk_list[j - begin].id;
                else
                    hnsw.neighbors[j] = -1;
            }
        }
    }
}

void IndexHNSW::search_level_0(
        idx_t n,
        const float* x,
        idx_t k,
        const storage_idx_t* nearest,
        const float* nearest_d,
        float* distances,
        idx_t* labels,
        int nprobe,
        int search_type,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(nprobe > 0);

    storage_idx_t ntotal = hnsw.levels.size();

    using RH = HeapBlockResultHandler<HNSW::C>;
    RH bres(n, distances, labels, k);

#pragma omp parallel
    {
        std::unique_ptr<DistanceComputer> qdis(
                storage_distance_computer(storage));
        HNSWStats search_stats;
        VisitedTable vt(ntotal);
        RH::SingleResultHandler res(bres);

#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            res.begin(i);
            qdis->set_query(x + i * d);

            hnsw.search_level_0(
                    *qdis.get(),
                    res,
                    nprobe,
                    nearest + i * nprobe,
                    nearest_d + i * nprobe,
                    search_type,
                    search_stats,
                    vt,
                    params);
            res.end();
            vt.advance();
        }
#pragma omp critical
        {
            hnsw_stats.combine(search_stats);
        }
    }
    if (is_similarity_metric(this->metric_type)) {
// we need to revert the negated distances
#pragma omp parallel for
        for (int64_t i = 0; i < k * n; i++) {
            distances[i] = -distances[i];
        }
    }
}

void IndexHNSW::init_level_0_from_knngraph(
        int k,
        const float* D,
        const idx_t* I) {
    int dest_size = hnsw.nb_neighbors(0);

#pragma omp parallel for
    for (idx_t i = 0; i < ntotal; i++) {
        DistanceComputer* qdis = storage_distance_computer(storage);
        std::vector<float> vec(d);
        storage->reconstruct(i, vec.data());
        qdis->set_query(vec.data());

        std::priority_queue<NodeDistFarther> initial_list;

        for (size_t j = 0; j < k; j++) {
            int v1 = I[i * k + j];
            if (v1 == i)
                continue;
            if (v1 < 0)
                break;
            initial_list.emplace(D[i * k + j], v1);
        }

        std::vector<NodeDistFarther> shrunk_list;
        HNSW::shrink_neighbor_list(*qdis, initial_list, shrunk_list, dest_size);

        size_t begin, end;
        hnsw.neighbor_range(i, 0, &begin, &end);

        for (size_t j = begin; j < end; j++) {
            if (j - begin < shrunk_list.size())
                hnsw.neighbors[j] = shrunk_list[j - begin].id;
            else
                hnsw.neighbors[j] = -1;
        }
    }
}

void IndexHNSW::init_level_0_from_entry_points(
        int n,
        const storage_idx_t* points,
        const storage_idx_t* nearests) {
    std::vector<omp_lock_t> locks(ntotal);
    for (int i = 0; i < ntotal; i++)
        omp_init_lock(&locks[i]);

#pragma omp parallel
    {
        VisitedTable vt(ntotal);

        std::unique_ptr<DistanceComputer> dis(
                storage_distance_computer(storage));
        std::vector<float> vec(storage->d);

#pragma omp for schedule(dynamic)
        for (int i = 0; i < n; i++) {
            storage_idx_t pt_id = points[i];
            storage_idx_t nearest = nearests[i];
            storage->reconstruct(pt_id, vec.data());
            dis->set_query(vec.data());

            hnsw.add_links_starting_from(
                    *dis, pt_id, nearest, (*dis)(nearest), 0, locks.data(), vt);

            if (verbose && i % 10000 == 0) {
                printf("  %d / %d\r", i, n);
                fflush(stdout);
            }
        }
    }
    if (verbose) {
        printf("\n");
    }

    for (int i = 0; i < ntotal; i++)
        omp_destroy_lock(&locks[i]);
}

void IndexHNSW::reorder_links() {
    int M = hnsw.nb_neighbors(0);

#pragma omp parallel
    {
        std::vector<float> distances(M);
        std::vector<size_t> order(M);
        std::vector<storage_idx_t> tmp(M);
        std::unique_ptr<DistanceComputer> dis(
                storage_distance_computer(storage));

#pragma omp for
        for (storage_idx_t i = 0; i < ntotal; i++) {
            size_t begin, end;
            hnsw.neighbor_range(i, 0, &begin, &end);

            for (size_t j = begin; j < end; j++) {
                storage_idx_t nj = hnsw.neighbors[j];
                if (nj < 0) {
                    end = j;
                    break;
                }
                distances[j - begin] = dis->symmetric_dis(i, nj);
                tmp[j - begin] = nj;
            }

            fvec_argsort(end - begin, distances.data(), order.data());
            for (size_t j = begin; j < end; j++) {
                hnsw.neighbors[j] = tmp[order[j - begin]];
            }
        }
    }
}

void IndexHNSW::link_singletons() {
    printf("search for singletons\n");

    std::vector<bool> seen(ntotal);

    for (size_t i = 0; i < ntotal; i++) {
        size_t begin, end;
        hnsw.neighbor_range(i, 0, &begin, &end);
        for (size_t j = begin; j < end; j++) {
            storage_idx_t ni = hnsw.neighbors[j];
            if (ni >= 0)
                seen[ni] = true;
        }
    }

    int n_sing = 0, n_sing_l1 = 0;
    std::vector<storage_idx_t> singletons;
    for (storage_idx_t i = 0; i < ntotal; i++) {
        if (!seen[i]) {
            singletons.push_back(i);
            n_sing++;
            if (hnsw.levels[i] > 1)
                n_sing_l1++;
        }
    }

    printf("  Found %d / %" PRId64 " singletons (%d appear in a level above)\n",
           n_sing,
           ntotal,
           n_sing_l1);

    std::vector<float> recons(singletons.size() * d);
    for (int i = 0; i < singletons.size(); i++) {
        FAISS_ASSERT(!"not implemented");
    }
}

void IndexHNSW::permute_entries(const idx_t* perm) {
    auto flat_storage = dynamic_cast<IndexFlatCodes*>(storage);
    FAISS_THROW_IF_NOT_MSG(
            flat_storage, "don't know how to permute this index");
    flat_storage->permute_entries(perm);
    hnsw.permute_entries(perm);
}

DistanceComputer* IndexHNSW::get_distance_computer() const {
    return new ZmqDistanceComputer(storage);
    // return storage->get_distance_computer();
}

/**************************************************************
 * IndexHNSWFlat implementation
 **************************************************************/

IndexHNSWFlat::IndexHNSWFlat() {
    is_trained = true;
}

IndexHNSWFlat::IndexHNSWFlat(int d, int M, MetricType metric, int M0)
        : IndexHNSW(
                  (metric == METRIC_L2) ? new IndexFlatL2(d)
                                        : new IndexFlat(d, metric),
                  M,
                  M0) {
    own_fields = true;
    is_trained = true;
}

/**************************************************************
 * IndexHNSWPQ implementation
 **************************************************************/

IndexHNSWPQ::IndexHNSWPQ() = default;

IndexHNSWPQ::IndexHNSWPQ(
        int d,
        int pq_m,
        int M,
        int pq_nbits,
        MetricType metric)
        : IndexHNSW(new IndexPQ(d, pq_m, pq_nbits, metric), M) {
    own_fields = true;
    is_trained = false;
}

void IndexHNSWPQ::train(idx_t n, const float* x) {
    IndexHNSW::train(n, x);
    (dynamic_cast<IndexPQ*>(storage))->pq.compute_sdc_table();
}

/**************************************************************
 * IndexHNSWSQ implementation
 **************************************************************/

IndexHNSWSQ::IndexHNSWSQ(
        int d,
        ScalarQuantizer::QuantizerType qtype,
        int M,
        MetricType metric)
        : IndexHNSW(new IndexScalarQuantizer(d, qtype, metric), M) {
    is_trained = this->storage->is_trained;
    own_fields = true;
}

IndexHNSWSQ::IndexHNSWSQ() = default;

/**************************************************************
 * IndexHNSW2Level implementation
 **************************************************************/

IndexHNSW2Level::IndexHNSW2Level(
        Index* quantizer,
        size_t nlist,
        int m_pq,
        int M)
        : IndexHNSW(new Index2Layer(quantizer, nlist, m_pq), M) {
    own_fields = true;
    is_trained = false;
}

IndexHNSW2Level::IndexHNSW2Level() = default;

namespace {

// same as search_from_candidates but uses v
// visno -> is in result list
// visno + 1 -> in result list + in candidates
int search_from_candidates_2(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        int k,
        idx_t* I,
        float* D,
        MinimaxHeap& candidates,
        VisitedTable& vt,
        HNSWStats& stats,
        int level,
        int nres_in = 0) {
    int nres = nres_in;
    for (int i = 0; i < candidates.size(); i++) {
        idx_t v1 = candidates.ids[i];
        FAISS_ASSERT(v1 >= 0);
        vt.visited[v1] = vt.visno + 1;
    }

    int nstep = 0;

    while (candidates.size() > 0) {
        float d0 = 0;
        int v0 = candidates.pop_min(&d0);

        size_t begin, end;
        hnsw.neighbor_range(v0, level, &begin, &end);

        for (size_t j = begin; j < end; j++) {
            int v1 = hnsw.neighbors[j];
            if (v1 < 0)
                break;
            if (vt.visited[v1] == vt.visno + 1) {
                // nothing to do
            } else {
                float d = qdis(v1);
                candidates.push(v1, d);

                // never seen before --> add to heap
                if (vt.visited[v1] < vt.visno) {
                    if (nres < k) {
                        faiss::maxheap_push(++nres, D, I, d, v1);
                    } else if (d < D[0]) {
                        faiss::maxheap_replace_top(nres, D, I, d, v1);
                    }
                }
                vt.visited[v1] = vt.visno + 1;
            }
        }

        nstep++;
        if (nstep > hnsw.efSearch) {
            break;
        }
    }

    stats.n1++;
    if (candidates.size() == 0)
        stats.n2++;

    return nres;
}

} // namespace

void IndexHNSW2Level::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT_MSG(
            !params, "search params not supported for this index");

    if (dynamic_cast<const Index2Layer*>(storage)) {
        IndexHNSW::search(n, x, k, distances, labels);

    } else { // "mixed" search
        size_t n1 = 0, n2 = 0, ndis = 0, nhops = 0;

        const IndexIVFPQ* index_ivfpq =
                dynamic_cast<const IndexIVFPQ*>(storage);

        int nprobe = index_ivfpq->nprobe;

        std::unique_ptr<idx_t[]> coarse_assign(new idx_t[n * nprobe]);
        std::unique_ptr<float[]> coarse_dis(new float[n * nprobe]);

        index_ivfpq->quantizer->search(
                n, x, nprobe, coarse_dis.get(), coarse_assign.get());

        index_ivfpq->search_preassigned(
                n,
                x,
                k,
                coarse_assign.get(),
                coarse_dis.get(),
                distances,
                labels,
                false);

#pragma omp parallel
        {
            VisitedTable vt(ntotal);
            std::unique_ptr<DistanceComputer> dis(
                    storage_distance_computer(storage));

            constexpr int candidates_size = 1;
            MinimaxHeap candidates(candidates_size);

#pragma omp for reduction(+ : n1, n2, ndis, nhops)
            for (idx_t i = 0; i < n; i++) {
                idx_t* idxi = labels + i * k;
                float* simi = distances + i * k;
                dis->set_query(x + i * d);

                // mark all inverted list elements as visited

                for (int j = 0; j < nprobe; j++) {
                    idx_t key = coarse_assign[j + i * nprobe];
                    if (key < 0)
                        break;
                    size_t list_length = index_ivfpq->get_list_size(key);
                    const idx_t* ids = index_ivfpq->invlists->get_ids(key);

                    for (int jj = 0; jj < list_length; jj++) {
                        vt.set(ids[jj]);
                    }
                }

                candidates.clear();

                for (int j = 0; j < k; j++) {
                    if (idxi[j] < 0)
                        break;
                    candidates.push(idxi[j], simi[j]);
                }

                // reorder from sorted to heap
                maxheap_heapify(k, simi, idxi, simi, idxi, k);

                HNSWStats search_stats;
                search_from_candidates_2(
                        hnsw,
                        *dis,
                        k,
                        idxi,
                        simi,
                        candidates,
                        vt,
                        search_stats,
                        0,
                        k);
                n1 += search_stats.n1;
                n2 += search_stats.n2;
                ndis += search_stats.ndis;
                nhops += search_stats.nhops;

                vt.advance();
                vt.advance();

                maxheap_reorder(k, simi, idxi);
            }
        }

        hnsw_stats.combine({n1, n2, ndis, nhops});
    }
}

void IndexHNSW2Level::flip_to_ivf() {
    Index2Layer* storage2l = dynamic_cast<Index2Layer*>(storage);

    FAISS_THROW_IF_NOT(storage2l);

    IndexIVFPQ* index_ivfpq = new IndexIVFPQ(
            storage2l->q1.quantizer,
            d,
            storage2l->q1.nlist,
            storage2l->pq.M,
            8);
    index_ivfpq->pq = storage2l->pq;
    index_ivfpq->is_trained = storage2l->is_trained;
    index_ivfpq->precompute_table();
    index_ivfpq->own_fields = storage2l->q1.own_fields;
    storage2l->transfer_to_IVFPQ(*index_ivfpq);
    index_ivfpq->make_direct_map(true);

    storage = index_ivfpq;
    delete storage2l;
}

/**************************************************************
 * IndexHNSWCagra implementation
 **************************************************************/

IndexHNSWCagra::IndexHNSWCagra() {
    is_trained = true;
}

IndexHNSWCagra::IndexHNSWCagra(int d, int M, MetricType metric)
        : IndexHNSW(
                  (metric == METRIC_L2)
                          ? static_cast<IndexFlat*>(new IndexFlatL2(d))
                          : static_cast<IndexFlat*>(new IndexFlatIP(d)),
                  M) {
    FAISS_THROW_IF_NOT_MSG(
            ((metric == METRIC_L2) || (metric == METRIC_INNER_PRODUCT)),
            "unsupported metric type for IndexHNSWCagra");
    own_fields = true;
    is_trained = true;
    init_level0 = true;
    keep_max_size_level0 = true;
}

void IndexHNSWCagra::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            !base_level_only,
            "Cannot add vectors when base_level_only is set to True");

    IndexHNSW::add(n, x);
}

void IndexHNSWCagra::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    if (!base_level_only) {
        IndexHNSW::search(n, x, k, distances, labels, params);
    } else {
        std::vector<storage_idx_t> nearest(n);
        std::vector<float> nearest_d(n);

#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            std::unique_ptr<DistanceComputer> dis(
                    storage_distance_computer(this->storage));
            dis->set_query(x + i * d);
            nearest[i] = -1;
            nearest_d[i] = std::numeric_limits<float>::max();

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<idx_t> distrib(0, this->ntotal - 1);

            for (idx_t j = 0; j < num_base_level_search_entrypoints; j++) {
                auto idx = distrib(gen);
                auto distance = (*dis)(idx);
                if (distance < nearest_d[i]) {
                    nearest[i] = idx;
                    nearest_d[i] = distance;
                }
            }
            FAISS_THROW_IF_NOT_MSG(
                    nearest[i] >= 0, "Could not find a valid entrypoint.");
        }

        search_level_0(
                n,
                x,
                k,
                nearest.data(),
                nearest_d.data(),
                distances,
                labels,
                1, // n_probes
                1, // search_type
                params);
    }
}

} // namespace faiss
