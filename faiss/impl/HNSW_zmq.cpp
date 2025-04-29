/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexHNSW.h>

#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <limits>
#include <memory>

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
#include <atomic>
#include <chrono>
#include <future>
#include <sstream> // For msgpack stringstream buffer
#include <thread>

#include "HNSW_zmq.h"

namespace faiss {
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

struct DistanceRequestMsgpack {
    std::vector<uint32_t> node_ids;
    std::vector<float> query_vector;
    MSGPACK_DEFINE_ARRAY(node_ids, query_vector); // [ [ids], [query_vector] ]
};

struct DistanceResponseMsgpack {
    std::vector<float> distances;    // Direct distances between query and nodes
    MSGPACK_DEFINE_ARRAY(distances); // [ [distances] ]
};

// --- ZMQ Fetch Function (Using MessagePack) ---
bool fetch_embeddings_zmq(
        const std::vector<uint32_t>& node_ids,
        std::vector<std::vector<float>>& out_embeddings,
        int zmq_port = 5557) // Default port kept
{
    EmbeddingRequestMsgpack req_msgpack;
    req_msgpack.node_ids = node_ids;

    std::stringstream buffer;
    try {
        msgpack::pack(buffer, req_msgpack);
    } catch (const std::exception& e) {
        std::cerr << "MessagePack pack failed: " << e.what() << std::endl;
        return false;
    }
    std::string req_str = buffer.str();

    void* context = zmq_ctx_new();
    if (!context) {
        // fprintf(stderr,
        //         "[fetch_zmq] zmq_ctx_new failed: %s\n",
        //         zmq_strerror(zmq_errno()));
        return false;
    }
    void* socket = zmq_socket(context, ZMQ_REQ);
    if (!socket) {
        // fprintf(stderr,
        //         "[fetch_zmq] zmq_socket failed: %s\n",
        //         zmq_strerror(zmq_errno()));
        zmq_ctx_destroy(context);
        return false;
    }
    int timeout = 30000;
    zmq_setsockopt(socket, ZMQ_RCVTIMEO, &timeout, sizeof(timeout));
    zmq_setsockopt(socket, ZMQ_SNDTIMEO, &timeout, sizeof(timeout));
    std::string endpoint = "tcp://127.0.0.1:" + std::to_string(zmq_port);
    if (zmq_connect(socket, endpoint.c_str()) != 0) {
        // fprintf(stderr,
        //         "[fetch_zmq] zmq_connect failed: %s\n",
        //         zmq_strerror(zmq_errno()));
        zmq_close(socket);
        zmq_ctx_destroy(context);
        return false;
    }

    if (zmq_send(socket, req_str.data(), req_str.size(), 0) < 0) { /*...*/
        // fprintf(stderr,
        //         "[fetch_zmq] zmq_msg_recv failed: %s\n",
        //         zmq_strerror(zmq_errno()));
        zmq_close(socket);
        zmq_ctx_destroy(context);
        return false;
    }

    zmq_msg_t response;
    zmq_msg_init(&response);
    if (zmq_msg_recv(&response, socket, 0) < 0) { /*...*/
        // fprintf(stderr,
        //         "[fetch_zmq] zmq_msg_recv failed: %s\n",
        //         zmq_strerror(zmq_errno()));
        zmq_msg_close(&response);
        zmq_close(socket);
        zmq_ctx_destroy(context);
        return false;
    }

    EmbeddingResponseMsgpack resp_msgpack;
    const char* resp_data = static_cast<const char*>(zmq_msg_data(&response));
    size_t resp_size = zmq_msg_size(&response);
    // printf("[fetch_zmq] Raw response bytes (first %d): ",
    //        (int)std::min((size_t)64, resp_size));

    msgpack::object_handle oh = msgpack::unpack(resp_data, resp_size);
    msgpack::object obj = oh.get();
    obj.convert(resp_msgpack); // Convert msgpack object to our struct
    // for (size_t k = 0; k < std::min((size_t)64, resp_size); ++k)
    //     printf("%02x ", (unsigned char)resp_data[k]);
    // printf("\n");

    // --- Print parsed values BEFORE NaN check ---
    // printf("[fetch_zmq] Parsed response. Dimensions: %d x %d. Data floats:
    // %zu\n",
    //        resp_msgpack.dimensions.empty() ? 0 : resp_msgpack.dimensions[0],
    //        resp_msgpack.dimensions.size() < 2 ? 0 :
    //        resp_msgpack.dimensions[1], resp_msgpack.embeddings_data.size());
    // printf("[fetch_zmq] Parsed embeddings_data (first %d floats): ",
    //        (int)std::min((size_t)10, resp_msgpack.embeddings_data.size()));
    // bool parse_contains_nan = false;
    // for (size_t k = 0;
    //      k < std::min((size_t)10, resp_msgpack.embeddings_data.size());
    //      ++k) {
    //     printf("%.6f ", resp_msgpack.embeddings_data[k]);
    //     if (std::isnan(resp_msgpack.embeddings_data[k]))
    //         parse_contains_nan = true;
    // }
    // printf("%s\n",
    //        parse_contains_nan ? "!!! CONTAINS NaN AFTER PARSE !!!"
    //                           : "(Checked first 10 for NaN)");

    if (resp_msgpack.dimensions.size() != 2) {
        // std::cerr << "Server response has invalid dimensions size: "
        //           << resp_msgpack.dimensions.size() << std::endl;
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

    size_t expected_floats = (size_t)batch_size * embedding_dim;
    if (resp_msgpack.embeddings_data.size() != expected_floats) {
        // std::cerr << "Embedding data size mismatch: Got "
        //           << resp_msgpack.embeddings_data.size() << " floats,
        //           expected "
        //           << expected_floats << " (" << batch_size << "x"
        //           << embedding_dim << ")" << std::endl;
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
        // fprintf(stderr,
        //         "!!! [fetch_zmq] ERROR: Final check confirms NaN values in
        //         parsed embeddings_data! First requested ID: %u !!!\n",
        //         node_ids.empty() ? 0 : node_ids[0]);
        return false; // Decide whether to fail here
    } else {
        // printf("[fetch_zmq] Final check confirms embeddings data appears
        // clean (no NaNs checked).\n"); // Can be verbose
    }

    out_embeddings.clear();
    out_embeddings.resize(batch_size);
    const float* flat_data_ptr = resp_msgpack.embeddings_data.data();
    for (int i = 0; i < batch_size; i++) {
        out_embeddings[i].assign(
                flat_data_ptr + (size_t)i * embedding_dim,
                flat_data_ptr + ((size_t)i + 1) * embedding_dim);
    }

    zmq_msg_close(&response);
    zmq_close(socket);
    zmq_ctx_destroy(context);

    return true;
}

const float* ZmqDistanceComputer::get_vector_zmq(idx_t id) {
    std::vector<uint32_t> ids_to_fetch = {(uint32_t)id};
    std::vector<std::vector<float>>
            fetched_embeddings; // fetch_embeddings_zmq expects this
                                // structure

    if (!fetch_embeddings_zmq(ids_to_fetch, fetched_embeddings, ZMQ_PORT)) {
        // fprintf(stderr,
        //         "!!! ERROR get_vector_zmq: fetch_embeddings_zmq call
        //         failed for ID %ld !!!\n", (long)id);
        // Fill member with NaN to indicate failure?
        std::fill(
                last_fetched_zmq_vector.begin(),
                last_fetched_zmq_vector.end(),
                std::numeric_limits<float>::quiet_NaN());
        return nullptr; // Indicate failure upstream
    }
    if (fetched_embeddings.empty() || fetched_embeddings[0].size() != d) {
        // fprintf(stderr,
        //         "!!! ERROR get_vector_zmq: fetch_embeddings_zmq returned
        //         incorrect data for ID %ld !!!\n", (long)id);
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

    // ---- Addition: Increment fetch count on success ----
    fetch_count++;
    // ---- End Addition ----

    // --- Log values RIGHT BEFORE returning pointer ---
    const float* return_ptr = last_fetched_zmq_vector.data();
    // bool has_nan_before_return = false;
    // printf("DEBUG get_vector_zmq: Fetched ID %ld. Values BEFORE return
    // (ptr %p) [0..%d]: ",
    //        (long)id,
    //        (void*)return_ptr,
    //        (int)std::min((size_t)4, d - 1));
    // for (size_t k = 0; k < std::min((size_t)5, d); ++k) {
    //     printf("%.6f ", return_ptr[k]);
    // if (std::isnan(return_ptr[k]) || std::isinf(return_ptr[k]))
    //         has_nan_before_return = true;
    // }
    // printf("%s\n",
    //        has_nan_before_return ? "!!! HAS NaN/Inf BEFORE RETURN !!!"
    //                              : "(OK Before Return)");
    // -------------------------------------------

    return return_ptr; // Return pointer to member data
}

// --- ZMQ Distance Calculation Function (Using MessagePack) ---
bool fetch_distances_zmq(
        const std::vector<uint32_t>& node_ids,
        const float* query_vector,
        size_t query_dim,
        std::vector<float>& out_distances,
        int zmq_port = 5557) {
    DistanceRequestMsgpack req_msgpack;
    req_msgpack.node_ids = node_ids;

    // Copy query vector
    req_msgpack.query_vector.resize(query_dim);
    memcpy(req_msgpack.query_vector.data(),
           query_vector,
           query_dim * sizeof(float));

    std::stringstream buffer;
    try {
        msgpack::pack(buffer, req_msgpack);
    } catch (const std::exception& e) {
        std::cerr << "MessagePack pack failed for distance request: "
                  << e.what() << std::endl;
        return false;
    }
    std::string req_str = buffer.str();

    void* context = zmq_ctx_new();
    if (!context) {
        return false;
    }
    void* socket = zmq_socket(context, ZMQ_REQ);
    if (!socket) {
        zmq_ctx_destroy(context);
        return false;
    }
    int timeout = 30000;
    zmq_setsockopt(socket, ZMQ_RCVTIMEO, &timeout, sizeof(timeout));
    zmq_setsockopt(socket, ZMQ_SNDTIMEO, &timeout, sizeof(timeout));
    std::string endpoint = "tcp://127.0.0.1:" + std::to_string(zmq_port);
    if (zmq_connect(socket, endpoint.c_str()) != 0) {
        zmq_close(socket);
        zmq_ctx_destroy(context);
        return false;
    }

    if (zmq_send(socket, req_str.data(), req_str.size(), 0) < 0) {
        zmq_close(socket);
        zmq_ctx_destroy(context);
        return false;
    }

    zmq_msg_t response;
    zmq_msg_init(&response);
    if (zmq_msg_recv(&response, socket, 0) < 0) {
        zmq_msg_close(&response);
        zmq_close(socket);
        zmq_ctx_destroy(context);
        return false;
    }

    DistanceResponseMsgpack resp_msgpack;
    const char* resp_data = static_cast<const char*>(zmq_msg_data(&response));
    size_t resp_size = zmq_msg_size(&response);

    try {
        msgpack::object_handle oh = msgpack::unpack(resp_data, resp_size);
        msgpack::object obj = oh.get();
        obj.convert(resp_msgpack); // Convert msgpack object to our struct
    } catch (const std::exception& e) {
        std::cerr << "MessagePack unpack failed for distance response: "
                  << e.what() << std::endl;
        zmq_msg_close(&response);
        zmq_close(socket);
        zmq_ctx_destroy(context);
        return false;
    }

    if (resp_msgpack.distances.size() != node_ids.size()) {
        std::cerr << "Distance response size mismatch: Got "
                  << resp_msgpack.distances.size() << " distances, expected "
                  << node_ids.size() << std::endl;
        zmq_msg_close(&response);
        zmq_close(socket);
        zmq_ctx_destroy(context);
        return false;
    }

    // Copy distances to output vector
    out_distances = resp_msgpack.distances;

    zmq_msg_close(&response);
    zmq_close(socket);
    zmq_ctx_destroy(context);

    return true;
}

void ZmqDistanceComputer::distances_batch(
        const std::vector<idx_t>& ids,
        std::vector<float>& distances_out) {
    if (ids.empty()) {
        distances_out.clear();
        return;
    }

    std::vector<uint32_t> node_ids(ids.size());
    for (size_t i = 0; i < ids.size(); i++) {
        node_ids[i] = (uint32_t)ids[i];
    }

    // Use server-side distance calculation instead of fetching embeddings
    auto start_time = std::chrono::high_resolution_clock::now();
    bool fetch_success = fetch_distances_zmq(
            node_ids, query.data(), d, distances_out, ZMQ_PORT);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                            end_time - start_time)
                            .count();

    std::cout << "fetch_distances_zmq took " << duration << " microseconds for "
              << ids.size() << " nodes" << std::endl;

    if (!fetch_success || distances_out.size() != ids.size()) {
        // Use fallback values on error
        distances_out.resize(ids.size());
        std::fill(
                distances_out.begin(),
                distances_out.end(),
                (metric_type == METRIC_INNER_PRODUCT)
                        ? -std::numeric_limits<float>::max()
                        : std::numeric_limits<float>::max());
        return;
    }

    // Update fetch count
    fetch_count += ids.size();
}
} // namespace faiss