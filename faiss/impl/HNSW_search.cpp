/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/HNSW.h>

#include <cstddef>
#include "faiss/IndexHNSW.h"

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/prefetch.h>
#include "HNSW_zmq.h"

#include <faiss/impl/platform_macros.h>

#include <fcntl.h>
#include <stdlib.h>
#include <sys/stat.h> // For file size check
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <set>
#include "faiss/impl/FaissAssert.h"
#include "faiss/impl/pq.h"

namespace faiss {

ssize_t read_direct_and_extract(
        int fd,
        void* user_buffer,
        size_t desired_bytes,
        off_t desired_offset) {
    struct stat stat_buf;

    // Get block size for alignment
    if (fstat(fd, &stat_buf) != 0) {
        return -1; // errno is set by fstat
    }

    size_t block_size = stat_buf.st_blksize;

    // Calculate aligned offsets
    off_t aligned_start_offset = (desired_offset / block_size) * block_size;
    off_t aligned_end_offset =
            std::ceil((desired_offset + desired_bytes) / (double)block_size) *
            block_size;
    size_t bytes_to_read_aligned = aligned_end_offset - aligned_start_offset;

    // Allocate aligned buffer
    void* temp_buffer = nullptr;
    if (posix_memalign(&temp_buffer, block_size, bytes_to_read_aligned) != 0) {
        errno = ENOMEM;
        return -1;
    }

    // Seek to the aligned offset
    if (lseek(fd, aligned_start_offset, SEEK_SET) == -1) {
        int saved_errno = errno;
        free(temp_buffer);
        errno = saved_errno;
        return -1;
    }

    // Read aligned data
    ssize_t actual_bytes_read = read(fd, temp_buffer, bytes_to_read_aligned);

    if (actual_bytes_read < 0) {
        int saved_errno = errno;
        free(temp_buffer);
        errno = saved_errno;
        return -1;
    }

    if (actual_bytes_read == 0 && desired_bytes > 0) {
        // End of file
        free(temp_buffer);
        return 0;
    }

    // Calculate internal offset and bytes to copy
    size_t internal_offset = desired_offset - aligned_start_offset;
    size_t bytes_to_copy = std::min(
            desired_bytes,
            (size_t)actual_bytes_read > internal_offset
                    ? (size_t)actual_bytes_read - internal_offset
                    : 0);

    // Copy the requested portion to user buffer
    memcpy(user_buffer, (char*)temp_buffer + internal_offset, bytes_to_copy);

    // Free the temporary buffer
    free(temp_buffer);

    return bytes_to_copy;
}

HNSW::~HNSW() {
    // Close file descriptor
    if (graph_fd != -1) {
        close(graph_fd);
    }
}

void HNSW::initialize_graph(const std::string& index_filename) {
    assert(neighbors_on_disk);
    this->hnsw_index_filename = index_filename;
    if (this->graph_fd != -1) {
        close(this->graph_fd);
    }
    this->graph_fd = open(
            this->hnsw_index_filename.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
    if (this->graph_fd == -1) {
        int RTERRNO = errno;
        FAISS_THROW_FMT(
                "Failed to open HNSW index file '%s' for pread during initialization. errno=%d (%s)",
                this->hnsw_index_filename.c_str(),
                RTERRNO,
                strerror(RTERRNO));
    }
    printf("[InitOnDemand] Opened HNSW index file descriptor: %d\n",
           this->graph_fd);
}

size_t HNSW::fetch_neighbors(
        idx_t node_id,
        int level,
        std::vector<storage_idx_t>& buffer) const {
    // Basic bounds check for node_id and level
    FAISS_THROW_IF_NOT_FMT(
            node_id >= 0 && (size_t)node_id < levels.size(),
            "fetch_neighbors: node_id %ld out of range [0, %zu)",
            (long)node_id,
            levels.size());
    FAISS_THROW_IF_NOT_FMT(
            level >= 0 && level <= max_level,
            "fetch_neighbors: level %d out of range [0, %d]",
            level,
            max_level);

    size_t begin_idx, end_idx;
    neighbor_range(node_id, level, &begin_idx, &end_idx);
    size_t num_neighbors = end_idx - begin_idx;
    buffer.resize(num_neighbors);

    if (num_neighbors == 0) {
        return 0;
    }

    if (!neighbors_on_disk) {
        // Case 1: Neighbors are in memory (either compact_neighbors_data or
        // original neighbors)
        if (!storage_is_compact) {
            // Original storage format
            size_t count = 0;
            for (size_t j = begin_idx; j < end_idx; j++) {
                if (neighbors[j] < 0)
                    break;
                buffer[count++] = neighbors[j];
            }
            buffer.resize(count);
            return count;
        } else {
            // Compact storage format with data in memory
            memcpy(buffer.data(),
                   compact_neighbors_data.data() + begin_idx,
                   num_neighbors * sizeof(storage_idx_t));
            return num_neighbors;
        }
    } else {
        // Case 2: Neighbors are either on disk or in mmap region
        FAISS_THROW_IF_NOT_MSG(
                storage_is_compact,
                "Disk/mmap neighbors access requires compact storage format");

        if (neighbors_use_mmap) {
            // Case 2a: Access via memory-mapped region
            FAISS_THROW_IF_NOT_MSG(
                    neighbors_mmap_ptr != nullptr,
                    "neighbors_use_mmap is true but neighbors_mmap_ptr is null");

            // Direct memory copy from mapped region
            memcpy(buffer.data(),
                   neighbors_mmap_ptr + begin_idx,
                   num_neighbors * sizeof(storage_idx_t));

            return num_neighbors;
        } else {
            // Case 2b: Access via pread from disk
            FAISS_THROW_IF_NOT_MSG(
                    graph_fd != -1,
                    "Graph file descriptor is not valid (file not opened?)");
            FAISS_THROW_IF_NOT_MSG(
                    neighbors_start_offset >= 0,
                    "Invalid neighbors_start_offset for pread");

            // Calculate file offsets
            // neighbors_start_offset points to the size field (uint64_t/size_t)
            off_t desired_offset = neighbors_start_offset + sizeof(size_t) +
                    (off_t)(begin_idx * sizeof(storage_idx_t));
            size_t desired_bytes = num_neighbors * sizeof(storage_idx_t);

            // Note: When using this method in a multi-threaded context, access
            // to graph_fd should be synchronized since lseek+read is not
            // thread-safe on the same file descriptor.
            ssize_t bytes_copied = read_direct_and_extract(
                    graph_fd, buffer.data(), desired_bytes, desired_offset);

            // Check for errors
            if (bytes_copied < 0) {
                int RTERRNO = errno;
                FAISS_THROW_FMT(
                        "read_direct_and_extract failed: node %ld, level %d, offset %ld. errno=%d (%s)",
                        (long)node_id,
                        level,
                        (long)desired_offset,
                        RTERRNO,
                        strerror(RTERRNO));
            }

            // Check if we got all the data we expected
            if ((size_t)bytes_copied != desired_bytes) {
                FAISS_THROW_FMT(
                        "Short copy from read_direct_and_extract: node %ld, level %d, offset %ld. Copied %zd bytes, expected %zu",
                        (long)node_id,
                        level,
                        (long)desired_offset,
                        bytes_copied,
                        desired_bytes);
            }

            return num_neighbors;
        }
    }
}

bool HNSW::load_pq_pruning_data(
        const std::string& pq_pivots_path,
        const std::string& pq_compressed_path) {
    pq_data_loader = nullptr;

    auto loader = std::make_shared<PQPrunerDataLoader>();

    if (!loader->load_pq_pivots(pq_pivots_path)) {
        std::cerr
                << "Failed to load PQ pivots for pruning. PQ pruning disabled."
                << std::endl;
        return false;
    }

    uint8_t* loaded_codes_ptr = nullptr;
    size_t num_vectors, codes_per_vector;
    if (!load_simple_bin<uint8_t>(
                pq_compressed_path,
                loaded_codes_ptr,
                num_vectors,
                codes_per_vector)) {
        std::cerr
                << "Failed to load PQ compressed codes for pruning. PQ pruning disabled."
                << std::endl;
        return false;
    }
    if (codes_per_vector != loader->get_num_chunks()) {
        std::cerr << "Error: Chunk count mismatch between pivots ("
                  << loader->get_num_chunks() << ") and compressed codes ("
                  << codes_per_vector << "). PQ pruning disabled." << std::endl;
        delete[] loaded_codes_ptr;
        return false;
    }

    code_size = codes_per_vector;
    pq_codes.resize(num_vectors * code_size);
    memcpy(pq_codes.data(), loaded_codes_ptr, num_vectors * code_size);
    delete[] loaded_codes_ptr;

    pq_data_loader = loader;
    pq_loaded = true;
    std::cout << "Successfully loaded data for PQ pruning: " << num_vectors
              << " vectors, " << code_size << " bytes/vector." << std::endl;
    return true;
}

void HNSW::neighbor_range(idx_t no, int layer_no, size_t* begin, size_t* end)
        const {
    if (storage_is_compact) {
        // CSR Format Logic
        // Basic bounds check for the node index itself
        FAISS_THROW_IF_NOT_FMT(
                no < compact_node_offsets.size() - 1,
                "Node index %ld out of bounds for compact_node_offsets (size %zd)",
                no,
                compact_node_offsets.size());

        size_t ptr_start = compact_node_offsets[no];
        size_t ptr_end =
                compact_node_offsets[no + 1]; // Exclusive end for level
                                              // pointers of this node

        // Calculate the number of levels stored for this node in the ptr array
        // Each level needs two entries in level_ptr (start and end offset),
        // so (ptr_end - ptr_start) gives num_levels + 1 entries.
        int num_level_entries = ptr_end - ptr_start;
        int num_levels_for_node =
                (num_level_entries > 0) ? (num_level_entries - 1) : 0;

        // Check if the requested layer exists for this node
        if (layer_no < 0 || layer_no >= num_levels_for_node) {
            // Layer does not exist for this node or invalid layer index
            *begin = 0;
            *end = 0; // Indicate empty range
        } else {
            // Valid layer, get data pointers from compact_level_ptr
            size_t level_ptr_index_begin = ptr_start + layer_no;
            size_t level_ptr_index_end = ptr_start + layer_no + 1;

            // Bounds check for level_ptr access
            FAISS_THROW_IF_NOT_FMT(
                    level_ptr_index_end < compact_level_ptr.size(),
                    "Level pointer index %zd out of bounds for compact_level_ptr (size %zd)",
                    level_ptr_index_end,
                    compact_level_ptr.size());

            *begin = compact_level_ptr[level_ptr_index_begin];
            *end = compact_level_ptr
                    [level_ptr_index_end]; // Exclusive end for neighbors at
                                           // this level in
                                           // compact_neighbors_data
        }
    } else {
        // Original Format Logic
        FAISS_THROW_IF_NOT_FMT(
                no < offsets.size() - 1,
                "Node index %ld out of bounds for offsets (size %zd)",
                no,
                offsets.size());
        FAISS_THROW_IF_NOT_FMT(
                layer_no + 1 < cum_nneighbor_per_level.size(),
                "Layer index %d out of bounds for cum_nneighbor_per_level (size %zd)",
                layer_no,
                cum_nneighbor_per_level.size());

        size_t o = offsets[no];
        *begin = o + cum_nb_neighbors(layer_no);
        *end = o + cum_nb_neighbors(layer_no + 1);
    }
}

// Candidates: Main Candidate Heap, stored the accurate distance
using MinimaxHeap = HNSW::MinimaxHeap;
using Node = HNSW::Node;
using C = HNSW::C;
/** Do a BFS on the candidates list */
int search_from_candidates(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        ResultHandler<C>& res,
        MinimaxHeap& candidates,
        VisitedTable& vt,
        HNSWStats& stats,
        int level,
        int nres_in,
        const SearchParameters* params,
        const IndexHNSW* hnsw_index) {
    FAISS_THROW_IF_NOT(level == 0);

    int nres = nres_in;
    int ndis = 0;
    int nfetch = 0;
    int npq = 0;

    int beam_size = 1; // Default beam width
    int batch_size = 0;
    bool use_batching = false; // Whether to use batching
    // bool cache_distances = false;

    // Original search settings
    bool do_dis_check = hnsw.check_relative_distance;
    int efSearch = hnsw.efSearch;
    const IDSelector* sel = nullptr;

    // PQ pruning setup
    float pq_select_ratio = 1;
    std::vector<float> pq_dists_lookup;
    std::vector<float> query_preprocessed;
    std::vector<uint8_t> pq_code_scratch;
    std::vector<float> pq_dists_out;

    std::vector<HNSW::storage_idx_t> neighbor_read_buffer; // For pread
    size_t max_deg_l0 = hnsw.nb_neighbors(0);

    if (params) {
        if (const SearchParametersHNSW* hnsw_params =
                    dynamic_cast<const SearchParametersHNSW*>(params)) {
            do_dis_check = hnsw_params->check_relative_distance;
            efSearch = hnsw_params->efSearch;

            // Setup beam search parameters
            if (hnsw_params->beam_size > 0) {
                beam_size = hnsw_params->beam_size;
            }

            // Setup batching parameters
            if (hnsw_params->batch_size > 0) {
                batch_size = hnsw_params->batch_size;
                use_batching = true;
            } else {
                use_batching = false; // Explicitly disable if batch_size <= 0
            }

            // PQ pruning settings
            pq_select_ratio = 1 - hnsw_params->pq_pruning_ratio;

            // cache_distances = hnsw_params->cache_distances;
        }
        sel = params->sel;
    }

    bool perform_pq_pruning =
            (hnsw.pq_data_loader && hnsw.pq_data_loader->is_initialized() &&
             pq_select_ratio < 1);
    // Initialize PQ data if needed
    if (perform_pq_pruning) {
        size_t dim = hnsw.pq_data_loader->get_dims();
        size_t n_chunks = hnsw.pq_data_loader->get_num_chunks();
        const float* original_query = qdis.get_query();

        query_preprocessed.resize(dim);
        memcpy(query_preprocessed.data(), original_query, dim * sizeof(float));
        hnsw.pq_data_loader->preprocess_query(
                query_preprocessed.data(), query_preprocessed.data());
        pq_dists_lookup.resize(256 * n_chunks);
        hnsw.pq_data_loader->populate_chunk_distances(
                query_preprocessed.data(), pq_dists_lookup.data());

        pq_code_scratch.resize(max_deg_l0 * hnsw.code_size);
        pq_dists_out.resize(max_deg_l0);
    }
    neighbor_read_buffer.resize(max_deg_l0);

    // Global PQ candidate queue (min-heap)
    using PQCandidate = std::pair<float, idx_t>; // (pq_distance, node_id)
    using PQCandidateQueue = std::priority_queue<
            PQCandidate,
            std::vector<PQCandidate>,
            std::greater<PQCandidate>>;
    PQCandidateQueue pq_candidate_queue;

    C::T threshold = res.threshold;
    for (int i = 0; i < candidates.size(); i++) {
        idx_t v1 = candidates.ids[i];
        float d = candidates.dis[i];
        FAISS_ASSERT(v1 >= 0);
        if (!sel || sel->is_member(v1)) {
            if (d < threshold) {
                if (res.add_result(d, v1)) {
                    threshold = res.threshold;
                }
            }
        }
        vt.set(v1);

        // Add initial candidates to PQ queue if using PQ pruning
        if (perform_pq_pruning) {
            pq_code_scratch.resize(hnsw.code_size);
            pq_dists_out.resize(1);

            size_t aggregated_count = aggregate_pq_codes(
                    &v1,
                    1,
                    hnsw.pq_codes.data(),
                    hnsw.levels.size(),
                    hnsw.code_size,
                    pq_code_scratch.data());

            FAISS_ASSERT(aggregated_count == 1);
            pq_distance_lookup(
                    pq_code_scratch.data(),
                    1,
                    hnsw.pq_data_loader->get_num_chunks(),
                    pq_dists_lookup.data(),
                    pq_dists_out.data());
            npq++;
            pq_candidate_queue.push({pq_dists_out[0], v1});
        }
    }

    int nstep = 0;

    while (candidates.size() > 0) {
        // Process nodes based on strategy
        std::vector<int> beam_nodes;
        std::vector<float> beam_distances;
        std::map<idx_t, std::vector<idx_t>> beam_fetched_neighbors;
        int total_neighbors = 0;

        // 1. Get all beam nodes - either batch mode or fixed beam mode
        if (use_batching) {
            // Batch mode - get nodes until we reach batch_size neighbors
            while (candidates.size() > 0 &&
                   (beam_nodes.empty() || total_neighbors < batch_size)) {
                float d0 = 0;
                int v0 = candidates.pop_min(&d0);
                assert(v0 >= 0);

                if (do_dis_check) {
                    // tricky stopping condition: there are more that ef
                    // distances that are processed already that are smaller
                    // than d0

                    int n_dis_below = candidates.count_below(d0);
                    if (n_dis_below >= efSearch) {
                        break;
                    }
                }

                size_t node_neighbor_count =
                        hnsw.fetch_neighbors(v0, 0, neighbor_read_buffer);
                nfetch++;

                std::vector<idx_t> current_node_neighbors;
                FAISS_ASSERT(node_neighbor_count >= 0);
                for (size_t i = 0; i < node_neighbor_count; ++i) {
                    if (!vt.get(neighbor_read_buffer[i])) {
                        current_node_neighbors.push_back(
                                static_cast<idx_t>(neighbor_read_buffer[i]));
                    }
                }

                beam_nodes.push_back(v0);
                beam_distances.push_back(d0);
                total_neighbors += current_node_neighbors.size();
                beam_fetched_neighbors[v0] = std::move(current_node_neighbors);
            }
            // printf("get beam_nodes: %d\n", beam_nodes.size());
            // printf("total_neighbors: %d\n", total_neighbors);
        } else {
            for (int b = 0; b < beam_size && candidates.size() > 0; b++) {
                float d0 = 0;
                int v0 = candidates.pop_min(&d0);
                FAISS_ASSERT(v0 >= 0);

                if (do_dis_check) {
                    // tricky stopping condition: there are more that ef
                    // distances that are processed already that are smaller
                    // than d0

                    int n_dis_below = candidates.count_below(d0);
                    if (n_dis_below >= efSearch) {
                        break;
                    }
                }

                std::vector<idx_t> current_node_neighbors;
                size_t node_neighbor_count =
                        hnsw.fetch_neighbors(v0, 0, neighbor_read_buffer);
                nfetch++;

                FAISS_ASSERT(node_neighbor_count >= 0);

                for (size_t i = 0; i < node_neighbor_count; ++i) {
                    if (!vt.get(neighbor_read_buffer[i])) {
                        current_node_neighbors.push_back(
                                static_cast<idx_t>(neighbor_read_buffer[i]));
                    }
                }
                beam_nodes.push_back(v0);
                beam_distances.push_back(d0);
                total_neighbors += current_node_neighbors.size();
                beam_fetched_neighbors[v0] = std::move(current_node_neighbors);
            }
            // printf("get beam_nodes: %d\n", beam_nodes.size());
            // printf("total_neighbors: %d\n", total_neighbors);
        }

        // Continue if we couldn't pop any valid nodes
        if (beam_nodes.empty()) {
            continue;
        }

        threshold = res.threshold;
        std::set<idx_t> all_new_neighbors_set;

        // 2. Process neighbors of all nodes in the beam
        for (size_t b = 0; b < beam_nodes.size(); b++) {
            int v0 = beam_nodes[b];

            const auto& neighbors_of_v0 = beam_fetched_neighbors[v0];
            for (idx_t v1 : neighbors_of_v0) {
                assert(v1 >= 0);
                assert(!vt.get(v1)); // Since the current_node_neighbors is
                                     // already filtered by vt
                all_new_neighbors_set.insert(v1);
            }
        }

        std::vector<idx_t> unique_new_neighbors(
                all_new_neighbors_set.begin(), all_new_neighbors_set.end());
        std::vector<idx_t> nodes_to_compute;

        // Calculate PQ distances for unvisited neighbors and add to global PQ
        // queue
        if (perform_pq_pruning) {
            size_t n_new = unique_new_neighbors.size();
            pq_code_scratch.resize(n_new * hnsw.code_size);
            pq_dists_out.resize(n_new);

            size_t aggregated_count = aggregate_pq_codes(
                    unique_new_neighbors.data(),
                    n_new,
                    hnsw.pq_codes.data(),
                    hnsw.levels.size(),
                    hnsw.code_size,
                    pq_code_scratch.data());

            FAISS_ASSERT(aggregated_count == unique_new_neighbors.size());
            pq_distance_lookup(
                    pq_code_scratch.data(),
                    aggregated_count,
                    hnsw.pq_data_loader->get_num_chunks(),
                    pq_dists_lookup.data(),
                    pq_dists_out.data());
            npq += aggregated_count;

            assert(pq_dists_out.size() == unique_new_neighbors.size());
            for (size_t i = 0; i < aggregated_count; i++) {
                pq_candidate_queue.push(
                        {pq_dists_out[i], unique_new_neighbors[i]});
            }

            // Another worker: select top candidates from PQ queue for exact
            // distance calculation
            int num_to_select = std::max(
                    1, int(pq_candidate_queue.size() * pq_select_ratio));
            std::vector<PQCandidate> popped_pq_nodes;
            popped_pq_nodes.reserve(num_to_select);

            for (int i = 0; i < num_to_select && !pq_candidate_queue.empty();
                 i++) {
                PQCandidate top_pq = pq_candidate_queue.top();
                pq_candidate_queue.pop();

                if (!vt.get(top_pq.second)) {
                    nodes_to_compute.push_back(top_pq.second);
                    vt.set(top_pq.second);
                }
                popped_pq_nodes.push_back(std::move(top_pq));
            }

            // Push back popped nodes
            for (const auto& pq_node : popped_pq_nodes) {
                pq_candidate_queue.push(pq_node);
            }
        } else {
            // If not using PQ pruning, process all new neighbors normally
            for (idx_t v1 : unique_new_neighbors) {
                if (!vt.get(v1)) {
                    nodes_to_compute.push_back(v1);
                    vt.set(v1);
                }
            }
        }

        std::vector<float> batch_distances(nodes_to_compute.size());
        qdis.distances_batch(nodes_to_compute, batch_distances);

        auto add_to_heap = [&](const size_t idx, const float dis) {
            if (!sel || sel->is_member(idx)) {
                if (dis < threshold) {
                    if (res.add_result(dis, idx)) {
                        threshold = res.threshold;
                        nres += 1;
                    }
                }
            }
            candidates.push(idx, dis);
        };

        for (size_t i = 0; i < nodes_to_compute.size(); i++) {
            add_to_heap(nodes_to_compute[i], batch_distances[i]);
        }

        ndis += nodes_to_compute.size();

        nstep += beam_nodes.size();
        if (!do_dis_check && nstep > efSearch) {
            break;
        }
    }

    printf("total_neigh_fetch: %d\n", ndis);

    if (level == 0) {
        stats.n1++;
        if (candidates.size() == 0) {
            stats.n2++;
        }
        stats.ndis += ndis;
        stats.nhops += nstep;
        stats.n_ios = nfetch;
        stats.n_pq_calcs = npq;
    }

    return nres;
}
} // namespace faiss