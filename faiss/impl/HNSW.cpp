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
#include <faiss/utils/prefetch.h>

#include <faiss/impl/platform_macros.h>

#include <fcntl.h>
#include <stdlib.h>
#include <sys/stat.h> // For file size check
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <set>
#include <unordered_map>
#include "faiss/impl/FaissAssert.h"
#include "faiss/impl/io.h"
#include "faiss/impl/io_macros.h"
#include "faiss/impl/pq.h"

#ifdef __AVX2__
#include <immintrin.h>

#include <limits>
#include <type_traits>
#endif

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
/**************************************************************
 * HNSW structure implementation
 **************************************************************/

// Minimal, level-0 only, non-compact edge deletion
void HNSW::delete_random_level0_edges_minimal(float prune_ratio) {
    // --- Assert assumptions ---
    FAISS_THROW_IF_NOT_FMT(
            !storage_is_compact,
            "Function %s requires non-compact storage",
            __func__);

    if (prune_ratio == 0) {
        return; // Nothing to do
    }
    if (levels.empty()) {
        return; // No nodes
    }

    // --- Step 1: Identify all valid level 0 edge locations ---
    std::vector<Level0EdgeLocation> candidates;
    // Simple estimate, could be refined
    candidates.reserve(levels.size() * nb_neighbors(0));

    for (storage_idx_t node_id = 0; node_id < levels.size(); ++node_id) {
        // Skip nodes that don't exist at level 0 (shouldn't happen if
        // levels[node_id] >= 1)
        if (levels[node_id] <= 0)
            continue;

        size_t begin, end;
        neighbor_range(node_id, 0, &begin, &end); // Get level 0 range

        for (size_t idx = begin; idx < end; ++idx) {
            if (neighbors[idx] != -1) {
                candidates.emplace_back(node_id, idx);
            } else {
                // Found the -1 sentinel, stop searching for this node's level 0
                // neighbors
                break;
            }
        }
    }

    // --- Step 2: Shuffle and Select ---
    if (candidates.empty()) {
        printf("No valid level 0 edges found to delete.\n");
        return;
    }

    size_t actual_num_to_delete = prune_ratio * candidates.size();
    printf("Found %zd level 0 edges. Attempting to delete %zd.\n",
           candidates.size(),
           actual_num_to_delete);

    std::random_device rd;
    std::default_random_engine engine(rd());
    std::shuffle(candidates.begin(), candidates.end(), engine);

    // --- Step 3: Delete using "swap with last" ---
    size_t deleted_count = 0;
    for (size_t i = 0; i < actual_num_to_delete; ++i) {
        const auto& loc = candidates[i];
        storage_idx_t node_id = loc.node_id;
        size_t index_to_delete = loc.neighbor_array_index;

        // Minimal check: Skip if it somehow already got deleted
        // if (neighbors[index_to_delete] == -1) {
        //     continue;
        // }
        while (neighbors[index_to_delete] == -1 && index_to_delete > 0) {
            index_to_delete--;
        }
        assert(neighbors[index_to_delete] != -1);
        size_t begin, end;
        neighbor_range(node_id, 0, &begin, &end); // Level 0 range

        // Find last valid neighbor index in [begin, end)
        size_t idx_last_valid = begin; // Initialize, will be overwritten
        bool found_last = false;
        for (size_t j = end; j > begin; --j) {
            if (neighbors[j - 1] != -1) {
                idx_last_valid = j - 1;
                found_last = true;
                break;
            }
        }

        // If no valid neighbor found (only possible if list was already all
        // -1s?) Or if somehow the target index is beyond last valid (should not
        // happen)
        if (!found_last || index_to_delete > idx_last_valid) {
            assert(false);
        }

        // Swap-and-delete
        if (index_to_delete == idx_last_valid) {
            neighbors[index_to_delete] = -1;
        } else {
            std::swap(neighbors[index_to_delete], neighbors[idx_last_valid]);
            neighbors[idx_last_valid] = -1;
        }
        deleted_count++;
    }

    printf("Minimal delete: %zd level 0 edges processed.\n", deleted_count);
}

int HNSW::nb_neighbors(int layer_no) const {
    FAISS_THROW_IF_NOT(layer_no + 1 < cum_nneighbor_per_level.size());
    return cum_nneighbor_per_level[layer_no + 1] -
            cum_nneighbor_per_level[layer_no];
}

void HNSW::set_nb_neighbors(int level_no, int n) {
    FAISS_THROW_IF_NOT(levels.size() == 0);
    int cur_n = nb_neighbors(level_no);
    for (int i = level_no + 1; i < cum_nneighbor_per_level.size(); i++) {
        cum_nneighbor_per_level[i] += n - cur_n;
    }
}

int HNSW::cum_nb_neighbors(int layer_no) const {
    return cum_nneighbor_per_level[layer_no];
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

HNSW::HNSW(int M, int M0) : rng(12345) {
    set_default_probas(M, 1.0 / log(M), M0);
    offsets.push_back(0);

    // Initialize disk/mmap access fields
    neighbors_on_disk = false;
    graph_fd = -1;
    neighbors_start_offset = -1;
    neighbors_mmap_ptr = nullptr;
    neighbors_use_mmap = false;
}

int HNSW::random_level() {
    double f = rng.rand_float();
    // could be a bit faster with bissection
    for (int level = 0; level < assign_probas.size(); level++) {
        if (f < assign_probas[level]) {
            return level;
        }
        f -= assign_probas[level];
    }
    // happens with exponentially low probability
    return assign_probas.size() - 1;
}

void HNSW::set_default_probas(int M, float levelMult, int M0) {
    int nn = 0;
    // printf("M0: %d\n", M0);
    // printf("M: %d\n", M);
    if (M0 == -1) {
        M0 = M * 2;
    }
    cum_nneighbor_per_level.push_back(0);
    for (int level = 0;; level++) {
        float proba = exp(-level / levelMult) * (1 - exp(-1 / levelMult));
        if (proba < 1e-9)
            break;
        assign_probas.push_back(proba);
        nn += level == 0 ? M0 : M;
        // printf("level %d, proba %f, nn %d\n", level, proba, nn);
        cum_nneighbor_per_level.push_back(nn);
    }
}

void HNSW::clear_neighbor_tables(int level) {
    for (int i = 0; i < levels.size(); i++) {
        size_t begin, end;
        neighbor_range(i, level, &begin, &end);
        for (size_t j = begin; j < end; j++) {
            neighbors[j] = -1;
        }
    }
}

void HNSW::reset() {
    max_level = -1;
    entry_point = -1;
    offsets.clear();
    offsets.push_back(0);
    levels.clear();
    neighbors.clear();

    // Clear compact storage fields
    storage_is_compact = false; // Reset flag to default
    compact_neighbors_data.clear();
    compact_level_ptr.clear();
    compact_node_offsets.clear();

    // Reset disk/mmap access fields
    neighbors_on_disk = false;
    if (graph_fd != -1) {
        close(graph_fd);
        graph_fd = -1;
    }
    hnsw_index_filename.clear();
    neighbors_start_offset = -1;
    neighbors_mmap_ptr = nullptr;
    neighbors_use_mmap = false;
}

void HNSW::save_degree_distribution(int level, const char* filename) const {
    // Check if level is valid
    if (level < 0 || level >= cum_nneighbor_per_level.size() - 1) {
        fprintf(stderr,
                "Invalid level %d (max level is %d)\n",
                level,
                (int)cum_nneighbor_per_level.size() - 2);
        return;
    }

    // Open file for writing
    FILE* f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Could not open %s for writing\n", filename);
        return;
    }

    // For each node, count actual neighbors at the specified level
    printf("Computing degree distribution for level %d\n", level);

    // Only consider nodes that exist at this level or above
    int nodes_at_level = 0;
    for (int i = 0; i < levels.size(); i++) {
        if (levels[i] > level) {
            nodes_at_level++;

            // Count actual neighbors (not -1)
            size_t begin, end;
            neighbor_range(i, level, &begin, &end);

            int degree = 0;
            for (size_t j = begin; j < end; j++) {
                if (neighbors[j] >= 0) {
                    degree++;
                } else {
                    break; // Stop at first -1
                }
            }

            // Write the degree to the file
            fprintf(f, "%d\n", degree);
        }
    }

    fclose(f);
    printf("Saved degree distribution for %d nodes at level %d to %s\n",
           nodes_at_level,
           level,
           filename);

    // Print command to generate the plot
    printf("To visualize the distribution, run:\n");
    printf("python -m faiss.contrib.plot_degree_distribution %s\n", filename);
}

void HNSW::print_neighbor_stats(int level) const {
    FAISS_THROW_IF_NOT(level < cum_nneighbor_per_level.size());
    printf("stats on level %d, max %d neighbors per vertex:\n",
           level,
           nb_neighbors(level));
    size_t tot_neigh = 0, tot_common = 0, tot_reciprocal = 0, n_node = 0;
#pragma omp parallel for reduction(+ : tot_neigh) reduction(+ : tot_common) \
        reduction(+ : tot_reciprocal) reduction(+ : n_node)
    for (int i = 0; i < levels.size(); i++) {
        if (levels[i] > level) {
            n_node++;
            size_t begin, end;
            neighbor_range(i, level, &begin, &end);
            std::unordered_set<int> neighset;
            for (size_t j = begin; j < end; j++) {
                if (neighbors[j] < 0)
                    break;
                neighset.insert(neighbors[j]);
            }
            int n_neigh = neighset.size();
            int n_common = 0;
            int n_reciprocal = 0;
            for (size_t j = begin; j < end; j++) {
                storage_idx_t i2 = neighbors[j];
                if (i2 < 0)
                    break;
                FAISS_ASSERT(i2 != i);
                size_t begin2, end2;
                neighbor_range(i2, level, &begin2, &end2);
                for (size_t j2 = begin2; j2 < end2; j2++) {
                    storage_idx_t i3 = neighbors[j2];
                    if (i3 < 0)
                        break;
                    if (i3 == i) {
                        n_reciprocal++;
                        continue;
                    }
                    if (neighset.count(i3)) {
                        neighset.erase(i3);
                        n_common++;
                    }
                }
            }
            tot_neigh += n_neigh;
            tot_common += n_common;
            tot_reciprocal += n_reciprocal;
        }
    }
    float normalizer = n_node;
    printf("   nb of nodes at that level %zd\n", n_node);
    printf("   neighbors per node: %.2f (%zd)\n",
           tot_neigh / normalizer,
           tot_neigh);
    printf("   nb of reciprocal neighbors: %.2f\n",
           tot_reciprocal / normalizer);
    printf("   nb of neighbors that are also neighbor-of-neighbors: %.2f (%zd)\n",
           tot_common / normalizer,
           tot_common);
}

void HNSW::fill_with_random_links(size_t n) {
    int max_level_2 = prepare_level_tab(n);
    RandomGenerator rng2(456);

    for (int level = max_level_2 - 1; level >= 0; --level) {
        std::vector<int> elts;
        for (int i = 0; i < n; i++) {
            if (levels[i] > level) {
                elts.push_back(i);
            }
        }
        printf("linking %zd elements in level %d\n", elts.size(), level);

        if (elts.size() == 1)
            continue;

        for (int ii = 0; ii < elts.size(); ii++) {
            int i = elts[ii];
            size_t begin, end;
            neighbor_range(i, 0, &begin, &end);
            for (size_t j = begin; j < end; j++) {
                int other = 0;
                do {
                    other = elts[rng2.rand_int(elts.size())];
                } while (other == i);

                neighbors[j] = other;
            }
        }
    }
}

int HNSW::prepare_level_tab(size_t n, bool preset_levels) {
    size_t n0 = offsets.size() - 1;

    if (preset_levels) {
        FAISS_ASSERT(n0 + n == levels.size());
    } else {
        FAISS_ASSERT(n0 == levels.size());
        for (int i = 0; i < n; i++) {
            int pt_level = random_level();
            levels.push_back(pt_level + 1);
        }
    }

    int max_level_2 = 0;
    for (int i = 0; i < n; i++) {
        int pt_level = levels[i + n0] - 1;
        if (pt_level > max_level_2)
            max_level_2 = pt_level;
        offsets.push_back(offsets.back() + cum_nb_neighbors(pt_level + 1));
    }
    neighbors.resize(offsets.back(), -1);

    return max_level_2;
}

/** Enumerate vertices from nearest to farthest from query, keep a
 * neighbor only if there is no previous neighbor that is closer to
 * that vertex than the query.
 */
void HNSW::shrink_neighbor_list(
        DistanceComputer& qdis,
        std::priority_queue<NodeDistFarther>& input,
        std::vector<NodeDistFarther>& output,
        int max_size,
        bool keep_max_size_level0) {
    // RNG pruning
    //  This prevents number of neighbors at
    //  level 0 from being shrunk to less than 2 * M.
    //  This is essential in making sure
    //  `faiss::gpu::GpuIndexCagra::copyFrom(IndexHNSWCagra*)` is functional
    std::vector<NodeDistFarther> outsiders;

    while (input.size() > 0) {
        NodeDistFarther v1 = input.top();
        input.pop();
        float dist_v1_q = v1.d;

        bool good = true;
        for (NodeDistFarther v2 : output) {
            float dist_v1_v2 = qdis.symmetric_dis(v2.id, v1.id);

            if (dist_v1_v2 < dist_v1_q) {
                good = false;
                break;
            }
        }

        if (good) {
            output.push_back(v1);
            if (output.size() >= max_size) {
                return;
            }
        } else if (keep_max_size_level0) {
            outsiders.push_back(v1);
        }
    }
    size_t idx = 0;
    while (keep_max_size_level0 && (output.size() < max_size) &&
           (idx < outsiders.size())) {
        output.push_back(outsiders[idx++]);
    }
}

namespace {

using storage_idx_t = HNSW::storage_idx_t;
using NodeDistCloser = HNSW::NodeDistCloser;
using NodeDistFarther = HNSW::NodeDistFarther;

/**************************************************************
 * Addition subroutines
 **************************************************************/

/// remove neighbors from the list to make it smaller than max_size
void shrink_neighbor_list(
        DistanceComputer& qdis,
        std::priority_queue<NodeDistCloser>& resultSet1,
        int max_size,
        bool keep_max_size_level0 = false) {
    // if remove this, which menas we will enforce RNG check no matter if the
    // degree is too high this does not work to save degree()may because nodes
    // with low degree already hvae strict go though RNG test
    if (resultSet1.size() < max_size) {
        return;
    }
    std::priority_queue<NodeDistFarther> resultSet;
    std::vector<NodeDistFarther> returnlist;

    while (resultSet1.size() > 0) {
        resultSet.emplace(resultSet1.top().d, resultSet1.top().id);
        resultSet1.pop();
    }

    HNSW::shrink_neighbor_list(
            qdis, resultSet, returnlist, max_size, keep_max_size_level0);

    for (NodeDistFarther curen2 : returnlist) {
        resultSet1.emplace(curen2.d, curen2.id);
    }
}

/// add a link between two elements, possibly shrinking the list
/// of links to make room for it.
void add_link(
        HNSW& hnsw,
        DistanceComputer& qdis,
        storage_idx_t src,
        storage_idx_t dest,
        int level,
        bool keep_max_size_level0 = false) {
    size_t begin, end;
    hnsw.neighbor_range(src, level, &begin, &end);
    if (hnsw.neighbors[end - 1] == -1) {
        // there is enough room, find a slot to add it
        size_t i = end;
        while (i > begin) {
            if (hnsw.neighbors[i - 1] != -1)
                break;
            i--;
        }
        hnsw.neighbors[i] = dest;
        return;
    }

    // otherwise we let them fight out which to keep

    // copy to resultSet...
    std::priority_queue<NodeDistCloser> resultSet;
    resultSet.emplace(qdis.symmetric_dis(src, dest), dest);
    for (size_t i = begin; i < end; i++) { // HERE WAS THE BUG
        storage_idx_t neigh = hnsw.neighbors[i];
        resultSet.emplace(qdis.symmetric_dis(src, neigh), neigh);
    }

    shrink_neighbor_list(qdis, resultSet, end - begin, keep_max_size_level0);

    // ...and back
    size_t i = begin;
    while (resultSet.size()) {
        hnsw.neighbors[i++] = resultSet.top().id;
        resultSet.pop();
    }
    // they may have shrunk more than just by 1 element
    while (i < end) {
        hnsw.neighbors[i++] = -1;
    }
}

/// add a link between two elements, possibly shrinking the list
/// of links to make room for it.
// this is a variant of add_link that prunes the list with 90% probability
void add_link_pruned(
        HNSW& hnsw,
        DistanceComputer& qdis,
        storage_idx_t src,
        storage_idx_t dest,
        int level,
        bool keep_max_size_level0 = false) {
    assert(level == 0);
    size_t begin, end;
    hnsw.neighbor_range(src, level, &begin, &end);

    int strict_end = end;
    if (hnsw.neighbors[end - 1] == -1) {
        // there is enough room, find a slot to add it
        size_t i = end;
        while (i > begin) {
            if (hnsw.neighbors[i - 1] != -1)
                break;
            i--;
        }
        strict_end = i;
        if (level == 0 && strict_end - begin < hnsw.ems[src]) {
            hnsw.neighbors[i] = dest;
            return;
        }
    }
    // printf("strict_end: %d\n", strict_end);

    // otherwise we let them fight out which to keep

    // copy to resultSet...
    std::priority_queue<NodeDistCloser> resultSet;
    resultSet.emplace(qdis.symmetric_dis(src, dest), dest);
    for (size_t i = begin; i < strict_end; i++) { // HERE WAS THE BUG
        storage_idx_t neigh = hnsw.neighbors[i];
        resultSet.emplace(qdis.symmetric_dis(src, neigh), neigh);
    }

    // printf("end - begin: %zd, strict_end - begin: %zd\n", end - begin,
    // strict_end - begin);
    int len = strict_end - begin;
    len = std::min(len, hnsw.ems[src]);
    printf("len: %d, ems[src]: %d\n", len, hnsw.ems[src]);
    // assert (len != 64);
    // assert (len==8);
    shrink_neighbor_list(qdis, resultSet, len, keep_max_size_level0);

    // ...and back
    size_t i = begin;
    while (resultSet.size()) {
        hnsw.neighbors[i++] = resultSet.top().id;
        resultSet.pop();
    }
    // they may have shrunk more than just by 1 element
    while (i < end) {
        hnsw.neighbors[i++] = -1;
    }
}

} // namespace

/// search neighbors on a single level, starting from an entry point
void search_neighbors_to_add(
        HNSW& hnsw,
        DistanceComputer& qdis,
        std::priority_queue<NodeDistCloser>& results,
        int entry_point,
        float d_entry_point,
        int level,
        VisitedTable& vt,
        bool reference_version) {
    // top is nearest candidate
    std::priority_queue<NodeDistFarther> candidates;

    NodeDistFarther ev(d_entry_point, entry_point);
    candidates.push(ev);
    results.emplace(d_entry_point, entry_point);
    vt.set(entry_point);

    while (!candidates.empty()) {
        // get nearest
        const NodeDistFarther& currEv = candidates.top();

        if (currEv.d > results.top().d) {
            break;
        }
        int currNode = currEv.id;
        candidates.pop();

        // loop over neighbors
        size_t begin, end;
        hnsw.neighbor_range(currNode, level, &begin, &end);

        // The reference version is not used, but kept here because:
        // 1. It is easier to switch back if the optimized version has a problem
        // 2. It serves as a starting point for new optimizations
        // 3. It helps understand the code
        // 4. It ensures the reference version is still compilable if the
        // optimized version changes
        // The reference and the optimized versions' results are compared in
        // test_hnsw.cpp
        if (reference_version) {
            // a reference version
            for (size_t i = begin; i < end; i++) {
                storage_idx_t nodeId = hnsw.neighbors[i];
                if (nodeId < 0)
                    break;
                if (vt.get(nodeId))
                    continue;
                vt.set(nodeId);

                float dis = qdis(nodeId);
                NodeDistFarther evE1(dis, nodeId);

                if (results.size() < hnsw.efConstruction ||
                    results.top().d > dis) {
                    results.emplace(dis, nodeId);
                    candidates.emplace(dis, nodeId);
                    if (results.size() > hnsw.efConstruction) {
                        results.pop();
                    }
                }
            }
        } else {
            // a faster version

            // the following version processes 4 neighbors at a time
            auto update_with_candidate = [&](const storage_idx_t idx,
                                             const float dis) {
                if (results.size() < hnsw.efConstruction ||
                    results.top().d > dis) {
                    results.emplace(dis, idx);
                    candidates.emplace(dis, idx);
                    if (results.size() > hnsw.efConstruction) {
                        results.pop();
                    }
                }
            };

            int n_buffered = 0;
            storage_idx_t buffered_ids[4];

            for (size_t j = begin; j < end; j++) {
                storage_idx_t nodeId = hnsw.neighbors[j];
                if (nodeId < 0)
                    break;
                if (vt.get(nodeId)) {
                    continue;
                }
                vt.set(nodeId);

                buffered_ids[n_buffered] = nodeId;
                n_buffered += 1;

                if (n_buffered == 4) {
                    float dis[4];
                    qdis.distances_batch_4(
                            buffered_ids[0],
                            buffered_ids[1],
                            buffered_ids[2],
                            buffered_ids[3],
                            dis[0],
                            dis[1],
                            dis[2],
                            dis[3]);

                    for (size_t id4 = 0; id4 < 4; id4++) {
                        update_with_candidate(buffered_ids[id4], dis[id4]);
                    }

                    n_buffered = 0;
                }
            }

            // process leftovers
            for (size_t icnt = 0; icnt < n_buffered; icnt++) {
                float dis = qdis(buffered_ids[icnt]);
                update_with_candidate(buffered_ids[icnt], dis);
            }
        }
    }

    vt.advance();
}

/// Finds neighbors and builds links with them, starting from an entry
/// point. The own neighbor list is assumed to be locked.
void HNSW::add_links_starting_from(
        DistanceComputer& ptdis,
        storage_idx_t pt_id,
        storage_idx_t nearest,
        float d_nearest,
        int level,
        omp_lock_t* locks,
        VisitedTable& vt,
        bool keep_max_size_level0) {
    std::priority_queue<NodeDistCloser> link_targets;

    search_neighbors_to_add(
            *this, ptdis, link_targets, nearest, d_nearest, level, vt);

    // but we can afford only this many neighbors
    int M = nb_neighbors(level);

    ::faiss::shrink_neighbor_list(
            ptdis, link_targets, ems[pt_id], keep_max_size_level0);

    std::vector<storage_idx_t> neighbors_to_add;
    neighbors_to_add.reserve(link_targets.size());
    bool prune_in_add_link = false;
    while (!link_targets.empty()) {
        storage_idx_t other_id = link_targets.top().id;
        if (level == 0 && M > ems[pt_id] && prune_in_add_link) {
            add_link_pruned(
                    *this, ptdis, pt_id, other_id, level, keep_max_size_level0);
        } else {
            add_link(
                    *this, ptdis, pt_id, other_id, level, keep_max_size_level0);
        }
        neighbors_to_add.push_back(other_id);
        link_targets.pop();
    }

    omp_unset_lock(&locks[pt_id]);
    for (storage_idx_t other_id : neighbors_to_add) {
        omp_set_lock(&locks[other_id]);
        if (level == 0 && M > ems[other_id] && prune_in_add_link) {
            add_link_pruned(
                    *this, ptdis, other_id, pt_id, level, keep_max_size_level0);
        } else {
            add_link(
                    *this, ptdis, other_id, pt_id, level, keep_max_size_level0);
        }
        omp_unset_lock(&locks[other_id]);
    }
    omp_set_lock(&locks[pt_id]);
}

/**************************************************************
 * Building, parallel
 **************************************************************/

void HNSW::add_with_locks(
        DistanceComputer& ptdis,
        int pt_level,
        int pt_id,
        std::vector<omp_lock_t>& locks,
        VisitedTable& vt,
        bool keep_max_size_level0) {
    //  greedy search on upper levels

    storage_idx_t nearest;
#pragma omp critical
    {
        nearest = entry_point;

        if (nearest == -1) {
            max_level = pt_level;
            entry_point = pt_id;
        }
    }

    if (nearest < 0) {
        return;
    }

    omp_set_lock(&locks[pt_id]);

    int level = max_level; // level at which we start adding neighbors
    float d_nearest = ptdis(nearest);

    for (; level > pt_level; level--) {
        greedy_update_nearest(*this, ptdis, level, nearest, d_nearest);
    }

    for (; level >= 0; level--) {
        add_links_starting_from(
                ptdis,
                pt_id,
                nearest,
                d_nearest,
                level,
                locks.data(),
                vt,
                keep_max_size_level0);
    }

    omp_unset_lock(&locks[pt_id]);

    if (pt_level > max_level) {
        max_level = pt_level;
        entry_point = pt_id;
    }
}

/**************************************************************
 * Searching
 **************************************************************/

inline HNSW::storage_idx_t get_neighbor_id_internal(
        const HNSW& hnsw,
        size_t index_in_neighbor_array) {
    // assumes index_in_neighbor_array is valid based on begin/end from
    // neighbor_range
    return hnsw.storage_is_compact
            ? hnsw.compact_neighbors_data[index_in_neighbor_array]
            : hnsw.neighbors[index_in_neighbor_array];
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

std::priority_queue<HNSW::Node> search_from_candidate_unbounded(
        const HNSW& hnsw,
        const Node& node,
        DistanceComputer& qdis,
        int ef,
        VisitedTable* vt,
        HNSWStats& stats) {
    int ndis = 0;
    std::priority_queue<Node> top_candidates;
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> candidates;

    top_candidates.push(node);
    candidates.push(node);

    vt->set(node.second);

    while (!candidates.empty()) {
        float d0;
        storage_idx_t v0;
        std::tie(d0, v0) = candidates.top();

        if (d0 > top_candidates.top().first) {
            break;
        }

        candidates.pop();

        size_t begin, end;
        hnsw.neighbor_range(v0, 0, &begin, &end);

        // a faster version: reference version in unit test test_hnsw.cpp
        // the following version processes 4 neighbors at a time
        size_t jmax = begin;
        for (size_t j = begin; j < end; j++) {
            // *** CSR Change: Conditional neighbor access ***
            int v1 = hnsw.storage_is_compact ? hnsw.compact_neighbors_data[j]
                                             : hnsw.neighbors[j];
            // *** CSR Change: Conditional break for original format ***
            if (!hnsw.storage_is_compact && v1 < 0)
                break;

            if (v1 >= 0) {
                prefetch_L2(vt->visited.data() + v1);
            }
            jmax += 1;
        }

        int counter = 0;
        size_t saved_j[4];

        auto add_to_heap = [&](const size_t idx, const float dis) {
            if (top_candidates.top().first > dis ||
                top_candidates.size() < ef) {
                candidates.emplace(dis, idx);
                top_candidates.emplace(dis, idx);

                if (top_candidates.size() > ef) {
                    top_candidates.pop();
                }
            }
        };

        std::vector<idx_t> ids_to_process;
        ids_to_process.reserve(jmax - begin);

        for (size_t j = begin; j < jmax; j++) {
            // *** CSR Change: Conditional neighbor access ***
            int v1 = hnsw.storage_is_compact ? hnsw.compact_neighbors_data[j]
                                             : hnsw.neighbors[j];

            bool vget = vt->get(v1);
            vt->set(v1);
            if (!vget) {
                vt->set(v1);
                ids_to_process.push_back(v1);
            }
        }

        if (!ids_to_process.empty()) {
            std::vector<float> batch_distances;
            qdis.distances_batch(ids_to_process, batch_distances);

            for (size_t i = 0; i < ids_to_process.size(); i++) {
                add_to_heap(ids_to_process[i], batch_distances[i]);
            }

            ndis += ids_to_process.size();
        }

        stats.nhops += 1;
    }

    ++stats.n1;
    if (candidates.size() == 0) {
        ++stats.n2;
    }
    stats.ndis += ndis;

    return top_candidates;
}

/// greedily update a nearest vector at a given level
HNSWStats greedy_update_nearest(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        int level,
        storage_idx_t& nearest,
        float& d_nearest) {
    HNSWStats stats;

    // printf("greedy_update_nearest level: %d\n", level);
    int level_neighbors = 0;

    size_t max_degree_at_level = hnsw.nb_neighbors(level);
    std::vector<HNSW::storage_idx_t> neighbor_read_buffer(max_degree_at_level);
    for (;;) {
        storage_idx_t prev_nearest = nearest;

        size_t ndis = 0;
        ssize_t neighbors_read_count =
                hnsw.fetch_neighbors(nearest, level, neighbor_read_buffer);
        stats.n_ios++;

        std::vector<idx_t> neighbors_to_process(neighbors_read_count);
        size_t valid_neighbor_count = 0;
        for (ssize_t i = 0; i < neighbors_read_count; ++i) {
            storage_idx_t v_storage = neighbor_read_buffer[i];
            assert(v_storage >= 0 && v_storage < hnsw.levels.size());
            neighbors_to_process[valid_neighbor_count++] =
                    static_cast<idx_t>(v_storage);
        }
        neighbors_to_process.resize(valid_neighbor_count);

        std::vector<float> batch_distances(valid_neighbor_count);
        qdis.distances_batch(neighbors_to_process, batch_distances);
        stats.ndis += valid_neighbor_count;

        for (size_t i = 0; i < valid_neighbor_count; ++i) {
            idx_t neighbor_id = neighbors_to_process[i];
            float neighbor_dist = batch_distances[i];

            if (neighbor_dist < d_nearest) {
                d_nearest = neighbor_dist;
                nearest = static_cast<storage_idx_t>(neighbor_id);
            }
        }

        stats.nhops++;

        if (nearest == prev_nearest) {
            return stats;
        }
    }
}

namespace {
using MinimaxHeap = HNSW::MinimaxHeap;
using Node = HNSW::Node;
using C = HNSW::C;

// just used as a lower bound for the minmaxheap, but it is set for heap search
int extract_k_from_ResultHandler(ResultHandler<C>& res) {
    using RH = HeapBlockResultHandler<C>;
    if (auto hres = dynamic_cast<RH::SingleResultHandler*>(&res)) {
        return hres->k;
    }
    return 1;
}

} // namespace

HNSWStats HNSW::search(
        DistanceComputer& qdis,
        ResultHandler<C>& res,
        VisitedTable& vt,
        const SearchParameters* params,
        const IndexHNSW* hnsw_index) const {
    HNSWStats stats;
    if (entry_point == -1) {
        return stats;
    }
    int k = extract_k_from_ResultHandler(res);

    bool bounded_queue = this->search_bounded_queue;
    int efSearch = this->efSearch;
    if (params) {
        if (const SearchParametersHNSW* hnsw_params =
                    dynamic_cast<const SearchParametersHNSW*>(params)) {
            bounded_queue = hnsw_params->bounded_queue;
            efSearch = hnsw_params->efSearch;
        }
    }

    //  greedy search on upper levels
    storage_idx_t nearest = entry_point;
    float d_nearest = qdis(nearest);

    // printf("max_level: %d\n", max_level);
    for (int level = max_level; level >= 1; level--) {
        HNSWStats local_stats =
                greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
        stats.combine(local_stats);
    }
    if (hnsw_index) {
        // printf("initial fetch count: %ld\n", qdis.get_fetch_count());
    }

    int ef = std::max(efSearch, k);
    if (bounded_queue) { // this is the most common branch
        MinimaxHeap candidates(ef);

        candidates.push(nearest, d_nearest);

        search_from_candidates(
                *this,
                qdis,
                res,
                candidates,
                vt,
                stats,
                0,
                0,
                params,
                hnsw_index);
    } else {
        std::priority_queue<Node> top_candidates =
                search_from_candidate_unbounded(
                        *this, Node(d_nearest, nearest), qdis, ef, &vt, stats);

        while (top_candidates.size() > k) {
            top_candidates.pop();
        }

        while (!top_candidates.empty()) {
            float d;
            storage_idx_t label;
            std::tie(d, label) = top_candidates.top();
            res.add_result(d, label);
            top_candidates.pop();
        }
    }

    vt.advance();

    return stats;
}

void HNSW::search_level_0(
        DistanceComputer& qdis,
        ResultHandler<C>& res,
        idx_t nprobe,
        const storage_idx_t* nearest_i,
        const float* nearest_d,
        int search_type,
        HNSWStats& search_stats,
        VisitedTable& vt,
        const SearchParameters* params) const {
    const HNSW& hnsw = *this;

    auto efSearch = hnsw.efSearch;
    if (params) {
        if (const SearchParametersHNSW* hnsw_params =
                    dynamic_cast<const SearchParametersHNSW*>(params)) {
            efSearch = hnsw_params->efSearch;
        }
    }

    int k = extract_k_from_ResultHandler(res);

    if (search_type == 1) {
        int nres = 0;

        for (int j = 0; j < nprobe; j++) {
            storage_idx_t cj = nearest_i[j];

            if (cj < 0)
                break;

            if (vt.get(cj))
                continue;

            int candidates_size = std::max(efSearch, k);
            MinimaxHeap candidates(candidates_size);

            candidates.push(cj, nearest_d[j]);

            nres = search_from_candidates(
                    hnsw,
                    qdis,
                    res,
                    candidates,
                    vt,
                    search_stats,
                    0,
                    nres,
                    params);
            nres = std::min(nres, candidates_size);
        }
    } else if (search_type == 2) {
        int candidates_size = std::max(efSearch, int(k));
        candidates_size = std::max(candidates_size, int(nprobe));

        MinimaxHeap candidates(candidates_size);
        for (int j = 0; j < nprobe; j++) {
            storage_idx_t cj = nearest_i[j];

            if (cj < 0)
                break;
            candidates.push(cj, nearest_d[j]);
        }

        search_from_candidates(
                hnsw, qdis, res, candidates, vt, search_stats, 0, 0, params);
    }
}

void HNSW::permute_entries(const idx_t* map) {
    // remap levels
    storage_idx_t ntotal = levels.size();
    std::vector<storage_idx_t> imap(ntotal); // inverse mapping
    // map: new index -> old index
    // imap: old index -> new index
    for (int i = 0; i < ntotal; i++) {
        assert(map[i] >= 0 && map[i] < ntotal);
        imap[map[i]] = i;
    }
    if (entry_point != -1) {
        entry_point = imap[entry_point];
    }
    std::vector<int> new_levels(ntotal);
    std::vector<size_t> new_offsets(ntotal + 1);
    std::vector<storage_idx_t> new_neighbors(neighbors.size());
    size_t no = 0;
    for (int i = 0; i < ntotal; i++) {
        storage_idx_t o = map[i]; // corresponding "old" index
        new_levels[i] = levels[o];
        for (size_t j = offsets[o]; j < offsets[o + 1]; j++) {
            storage_idx_t neigh = neighbors[j];
            new_neighbors[no++] = neigh >= 0 ? imap[neigh] : neigh;
        }
        new_offsets[i + 1] = no;
    }
    assert(new_offsets[ntotal] == offsets[ntotal]);
    // swap everyone
    std::swap(levels, new_levels);
    std::swap(offsets, new_offsets);
    neighbors = std::move(new_neighbors);
}

/**************************************************************
 * MinimaxHeap
 **************************************************************/

void HNSW::MinimaxHeap::push(storage_idx_t i, float v) {
    if (k == n) {
        if (v >= dis[0])
            return;
        if (ids[0] != -1) {
            --nvalid;
        }
        faiss::heap_pop<HC>(k--, dis.data(), ids.data());
    }
    faiss::heap_push<HC>(++k, dis.data(), ids.data(), v, i);
    ++nvalid;
}

float HNSW::MinimaxHeap::max() const {
    return dis[0];
}

int HNSW::MinimaxHeap::size() const {
    return nvalid;
}

void HNSW::MinimaxHeap::clear() {
    nvalid = k = 0;
}

#ifdef __AVX512F__

int HNSW::MinimaxHeap::pop_min(float* vmin_out) {
    assert(k > 0);
    static_assert(
            std::is_same<storage_idx_t, int32_t>::value,
            "This code expects storage_idx_t to be int32_t");

    int32_t min_idx = -1;
    float min_dis = std::numeric_limits<float>::infinity();

    __m512i min_indices = _mm512_set1_epi32(-1);
    __m512 min_distances =
            _mm512_set1_ps(std::numeric_limits<float>::infinity());
    __m512i current_indices = _mm512_setr_epi32(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    __m512i offset = _mm512_set1_epi32(16);

    // The following loop tracks the rightmost index with the min distance.
    // -1 index values are ignored.
    const int k16 = (k / 16) * 16;
    for (size_t iii = 0; iii < k16; iii += 16) {
        __m512i indices =
                _mm512_loadu_si512((const __m512i*)(ids.data() + iii));
        __m512 distances = _mm512_loadu_ps(dis.data() + iii);

        // This mask filters out -1 values among indices.
        __mmask16 m1mask =
                _mm512_cmpgt_epi32_mask(_mm512_setzero_si512(), indices);

        __mmask16 dmask =
                _mm512_cmp_ps_mask(min_distances, distances, _CMP_LT_OS);
        __mmask16 finalmask = m1mask | dmask;

        const __m512i min_indices_new = _mm512_mask_blend_epi32(
                finalmask, current_indices, min_indices);
        const __m512 min_distances_new =
                _mm512_mask_blend_ps(finalmask, distances, min_distances);

        min_indices = min_indices_new;
        min_distances = min_distances_new;

        current_indices = _mm512_add_epi32(current_indices, offset);
    }

    // leftovers
    if (k16 != k) {
        const __mmask16 kmask = (1 << (k - k16)) - 1;

        __m512i indices = _mm512_mask_loadu_epi32(
                _mm512_set1_epi32(-1), kmask, ids.data() + k16);
        __m512 distances = _mm512_maskz_loadu_ps(kmask, dis.data() + k16);

        // This mask filters out -1 values among indices.
        __mmask16 m1mask =
                _mm512_cmpgt_epi32_mask(_mm512_setzero_si512(), indices);

        __mmask16 dmask =
                _mm512_cmp_ps_mask(min_distances, distances, _CMP_LT_OS);
        __mmask16 finalmask = m1mask | dmask;

        const __m512i min_indices_new = _mm512_mask_blend_epi32(
                finalmask, current_indices, min_indices);
        const __m512 min_distances_new =
                _mm512_mask_blend_ps(finalmask, distances, min_distances);

        min_indices = min_indices_new;
        min_distances = min_distances_new;
    }

    // grab min distance
    min_dis = _mm512_reduce_min_ps(min_distances);
    // blend
    __mmask16 mindmask =
            _mm512_cmpeq_ps_mask(min_distances, _mm512_set1_ps(min_dis));
    // pick the max one
    min_idx = _mm512_mask_reduce_max_epi32(mindmask, min_indices);

    if (min_idx == -1) {
        return -1;
    }

    if (vmin_out) {
        *vmin_out = min_dis;
    }
    int ret = ids[min_idx];
    ids[min_idx] = -1;
    --nvalid;
    return ret;
}

#elif __AVX2__

int HNSW::MinimaxHeap::pop_min(float* vmin_out) {
    assert(k > 0);
    static_assert(
            std::is_same<storage_idx_t, int32_t>::value,
            "This code expects storage_idx_t to be int32_t");

    int32_t min_idx = -1;
    float min_dis = std::numeric_limits<float>::infinity();

    size_t iii = 0;

    __m256i min_indices = _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, -1, -1);
    __m256 min_distances =
            _mm256_set1_ps(std::numeric_limits<float>::infinity());
    __m256i current_indices = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256i offset = _mm256_set1_epi32(8);

    // The baseline version is available in non-AVX2 branch.

    // The following loop tracks the rightmost index with the min distance.
    // -1 index values are ignored.
    const int k8 = (k / 8) * 8;
    for (; iii < k8; iii += 8) {
        __m256i indices =
                _mm256_loadu_si256((const __m256i*)(ids.data() + iii));
        __m256 distances = _mm256_loadu_ps(dis.data() + iii);

        // This mask filters out -1 values among indices.
        __m256i m1mask = _mm256_cmpgt_epi32(_mm256_setzero_si256(), indices);

        __m256i dmask = _mm256_castps_si256(
                _mm256_cmp_ps(min_distances, distances, _CMP_LT_OS));
        __m256 finalmask = _mm256_castsi256_ps(_mm256_or_si256(m1mask, dmask));

        const __m256i min_indices_new = _mm256_castps_si256(_mm256_blendv_ps(
                _mm256_castsi256_ps(current_indices),
                _mm256_castsi256_ps(min_indices),
                finalmask));

        const __m256 min_distances_new =
                _mm256_blendv_ps(distances, min_distances, finalmask);

        min_indices = min_indices_new;
        min_distances = min_distances_new;

        current_indices = _mm256_add_epi32(current_indices, offset);
    }

    // Vectorizing is doable, but is not practical
    int32_t vidx8[8];
    float vdis8[8];
    _mm256_storeu_ps(vdis8, min_distances);
    _mm256_storeu_si256((__m256i*)vidx8, min_indices);

    for (size_t j = 0; j < 8; j++) {
        if (min_dis > vdis8[j] || (min_dis == vdis8[j] && min_idx < vidx8[j])) {
            min_idx = vidx8[j];
            min_dis = vdis8[j];
        }
    }

    // process last values. Vectorizing is doable, but is not practical
    for (; iii < k; iii++) {
        if (ids[iii] != -1 && dis[iii] <= min_dis) {
            min_dis = dis[iii];
            min_idx = iii;
        }
    }

    if (min_idx == -1) {
        return -1;
    }

    if (vmin_out) {
        *vmin_out = min_dis;
    }
    int ret = ids[min_idx];
    ids[min_idx] = -1;
    --nvalid;
    return ret;
}

#else

// baseline non-vectorized version
int HNSW::MinimaxHeap::pop_min(float* vmin_out) {
    assert(k > 0);
    // returns min. This is an O(n) operation
    int i = k - 1;
    while (i >= 0) {
        if (ids[i] != -1) {
            break;
        }
        i--;
    }
    if (i == -1) {
        return -1;
    }
    int imin = i;
    float vmin = dis[i];
    i--;
    while (i >= 0) {
        if (ids[i] != -1 && dis[i] < vmin) {
            vmin = dis[i];
            imin = i;
        }
        i--;
    }
    if (vmin_out) {
        *vmin_out = vmin;
    }
    int ret = ids[imin];
    ids[imin] = -1;
    --nvalid;

    return ret;
}
#endif

int HNSW::MinimaxHeap::count_below(float thresh) {
    int n_below = 0;
    for (int i = 0; i < k; i++) {
        if (dis[i] < thresh) {
            n_below++;
        }
    }

    return n_below;
}

} // namespace faiss
