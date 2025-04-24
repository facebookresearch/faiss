#include "pq.h"
#include <xmmintrin.h> // For _mm_prefetch
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace faiss {

PQPrunerDataLoader::PQPrunerDataLoader()
        : tables_tr(nullptr),
          centroid(nullptr),
          chunk_offsets(nullptr),
          ndims(0),
          n_chunks(0),
          initialized(false) {}

PQPrunerDataLoader::~PQPrunerDataLoader() {
    delete[] tables_tr;
    delete[] centroid;
    delete[] chunk_offsets;
}

size_t PQPrunerDataLoader::get_num_chunks() const {
    return n_chunks;
}
size_t PQPrunerDataLoader::get_dims() const {
    return ndims;
}
bool PQPrunerDataLoader::is_initialized() const {
    return initialized;
}

bool PQPrunerDataLoader::load_pq_pivots(const std::string& pq_pivots_path) {
    initialized = false;
    delete[] tables_tr;
    tables_tr = nullptr;
    delete[] centroid;
    centroid = nullptr;
    delete[] chunk_offsets;
    chunk_offsets = nullptr;
    ndims = 0;
    n_chunks = 0;

    std::ifstream reader(pq_pivots_path, std::ios::binary);
    if (!reader) { /* ... */
        return false;
    }

    // 1. Read offset metadata
    size_t offset_metadata_size = 0;
    reader.read(reinterpret_cast<char*>(&offset_metadata_size), sizeof(size_t));
    if (!reader ||
        (offset_metadata_size != 4 && offset_metadata_size != 5)) { /* ... */
        return false;
    }
    std::vector<size_t> file_offsets(offset_metadata_size);
    reader.read(
            reinterpret_cast<char*>(file_offsets.data()),
            offset_metadata_size * sizeof(size_t));
    if (!reader) { /* ... */
        return false;
    }
    bool use_old_filetype = (offset_metadata_size == 5);

    // 2. Load original codebook
    constexpr size_t NUM_PQ_CENTROIDS = 256;
    size_t table_offset = file_offsets[0];
    reader.seekg(table_offset, std::ios::beg);
    uint32_t file_rows, file_cols;
    reader.read(reinterpret_cast<char*>(&file_rows), sizeof(uint32_t));
    reader.read(reinterpret_cast<char*>(&file_cols), sizeof(uint32_t));
    if (!reader || file_rows != NUM_PQ_CENTROIDS || file_cols == 0) { /* ... */
        return false;
    }
    ndims = file_cols;
    float* tables_original = new (std::nothrow) float[NUM_PQ_CENTROIDS * ndims];
    if (!tables_original) { /* ... */
        return false;
    }
    reader.read(
            reinterpret_cast<char*>(tables_original),
            NUM_PQ_CENTROIDS * ndims * sizeof(float));
    if (!reader) {
        delete[] tables_original;
        return false;
    }

    // 3. Load centroid
    size_t centroid_offset = file_offsets[1];
    reader.seekg(centroid_offset, std::ios::beg);
    reader.read(reinterpret_cast<char*>(&file_rows), sizeof(uint32_t));
    reader.read(reinterpret_cast<char*>(&file_cols), sizeof(uint32_t));
    if (!reader || file_rows != ndims || file_cols != 1) {
        delete[] tables_original;
        return false;
    }
    centroid = new (std::nothrow) float[ndims];
    if (!centroid) {
        delete[] tables_original;
        return false;
    }
    reader.read(reinterpret_cast<char*>(centroid), ndims * sizeof(float));
    if (!reader) {
        delete[] tables_original;
        delete[] centroid;
        centroid = nullptr;
        return false;
    }

    // 4. Load chunk offsets
    int chunk_offsets_index = use_old_filetype ? 3 : 2;
    size_t chunk_offset_offset = file_offsets[chunk_offsets_index];
    reader.seekg(chunk_offset_offset, std::ios::beg);
    reader.read(reinterpret_cast<char*>(&file_rows), sizeof(uint32_t));
    reader.read(reinterpret_cast<char*>(&file_cols), sizeof(uint32_t));
    if (!reader || file_cols != 1 || file_rows < 2) {
        delete[] tables_original;
        delete[] centroid;
        return false;
    }
    n_chunks = file_rows - 1;
    chunk_offsets = new (std::nothrow) uint32_t[file_rows];
    if (!chunk_offsets) {
        delete[] tables_original;
        delete[] centroid;
        return false;
    }
    reader.read(
            reinterpret_cast<char*>(chunk_offsets),
            file_rows * sizeof(uint32_t));
    if (!reader) {
        delete[] tables_original;
        delete[] centroid;
        delete[] chunk_offsets;
        chunk_offsets = nullptr;
        return false;
    }

    // 5. Transpose codebook
    tables_tr = new (std::nothrow) float[ndims * NUM_PQ_CENTROIDS];
    if (!tables_tr) {
        delete[] tables_original;
        delete[] centroid;
        delete[] chunk_offsets;
        return false;
    }
    for (size_t i = 0; i < NUM_PQ_CENTROIDS; i++) {
        for (size_t j = 0; j < ndims; j++) {
            tables_tr[j * NUM_PQ_CENTROIDS + i] =
                    tables_original[i * ndims + j];
        }
    }
    delete[] tables_original;

    initialized = true;
    std::cout << "PQPrunerDataLoader initialized from .cpp. Dims: " << ndims
              << ", Chunks: " << n_chunks << std::endl;
    return true;
}

void PQPrunerDataLoader::preprocess_query(
        const float* query,
        float* query_preprocessed) const {
    if (!initialized)
        return;
    for (size_t d = 0; d < ndims; d++) {
        query_preprocessed[d] = query[d] - centroid[d];
    }
}

void PQPrunerDataLoader::populate_chunk_distances(
        const float* query_preprocessed,
        float* pq_dists_lookup_table) const {
    if (!initialized || !tables_tr || !chunk_offsets)
        return;
    constexpr size_t NUM_PQ_CENTROIDS = 256;
    memset(pq_dists_lookup_table,
           0,
           NUM_PQ_CENTROIDS * n_chunks * sizeof(float));
    for (size_t chunk = 0; chunk < n_chunks; chunk++) {
        float* chunk_dists = pq_dists_lookup_table + (NUM_PQ_CENTROIDS * chunk);
        uint32_t start_dim = chunk_offsets[chunk];
        uint32_t end_dim = chunk_offsets[chunk + 1];
        for (size_t j = start_dim; j < end_dim; j++) {
            const float* centers_dim_vec = tables_tr + (NUM_PQ_CENTROIDS * j);
            float query_val_dim_j = query_preprocessed[j];
            for (size_t idx = 0; idx < NUM_PQ_CENTROIDS; idx++) {
                double diff =
                        (double)centers_dim_vec[idx] - (double)query_val_dim_j;
                chunk_dists[idx] += (float)(diff * diff);
            }
        }
    }
}

size_t aggregate_pq_codes(
        const idx_t* ids,
        size_t num_ids,
        const uint8_t* all_pq_codes,
        size_t total_codes_count,
        size_t code_size,
        uint8_t* out_aggregated_codes) {
    size_t count = 0;
    for (size_t i = 0; i < num_ids; i++) {
        idx_t id = ids[i];
        if (id < 0 || (size_t)id >= total_codes_count) {
            std::cerr << "Warning: Invalid ID " << id
                      << " during PQ code aggregation." << std::endl;
            continue;
        }
        const uint8_t* code_ptr = all_pq_codes + (size_t)id * code_size;
        memcpy(out_aggregated_codes + count * code_size, code_ptr, code_size);
        count++;
    }
    return count;
}

void pq_distance_lookup(
        const uint8_t* aggregated_codes,
        size_t n_pts,
        size_t n_chunks,
        const float* pq_dists_lookup_table,
        float* dists_out) {
    memset(dists_out, 0, n_pts * sizeof(float));
    for (size_t chunk = 0; chunk < n_chunks; chunk++) {
        const float* chunk_dists = pq_dists_lookup_table + 256 * chunk;
        if (chunk < n_chunks - 1) {
            _mm_prefetch((const char*)(chunk_dists + 256), _MM_HINT_T0);
        }
        for (size_t idx = 0; idx < n_pts; idx++) {
            uint8_t pq_centerid = aggregated_codes[idx * n_chunks + chunk];
            dists_out[idx] += chunk_dists[pq_centerid];
        }
    }
}
} // namespace faiss