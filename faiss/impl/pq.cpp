#include "pq.h"
#include <xmmintrin.h> // For _mm_prefetch
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
#include "faiss/impl/FaissAssert.h"

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

    std::ifstream reader(pq_pivots_path, std::ios::binary);

    int32_t offset_nr_i32, offset_nc_i32;
    reader.read(reinterpret_cast<char*>(&offset_nr_i32), sizeof(int32_t));
    reader.read(reinterpret_cast<char*>(&offset_nc_i32), sizeof(int32_t));

    size_t offset_rows = (size_t)offset_nr_i32;
    size_t offset_cols = (size_t)offset_nc_i32;
    // FAISS_ASSERT(offset_cols == 1 && (offset_rows == 4 || offset_rows == 5));

    size_t num_offsets = offset_rows * offset_cols;
    std::unique_ptr<size_t[]> file_offsets(new size_t[num_offsets]);
    reader.read(
            reinterpret_cast<char*>(file_offsets.get()),
            num_offsets * sizeof(size_t));

    bool use_old_filetype = (offset_rows == 5);

    constexpr size_t NUM_PQ_CENTROIDS = 256;
    float* tables_original = nullptr;
    size_t pivot_rows = 0, pivot_cols = 0;
    {
        reader.seekg(file_offsets[0], std::ios::beg);
        int32_t r_i32, c_i32;
        reader.read(reinterpret_cast<char*>(&r_i32), sizeof(int32_t));
        reader.read(reinterpret_cast<char*>(&c_i32), sizeof(int32_t));
        pivot_rows = (size_t)r_i32;
        pivot_cols = (size_t)c_i32;
        // FAISS_ASSERT(pivot_rows == NUM_PQ_CENTROIDS && pivot_cols > 0); //
        ndims = pivot_cols;
        tables_original = new float[pivot_rows * pivot_cols];
        reader.read(
                reinterpret_cast<char*>(tables_original),
                pivot_rows * pivot_cols * sizeof(float));
    }
    std::unique_ptr<float[]> tables_original_ptr(tables_original);

    float* centroid_raw = nullptr;
    size_t centroid_rows = 0, centroid_cols = 0;
    {
        reader.seekg(file_offsets[1], std::ios::beg);
        int32_t r_i32, c_i32;
        reader.read(reinterpret_cast<char*>(&r_i32), sizeof(int32_t));
        reader.read(reinterpret_cast<char*>(&c_i32), sizeof(int32_t));
        centroid_rows = (size_t)r_i32;
        centroid_cols = (size_t)c_i32;
        // FAISS_ASSERT(centroid_rows == ndims && centroid_cols == 1); //
        centroid_raw = new float[centroid_rows * centroid_cols]; // 直接 new
        reader.read(
                reinterpret_cast<char*>(centroid_raw),
                centroid_rows * centroid_cols * sizeof(float));
    }
    this->centroid = centroid_raw;

    int chunk_offsets_file_idx = use_old_filetype ? 3 : 2;
    uint32_t* chunk_offsets_raw = nullptr;
    size_t chunk_rows = 0, chunk_cols = 0;
    {
        reader.seekg(file_offsets[chunk_offsets_file_idx], std::ios::beg);
        int32_t r_i32, c_i32;
        reader.read(reinterpret_cast<char*>(&r_i32), sizeof(int32_t));
        reader.read(reinterpret_cast<char*>(&c_i32), sizeof(int32_t));
        chunk_rows = (size_t)r_i32;
        chunk_cols = (size_t)c_i32;
        // FAISS_ASSERT(chunk_cols == 1 && chunk_rows >= 2);
        n_chunks = chunk_rows - 1;
        chunk_offsets_raw = new uint32_t[chunk_rows * chunk_cols];
        reader.read(
                reinterpret_cast<char*>(chunk_offsets_raw),
                chunk_rows * chunk_cols * sizeof(uint32_t));
    }
    this->chunk_offsets = chunk_offsets_raw;

    tables_tr = new float[ndims * NUM_PQ_CENTROIDS];
    for (size_t i = 0; i < NUM_PQ_CENTROIDS; i++) {
        for (size_t j = 0; j < ndims; j++) {
            tables_tr[j * NUM_PQ_CENTROIDS + i] =
                    tables_original[i * ndims + j];
        }
    }

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

    // Calculate L2 norm of the original query vector
    double norm_sq = 0.0;
    for (size_t d = 0; d < ndims; d++) {
        norm_sq += (double)query[d] * query[d];
    }
    double norm = std::sqrt(norm_sq);

    // Handle zero norm case (avoid division by zero)
    if (norm <= 0.0) {
        norm = 1.0;
    }

    // First normalize the query vector, then subtract the centroid
    // This adapts L2-based PQ distance for IP/Cosine search approximation,
    // assuming the loaded PQ codebook is compatible with this approach
    for (size_t d = 0; d < ndims; d++) {
        float normalized_query_d = query[d] / (float)norm;
        query_preprocessed[d] = normalized_query_d - centroid[d];
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