#pragma once

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

#include "faiss/Index.h"

namespace faiss {

class PQPrunerDataLoader {
   public:
    float* tables_tr = nullptr;        // transposed codebook (ndims * 256)
    float* centroid = nullptr;         // centered vector (ndims)
    uint32_t* chunk_offsets = nullptr; // chunk offsets (n_chunks + 1)
    size_t ndims = 0;                  // original dimension
    size_t n_chunks = 0;               // number of PQ chunks
    bool initialized = false;

    PQPrunerDataLoader();
    ~PQPrunerDataLoader();

    PQPrunerDataLoader(const PQPrunerDataLoader&) = delete;
    PQPrunerDataLoader& operator=(const PQPrunerDataLoader&) = delete;
    PQPrunerDataLoader(PQPrunerDataLoader&&) = default;
    PQPrunerDataLoader& operator=(PQPrunerDataLoader&&) = default;

    bool load_pq_pivots(const std::string& pq_pivots_path);

    // Preprocess the query (only centering)
    void preprocess_query(const float* query, float* query_preprocessed) const;

    // Calculate the distance lookup table from the query vector to all codebook
    // centroids (L2 distance)
    void populate_chunk_distances(
            const float* query_preprocessed,
            float* pq_dists_lookup_table) const;

    size_t get_num_chunks() const;
    size_t get_dims() const;
    bool is_initialized() const;
};

size_t aggregate_pq_codes(
        const idx_t* ids,
        size_t num_ids,
        const uint8_t* all_pq_codes,
        size_t total_codes_count,
        size_t code_size,
        uint8_t* out_aggregated_codes);

void pq_distance_lookup(
        const uint8_t* aggregated_codes,
        size_t n_pts,
        size_t n_chunks,
        const float* pq_dists_lookup_table,
        float* dists_out);

template <typename T>
bool load_simple_bin(
        const std::string& filename,
        T*& data,
        size_t& n_rows,
        size_t& n_cols);

template <typename T>
bool load_simple_bin(
        const std::string& filename,
        T*& data,
        size_t& n_rows,
        size_t& n_cols) {
    data = nullptr;
    n_rows = 0;
    n_cols = 0;
    std::ifstream reader(filename, std::ios::binary);
    if (!reader) {
        std::cerr << "Error: Cannot open simple binary file: " << filename
                  << std::endl;
        return false;
    }
    uint32_t r, c;
    reader.read(reinterpret_cast<char*>(&r), sizeof(uint32_t));
    reader.read(reinterpret_cast<char*>(&c), sizeof(uint32_t));
    if (!reader) { /* ... */
        return false;
    }
    n_rows = r;
    n_cols = c;
    size_t num_elements = n_rows * n_cols;
    if (num_elements == 0)
        return true;
    try {
        data = new T[num_elements];
    } catch (const std::bad_alloc& e) { /* ... */
        return false;
    }
    reader.read(reinterpret_cast<char*>(data), num_elements * sizeof(T));
    if (!reader) { /* ... */
        delete[] data;
        data = nullptr;
        return false;
    }
    return true;
}

} // namespace faiss