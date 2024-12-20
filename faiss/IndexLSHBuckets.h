#ifndef INDEX_LSH_H
#define INDEX_LSH_H

#include <vector>

#include <faiss/IndexFlatCodes.h>
#include <faiss/VectorTransform.h>

namespace faiss {
 /** The sign of each vector component is put in a binary signature */
struct IndexLSHBuckets : IndexFlatCodes {

    int nbits;             ///< nb of bits per vector
    bool rotate_data;      ///< whether to apply a random rotation to input
    bool train_thresholds; ///< whether we train thresholds or use 0

    RandomRotationMatrix rrot; ///< optional random rotation

    std::vector<float> thresholds; ///< thresholds to compare with

    IndexLSHBuckets(
            idx_t d,
            int nbits,
            bool rotate_data = true,
            bool train_thresholds = false);
            
    const float* apply_preprocess(idx_t n, const float* x) const;
    void train(idx_t n, const float* x) override;

    void Hash_code(idx_t n, const float* x, uint8_t* bytes, int bucket_count, std::vector<std::vector<uint64_t>>& bucket_mapping) const;
};
}

#endif