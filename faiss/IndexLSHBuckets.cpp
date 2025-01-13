#include <faiss/IndexLSHBuckets.h>
#include <faiss/IndexLSH.h>
#include <cstdio>
#include <cstring>

#include <algorithm>
#include <memory>

#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/hamming.h>

namespace faiss {
    IndexLSHBuckets::IndexLSHBuckets(idx_t d, int nbits, bool rotate_data, bool train_thresholds)
        : IndexFlatCodes((nbits + 7) / 8, d),
          nbits(nbits),
          rotate_data(rotate_data),
          train_thresholds(train_thresholds),
          rrot(d, nbits) {
    is_trained = !train_thresholds;

    if (rotate_data) {
        rrot.init(5);
    } else {
        FAISS_THROW_IF_NOT(d >= nbits);
    }

}

const float* IndexLSHBuckets::apply_preprocess(idx_t n, const float* x) const {
    float* xt = nullptr;
    if (rotate_data) {
        // also applies bias if exists
        xt = rrot.apply(n, x);
    } else if (d != nbits) {
        assert(nbits < d);
        xt = new float[nbits * n];
        float* xp = xt;
        for (idx_t i = 0; i < n; i++) {
            const float* xl = x + i * d;
            for (int j = 0; j < nbits; j++)
                *xp++ = xl[j];
        }
    }

    if (train_thresholds) {
        if (xt == nullptr) {
            xt = new float[nbits * n];
            memcpy(xt, x, sizeof(*x) * n * nbits);
        }

        float* xp = xt;
        for (idx_t i = 0; i < n; i++)
            for (int j = 0; j < nbits; j++)
                *xp++ -= thresholds[j];
    }

    return xt ? xt : x;
}

void IndexLSHBuckets::train(idx_t n, const float* x) {
    if (train_thresholds) {
        thresholds.resize(nbits);
        train_thresholds = false;
        const float* xt = apply_preprocess(n, x);
        std::unique_ptr<const float[]> del(xt == x ? nullptr : xt);
        train_thresholds = true;

        std::unique_ptr<float[]> transposed_x(new float[n * nbits]);

        for (idx_t i = 0; i < n; i++)
            for (idx_t j = 0; j < nbits; j++)
                transposed_x[j * n + i] = xt[i * nbits + j];

        for (idx_t i = 0; i < nbits; i++) {
            float* xi = transposed_x.get() + i * n;
            // std::nth_element
            std::sort(xi, xi + n);
            if (n % 2 == 1)
                thresholds[i] = xi[n / 2];
            else
                thresholds[i] = (xi[n / 2 - 1] + xi[n / 2]) / 2;
        }
    }
    is_trained = true;
}

/**
 * @brief Computes the hash bucket mappings for a set of data points.
 * 
 * This method applies preprocessing to the input data, computes the hash values 
 * using a Locality-Sensitive Hashing (LSH) approach, and stores the resulting hash 
 * bucket numbers in a 2D vector. Each data point's hash bucket number is calculated 
 * based on the hash of its corresponding feature vector.
 * 
 * @param n The number of data points to be processed.
 * @param x A pointer to the input feature matrix (a 2D array of floats).
 * @param bytes A pointer to an array of bytes where the bit representation of 
 *              the feature vectors will be stored.
 * @param bucket_count The number of hash buckets to be used.
 * @param bucket_mapping A reference to a 2D vector (std::vector<std::vector<uint64_t>>) 
 *                       where the resulting hash bucket mappings will be stored. 
 *                       Each row in the vector corresponds to one data point, 
 *                       and each entry in a row represents a hash bucket number.
 * 
 * @note This method assumes that the LSH model is already trained. It also assumes 
 *       that the number of buckets is defined by `bucket_count` and that the 
 *       hash codes are computed using a bit representation of the input data.
 */
void IndexLSHBuckets::Hash_code(idx_t n, const float* x, uint8_t* bytes, int bucket_count,std::vector<std::vector<uint64_t>>& bucket_mapping) const {
    FAISS_THROW_IF_NOT(is_trained);
    const float* xt = apply_preprocess(n, x);
    std::unique_ptr<const float[]> del(xt == x ? nullptr : xt);
    fvecs2bitvecs(xt, bytes, nbits, n);
     for (idx_t i = 0; i < n; ++i) {
        uint64_t hash_value = 0;
        for (int j = 0; j < nbits / 8; ++j) {
            hash_value |= static_cast<uint64_t>(bytes[i * (nbits / 8) + j]) << (8 * j);
        }
        uint64_t bucket_number = hash_value % bucket_count;

        bucket_mapping[i].push_back(bucket_number);
    }
}


}