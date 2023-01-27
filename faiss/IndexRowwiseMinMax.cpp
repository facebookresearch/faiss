#include <faiss/IndexRowwiseMinMax.h>

#include <cstdint>
#include <cstring>
#include <limits>

#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/fp16.h>

namespace faiss {

namespace {

using idx_t = faiss::idx_t;

struct StorageMinMaxFP16 {
    uint16_t scaler;
    uint16_t minv;

    inline void from_floats(const float float_scaler, const float float_minv) {
        scaler = encode_fp16(float_scaler);
        minv = encode_fp16(float_minv);
    }

    inline void to_floats(float& float_scaler, float& float_minv) const {
        float_scaler = decode_fp16(scaler);
        float_minv = decode_fp16(minv);
    }
};

struct StorageMinMaxFP32 {
    float scaler;
    float minv;

    inline void from_floats(const float float_scaler, const float float_minv) {
        scaler = float_scaler;
        minv = float_minv;
    }

    inline void to_floats(float& float_scaler, float& float_minv) const {
        float_scaler = scaler;
        float_minv = minv;
    }
};

template <typename StorageMinMaxT>
void sa_encode_impl(
        const IndexRowwiseMinMaxBase* const index,
        const idx_t n_input,
        const float* x_input,
        uint8_t* bytes_output) {
    // process chunks
    const size_t chunk_size = rowwise_minmax_sa_encode_bs;

    // useful variables
    const Index* const sub_index = index->index;
    const int d = index->d;

    // the code size of the subindex
    const size_t old_code_size = sub_index->sa_code_size();
    // the code size of the index
    const size_t new_code_size = index->sa_code_size();

    // allocate tmp buffers
    std::vector<float> tmp(chunk_size * d);
    std::vector<StorageMinMaxT> minmax(chunk_size);

    // all the elements to process
    size_t n_left = n_input;

    const float* __restrict x = x_input;
    uint8_t* __restrict bytes = bytes_output;

    while (n_left > 0) {
        // current portion to be processed
        const idx_t n = std::min(n_left, chunk_size);

        // allocate a temporary buffer and do the rescale
        for (idx_t i = 0; i < n; i++) {
            // compute min & max values
            float minv = std::numeric_limits<float>::max();
            float maxv = std::numeric_limits<float>::lowest();

            const float* const vec_in = x + i * d;
            for (idx_t j = 0; j < d; j++) {
                minv = std::min(minv, vec_in[j]);
                maxv = std::max(maxv, vec_in[j]);
            }

            // save the coefficients
            const float scaler = maxv - minv;
            minmax[i].from_floats(scaler, minv);

            // and load them back, because the coefficients might
            // be modified.
            float actual_scaler = 0;
            float actual_minv = 0;
            minmax[i].to_floats(actual_scaler, actual_minv);

            float* const vec_out = tmp.data() + i * d;
            if (actual_scaler == 0) {
                for (idx_t j = 0; j < d; j++) {
                    vec_out[j] = 0;
                }
            } else {
                float inv_actual_scaler = 1.0f / actual_scaler;
                for (idx_t j = 0; j < d; j++) {
                    vec_out[j] = (vec_in[j] - actual_minv) * inv_actual_scaler;
                }
            }
        }

        // do the coding
        sub_index->sa_encode(n, tmp.data(), bytes);

        // rearrange
        for (idx_t i = n; (i--) > 0;) {
            // move a single index
            std::memmove(
                    bytes + i * new_code_size + (new_code_size - old_code_size),
                    bytes + i * old_code_size,
                    old_code_size);

            // save min & max values
            StorageMinMaxT* fpv = reinterpret_cast<StorageMinMaxT*>(
                    bytes + i * new_code_size);
            *fpv = minmax[i];
        }

        // next chunk
        x += n * d;
        bytes += n * new_code_size;

        n_left -= n;
    }
}

template <typename StorageMinMaxT>
void sa_decode_impl(
        const IndexRowwiseMinMaxBase* const index,
        const idx_t n_input,
        const uint8_t* bytes_input,
        float* x_output) {
    // process chunks
    const size_t chunk_size = rowwise_minmax_sa_decode_bs;

    // useful variables
    const Index* const sub_index = index->index;
    const int d = index->d;

    // the code size of the subindex
    const size_t old_code_size = sub_index->sa_code_size();
    // the code size of the index
    const size_t new_code_size = index->sa_code_size();

    // allocate tmp buffers
    std::vector<uint8_t> tmp(
            (chunk_size < n_input ? chunk_size : n_input) * old_code_size);
    std::vector<StorageMinMaxFP16> minmax(
            (chunk_size < n_input ? chunk_size : n_input));

    // all the elements to process
    size_t n_left = n_input;

    const uint8_t* __restrict bytes = bytes_input;
    float* __restrict x = x_output;

    while (n_left > 0) {
        // current portion to be processed
        const idx_t n = std::min(n_left, chunk_size);

        // rearrange
        for (idx_t i = 0; i < n; i++) {
            std::memcpy(
                    tmp.data() + i * old_code_size,
                    bytes + i * new_code_size + (new_code_size - old_code_size),
                    old_code_size);
        }

        // decode
        sub_index->sa_decode(n, tmp.data(), x);

        // scale back
        for (idx_t i = 0; i < n; i++) {
            const uint8_t* const vec_in = bytes + i * new_code_size;
            StorageMinMaxT fpv =
                    *(reinterpret_cast<const StorageMinMaxT*>(vec_in));

            float scaler = 0;
            float minv = 0;
            fpv.to_floats(scaler, minv);

            float* const __restrict vec = x + d * i;

            for (idx_t j = 0; j < d; j++) {
                vec[j] = vec[j] * scaler + minv;
            }
        }

        // next chunk
        bytes += n * new_code_size;
        x += n * d;

        n_left -= n;
    }
}

//
template <typename StorageMinMaxT>
void train_inplace_impl(
        IndexRowwiseMinMaxBase* const index,
        idx_t n,
        float* x) {
    // useful variables
    Index* const sub_index = index->index;
    const int d = index->d;

    // save normalizing coefficients
    std::vector<StorageMinMaxT> minmax(n);

    // normalize
#pragma omp for
    for (idx_t i = 0; i < n; i++) {
        // compute min & max values
        float minv = std::numeric_limits<float>::max();
        float maxv = std::numeric_limits<float>::lowest();

        float* const vec = x + i * d;
        for (idx_t j = 0; j < d; j++) {
            minv = std::min(minv, vec[j]);
            maxv = std::max(maxv, vec[j]);
        }

        // save the coefficients
        const float scaler = maxv - minv;
        minmax[i].from_floats(scaler, minv);

        // and load them back, because the coefficients might
        // be modified.
        float actual_scaler = 0;
        float actual_minv = 0;
        minmax[i].to_floats(actual_scaler, actual_minv);

        if (actual_scaler == 0) {
            for (idx_t j = 0; j < d; j++) {
                vec[j] = 0;
            }
        } else {
            float inv_actual_scaler = 1.0f / actual_scaler;
            for (idx_t j = 0; j < d; j++) {
                vec[j] = (vec[j] - actual_minv) * inv_actual_scaler;
            }
        }
    }

    // train the subindex
    sub_index->train(n, x);

    // rescale data back
    for (idx_t i = 0; i < n; i++) {
        float scaler = 0;
        float minv = 0;
        minmax[i].to_floats(scaler, minv);

        float* const vec = x + i * d;

        for (idx_t j = 0; j < d; j++) {
            vec[j] = vec[j] * scaler + minv;
        }
    }
}

//
template <typename StorageMinMaxT>
void train_impl(IndexRowwiseMinMaxBase* const index, idx_t n, const float* x) {
    // the default training that creates a copy of the input data

    // useful variables
    Index* const sub_index = index->index;
    const int d = index->d;

    // temp buffer
    std::vector<float> tmp(n * d);

#pragma omp for
    for (idx_t i = 0; i < n; i++) {
        // compute min & max values
        float minv = std::numeric_limits<float>::max();
        float maxv = std::numeric_limits<float>::lowest();

        const float* const __restrict vec_in = x + i * d;
        for (idx_t j = 0; j < d; j++) {
            minv = std::min(minv, vec_in[j]);
            maxv = std::max(maxv, vec_in[j]);
        }

        const float scaler = maxv - minv;

        // save the coefficients
        StorageMinMaxT storage;
        storage.from_floats(scaler, minv);

        // and load them back, because the coefficients might
        // be modified.
        float actual_scaler = 0;
        float actual_minv = 0;
        storage.to_floats(actual_scaler, actual_minv);

        float* const __restrict vec_out = tmp.data() + i * d;
        if (actual_scaler == 0) {
            for (idx_t j = 0; j < d; j++) {
                vec_out[j] = 0;
            }
        } else {
            float inv_actual_scaler = 1.0f / actual_scaler;
            for (idx_t j = 0; j < d; j++) {
                vec_out[j] = (vec_in[j] - actual_minv) * inv_actual_scaler;
            }
        }
    }

    sub_index->train(n, tmp.data());
}

} // namespace

// block size for performing sa_encode and sa_decode
int rowwise_minmax_sa_encode_bs = 16384;
int rowwise_minmax_sa_decode_bs = 16384;

/*********************************************************
 * IndexRowwiseMinMaxBase implementation
 ********************************************************/

IndexRowwiseMinMaxBase::IndexRowwiseMinMaxBase(Index* index)
        : Index(index->d, index->metric_type),
          index{index},
          own_fields{false} {}

IndexRowwiseMinMaxBase::IndexRowwiseMinMaxBase()
        : index{nullptr}, own_fields{false} {}

IndexRowwiseMinMaxBase::~IndexRowwiseMinMaxBase() {
    if (own_fields) {
        delete index;
        index = nullptr;
    }
}

void IndexRowwiseMinMaxBase::add(idx_t, const float*) {
    FAISS_THROW_MSG("add not implemented for this type of index");
}

void IndexRowwiseMinMaxBase::search(
        idx_t,
        const float*,
        idx_t,
        float*,
        idx_t*,
        const SearchParameters*) const {
    FAISS_THROW_MSG("search not implemented for this type of index");
}

void IndexRowwiseMinMaxBase::reset() {
    FAISS_THROW_MSG("reset not implemented for this type of index");
}

/*********************************************************
 * IndexRowwiseMinMaxFP16 implementation
 ********************************************************/

IndexRowwiseMinMaxFP16::IndexRowwiseMinMaxFP16(Index* index)
        : IndexRowwiseMinMaxBase(index) {}

IndexRowwiseMinMaxFP16::IndexRowwiseMinMaxFP16() : IndexRowwiseMinMaxBase() {}

size_t IndexRowwiseMinMaxFP16::sa_code_size() const {
    return index->sa_code_size() + 2 * sizeof(uint16_t);
}

void IndexRowwiseMinMaxFP16::sa_encode(
        idx_t n_input,
        const float* x_input,
        uint8_t* bytes_output) const {
    sa_encode_impl<StorageMinMaxFP16>(this, n_input, x_input, bytes_output);
}

void IndexRowwiseMinMaxFP16::sa_decode(
        idx_t n_input,
        const uint8_t* bytes_input,
        float* x_output) const {
    sa_decode_impl<StorageMinMaxFP16>(this, n_input, bytes_input, x_output);
}

void IndexRowwiseMinMaxFP16::train(idx_t n, const float* x) {
    train_impl<StorageMinMaxFP16>(this, n, x);
}

void IndexRowwiseMinMaxFP16::train_inplace(idx_t n, float* x) {
    train_inplace_impl<StorageMinMaxFP16>(this, n, x);
}

/*********************************************************
 * IndexRowwiseMinMax implementation
 ********************************************************/

IndexRowwiseMinMax::IndexRowwiseMinMax(Index* index)
        : IndexRowwiseMinMaxBase(index) {}

IndexRowwiseMinMax::IndexRowwiseMinMax() : IndexRowwiseMinMaxBase() {}

size_t IndexRowwiseMinMax::sa_code_size() const {
    return index->sa_code_size() + 2 * sizeof(float);
}

void IndexRowwiseMinMax::sa_encode(
        idx_t n_input,
        const float* x_input,
        uint8_t* bytes_output) const {
    sa_encode_impl<StorageMinMaxFP32>(this, n_input, x_input, bytes_output);
}

void IndexRowwiseMinMax::sa_decode(
        idx_t n_input,
        const uint8_t* bytes_input,
        float* x_output) const {
    sa_decode_impl<StorageMinMaxFP32>(this, n_input, bytes_input, x_output);
}

void IndexRowwiseMinMax::train(idx_t n, const float* x) {
    train_impl<StorageMinMaxFP32>(this, n, x);
}

void IndexRowwiseMinMax::train_inplace(idx_t n, float* x) {
    train_inplace_impl<StorageMinMaxFP32>(this, n, x);
}

} // namespace faiss
