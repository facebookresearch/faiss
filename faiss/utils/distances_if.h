/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>
#include <cstddef>
#include <optional>
#include <tuple>

#include <faiss/impl/DistanceComputer.h>
#include <faiss/utils/distances.h>

namespace faiss {

/*********************************************************
 * Facilities that are used for batch distance computation
 *   for the case of a presence of a condition for the 
 *   acceptable elements.
 *********************************************************/

namespace {

constexpr size_t DEFAULT_BUFFER_SIZE = 8;

// Checks groups of BUFFER_SIZE elements and process acceptable
//   ones in groups of N. Process leftovers elements one by one. 
// This can be rewritten using <ranges> once an appropriate 
//   C++ standard is used.
// Concept constraints may be added once an appropriate 
//   C++ standard is used.
template<
    // A predicate for filtering elements. 
    //   std::optional<bool> Pred(const size_t idx);
    // * return true to accept an element.
    // * return false to reject an element.
    // * return std::nullopt to break the iteration loop.
    typename Pred, 
    // process 1 element. 
    //   void Process1(const size_t idx);
    typename Process1, 
    // process N elements. 
    //   void ProcessN(const std::array<size_t, N> ids);
    typename ProcessN,
    size_t N, 
    size_t BUFFER_SIZE>
void buffered_if(
        const size_t ny,
        Pred pred,
        Process1 process1,
        ProcessN processN) {
    static_assert((BUFFER_SIZE % N) == 0);

    // // the most generic version of the following code that is
    // //   suitable for the debugging is the following:
    //
    // for (size_t j = 0; j < ny; j++) {
    //     const std::optional<bool> outcome = pred(j);
    //     if (!outcome.has_value()) {
    //         break;
    //     }
    //     if (outcome.value()) {
    //         process1(j);
    //     }
    // }

    // todo: maybe add a special case "ny < N" right here

    const size_t ny_buffer_size = (ny / BUFFER_SIZE) * BUFFER_SIZE;
    size_t saved_j[2 * BUFFER_SIZE + N];
    size_t counter = 0;
    
    for (size_t j = 0; j < ny_buffer_size; j += BUFFER_SIZE) {
        for (size_t jj = 0; jj < BUFFER_SIZE; jj++) {
            const std::optional<bool> outcome = pred(j + jj);
            if (!outcome.has_value()) {
                // pred() wants to stop the iteration. 
                // It is a bad code style, but it makes clear 
                //   of what happens next.
                goto leftovers;
            }

            const bool is_acceptable = outcome.value();
            saved_j[counter] = j + jj; counter += is_acceptable ? 1 : 0;
        }

        if (counter >= N) {
            const size_t counter_n = (counter / N) * N;
            for (size_t i_counter = 0; i_counter < counter_n; i_counter += N) {
                std::array<size_t, N> tmp;
                std::copy(saved_j + i_counter, saved_j + i_counter + N, tmp.begin());
                
                processN(tmp);
            }

            // copy leftovers to the beginning of the buffer.
            // todo: use ring buffer instead, maybe?
            // for (size_t jk = counter_n; jk < counter; jk++) {
            //     saved_j[jk - counter_n] = saved_j[jk];
            // }
            for (size_t jk = counter_n; jk < counter_n + N; jk++) {
                saved_j[jk - counter_n] = saved_j[jk];
            }

            // rewind
            counter -= counter_n;
        }
    }

    for (size_t j = ny_buffer_size; j < ny; j++) {
        const std::optional<bool> outcome = pred(j);
        if (!outcome.has_value()) {
            // pred() wants to stop the iteration. 
            break;
        }

        const bool is_acceptable = outcome.value();
        saved_j[counter] = j; counter += is_acceptable ? 1 : 0;
    }

    // process leftovers
leftovers:
    for (size_t jj = 0; jj < counter; jj++) {
        const size_t j = saved_j[jj];
        process1(j);
    }
}

// does nothing
struct NoRemapping {
    inline size_t operator()(const size_t idx) const {
        return idx;
    }
};

// maps idx to indices[idx]
template<typename IdxT>
struct ByIdxRemapping {
    const IdxT* const mapping;
    inline IdxT operator()(const size_t idx) const {
        return mapping[idx];
    }    
};

} // namespace

template<
    // A predicate for filtering elements. 
    //   std::optional<bool> Pred(const size_t idx);
    // * return true to accept an element.
    // * return false to reject an element.
    // * return std::nullopt to break the iteration loop.
    typename Pred, 
    // Compute distance from a query vector to 1 element.
    //   float Distance1(const idx_t idx);
    typename Distance1, 
    // Compute distance from a query vector to N elements
    //   void DistanceN(
    //      const std::array<idx_t, N> idx,
    //      std::array<float, N>& dis);
    typename DistanceN,
    // Maps an iteration for-loop index to a database index.
    // It is needed for calls with indirect indexing like fvec_L2sqr_by_idx().
    //   auto IndexRemapper(const size_t idx);
    typename IndexRemapper,
    // Apply an element.
    //   void Apply(const float dis, const auto idx);
    typename Apply,
    size_t N,
    size_t BUFFER_SIZE>
void fvec_distance_ny_if(
        const size_t ny,
        Pred pred,
        Distance1 distance1,
        DistanceN distanceN,
        IndexRemapper remapper,
        Apply apply
) {
    using idx_type = std::invoke_result_t<IndexRemapper, size_t>;

    // process 1 element
    auto process1 = [&](const size_t idx) {
        const auto remapped_idx = remapper(idx);
        const float distance = distance1(remapped_idx);
        apply(distance, idx);
    };

    // process N elements
    auto processN = [&](const std::array<size_t, N> indices) {
        std::array<float, N> dis;
        std::array<idx_type, N> remapped_indices;
        for (size_t i = 0; i < N; i++) {
            remapped_indices[i] = remapper(indices[i]);
        }

        distanceN(remapped_indices, dis);

        for (size_t i = 0; i < N; i++) {
            apply(dis[i], indices[i]);
        }
    };

    // process
    buffered_if<Pred, decltype(process1), decltype(processN), N, BUFFER_SIZE>(
        ny,
        pred,
        process1,
        processN
    );
}

// an internal implementation
namespace {
// compute ny inner product between x vectors x and a set of contiguous y vectors
//   with filtering and applying filtered elements.
template<
    // A predicate for filtering elements. 
    //   std::optional<bool> Pred(const size_t idx);
    // * return true to accept an element.
    // * return false to reject an element.
    // * return std::nullopt to break the iteration loop.
    typename Pred,
    // Maps an iteration for-loop index to a database index.
    // It is needed for calls with indirect indexing like fvec_L2sqr_by_idx().
    //   auto IndexRemapper(const size_t idx);
    typename IndexRemapper,
    // Apply an element.
    //   void Apply(const float dis, const auto idx);
    typename Apply>
void internal_fvec_inner_products_ny_if(
        const float* __restrict x,
        const float* __restrict y,
        size_t d,
        const size_t ny,
        Pred pred,
        IndexRemapper remapper,
        Apply apply) {
    using idx_type = std::invoke_result_t<IndexRemapper, size_t>;

    // compute a distance from the query to 1 element
    auto distance1 = [x, y, d](const idx_type idx) { 
        return fvec_inner_product(x, y + idx * d, d); 
    };

    // compute distances from the query to 4 elements
    auto distance4 = [x, y, d](const std::array<idx_type, 4> indices, std::array<float, 4>& dis) { 
        fvec_inner_product_batch_4(
            x,
            y + indices[0] * d,
            y + indices[1] * d,
            y + indices[2] * d,
            y + indices[3] * d,
            d,
            dis[0],
            dis[1],
            dis[2],
            dis[3]
        );
    };

    fvec_distance_ny_if<Pred, decltype(distance1), decltype(distance4), IndexRemapper, Apply, 4, DEFAULT_BUFFER_SIZE>(
        ny,
        pred,
        distance1,
        distance4,
        remapper,
        apply
    );
}

// compute ny square L2 distance between x vectors x and a set of contiguous y vectors
//   with filtering and applying filtered elements.
template<
    // A predicate for filtering elements. 
    //   std::optional<bool> Pred(const size_t idx);
    // * return true to accept an element.
    // * return false to reject an element.
    // * return std::nullopt to break the iteration loop.
    typename Pred, 
    // Maps an iteration for-loop index to a database index.
    // It is needed for calls with indirect indexing like fvec_L2sqr_by_idx().
    //   auto IndexRemapper(const size_t idx);
    typename IndexRemapper,
    // Apply an element.
    //   void Apply(const float dis, const auto idx);
    typename Apply>
void internal_fvec_L2sqr_ny_if(
        const float* __restrict x,
        const float* __restrict y,
        size_t d,
        const size_t ny,
        Pred pred,
        IndexRemapper remapper,
        Apply apply) {    
    using idx_type = std::invoke_result_t<IndexRemapper, size_t>;

    // compute a distance from the query to 1 element
    auto distance1 = [x, y, d](const idx_type idx) { 
        return fvec_L2sqr(x, y + idx * d, d); 
    };

    // compute distances from the query to 4 elements
    auto distance4 = [x, y, d](const std::array<idx_type, 4> indices, std::array<float, 4>& dis) { 
        fvec_L2sqr_batch_4(
            x,
            y + indices[0] * d,
            y + indices[1] * d,
            y + indices[2] * d,
            y + indices[3] * d,
            d,
            dis[0],
            dis[1],
            dis[2],
            dis[3]
        );
    };

    fvec_distance_ny_if<Pred, decltype(distance1), decltype(distance4), IndexRemapper, Apply, 4, DEFAULT_BUFFER_SIZE>(
        ny,
        pred,
        distance1,
        distance4,
        remapper,
        apply
    );
}


// compute ny distance between x vectors x and a set of contiguous y vectors
//   with filtering and applying filtered elements.
template<
    // A predicate for filtering elements. 
    //   std::optional<bool> Pred(const size_t idx);
    // * return true to accept an element.
    // * return false to reject an element.
    // * return std::nullopt to break the iteration loop.
    typename Pred, 
    // Maps an iteration for-loop index to a database index.
    // It is needed for calls with indirect indexing like fvec_L2sqr_by_idx().
    //   auto IndexRemapper(const size_t idx);
    typename IndexRemapper,
    // Apply an element.
    //   void Apply(const float dis, const idx_t idx);
    typename Apply>
void internal_distance_compute_if(
        const size_t ny,
        DistanceComputer* __restrict dc,
        Pred pred,
        IndexRemapper remapper,
        Apply apply) {
    //using idx_type = typename IndexRemapper::idx_type;
    using idx_type = std::invoke_result_t<IndexRemapper, size_t>;

    // compute a distance from the query to 1 element
    auto distance1 = [dc](const idx_type idx) { 
        return dc->operator()(idx);
    };

    // compute distances from the query to 4 elements
    auto distance4 = [dc](const std::array<idx_type, 4> indices, std::array<float, 4>& dis) { 
        dc->distances_batch_4(
            indices[0],
            indices[1],
            indices[2],
            indices[3],
            dis[0],
            dis[1],
            dis[2],
            dis[3]
        );
    };

    fvec_distance_ny_if<Pred, decltype(distance1), decltype(distance4), IndexRemapper, Apply, 4, DEFAULT_BUFFER_SIZE>(
        ny,
        pred,
        distance1,
        distance4,
        remapper,
        apply
    );
}

}

// compute ny inner product between x vectors x and a set of contiguous y vectors
//   with filtering and applying filtered elements.
template<
    // A predicate for filtering elements. 
    //   std::optional<bool> Pred(const size_t idx);
    // * return true to accept an element.
    // * return false to reject an element.
    // * return std::nullopt to break the iteration loop.
    typename Pred, 
    // Apply an element.
    //   void Apply(const float dis, const size_t idx);
    typename Apply>
void fvec_inner_products_ny_if(
        const float* __restrict x,
        const float* __restrict y,
        size_t d,
        const size_t ny,
        Pred pred,
        Apply apply) {
    internal_fvec_inner_products_ny_if(x, y, d, ny, pred, NoRemapping(), apply);
}

// compute ny square L2 distance between x vectors x and a set of contiguous y vectors
//   with filtering and applying filtered elements.
template<
    // A predicate for filtering elements. 
    //   std::optional<bool> Pred(const size_t idx);
    // * return true to accept an element.
    // * return false to reject an element.
    // * return std::nullopt to break the iteration loop.
    typename Pred, 
    // Apply an element.
    //   void Apply(const float dis, const size_t idx);
    typename Apply>
void fvec_L2sqr_ny_if(
        const float* __restrict x,
        const float* __restrict y,
        size_t d,
        const size_t ny,
        Pred pred,
        Apply apply) {    
    internal_fvec_L2sqr_ny_if(x, y, d, ny, pred, NoRemapping(), apply);
}

// compute ny inner product between x vectors x and a set of contiguous y vectors
//   whose indices are given by idy with filtering and applying filtered elements.
template<
    // A predicate for filtering elements. 
    //   std::optional<bool> Pred(const size_t idx);
    // * return true to accept an element.
    // * return false to reject an element.
    // * return std::nullopt to break the iteration loop.
    typename Pred, 
    // Apply an element.
    //   void Apply(const float dis, const int64_t idx);
    typename Apply>
void fvec_inner_products_ny_by_idx_if(
        const float* __restrict x,
        const float* __restrict y,
        const int64_t* __restrict ids, /* ids of y vecs */
        size_t d,
        const size_t ny,
        Pred pred,
        Apply apply) {
    ByIdxRemapping<int64_t> remapper{ids};
    internal_fvec_inner_products_ny_if(x, y, d, ny, pred, remapper, apply);
}

// compute ny square L2 distance between x vectors x and a set of contiguous y vectors
//   whose indices are given by idy with filtering and applying filtered elements.
template<
    // A predicate for filtering elements. 
    //   std::optional<bool> Pred(const size_t idx);
    // * return true to accept an element.
    // * return false to reject an element.
    // * return std::nullopt to break the iteration loop.
    typename Pred, 
    // Apply an element.
    //   void Apply(const float dis, const int64_t idx);
    typename Apply>
void fvec_L2sqr_ny_by_idx_if(
        const float* __restrict x,
        const float* __restrict y,
        const int64_t* __restrict ids, /* ids of y vecs */
        size_t d,
        const size_t ny,
        Pred pred,
        Apply apply) {    
    ByIdxRemapping<int64_t> remapper{ids};
    internal_fvec_L2sqr_ny_if(x, y, d, ny, pred, remapper, apply);
}

// compute ny distance between x vectors x and a set of contiguous y vectors
//   with filtering and applying filtered elements.
template<
    // A predicate for filtering elements. 
    //   std::optional<bool> Pred(const size_t idx);
    // * return true to accept an element.
    // * return false to reject an element.
    // * return std::nullopt to break the iteration loop.
    typename Pred, 
    // Apply an element.
    //   void Apply(const float dis, const idx_t idx);
    typename Apply>
void internal_distance_compute_if(
        const idx_t* __restrict query_indices,
        const size_t ny,
        DistanceComputer* __restrict dc,
        Pred pred,
        Apply apply) {
    // compute a distance from the query to 1 element
    auto distance1 = [dc](const idx_t idx) { 
        return dc->operator()(idx);
    };

    // compute distances from the query to 4 elements
    auto distance4 = [dc](const std::array<idx_t, 4> indices, std::array<float, 4>& dis) { 
        dc->distances_batch_4(
            indices[0],
            indices[1],
            indices[2],
            indices[3],
            dis[0],
            dis[1],
            dis[2],
            dis[3]
        );
    };

    ByIdxRemapping<idx_t> remapper{query_indices};
    fvec_distance_ny_if<Pred, decltype(distance1), decltype(distance4), Apply, 4, DEFAULT_BUFFER_SIZE>(
        ny,
        pred,
        distance1,
        distance4,
        remapper,
        apply
    );
}

// compute ny distance between x vectors x and a set of contiguous y vectors
//   with filtering and applying filtered elements.
template<
    // A predicate for filtering elements. 
    //   std::optional<bool> Pred(const size_t idx);
    // * return true to accept an element.
    // * return false to reject an element.
    // * return std::nullopt to break the iteration loop.
    typename Pred, 
    // Apply an element.
    //   void Apply(const float dis, const idx_t idx);
    typename Apply>
void distance_compute_if(
        const size_t ny,
        DistanceComputer* const __restrict dc,
        Pred pred,
        Apply apply) {
    NoRemapping remapper;
    internal_distance_compute_if(ny, dc, pred, remapper, apply);
}

// compute ny distance between x vectors x and a set of contiguous y vectors
//   whose indices are given by query_indices
//   with filtering and applying filtered elements.
template<
    // A predicate for filtering elements. 
    //   std::optional<bool> Pred(const size_t idx);
    // * return true to accept an element.
    // * return false to reject an element.
    // * return std::nullopt to break the iteration loop.
    typename Pred, 
    // Apply an element.
    //   void Apply(const float dis, const idx_t idx);
    typename Apply>
void distance_compute_by_idx_if(
        const idx_t* const __restrict query_indices,
        const size_t ny,
        DistanceComputer* const __restrict dc,
        Pred pred,
        Apply apply) {
    ByIdxRemapping<idx_t> remapper{query_indices};
    internal_distance_compute_if(ny, dc, pred, remapper, apply);
}

} //namespace faiss

