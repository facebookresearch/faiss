/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "partition.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include <immintrin.h>

// Set to 1 to enable debug output
#define DEBUG_PARTITION 0

namespace {

// Scalar Hoare partition - used for small arrays and as reference
size_t scalar_partition(float* vals, int32_t* idxs, size_t n, float pivot) {
    float* left = vals;
    float* right = vals + n - 1;
    int32_t* left_idx = idxs;
    int32_t* right_idx = idxs + n - 1;

    while (true) {
        while (left <= right && *left < pivot) {
            left++;
            left_idx++;
        }
        while (left <= right && *right >= pivot) {
            right--;
            right_idx--;
        }
        if (left >= right) {
            break;
        }
        std::swap(*left, *right);
        std::swap(*left_idx, *right_idx);
        left++;
        left_idx++;
        right--;
        right_idx--;
    }
    return left - vals;
}

constexpr int B = 16; // AVX-512 processes 16 floats at a time

// simple full buffer (after loading)
struct KVBufferB {
    __m512 val;
    __m512i idx;
    static constexpr uint32_t nval = B; // always full
};

struct KVPartialBuffer {
    __m512 val;
    __m512i idx;
    uint32_t nval = 0; // number of values, size <= B
};

struct PartialBuffer2B {
    KVBufferB b0;
    KVBufferB b1;
    uint32_t nval = 0; // number of values, size <= 2 * B

    // Append a buffer (KVBufferB or KVPartialBuffer) to this 2B buffer
    template <typename BufferT>
    inline void append(const BufferT& partial) {
        const uint32_t partial_nval = partial.nval;

        if (nval == 0) {
            b0.val = partial.val;
            b0.idx = partial.idx;
            nval = partial_nval;
        } else if (nval + partial_nval <= B) {
            // Merge into b0: shift existing and add new
            // Use compress store and load to merge
            __mmask16 mask_new = ((1u << partial_nval) - 1) << nval;

            // Expand partial to start at position nval
            b0.val = _mm512_mask_expand_ps(b0.val, mask_new, partial.val);
            b0.idx = _mm512_mask_expand_epi32(b0.idx, mask_new, partial.idx);
            nval += partial_nval;
        } else if (nval <= B) {
            // Need to split between b0 and b1
            uint32_t space_in_b0 = B - nval;

            if (space_in_b0 > 0) {
                // Fill remaining space in b0
                __mmask16 mask_fill = ((1u << space_in_b0) - 1) << nval;
                b0.val = _mm512_mask_expand_ps(b0.val, mask_fill, partial.val);
                b0.idx = _mm512_mask_expand_epi32(
                        b0.idx, mask_fill, partial.idx);
            }

            // Put the rest in b1
            uint32_t remaining = partial_nval - space_in_b0;
            if (remaining > 0) {
                // Compress out the first space_in_b0 elements from partial
                __mmask16 mask_rest = ((1u << remaining) - 1) << space_in_b0;
                b1.val = _mm512_maskz_compress_ps(mask_rest, partial.val);
                b1.idx = _mm512_maskz_compress_epi32(mask_rest, partial.idx);
            }
            nval = B + remaining;
        } else {
            // nval > B, append to b1
            uint32_t pos_in_b1 = nval - B;
            __mmask16 mask_new = ((1u << partial_nval) - 1) << pos_in_b1;
            b1.val = _mm512_mask_expand_ps(b1.val, mask_new, partial.val);
            b1.idx = _mm512_mask_expand_epi32(b1.idx, mask_new, partial.idx);
            nval += partial_nval;
        }
    }

    // Extract the first B elements, shifting the rest
    inline KVBufferB pop_front_B() {
        assert(nval >= B);
        KVBufferB ret = b0;
        b0 = b1;
        nval -= B;
        return ret;
    }
};

size_t partition_avx512(
        float* __restrict__ vals,
        int32_t* __restrict__ idxs,
        size_t n,
        float pivot) {
    if (n < B) {
        return scalar_partition(vals, idxs, n, pivot);
    }

    __m512 pivot_vec = _mm512_set1_ps(pivot);

    PartialBuffer2B buffer_left;
    buffer_left.nval = 0;
    PartialBuffer2B buffer_right;
    buffer_right.nval = 0;

    auto load = [vals, idxs](size_t i) {
        KVBufferB ret;
        ret.val = _mm512_loadu_ps(vals + i);
        ret.idx = _mm512_loadu_epi32(idxs + i);
        return ret;
    };

    auto store = [vals, idxs](size_t i, const KVBufferB& x) {
        _mm512_storeu_ps(vals + i, x.val);
        _mm512_storeu_epi32(idxs + i, x.idx);
    };

    auto partial_load = [vals, idxs](size_t addr, size_t count) {
        KVPartialBuffer ret;
        ret.nval = count;
        if (count == 0) {
            return ret;
        }
        __mmask16 mask = (1u << count) - 1;
        ret.val = _mm512_maskz_loadu_ps(mask, vals + addr);
        ret.idx = _mm512_maskz_loadu_epi32(mask, idxs + addr);
        return ret;
    };

    auto partial_store = [vals, idxs](size_t addr, const KVPartialBuffer& buf) {
        if (buf.nval == 0) {
            return;
        }
        __mmask16 mask = (1u << buf.nval) - 1;
        _mm512_mask_storeu_ps(vals + addr, mask, buf.val);
        _mm512_mask_storeu_epi32(idxs + addr, mask, buf.idx);
    };

    // Partition a block into left/right buffers based on pivot comparison
    auto fill_buffers = [&](const KVBufferB& x) {
        __mmask16 mask_lt = _mm512_cmp_ps_mask(x.val, pivot_vec, _CMP_LT_OS);
        __mmask16 mask_ge = ~mask_lt;

        KVPartialBuffer left_part;
        left_part.val = _mm512_maskz_compress_ps(mask_lt, x.val);
        left_part.idx = _mm512_maskz_compress_epi32(mask_lt, x.idx);
        left_part.nval = _mm_popcnt_u32(mask_lt);

        KVPartialBuffer right_part;
        right_part.val = _mm512_maskz_compress_ps(mask_ge, x.val);
        right_part.idx = _mm512_maskz_compress_epi32(mask_ge, x.idx);
        right_part.nval = _mm_popcnt_u32(mask_ge);

        buffer_left.append(left_part);
        buffer_right.append(right_part);
    };

    // Same for partial buffer
    auto fill_buffers_partial = [&](const KVPartialBuffer& x) {
        if (x.nval == 0) {
            return;
        }

        __mmask16 valid_mask = (1u << x.nval) - 1;
        __mmask16 mask_lt =
                _mm512_cmp_ps_mask(x.val, pivot_vec, _CMP_LT_OS) & valid_mask;
        __mmask16 mask_ge = (~mask_lt) & valid_mask;

        KVPartialBuffer left_part;
        left_part.val = _mm512_maskz_compress_ps(mask_lt, x.val);
        left_part.idx = _mm512_maskz_compress_epi32(mask_lt, x.idx);
        left_part.nval = _mm_popcnt_u32(mask_lt);

        KVPartialBuffer right_part;
        right_part.val = _mm512_maskz_compress_ps(mask_ge, x.val);
        right_part.idx = _mm512_maskz_compress_epi32(mask_ge, x.idx);
        right_part.nval = _mm_popcnt_u32(mask_ge);

        buffer_left.append(left_part);
        buffer_right.append(right_part);
    };

    size_t left_ptr = 0;
    size_t right_ptr = n;
    KVPartialBuffer x_partial;
    x_partial.nval = 0;

    if (n >= 2 * B) {
        // Initialization: load first and last blocks
        KVBufferB x = load(left_ptr);
        fill_buffers(x);

        x = load(right_ptr - B);
        fill_buffers(x);

        // Main loop
        while (left_ptr + 3 * B <= right_ptr) {
            assert(buffer_left.nval <= 2 * B);
            assert(buffer_right.nval <= 2 * B);
            assert(buffer_left.nval + buffer_right.nval == 2 * B);

            if (buffer_left.nval >= B) {
                KVBufferB to_store = buffer_left.pop_front_B();
                store(left_ptr, to_store);
                left_ptr += B;
                x = load(left_ptr);
                fill_buffers(x);
            } else {
                assert(buffer_right.nval >= B);
                KVBufferB to_store = buffer_right.pop_front_B();
                store(right_ptr - B, to_store);
                right_ptr -= B;
                x = load(right_ptr - B);
                fill_buffers(x);
            }
        }

        // Load remaining data (size < B)
        assert(left_ptr + B <= right_ptr - B);
        size_t remaining = right_ptr - left_ptr - 2 * B;
        x_partial = partial_load(left_ptr + B, remaining);

        // Make room by flushing full buffers
        if (buffer_left.nval >= B) {
            KVBufferB to_store = buffer_left.pop_front_B();
            store(left_ptr, to_store);
            left_ptr += B;
        }
        if (buffer_right.nval >= B) {
            KVBufferB to_store = buffer_right.pop_front_B();
            store(right_ptr - B, to_store);
            right_ptr -= B;
        }

    } else if (n >= B) {
        // Load a single full batch and a partial batch
        KVBufferB x = load(0);
        fill_buffers(x);
        x_partial = partial_load(B, n - B);
    } else { // for completeness, was handled by scalar code
        // n < B: load a partial batch
        x_partial = partial_load(0, n);
    }

    // Process remaining partial data
    fill_buffers_partial(x_partial);

    // Write out the left buffer
    if (buffer_left.nval >= B) {
        KVBufferB to_store = buffer_left.pop_front_B();
        store(left_ptr, to_store);
        left_ptr += B;
    }

    // Write partial left buffer
    KVPartialBuffer left_remainder;
    left_remainder.val = buffer_left.b0.val;
    left_remainder.idx = buffer_left.b0.idx;
    left_remainder.nval = buffer_left.nval;
    partial_store(left_ptr, left_remainder);
    left_ptr += left_remainder.nval;

    // Remember partition point
    size_t ret = left_ptr;

    // Write out the right buffer
    if (buffer_right.nval >= B) {
        KVBufferB to_store = buffer_right.pop_front_B();
        store(left_ptr, to_store);
        left_ptr += B;
    }

    // Write partial right buffer
    KVPartialBuffer right_remainder;
    right_remainder.val = buffer_right.b0.val;
    right_remainder.idx = buffer_right.b0.idx;
    right_remainder.nval = buffer_right.nval;
    partial_store(left_ptr, right_remainder);
    left_ptr += right_remainder.nval;

    assert(left_ptr == right_ptr);

    return ret;
}

// Median of three for pivot selection
inline float median3(float a, float b, float c) {
    return std::max(std::min(a, b), std::min(std::max(a, b), c));
}

// Heap-based selection fallback for worst-case protection
void heap_select_largest_k(float* vals, int32_t* idxs, size_t n, size_t k) {
    // Build min-heap of first k elements
    auto cmp = [&vals](size_t a, size_t b) { return vals[a] > vals[b]; };

    std::vector<size_t> heap_idx(k);
    std::iota(heap_idx.begin(), heap_idx.end(), 0);
    std::make_heap(heap_idx.begin(), heap_idx.end(), cmp);

    // Process remaining elements
    for (size_t i = k; i < n; i++) {
        if (vals[i] > vals[heap_idx[0]]) {
            std::pop_heap(heap_idx.begin(), heap_idx.end(), cmp);
            heap_idx[k - 1] = i;
            std::push_heap(heap_idx.begin(), heap_idx.end(), cmp);
        }
    }

    // Rearrange so top-k are at the end
    std::vector<float> tmp_vals(n);
    std::vector<int32_t> tmp_idxs(n);
    std::copy(vals, vals + n, tmp_vals.begin());
    std::copy(idxs, idxs + n, tmp_idxs.begin());

    // Mark which indices are in top-k
    std::vector<bool> in_topk(n, false);
    for (size_t i = 0; i < k; i++) {
        in_topk[heap_idx[i]] = true;
    }

    // Write non-top-k to front, top-k to back
    size_t front = 0, back = n - k;
    for (size_t i = 0; i < n; i++) {
        if (in_topk[i]) {
            vals[back] = tmp_vals[i];
            idxs[back] = tmp_idxs[i];
            back++;
        } else {
            vals[front] = tmp_vals[i];
            idxs[front] = tmp_idxs[i];
            front++;
        }
    }
}

// Introselect: quickselect with depth limit and heap fallback
void introselect_avx512(
        float* vals,
        int32_t* idxs,
        size_t n,
        size_t k,
        int depth_limit) {
    while (n > 1) {
        if (depth_limit == 0) {
            // Fall back to heap selection to guarantee O(n log n)
            heap_select_largest_k(vals, idxs, n, k);
            return;
        }
        depth_limit--;

        // Select pivot using median-of-three
        float pivot = median3(vals[0], vals[n / 2], vals[n - 1]);

        // Partition around pivot
        size_t left_size = partition_avx512(vals, idxs, n, pivot);
        size_t right_size = n - left_size;

        // Determine which partition contains the k-th largest element
        // We want the k largest at positions [n-k, n)
        // So we need (n - k) elements in the "less than" partition

        if (left_size == n || right_size == n) {
            // All elements equal to pivot or bad pivot - use different strategy
            float new_pivot = pivot;
            for (size_t i = 0; i < n; i++) {
                if (vals[i] != pivot) {
                    new_pivot = (vals[i] + pivot) / 2.0f;
                    break;
                }
            }
            if (new_pivot == pivot) {
                // All elements are equal, nothing to do
                return;
            }
            pivot = new_pivot;
            left_size = partition_avx512(vals, idxs, n, pivot);
            right_size = n - left_size;
        }

        size_t target_left = n - k; // How many elements should be on the left

        if (left_size == target_left) {
            // Perfect partition - done!
            return;
        } else if (left_size < target_left) {
            // Need more elements on the left side
            // Recurse on the right partition to find more elements for left
            size_t need_from_right = target_left - left_size;
            introselect_avx512(
                    vals + left_size,
                    idxs + left_size,
                    right_size,
                    right_size - need_from_right,
                    depth_limit);
            return;
        } else {
            // Too many elements on left side, recurse on left partition
            n = left_size;
            k = k - right_size; // Adjust k since we're dropping right_size
                                // elements
        }
    }
}

} // anonymous namespace

void argpartition(size_t n, float* val, int32_t* idx, size_t k) {
    assert(k <= n);

    if (k == 0 || k == n) {
        return;
    }

    // Depth limit for introselect: 2 * log2(n) iterations before heap fallback
    int depth_limit = 2 * (32 - __builtin_clz((uint32_t)n));

    // Run introselect to partition so k largest are at positions [n-k, n)
    introselect_avx512(val, idx, n, k, depth_limit);
}

/*****************************************
 *
 */

KVBufferB flip_with_step(KVBufferB a, int log2_p) {
    KVBufferB ret;
    switch (log2_p) {
        case 0:
            // p=1: swap adjacent elements (0,1), (2,3), etc.
            ret.val = _mm512_shuffle_ps(a.val, a.val, 0b10110001);
            ret.idx = _mm512_shuffle_epi32(a.idx, _MM_PERM_CDAB);
            break;
        case 1:
            // p=2: swap pairs (0,1)<->(2,3), (4,5)<->(6,7), etc.
            ret.val = _mm512_shuffle_ps(a.val, a.val, 0b01001110);
            ret.idx = _mm512_shuffle_epi32(a.idx, _MM_PERM_BADC);
            break;
        case 2:
            // p=4: swap 128-bit lanes within each 256-bit half
            ret.val = _mm512_shuffle_f32x4(a.val, a.val, 0b10110001);
            ret.idx = _mm512_shuffle_i32x4(a.idx, a.idx, 0b10110001);
            break;
        case 3:
            // p=8: swap 256-bit halves
            ret.val = _mm512_shuffle_f32x4(a.val, a.val, 0b01001110);
            ret.idx = _mm512_shuffle_i32x4(a.idx, a.idx, 0b01001110);
            break;
        default:
            assert(false);
    }
    return ret;
}

__mmask16 mask_with_step(int step) {
    switch (step) {
        case 0:
            return 0xAAAA;
        case 1:
            return 0xCCCC;
        case 2:
            return 0xF0F0;
        case 3:
            return 0xFF00;
        case 4:
            return 0x0000;
        default:
            assert(false);
    }
    return 0;
}

KVBufferB bitonic_merge(KVBufferB tab, int step, int stepk) {
    KVBufferB inv_tab = flip_with_step(tab, step);
    __mmask16 mask = _mm512_cmp_ps_mask(tab.val, inv_tab.val, _CMP_GT_OQ);
    __mmask16 eq_mask = _mm512_cmp_ps_mask(tab.val, inv_tab.val, _CMP_EQ_OQ);
    mask ^= mask_with_step(step) ^ mask_with_step(stepk);
    // When values are equal, don't swap - prevents index duplication
    mask &= ~eq_mask;
    KVBufferB res;
    res.val = _mm512_mask_blend_ps(mask, tab.val, inv_tab.val);
    res.idx = _mm512_mask_blend_epi32(mask, tab.idx, inv_tab.idx);
    return res;
}

KVBufferB bitonic_sort(KVBufferB tab) {
    for (int stepk = 1; stepk < 5; stepk++) {
        for (int step = stepk - 1; step >= 0; step--) {
            tab = bitonic_merge(tab, step, stepk);
        }
    }
    return tab;
}

void argsort(size_t n, float* val, int32_t* idx) {
    if (n <= 1) {
        return;
    }

    if (n <= B) {
        // Use bitonic sort for small arrays (<=16 elements)
        // Load data into KVBufferB, padding with max float values
        KVBufferB buf;
        if (n == B) {
            buf.val = _mm512_loadu_ps(val);
            buf.idx = _mm512_loadu_epi32(idx);
        } else {
            // Partial load: fill unused slots with max float to push them to
            // end
            __mmask16 mask = (1u << n) - 1;
            buf.val = _mm512_mask_loadu_ps(
                    _mm512_set1_ps(std::numeric_limits<float>::max()),
                    mask,
                    val);
            buf.idx = _mm512_mask_loadu_epi32(_mm512_set1_epi32(-1), mask, idx);
        }

        buf = bitonic_sort(buf);

        // Store back only the valid elements
        if (n == B) {
            _mm512_storeu_ps(val, buf.val);
            _mm512_storeu_epi32(idx, buf.idx);
        } else {
            __mmask16 mask = (1u << n) - 1;
            _mm512_mask_storeu_ps(val, mask, buf.val);
            _mm512_mask_storeu_epi32(idx, mask, buf.idx);
        }
        return;
    }

    // Use quicksort with partition_avx512 for larger arrays
    // Select pivot using median-of-three
    float pivot = median3(val[0], val[n / 2], val[n - 1]);

    // Partition around pivot
    size_t left_size = partition_avx512(val, idx, n, pivot);

    // Handle degenerate case where all elements are equal to pivot
    if (left_size == 0 || left_size == n) {
        // Find a different pivot
        float new_pivot = pivot;
        for (size_t i = 0; i < n; i++) {
            if (val[i] != pivot) {
                new_pivot = (val[i] + pivot) / 2.0f;
                break;
            }
        }
        if (new_pivot == pivot) {
            // All elements are equal, already sorted
            return;
        }
        left_size = partition_avx512(val, idx, n, new_pivot);
    }

    // Recursively sort both partitions
    argsort(left_size, val, idx);
    argsort(n - left_size, val + left_size, idx + left_size);
}
