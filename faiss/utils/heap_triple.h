/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stddef.h>
#include <cstdint>
#include <utility>

#include <faiss/utils/ordered_key_value.h>

namespace faiss {

/** Helper for managing a heap with three parallel arrays:
 * - values (distances)
 * - indices
 * - pointers to codes
 *
 * This is used for search_and_reconstruct() in IndexBinaryHash to avoid a
 * second pass to retrieve the codes.
 */
template <typename C>
struct HeapTriple {
    using T = typename C::T;   // value type (distances)
    using TI = typename C::TI; // index type

    size_t k;             // heap size
    T* vals;              // values array
    TI* ids;              // indices array
    const uint8_t** ptrs; // pointers array

    static constexpr TI kInvalidId = -1;

    HeapTriple(
            size_t capacity,
            T* distances,
            TI* indices,
            const uint8_t** code_ptrs)
            : k(capacity), vals(distances), ids(indices), ptrs(code_ptrs) {}

    /** initialize heap from optional seed arrays */
    void heapify(
            const T* seed_vals = nullptr,
            const TI* seed_ids = nullptr,
            const uint8_t* const* seed_ptrs = nullptr,
            size_t k0 = 0) {
        // Initialize with seed values
        for (size_t i = 0; i < k0; i++) {
            vals[i] = seed_vals ? seed_vals[i] : C::neutral();
            ids[i] = seed_ids ? seed_ids[i] : (TI)i;
            ptrs[i] = seed_ptrs ? seed_ptrs[i] : nullptr;
        }

        // Fill remaining slots with sentinel values
        for (size_t i = k0; i < k; i++) {
            vals[i] = C::neutral();
            ids[i] = kInvalidId;
            ptrs[i] = nullptr;
        }

        // Heapify the initial elements
        if (k0 > 0) {
            for (int i = (k0 - 2) / 2; i >= 0; i--) {
                size_t pos = i;
                while (true) {
                    size_t left = (pos << 1) + 1;
                    size_t right = left + 1;
                    size_t largest = pos;

                    if (left < k0 &&
                        C::cmp2(vals[left],
                                vals[largest],
                                ids[left],
                                ids[largest])) {
                        largest = left;
                    }
                    if (right < k0 &&
                        C::cmp2(vals[right],
                                vals[largest],
                                ids[right],
                                ids[largest])) {
                        largest = right;
                    }
                    if (largest == pos)
                        break;

                    std::swap(vals[pos], vals[largest]);
                    std::swap(ids[pos], ids[largest]);
                    std::swap(ptrs[pos], ptrs[largest]);
                    pos = largest;
                }
            }
        }
    }

    /** push element on the heap if it is better than the top element */
    void push(T val, TI id, const uint8_t* ptr) {
        // For CMax heap keeping smallest values: replace if val < vals[0]
        // C::cmp(vals[0], val) returns vals[0] > val for CMax
        if (C::cmp(vals[0], val)) {
            // Replace top element and heapify down
            vals[0] = val;
            ids[0] = id;
            ptrs[0] = ptr;
            heapify_down(0);
        }
    }

    /** add element to a heap of size heap_size */
    void push_to_heap(size_t heap_size, T val, TI id, const uint8_t* ptr) {
        size_t i = heap_size - 1;
        while (i > 0) {
            size_t parent = (i - 1) >> 1;
            if (!C::cmp2(val, vals[parent], id, ids[parent])) {
                break;
            }
            vals[i] = vals[parent];
            ids[i] = ids[parent];
            ptrs[i] = ptrs[parent];
            i = parent;
        }
        vals[i] = val;
        ids[i] = id;
        ptrs[i] = ptr;
    }

    /** heapify down from position i */
    void heapify_down(size_t i) {
        while (true) {
            size_t left = (i << 1) + 1;
            size_t right = left + 1;
            size_t largest = i;

            if (left < k &&
                C::cmp2(vals[left], vals[largest], ids[left], ids[largest])) {
                largest = left;
            }
            if (right < k &&
                C::cmp2(vals[right], vals[largest], ids[right], ids[largest])) {
                largest = right;
            }
            if (largest == i)
                break;

            std::swap(vals[i], vals[largest]);
            std::swap(ids[i], ids[largest]);
            std::swap(ptrs[i], ptrs[largest]);
            i = largest;
        }
    }

    /** remove top element */
    void pop() {
        if (k <= 1)
            return;

        vals[0] = vals[k - 1];
        ids[0] = ids[k - 1];
        ptrs[0] = ptrs[k - 1];

        vals[k - 1] = C::neutral();
        ids[k - 1] = kInvalidId;
        ptrs[k - 1] = nullptr;

        heapify_down(0);
    }

    /** sort the heap, returns number of valid elements */
    size_t reorder() {
        // Standard heap sort
        for (size_t end = k; end > 1; --end) {
            std::swap(vals[0], vals[end - 1]);
            std::swap(ids[0], ids[end - 1]);
            std::swap(ptrs[0], ptrs[end - 1]);

            // Heapify the reduced heap
            size_t i = 0;
            size_t heap_size = end - 1;
            while (true) {
                size_t left = (i << 1) + 1;
                size_t right = left + 1;
                size_t largest = i;

                if (left < heap_size &&
                    C::cmp2(vals[left],
                            vals[largest],
                            ids[left],
                            ids[largest])) {
                    largest = left;
                }
                if (right < heap_size &&
                    C::cmp2(vals[right],
                            vals[largest],
                            ids[right],
                            ids[largest])) {
                    largest = right;
                }
                if (largest == i)
                    break;

                std::swap(vals[i], vals[largest]);
                std::swap(ids[i], ids[largest]);
                std::swap(ptrs[i], ptrs[largest]);
                i = largest;
            }
        }

        // Count valid elements
        size_t nvalid = 0;
        while (nvalid < k && ids[nvalid] != kInvalidId) {
            ++nvalid;
        }

        // Clear sentinel values
        for (size_t i = nvalid; i < k; i++) {
            vals[i] = C::neutral();
            ids[i] = kInvalidId;
            ptrs[i] = nullptr;
        }

        return nvalid;
    }
};

} // namespace faiss
