/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


#pragma once

#include <cstdint>
#include <cstdlib>
#include <cassert>
#include <cstring>



namespace faiss {

template<int A=32>
inline bool is_aligned_pointer(const void* x)
{
    size_t xi = (size_t)x;
    return xi % A == 0;
}

// class that manages suitably aligned arrays for SIMD
// T should be a POV type. The default alignment is 32 for AVX
template<class T, int A=32>
struct AlignedTable {
    T * ptr;
    size_t numel;

    AlignedTable(): ptr(nullptr), numel(0)
    { }

    AlignedTable(size_t n): ptr(nullptr), numel(0)
    { resize(n); }

    size_t itemsize() const {return sizeof(T); }

    void resize(size_t n) {
        if (numel == n) {
            return;
        }
        T * new_ptr;
        if (n > 0) {
            int ret = posix_memalign((void**)&new_ptr, A, n * sizeof(T));
            if (ret != 0) {
                throw std::bad_alloc();
            }
            memcpy(new_ptr, ptr, sizeof(T) * std::min(numel, n));
        } else {
            new_ptr = nullptr;
        }
        numel = n;
        free(ptr);
        ptr = new_ptr;
    }

    void clear() {memset(ptr, 0, nbytes()); }
    size_t size() const {return numel; }
    size_t nbytes() const {return numel * sizeof(T); }
    T * get() {return ptr; }
    const T * get() const {return ptr; }
    T operator [] (size_t i) const {return ptr[i]; }
    T & operator [] (size_t i)  {return ptr[i]; }
    ~AlignedTable() {free(ptr); }

    AlignedTable<T, A> & operator = (const AlignedTable<T, A> & other) {
        resize(other.numel);
        memcpy(ptr, other.ptr, sizeof(T) * numel);
        return *this;
    }

    AlignedTable(const AlignedTable<T, A> & other) {
        *this = other;
    }

};


} // namespace faiss