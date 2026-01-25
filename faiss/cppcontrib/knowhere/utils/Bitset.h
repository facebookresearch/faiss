// Copyright (C) 2019-2024 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>

namespace faiss {
namespace cppcontrib {
namespace knowhere {

struct Bitset final {
    struct Proxy {
        uint8_t& element;
        uint8_t mask;

        inline Proxy(uint8_t& _element, const size_t _shift)
                : element{_element}, mask(uint8_t(1) << _shift) {}

        inline operator bool() const {
            return ((element & mask) != 0);
        }

        inline Proxy& operator=(const bool value) {
            if (value) {
                set();
            } else {
                reset();
            }
            return *this;
        }

        inline void set() {
            element |= mask;
        }

        inline void reset() {
            element &= ~mask;
        }
    };

    inline Bitset() {}

    // create an uncleared bitset
    inline static Bitset create_uninitialized(const size_t initial_size) {
        Bitset bitset;

        const size_t nbytes = (initial_size + 7) / 8;

        bitset.bits = std::make_unique<uint8_t[]>(nbytes);
        bitset.size = initial_size;

        return bitset;
    }

    // create an initialized bitset
    inline static Bitset create_cleared(const size_t initial_size) {
        Bitset bitset = create_uninitialized(initial_size);
        bitset.clear();

        return bitset;
    }

    Bitset(const Bitset&) = delete;
    Bitset(Bitset&&) = default;
    Bitset& operator=(const Bitset&) = delete;
    Bitset& operator=(Bitset&&) = default;

    inline bool get(const size_t index) const {
        return (bits[index >> 3] & (0x1 << (index & 0x7)));
    }

    inline void set(const size_t index) {
        bits[index >> 3] |= uint8_t(0x1 << (index & 0x7));
    }

    inline void reset(const size_t index) {
        bits[index >> 3] &= (~uint8_t(0x1 << (index & 0x7)));
    }

    inline const uint8_t* get_ptr(const size_t index) const {
        return bits.get() + index / 8;
    }

    inline uint8_t* get_ptr(const size_t index) {
        return bits.get() + index / 8;
    }

    inline void clear() {
        const size_t nbytes = (size + 7) / 8;
        std::memset(bits.get(), 0, nbytes);
    }

    inline Proxy operator[](const size_t bit_idx) {
        uint8_t& element = bits[bit_idx / 8];
        const size_t shift = bit_idx & 7;
        return Proxy{element, shift};
    }

    inline bool operator[](const size_t bit_idx) const {
        return get(bit_idx);
    }

    std::unique_ptr<uint8_t[]> bits;
    size_t size = 0;
};

} // namespace knowhere
} // namespace cppcontrib
} // namespace faiss
