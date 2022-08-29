#pragma once

namespace faiss {
namespace cppcontrib {
namespace detail {

namespace {

template <
        intptr_t DIM,
        intptr_t CODE_SIZE,
        intptr_t CPOS,
        bool = DIM / CODE_SIZE <= 3>
struct Uint8ReaderImpl {
    static intptr_t get(const uint8_t* const __restrict codes) {
        // Read 1 byte (movzx).
        return codes[CPOS];
    }
};
template <intptr_t DIM, intptr_t CODE_SIZE, intptr_t CPOS>
struct Uint8ReaderImpl<DIM, CODE_SIZE, CPOS, false> {
    static intptr_t get(const uint8_t* const __restrict codes) {
        // Read using 4-bytes.
        // Reading using 8-byte takes too many registers somewhy.
        const uint32_t* __restrict codes32 =
                reinterpret_cast<const uint32_t*>(codes);

        constexpr intptr_t ELEMENT_TO_READ = CPOS / 4;
        constexpr intptr_t SUB_ELEMENT = CPOS % 4;
        const uint32_t code32 = codes32[ELEMENT_TO_READ];

        switch (SUB_ELEMENT) {
            case 0:
                return (code32 & 0x000000FF);
            case 1:
                return (code32 & 0x0000FF00) >> 8;
            case 2:
                return (code32 & 0x00FF0000) >> 16;
            case 3:
                return (code32) >> 24;
        }
    }
};

template <intptr_t DIM, intptr_t CODE_SIZE, intptr_t CPOS>
using Uint8Reader = Uint8ReaderImpl<DIM, CODE_SIZE, CPOS>;

// reduces the number of read operations from RAM
template <
        intptr_t DIM,
        intptr_t CODE_SIZE,
        intptr_t CPOS,
        bool = DIM / CODE_SIZE <= 1>
struct Uint16ReaderImpl {
    static intptr_t get(const uint8_t* const __restrict codes) {
        // Read 2 bytes.
        const uint16_t* const __restrict codesFp16 =
                reinterpret_cast<const uint16_t*>(codes);
        return codesFp16[CPOS];
    }
};
template <intptr_t DIM, intptr_t CODE_SIZE, intptr_t CPOS>
struct Uint16ReaderImpl<DIM, CODE_SIZE, CPOS, false> {
    static intptr_t get(const uint8_t* const __restrict codes) {
        // Read using 4-bytes.
        // Reading using 8-byte takes too many registers somewhy.
        const uint32_t* __restrict codes32 =
                reinterpret_cast<const uint32_t*>(codes);

        constexpr intptr_t ELEMENT_TO_READ = CPOS / 2;
        constexpr intptr_t SUB_ELEMENT = CPOS % 2;
        const uint32_t code32 = codes32[ELEMENT_TO_READ];

        switch (SUB_ELEMENT) {
            case 0:
                return (code32 & 0x0000FFFF);
            case 1:
                return code32 >> 16;
        }
    }
};

template <intptr_t DIM, intptr_t CODE_SIZE, intptr_t CPOS>
using Uint16Reader = Uint16ReaderImpl<DIM, CODE_SIZE, CPOS>;

//
template <intptr_t DIM, intptr_t CODE_SIZE, intptr_t CODE_BITS, intptr_t CPOS>
struct UintReaderImplType {};

template <intptr_t DIM, intptr_t CODE_SIZE, intptr_t CPOS>
struct UintReaderImplType<DIM, CODE_SIZE, 8, CPOS> {
    using reader_type = Uint8Reader<DIM, CODE_SIZE, CPOS>;
};

template <intptr_t DIM, intptr_t CODE_SIZE, intptr_t CPOS>
struct UintReaderImplType<DIM, CODE_SIZE, 16, CPOS> {
    using reader_type = Uint16Reader<DIM, CODE_SIZE, CPOS>;
};

} // namespace

// reduces the number of read operations from RAM
template <intptr_t DIM, intptr_t CODE_SIZE, intptr_t CODE_BITS, intptr_t CPOS>
using UintReader =
        typename UintReaderImplType<DIM, CODE_SIZE, CODE_BITS, CPOS>::
                reader_type;

} // namespace detail
} // namespace cppcontrib
} // namespace faiss
