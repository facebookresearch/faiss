/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


#pragma once

#include <faiss/gpu/utils/PtxUtils.cuh>
#include <faiss/gpu/utils/WarpShuffles.cuh>

namespace faiss { namespace gpu {

//
// Warp-coalesced parallel reading and writing of packed bits
//

// Read/write native word sizes
template <typename WordT, int Bits>
struct WarpPackedBits {
  static __device__ void write(int laneId, WordT v, bool valid, WordT* out) {
    static_assert(sizeof(WordT) == Bits / 8 &&
                  (Bits % 8) == 0, "");
    // We can just write directly
    if (valid) {
      out[laneId] = v;
    }
  }

  static inline __device__ WordT read(int laneId, WordT* in) {
    return in[laneId];
  }

  static inline __device__ WordT postRead(int laneId, WordT v) {
    return v;
  }
};

// Read/write 6 bit fields, packed across the warp into 24 bytes
template <>
struct WarpPackedBits<uint8_t, 6> {
  static __device__ void write(int laneId, uint8_t v, bool valid, uint8_t* out) {
    // Lower 24 lanes wwrite out packed data
    int laneFrom = (laneId * 8) / 6;

    v = valid ? v : 0;
    v &= 0x3f; // ensure we have only 6 bits

    uint8_t vLower = (uint8_t) shfl((unsigned int) v, laneFrom);
    uint8_t vUpper = (uint8_t) shfl((unsigned int) v, laneFrom + 1);

    // lsb     ...    msb
    // 0: 0 0 0 0 0 0 1 1
    // 1: 1 1 1 1 2 2 2 2
    // 2: 2 2 3 3 3 3 3 3
    int typeLane = laneId % 3;
    uint8_t vOut = 0;
    switch (typeLane) {
      case 0:
        // 6 msbs of lower as vOut lsbs
        // 2 lsbs of upper as vOut msbs
        vOut = vLower | (vUpper << 6);
        break;
      case 1:
        // 4 msbs of lower as vOut lsbs
        // 4 lsbs of upper as vOut msbs
        vOut = (vLower >> 2) | (vUpper << 4);
        break;
      case 2:
        // 2 msbs of lower as vOut lsbs
        // 6 lsbs of upper as vOut msbs
        vOut = (vLower >> 4) | (vUpper << 2);
        break;
    }

    if (laneId < 24) {
      // There could be prior data
      out[laneId] |= vOut;
    }
  }

  static inline __device__ uint8_t read(int laneId, uint8_t* in) {
    uint8_t v = 0;

    if (laneId < 24) {
      v = in[laneId];
    }

    return v;
  }

  static inline __device__ uint8_t postRead(int laneId, uint8_t v) {
    int laneFrom = (laneId * 6) / 8;

    // auto vLower = shfl((unsigned int) v, laneFrom);
    // auto vUpper = shfl((unsigned int) v, laneFrom + 1);
    auto vLower = SHFL_SYNC((unsigned int) v, laneFrom, kWarpSize);
    auto vUpper = SHFL_SYNC((unsigned int) v, laneFrom + 1, kWarpSize);
    auto vConcat = (vUpper << 8) | vLower;

    // Now, this is weird. Each lane reads two uint8, but we wish to use the
    // bfe.u32 instruction to read a 6 bit value from the concatenated uint32.
    // The offset in which we wish to read in the concatenated word is the
    // following:
    //
    // 0: 0, 1: offset 0 size 6
    // 1: 0, 1: offset 6 size 6
    // 2: 1, 2: offset 4 size 6
    // 3: 2, 3: offset 2 size 6
    //
    // In binary, each of the offsets are the following (concatenated together):
    // 0b0010'0100'0110'0000 or 0x2460
    // We can thus use bfe.u32 as a lookup table for the above sequence.
    unsigned int pos;
    GET_BITFIELD_U32(pos, 0x2460, (laneId & 0x3) * 4, 4);

    unsigned int out;
    GET_BITFIELD_U32(out, vConcat, pos, 6);

    return out;
  }
};

// Read/write 4 bit fields, packed across the warp into 16 bytes
template <>
struct WarpPackedBits<uint8_t, 4> {
  static __device__ void write(int laneId, uint8_t v, bool valid, uint8_t* out) {
    // Lower 16 lanes write out packed data
    int laneFrom = laneId * 2;

    v = valid ? v : 0;

    uint8_t vLower = (uint8_t) shfl((unsigned int) v, laneFrom);
    uint8_t vUpper = (uint8_t) shfl((unsigned int) v, laneFrom + 1);

    uint8_t vOut = (vLower & 0xf) | (vUpper << 4);

    if (laneId < 16) {
      // There could be prior data
      out[laneId] |= vOut;
    }
  }

  static inline __device__ uint8_t read(int laneId, uint8_t* in) {
    uint8_t v = 0;

    if (laneId < 16) {
      v = in[laneId];
    }

    return v;
  }

  static inline __device__ uint8_t postRead(int laneId, uint8_t v) {
    int laneFrom = laneId / 2;
    auto v2 = shfl((unsigned int) v, laneFrom);
    return getBitfield(v2, (laneId & 0x1) * 4, 4);
  }
};

} } // namespace
