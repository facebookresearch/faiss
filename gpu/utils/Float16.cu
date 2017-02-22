
/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#include "Float16.cuh"
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

#ifdef FAISS_USE_FLOAT16

namespace faiss { namespace gpu {

bool getDeviceSupportsFloat16Math(int device) {
  const auto& prop = getDeviceProperties(device);

  return (prop.major >= 6 ||
          (prop.major == 5 && prop.minor >= 3));
}

struct FloatToHalf {
  __device__ half operator()(float v) const { return __float2half(v); }
};

struct HalfToFloat {
  __device__ float operator()(half v) const { return __half2float(v); }
};

void runConvertToFloat16(half* out,
                         const float* in,
                         size_t num,
                         cudaStream_t stream) {
  thrust::transform(thrust::cuda::par.on(stream),
                    in, in + num, out, FloatToHalf());
}

void runConvertToFloat32(float* out,
                         const half* in,
                         size_t num,
                         cudaStream_t stream) {
  thrust::transform(thrust::cuda::par.on(stream),
                    in, in + num, out, HalfToFloat());
}

// FIXME: replace
/*
  Copyright (c) 2015, Norbert Juffa
  All rights reserved.
  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:
  1. Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
  2. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

half hostFloat2Half(float a) {
  uint32_t ia;
  uint16_t ir;
  memcpy(&ia, &a, sizeof(float));

  ir = (ia >> 16) & 0x8000;
  if ((ia & 0x7f800000) == 0x7f800000) {
    if ((ia & 0x7fffffff) == 0x7f800000) {
      ir |= 0x7c00; /* infinity */
    } else {
      ir = 0x7fff; /* canonical NaN */
    }
  } else if ((ia & 0x7f800000) >= 0x33000000) {
    int shift = (int)((ia >> 23) & 0xff) - 127;
    if (shift > 15) {
      ir |= 0x7c00; /* infinity */
    } else {
      ia = (ia & 0x007fffff) | 0x00800000; /* extract mantissa */
      if (shift < -14) { /* denormal */
        ir |= ia >> (-1 - shift);
        ia = ia << (32 - (-1 - shift));
      } else { /* normal */
        ir |= ia >> (24 - 11);
        ia = ia << (32 - (24 - 11));
        ir = ir + ((14 + shift) << 10);
      }
      /* IEEE-754 round to nearest of even */
      if ((ia > 0x80000000) || ((ia == 0x80000000) && (ir & 1))) {
        ir++;
      }
    }
  }

  half ret;
  memcpy(&ret, &ir, sizeof(half));
  return ret;
}

} } // namespace

#endif // FAISS_USE_FLOAT16
