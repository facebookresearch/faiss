/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <cstdlib>

#include <faiss/utils/simdlib.h>

namespace faiss {

struct DummyScaler {
    
    inline simd16uint16 scale(int nsq, const simd16uint16 &x) const {
        return x;
    }
};

} // faiss
