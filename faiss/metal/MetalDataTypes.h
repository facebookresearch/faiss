#pragma once

#include <cstdint>

namespace faiss {
namespace metal {

struct DistanceLabel {
    float distance;
    int32_t label;
};

} // namespace metal
} // namespace faiss
