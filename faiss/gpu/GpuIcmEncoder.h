/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/LocalSearchQuantizer.h>

namespace faiss {
namespace gpu {

class GpuResourcesProvider;
struct IcmEncoderImpl;
struct IcmEncoderShards;

class GpuIcmEncoder : public lsq::IcmEncoder {
   public:
    GpuIcmEncoder(
            const LocalSearchQuantizer* lsq,
            const std::vector<GpuResourcesProvider*>& provs,
            const std::vector<int>& devices);

    ~GpuIcmEncoder();

    void set_binary_term(float* binaries) override;

    void encode(
            int32_t* codes,
            const float* x,
            std::mt19937& gen,
            size_t n,
            size_t ils_iters) const override;

   private:
    // IcmEncoderImpl* encoder;
    IcmEncoderShards* shards = nullptr;
};

struct GpuIcmEncoderFactory : public lsq::IcmEncoderFactory {
    GpuIcmEncoderFactory(int ngpus = 1);

    lsq::IcmEncoder* get(const LocalSearchQuantizer* lsq) override;

    std::vector<GpuResourcesProvider*> provs;
    std::vector<int> devices;
};

} // namespace gpu
} // namespace faiss
