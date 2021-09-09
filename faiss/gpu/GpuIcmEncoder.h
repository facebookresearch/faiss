/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/LocalSearchQuantizer.h>

#include <memory>

namespace faiss {
namespace gpu {

class GpuResourcesProvider;
struct IcmEncoderShards;

/** Perform LSQ encoding on GPU.
 *
 * Split input vectors to different devices and call IcmEncoderImpl::encode
 * to encode them
 */
class GpuIcmEncoder : public lsq::IcmEncoder {
   public:
    GpuIcmEncoder(
            const LocalSearchQuantizer* lsq,
            const std::vector<GpuResourcesProvider*>& provs,
            const std::vector<int>& devices);

    ~GpuIcmEncoder();

    GpuIcmEncoder(const GpuIcmEncoder&) = delete;
    GpuIcmEncoder& operator=(const GpuIcmEncoder&) = delete;

    void set_binary_term() override;

    void encode(
            int32_t* codes,
            const float* x,
            std::mt19937& gen,
            size_t n,
            size_t ils_iters) const override;

   private:
    std::unique_ptr<IcmEncoderShards> shards;
};

struct GpuIcmEncoderFactory : public lsq::IcmEncoderFactory {
    explicit GpuIcmEncoderFactory(int ngpus = 1);

    lsq::IcmEncoder* get(const LocalSearchQuantizer* lsq) override;

    std::vector<GpuResourcesProvider*> provs;
    std::vector<int> devices;
};

} // namespace gpu
} // namespace faiss
