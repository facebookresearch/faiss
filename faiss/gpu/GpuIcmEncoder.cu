/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/GpuIcmEncoder.h>

#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/utils/WorkerThread.h>
#include <faiss/gpu/impl/IcmEncoder.cuh>

#include <algorithm>

namespace faiss {
namespace gpu {

///< A helper structure to support multi-GPU
struct IcmEncoderShards {
    std::vector<std::pair<
            std::unique_ptr<IcmEncoderImpl>,
            std::unique_ptr<WorkerThread>>>
            workers;

    void add(IcmEncoderImpl* encoder) {
        workers.emplace_back(std::make_pair(
                std::unique_ptr<IcmEncoderImpl>(encoder),
                std::unique_ptr<WorkerThread>(new WorkerThread)));
    }

    IcmEncoderImpl* at(int idx) {
        return workers[idx].first.get();
    }

    ///< call f(idx, encoder) for each encoder
    void runOnShards(std::function<void(int, IcmEncoderImpl*)> f) {
        std::vector<std::future<bool>> v;

        for (int i = 0; i < this->workers.size(); ++i) {
            auto& p = this->workers[i];
            auto encoder = p.first.get();
            v.emplace_back(p.second->add([f, i, encoder]() { f(i, encoder); }));
        }

        for (int i = 0; i < v.size(); ++i) {
            auto& fut = v[i];
            fut.get(); // no exception handle, crash if any thread down
        }
    }

    size_t size() {
        return workers.size();
    }
};

GpuIcmEncoder::GpuIcmEncoder(
        const LocalSearchQuantizer* lsq,
        const std::vector<GpuResourcesProvider*>& provs,
        const std::vector<int>& devices)
        : lsq::IcmEncoder(lsq), shards(new IcmEncoderShards()) {
    // create an IcmEncoderImpl instance for each device.
    for (size_t i = 0; i < provs.size(); i++) {
        shards->add(new IcmEncoderImpl(
                lsq->M, lsq->K, lsq->d, provs[i], devices[i]));
    }
}

GpuIcmEncoder::~GpuIcmEncoder() {}

void GpuIcmEncoder::set_binary_term() {
    auto fn = [=](int idx, IcmEncoderImpl* encoder) {
        encoder->setBinaryTerm(lsq->codebooks.data());
    };
    shards->runOnShards(fn);
}

void GpuIcmEncoder::encode(
        int32_t* codes,
        const float* x,
        std::mt19937& gen,
        size_t n,
        size_t ils_iters) const {
    size_t nshards = shards->size();
    size_t shard_size = (n + nshards - 1) / nshards;

    auto codebooks = lsq->codebooks.data();
    auto M = lsq->M;
    auto d = lsq->d;
    auto nperts = lsq->nperts;
    auto icm_iters = lsq->icm_iters;

    auto seed = gen();

    // split input data
    auto fn = [=](int idx, IcmEncoderImpl* encoder) {
        size_t i0 = idx * shard_size;
        size_t ni = std::min(shard_size, n - i0);
        auto xi = x + i0 * d;
        auto ci = codes + i0 * M;
        std::mt19937 geni(idx + seed); // different seed for each shard
        encoder->encode(
                ci, xi, codebooks, geni, ni, nperts, ils_iters, icm_iters);
    };
    shards->runOnShards(fn);
}

GpuIcmEncoderFactory::GpuIcmEncoderFactory(int ngpus) {
    for (int i = 0; i < ngpus; i++) {
        provs.push_back(new StandardGpuResources());
        devices.push_back(i);
    }
}

lsq::IcmEncoder* GpuIcmEncoderFactory::get(const LocalSearchQuantizer* lsq) {
    return new GpuIcmEncoder(lsq, provs, devices);
}

} // namespace gpu
} // namespace faiss
