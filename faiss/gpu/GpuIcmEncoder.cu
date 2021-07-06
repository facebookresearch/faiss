/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/GpuIcmEncoder.h>

#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/impl/IcmEncoder.cuh>

#include <algorithm>

namespace faiss {
namespace gpu {

struct IcmEncoderShards {
    std::vector<std::pair<IcmEncoderImpl*, std::unique_ptr<WorkerThread>>>
            workers;

    void add(IcmEncoderImpl* encoder) {
        workers.emplace_back(std::make_pair(
                encoder, std::unique_ptr<WorkerThread>(new WorkerThread)));
    }

    IcmEncoderImpl* at(int idx) {
        return workers[idx].first;
    }

    void runOnShards(std::function<void(int, IcmEncoderImpl*)> f) {
        std::vector<std::future<bool>> v;

        for (int i = 0; i < this->workers.size(); ++i) {
            auto& p = this->workers[i];
            auto encoder = p.first;
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
        size_t M,
        size_t K,
        const std::vector<GpuResourcesProvider*>& provs,
        const std::vector<int>& devices)
        : lsq::IcmEncoder(M, K) {
    shards = new IcmEncoderShards();
    for (size_t i = 0; i < provs.size(); i++) {
        shards->add(new IcmEncoderImpl(M, K, provs[i], devices[i]));
    }
}

GpuIcmEncoder::~GpuIcmEncoder() {
    delete shards;
}

void GpuIcmEncoder::set_unary_term(size_t n, const float* unaries) {
    size_t nshards = shards->size();
    size_t shard_size = (n + nshards - 1) / nshards;

    auto fn = [=](int idx, IcmEncoderImpl* encoder) {
        size_t ni = std::min(shard_size, n - idx * shard_size);
        auto ui = unaries + idx * shard_size * M * K;
        encoder->setUnaryTerm(ni, ui);
    };
    shards->runOnShards(fn);
}

void GpuIcmEncoder::set_binary_term(const float* binaries) {
    auto fn = [=](int idx, IcmEncoderImpl* encoder) {
        encoder->setBinaryTerm(binaries);
    };
    shards->runOnShards(fn);
}

void GpuIcmEncoder::encode(
        const float* x,
        const float* codebooks,
        int32_t* codes,
        std::mt19937& gen,
        size_t n,
        size_t d,
        size_t nperts,
        size_t ils_iters,
        size_t icm_iters) const {
    size_t nshards = shards->size();
    size_t shard_size = (n + nshards - 1) / nshards;
    auto fn = [=](int idx, IcmEncoderImpl* encoder) {
        size_t i0 = idx * shard_size;
        size_t ni = std::min(shard_size, n - i0);
        auto xi = x + i0 * d;
        auto ci = codes + i0 * M;
        std::mt19937 geni(idx);
        encoder->encode(
                xi, codebooks, ci, geni, ni, d, nperts, ils_iters, icm_iters);
    };
    shards->runOnShards(fn);
}

GpuIcmEncoderFactory::GpuIcmEncoderFactory(int ngpus) {
    for (int i = 0; i < ngpus; i++) {
        provs.push_back(new StandardGpuResources());
        devices.push_back(i);
    }
}

lsq::IcmEncoder* GpuIcmEncoderFactory::get(size_t M, size_t K) {
    return new GpuIcmEncoder(M, K, provs, devices);
}

} // namespace gpu
} // namespace faiss
