// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "MetalCloner.h"
#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <cstring>
#import "MetalIndexFlat.h"
#import "StandardMetalResources.h"

namespace faiss {
namespace gpu_metal {

int get_num_gpus() {
    auto res = std::make_shared<MetalResources>();
    return res->isAvailable() ? 1 : 0;
}

faiss::Index* index_cpu_to_metal_gpu(
        StandardMetalResources* res,
        int device,
        const faiss::Index* index) {
    FAISS_THROW_IF_NOT(res != nullptr);
    FAISS_THROW_IF_NOT(res->getResources() != nullptr);
    FAISS_THROW_IF_NOT(res->getResources()->isAvailable());
    FAISS_THROW_IF_NOT_MSG(device == 0, "Metal backend supports only device 0");

    const auto* flat = dynamic_cast<const faiss::IndexFlat*>(index);
    if (!flat) {
        FAISS_THROW_MSG(
                "index_cpu_to_metal_gpu: only IndexFlat (and L2/IP) supported");
    }
    FAISS_THROW_IF_NOT(
            flat->metric_type == METRIC_L2 ||
            flat->metric_type == METRIC_INNER_PRODUCT);

    MetalIndexConfig config;
    config.device = 0;
    auto* metal = new MetalIndexFlat(
            res->getResources(),
            flat->d,
            flat->metric_type,
            flat->metric_arg,
            config);
    if (flat->ntotal > 0) {
        const float* xb = flat->get_xb();
        metal->add(flat->ntotal, xb);
    }
    return metal;
}

faiss::Index* index_metal_gpu_to_cpu(const faiss::Index* index) {
    const auto* metal = dynamic_cast<const MetalIndexFlat*>(index);
    if (!metal) {
        FAISS_THROW_MSG(
                "index_metal_gpu_to_cpu: only MetalIndexFlat supported");
    }
    faiss::IndexFlat* cpu = (metal->metric_type == METRIC_INNER_PRODUCT)
            ? (faiss::IndexFlat*)new faiss::IndexFlatIP(metal->d)
            : (faiss::IndexFlat*)new faiss::IndexFlatL2(metal->d);
    cpu->metric_arg = metal->metric_arg;
    metal->copyTo(cpu);
    return cpu;
}

} // namespace gpu_metal
} // namespace faiss
