// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "StandardMetalResources.h"

namespace faiss {
namespace gpu_metal {

StandardMetalResources::StandardMetalResources()
        : res_(std::make_shared<MetalResources>()) {}

} // namespace gpu_metal
} // namespace faiss
