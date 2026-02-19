/*
 * Portions Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Portions Copyright 2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <faiss/svs/IndexSVSIVFLVQ.h>

namespace faiss {

IndexSVSIVFLVQ::IndexSVSIVFLVQ() : IndexSVSIVF() {
    storage_kind = SVSStorageKind::SVS_LVQ4x0;
}

IndexSVSIVFLVQ::IndexSVSIVFLVQ(
        idx_t d,
        size_t nlist,
        MetricType metric,
        SVSStorageKind storage)
        : IndexSVSIVF(d, nlist, metric, storage) {}

} // namespace faiss
