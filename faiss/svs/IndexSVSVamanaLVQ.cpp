/*
 * Portions Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Portions Copyright 2025 Intel Corporation
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

#include <faiss/svs/IndexSVSVamanaLVQ.h>

namespace faiss {

IndexSVSVamanaLVQ::IndexSVSVamanaLVQ() : IndexSVSVamana() {
    storage_kind = SVSStorageKind::SVS_LVQ4x0;
}

IndexSVSVamanaLVQ::IndexSVSVamanaLVQ(
        idx_t d,
        size_t degree,
        MetricType metric,
        SVSStorageKind storage)
        : IndexSVSVamana(d, degree, metric, storage) {}

} // namespace faiss
