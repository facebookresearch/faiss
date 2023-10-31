/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <faiss/gpu/utils/RaftUtils.h>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/linalg/map.cuh>
#include <raft/matrix/gather.cuh>

#include <thrust/copy.h>
#include <thrust/reduce.h>

namespace faiss {
namespace gpu {

/// Filter out matrix rows containing NaN values
void validRowIndices(
        GpuResources* res,
        Tensor<float, 2, true>& vecs,
        bool* nan_flag) {
    raft::device_resources& raft_handle = res->getRaftHandleCurrentDevice();
    idx_t n_rows = vecs.getSize(0);
    idx_t dim = vecs.getSize(1);

    thrust::fill_n(raft_handle.get_thrust_policy(), nan_flag, n_rows, true);
    raft::linalg::map_offset(
            raft_handle,
            raft::make_device_vector_view<bool, idx_t>(nan_flag, n_rows),
            [vecs = vecs.data(), dim] __device__(idx_t i) {
                for (idx_t col = 0; col < dim; col++) {
                    if (!isfinite(vecs[i * dim + col])) {
                        return false;
                    }
                }
                return true;
            });
}

idx_t inplaceGatherFilteredRows(
        GpuResources* res,
        Tensor<float, 2, true>& vecs,
        Tensor<idx_t, 1, true>& indices) {
    raft::device_resources& raft_handle = res->getRaftHandleCurrentDevice();
    idx_t n_rows = vecs.getSize(0);
    idx_t dim = vecs.getSize(1);

    auto nan_filter =
            raft::make_device_vector<bool, idx_t>(raft_handle, n_rows);

    validRowIndices(res, vecs, nan_filter.data_handle());

    idx_t n_rows_valid = thrust::reduce(
            raft_handle.get_thrust_policy(),
            nan_filter.data_handle(),
            nan_filter.data_handle() + n_rows,
            0);

    if (n_rows_valid < n_rows) {
        auto gather_indices = raft::make_device_vector<idx_t, idx_t>(
                raft_handle, n_rows_valid);

        auto count = thrust::make_counting_iterator(0);

        thrust::copy_if(
                raft_handle.get_thrust_policy(),
                count,
                count + n_rows,
                gather_indices.data_handle(),
                [nan_filter = nan_filter.data_handle()] __device__(auto i) {
                    return nan_filter[i];
                });

        raft::matrix::gather(
                raft_handle,
                raft::make_device_matrix_view<float, idx_t>(
                        vecs.data(), n_rows, dim),
                raft::make_const_mdspan(gather_indices.view()),
                (idx_t)16);

        auto valid_indices = raft::make_device_vector<idx_t, idx_t>(
                raft_handle, n_rows_valid);

        raft::matrix::gather(
                raft_handle,
                raft::make_device_matrix_view<idx_t>(
                        indices.data(), n_rows, (idx_t)1),
                raft::make_const_mdspan(gather_indices.view()));
    }

    return n_rows_valid;
}
} // namespace gpu
} // namespace faiss
