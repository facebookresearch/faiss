// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <faiss/IndexHNSW.h>
#include <faiss/MetricType.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/test/TestUtils.h>
#include <faiss/gpu/utils/CuvsFilterConvert.h>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <optional>
#include <vector>

#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/bitset.cuh>
#include <raft/core/copy.cuh>

struct Options {
    Options() {
        bitset_len = faiss::gpu::randVal(100, 100000);
    }

    std::string toString() const {
        std::stringstream str;
        str << "bitset_len " << bitset_len;

        return str.str();
    }

    size_t bitset_len;
};

void run_complex() {
    Options spec;
    faiss::gpu::StandardGpuResources res;
    res.noTempMemory();
    auto gpuRes = res.getResources();
    const raft::device_resources& raft_handle =
            gpuRes->getRaftHandleCurrentDevice();
    // generate random selectors
    auto imin = faiss::gpu::randVal(0, spec.bitset_len - 2);
    auto imax = faiss::gpu::randVal(1, spec.bitset_len - 1);
    if (imin > imax)
        std::swap(imin, imax);
    auto range_selector = faiss::IDSelectorRange(imin, imax);

    std::vector<faiss::idx_t> array_selector_indices(
            faiss::gpu::randVal(10, 50));
    for (int i = 0; i < array_selector_indices.size(); i++) {
        array_selector_indices[i] = faiss::gpu::randVal(0, spec.bitset_len - 1);
    }
    auto array_selector = faiss::IDSelectorArray(
            array_selector_indices.size(), array_selector_indices.data());

    auto or_selector = faiss::IDSelectorOr(&range_selector, &array_selector);

    auto bitmap_faiss_cpu = std::vector<uint8_t>((spec.bitset_len + 8) / 8);
    for (uint32_t i = 0; i < bitmap_faiss_cpu.size(); i++) {
        bitmap_faiss_cpu[i] = (uint8_t)faiss::gpu::randVal(0, 255);
    }
    auto bitmap_selector =
            faiss::IDSelectorBitmap(spec.bitset_len, bitmap_faiss_cpu.data());
    auto not_bitmap_selector = faiss::IDSelectorNot(&bitmap_selector);

    auto xor_selector =
            faiss::IDSelectorXOr(&or_selector, &not_bitmap_selector);

    // convert to cuVS bitset
    auto bitset = cuvs::core::bitset<uint32_t, uint32_t>(
            raft_handle, spec.bitset_len, false);
    faiss::gpu::convert_to_bitset(gpuRes.get(), xor_selector, bitset.view());

    // verify
    auto bitset_converted_cpu =
            raft::make_host_vector<uint32_t, uint32_t>(bitset.n_elements());
    auto bitset_converted_cpu_view =
            cuvs::core::bitset_view<uint32_t, uint32_t>(
                    bitset_converted_cpu.data_handle(), spec.bitset_len);
    raft::copy(raft_handle, bitset_converted_cpu.view(), bitset.to_mdspan());
    raft::resource::sync_stream(raft_handle);
    for (uint32_t i = 0; i < spec.bitset_len; i++) {
        if (bitset_converted_cpu_view.test(i) != xor_selector.is_member(i)) {
            ASSERT_TRUE(
                    testing::AssertionFailure()
                    << "actual=" << bitset_converted_cpu_view.test(i)
                    << " != expected=" << xor_selector.is_member(i) << " @" << i
                    << " bitset_len: " << spec.bitset_len);
        }
    }
}

void run_range() {
    Options spec;
    faiss::gpu::StandardGpuResources res;
    res.noTempMemory();
    auto gpuRes = res.getResources();
    const raft::device_resources& raft_handle =
            gpuRes->getRaftHandleCurrentDevice();
    // take random imin and imax, check all ids
    using bitset_t = uint32_t;
    auto imin = faiss::gpu::randVal(0, spec.bitset_len - 2);
    auto imax = faiss::gpu::randVal(1, spec.bitset_len - 1);
    if (imin > imax)
        std::swap(imin, imax);
    auto selector = faiss::IDSelectorRange(imin, imax);
    auto bitset = cuvs::core::bitset<bitset_t, uint32_t>(
            raft_handle, spec.bitset_len, false);
    auto nbits = sizeof(bitset_t) * 8;

    faiss::gpu::convert_to_bitset(gpuRes.get(), selector, bitset.view());
    auto bitset_converted_cpu =
            raft::make_host_vector<bitset_t, uint32_t>(bitset.n_elements());
    raft::copy(raft_handle, bitset_converted_cpu.view(), bitset.to_mdspan());
    raft::resource::sync_stream(raft_handle);
    auto bitset_view_cpu = cuvs::core::bitset_view<bitset_t, uint32_t>(
            bitset_converted_cpu.data_handle(), spec.bitset_len);
    for (uint64_t i = 0; i < spec.bitset_len; i++) {
        if (bitset_view_cpu.test(i) != selector.is_member(i)) {
            ASSERT_TRUE(
                    testing::AssertionFailure()
                    << "actual=" << bitset_view_cpu.test(i)
                    << " != expected=" << selector.is_member(i) << " @" << i
                    << " bitset_len: " << spec.bitset_len << " imin: " << imin
                    << " imax: " << imax
                    << " bit_element: " << bitset_converted_cpu(i / nbits));
        }
    }
}

void run_bitmap() {
    Options spec;

    faiss::gpu::StandardGpuResources res;
    res.noTempMemory();
    auto gpuRes = res.getResources();
    const raft::device_resources& raft_handle =
            gpuRes->getRaftHandleCurrentDevice();
    // generate random bitmap selector
    auto bitmap_faiss_cpu = std::vector<uint8_t>((spec.bitset_len + 8) / 8);
    for (uint32_t i = 0; i < bitmap_faiss_cpu.size(); i++) {
        bitmap_faiss_cpu[i] = (uint8_t)faiss::gpu::randVal(0, 255);
    }
    auto bitmap_selector =
            faiss::IDSelectorBitmap(spec.bitset_len, bitmap_faiss_cpu.data());
    auto bitset = cuvs::core::bitset<uint32_t, uint32_t>(
            raft_handle, spec.bitset_len, false);
    faiss::gpu::convert_to_bitset(gpuRes.get(), bitmap_selector, bitset.view());

    auto bitset_converted_cpu =
            raft::make_host_vector<uint32_t, uint32_t>(bitset.n_elements());
    raft::copy(raft_handle, bitset_converted_cpu.view(), bitset.to_mdspan());
    raft::resource::sync_stream(raft_handle);
    auto bitset_converted_cpu_view =
            cuvs::core::bitset_view<uint32_t, uint32_t>(
                    bitset_converted_cpu.data_handle(), spec.bitset_len);
    for (uint32_t i = 0; i < spec.bitset_len; i++) {
        if (bitset_converted_cpu_view.test(i) != bitmap_selector.is_member(i)) {
            ASSERT_TRUE(
                    testing::AssertionFailure()
                    << "actual=" << bitset_converted_cpu_view.test(i)
                    << " != expected=" << bitmap_selector.is_member(i) << " @"
                    << i << " bitset_len: " << spec.bitset_len
                    << " bitset_expected: " << bitmap_faiss_cpu[i / 8]
                    << " bitset_actual: " << bitset_converted_cpu(i / 32));
        }
    }
}

void run_array() {
    Options spec;
    faiss::gpu::StandardGpuResources res;
    res.noTempMemory();
    auto gpuRes = res.getResources();
    const raft::device_resources& raft_handle =
            gpuRes->getRaftHandleCurrentDevice();
    // generate random array selector
    int n = spec.bitset_len / 20; // select 5% of the bitset length
    std::vector<faiss::idx_t> array_selector_indices(n);
    for (int i = 0; i < n; i++) {
        array_selector_indices[i] = faiss::gpu::randVal(0, spec.bitset_len - 1);
    }
    auto array_selector =
            faiss::IDSelectorArray(n, array_selector_indices.data());
    auto bitset = cuvs::core::bitset<uint32_t, uint32_t>(
            raft_handle, spec.bitset_len, false);
    faiss::gpu::convert_to_bitset(gpuRes.get(), array_selector, bitset.view());

    auto bitset_converted_cpu =
            raft::make_host_vector<uint32_t, uint32_t>(bitset.n_elements());
    raft::copy(raft_handle, bitset_converted_cpu.view(), bitset.to_mdspan());
    raft::resource::sync_stream(raft_handle);
    auto bitset_converted_cpu_view =
            cuvs::core::bitset_view<uint32_t, uint32_t>(
                    bitset_converted_cpu.data_handle(), spec.bitset_len);
    for (uint32_t i = 0; i < spec.bitset_len; i++) {
        if (bitset_converted_cpu_view.test(i) != array_selector.is_member(i)) {
            ASSERT_TRUE(
                    testing::AssertionFailure()
                    << "actual=" << bitset_converted_cpu_view.test(i)
                    << " != expected=" << array_selector.is_member(i) << " @"
                    << i << " bitset_len: " << spec.bitset_len);
        }
    }
}

TEST(TestGpuFilterConvert, Complex) {
    run_complex();
}

TEST(TestGpuFilterConvert, Bitmap) {
    run_bitmap();
}

TEST(TestGpuFilterConvert, Array) {
    run_array();
}

TEST(TestGpuFilterConvert, Range) {
    run_range();
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);

    // just run with a fixed test seed
    faiss::gpu::setTestSeed(100);

    return RUN_ALL_TESTS();
}
