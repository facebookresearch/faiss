/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* All distance functions for L2 and IP distances.
 * The actual functions are implemented in distances.cpp and distances_simd.cpp
 */

#pragma once
#include <stdlib.h>
#include <mutex>
#include <shared_mutex>
#include "oneapi/dnnl/dnnl.hpp"

namespace faiss {
static dnnl::engine cpu_engine;
static dnnl::stream engine_stream;
static bool is_onednn_init = false;
static std::mutex init_mutex;

static bool is_amxbf16_supported() {
    unsigned int eax, ebx, ecx, edx;
    __asm__ __volatile__("cpuid"
                         : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                         : "a"(7), "c"(0));
    return edx & (1 << 22);
}

static void init_onednn() {
    std::unique_lock<std::mutex> lock(init_mutex);

    if (is_onednn_init) {
        return;
    }

    // init dnnl engine
    cpu_engine = dnnl::engine(dnnl::engine::kind::cpu, 0);
    engine_stream = dnnl::stream(cpu_engine);

    is_onednn_init = true;
}

__attribute__((constructor)) static void library_load() {
    // this functionn will be automatically called when the library is loaded
    // printf("Library loaded.\n");
    init_onednn();
}

/**
 * @brief Compute float32 matrix inner product with bf16 intermediate results to
 * accelerate
 * @details The main idea is:
 * 1. Define float32 memory layout for input and output
 * 2. Create low precision bf16 memory descriptors as inner product input
 * 3. Generate inner product primitive descriptor
 * 4. Execute float32 => (reorder) => bf16 => (inner product) => float32
 *    chain operation, isolate different precision data, accelerate inner
 * product
 * 5. Pipeline execution via streams for asynchronous scheduling
 *
 * @param xrow Row number of input matrix X
 * @param xcol Column number of input matrix X
 * @param yrow Row number of weight matrix Y
 * @param ycol Column number of weight matrix Y
 * @param in_f32_1 Input matrix pointer in float32 type
 * @param in_f32_2 Weight matrix pointer in float32 type
 * @param out_f32 Output matrix pointer for result in float32 type
 * @return None
 */
static void comput_f32bf16f32_inner_product(
        uint32_t xrow,
        uint32_t xcol,
        uint32_t yrow,
        uint32_t ycol,
        float* in_f32_1,
        float* in_f32_2,
        float* out_f32) {
    dnnl::memory::desc f32_md1 = dnnl::memory::desc(
            {xrow, xcol},
            dnnl::memory::data_type::f32,
            dnnl::memory::format_tag::ab);
    dnnl::memory::desc f32_md2 = dnnl::memory::desc(
            {yrow, ycol},
            dnnl::memory::data_type::f32,
            dnnl::memory::format_tag::ab);
    dnnl::memory::desc f32_dst_md2 = dnnl::memory::desc(
            {xrow, yrow},
            dnnl::memory::data_type::f32,
            dnnl::memory::format_tag::ab);

    dnnl::memory f32_mem1 = dnnl::memory(f32_md1, cpu_engine, in_f32_1);
    dnnl::memory f32_mem2 = dnnl::memory(f32_md2, cpu_engine, in_f32_2);
    dnnl::memory f32_dst_mem = dnnl::memory(f32_dst_md2, cpu_engine, out_f32);

    // inner memory bf16
    dnnl::memory::desc bf16_md1 = dnnl::memory::desc(
            {xrow, xcol},
            dnnl::memory::data_type::bf16,
            dnnl::memory::format_tag::any);
    dnnl::memory::desc bf16_md2 = dnnl::memory::desc(
            {yrow, ycol},
            dnnl::memory::data_type::bf16,
            dnnl::memory::format_tag::any);

    dnnl::inner_product_forward::primitive_desc inner_product_pd =
            dnnl::inner_product_forward::primitive_desc(
                    cpu_engine,
                    dnnl::prop_kind::forward_training,
                    bf16_md1,
                    bf16_md2,
                    f32_dst_md2);

    dnnl::inner_product_forward inner_product_prim =
            dnnl::inner_product_forward(inner_product_pd);

    dnnl::memory bf16_mem1 =
            dnnl::memory(inner_product_pd.src_desc(), cpu_engine);
    dnnl::reorder(f32_mem1, bf16_mem1)
            .execute(engine_stream, f32_mem1, bf16_mem1);

    dnnl::memory bf16_mem2 =
            dnnl::memory(inner_product_pd.weights_desc(), cpu_engine);
    dnnl::reorder(f32_mem2, bf16_mem2)
            .execute(engine_stream, f32_mem2, bf16_mem2);

    inner_product_prim.execute(
            engine_stream,
            {{DNNL_ARG_SRC, bf16_mem1},
             {DNNL_ARG_WEIGHTS, bf16_mem2},
             {DNNL_ARG_DST, f32_dst_mem}});

    // Wait for the computation to finalize.
    engine_stream.wait();

    // printf("comput_f32bf16f32_inner_product finished#######>\n");
}

} // namespace faiss