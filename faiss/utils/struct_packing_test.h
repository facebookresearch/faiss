/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* This file and macro is there to catch discrepancies on the different
 * compilers used in Faiss align and pack structures. This is a difficult to
 * catch bug. It works by declaring a structure and measuring its size in
 * diffetent compilers from a function. The function can be called from a test.
 * If the size as seen from different compilers changes, it's a guarantee for
 * fireworks at runtime. */

#pragma once
#include <vector>

namespace faiss {
namespace struct_packing_test {

/*******************************************************
 * Fake structure to detect structure packing/alignment
 * issues.
 *******************************************************/

struct StructPackingTestD {
    int a;
    bool b, c, d;
    long e;
    bool f, g, h, i, j;
};

struct StructPackingTestC : StructPackingTestD {
    bool k;
    int l;
    bool m;
};

struct StructPackingTestA {
    virtual void* operator()(int) {
        return nullptr;
    }

    virtual ~StructPackingTestA() {}
};

struct StructPackingTestB : StructPackingTestA {
    StructPackingTestC options;
    std::vector<void*> vres;
    std::vector<int> devices;
    int ncall;

    void* operator()(int) override {
        return nullptr;
    }
    virtual ~StructPackingTestB() {}
};

// This is the object hierachy of GpuProgressiveDimIndexFactory that initially
// triggered this error (see PR #4135 and #4136)

enum IndicesOptionsB {
    INDICES_CPU_B = 0,
    INDICES_IVF_B = 1,
    INDICES_32_BIT_B = 2,
    INDICES_64_BIT_B = 3,
};

struct GpuClonerOptionsB {
    IndicesOptionsB indicesOptions = INDICES_64_BIT_B;

    bool useFloat16CoarseQuantizer = false;

    bool useFloat16 = false;

    bool usePrecomputed = false;

    long reserveVecs = 0;

    bool storeTransposed = false;

    bool verbose = false;

    bool use_cuvs = false;
    bool allowCpuCoarseQuantizer = false;
};

struct GpuMultipleClonerOptionsB : public GpuClonerOptionsB {
    bool shard = false;
    int shard_type = 1;
    bool common_ivf_quantizer = false;
};

struct ProgressiveDimIndexFactoryB {

    virtual void* operator()(int dim) {
        return nullptr;
    }

    virtual ~ProgressiveDimIndexFactoryB() {}
};

struct GpuProgressiveDimIndexFactoryB : ProgressiveDimIndexFactoryB {
    GpuMultipleClonerOptionsB options;
    std::vector<void*> vres;
    std::vector<int> devices;
    int ncall;

    explicit GpuProgressiveDimIndexFactoryB(int ngpu) {}

    void* operator()(int dim) override {
        return nullptr;
    }

    virtual ~GpuProgressiveDimIndexFactoryB() override {}
};

} // namespace struct_packing_test

} // namespace faiss

// body of function should be
// int function_name (int q) STRUCT_PACKING_FUNCTION_BODY

#define STRUCT_PACKING_FUNCTION_BODY                                           \
    {                                                                          \
        struct_packing_test::StructPackingTestB sb;                            \
        switch (q) {                                                           \
            case 0:                                                            \
                return sizeof(struct_packing_test::StructPackingTestB);        \
            case 1:                                                            \
                return (char*)&sb.ncall - (char*)&sb;                          \
            case 2:                                                            \
                return sizeof(struct_packing_test::StructPackingTestD);        \
            case 3:                                                            \
                return sizeof(struct_packing_test::StructPackingTestC);        \
            case 4:                                                            \
                return sizeof(struct_packing_test::StructPackingTestB);        \
            case 5:                                                            \
                return sizeof(struct_packing_test::StructPackingTestA);        \
            case 6:                                                            \
                return sizeof(struct_packing_test::IndicesOptionsB);           \
            case 7:                                                            \
                return sizeof(struct_packing_test::GpuMultipleClonerOptionsB); \
            case 8:                                                            \
                return sizeof(struct_packing_test::GpuClonerOptionsB);         \
            case 9:                                                            \
                return sizeof(                                                 \
                        struct_packing_test::ProgressiveDimIndexFactoryB);     \
            case 10:                                                           \
                return sizeof(                                                 \
                        struct_packing_test::GpuProgressiveDimIndexFactoryB);  \
            default:                                                           \
                return -1;                                                     \
        }                                                                      \
    }
