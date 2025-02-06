/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


/* This file and macro is there to catch discrepancies on the different compilers 
 * used in Faiss align and pack structures. This is a difficult to catch bug. 
 * It works by declaring a structure and measuring its size in diffetent compilers 
 * from a function. The function can be called from a test. If the size as seen 
 * from different compilers changes, it's a guarantee for fireworks at runtime. */

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
    virtual void* operator()(int) {return nullptr;}

    virtual ~StructPackingTestA() {}
};

struct StructPackingTestB : StructPackingTestA {
    StructPackingTestC options;
    std::vector<void*> vres;
    std::vector<int> devices;
    int ncall;

    void* operator()(int) override { return nullptr;}
    virtual ~StructPackingTestB() {}

};


// This is the object hierachy of GpuProgressiveDimIndexFactory that initially triggered 
// this error (see PR #4135 and #4136)

/// How user vector index data is stored on the GPU
enum IndicesOptionsB {
    /// The user indices are only stored on the CPU; the GPU returns
    /// (inverted list, offset) to the CPU which is then translated to
    /// the real user index.
    INDICES_CPU_B = 0,
    /// The indices are not stored at all, on either the CPU or
    /// GPU. Only (inverted list, offset) is returned to the user as the
    /// index.
    INDICES_IVF_B = 1,
    /// Indices are stored as 32 bit integers on the GPU, but returned
    /// as 64 bit integers
    INDICES_32_BIT_B = 2,
    /// Indices are stored as 64 bit integers on the GPU
    INDICES_64_BIT_B = 3,
};

/// set some options on how to copy to GPU
struct GpuClonerOptionsB {
    /// how should indices be stored on index types that support indices
    /// (anything but GpuIndexFlat*)?
    IndicesOptionsB indicesOptions = INDICES_64_BIT_B;

    /// is the coarse quantizer in float16?
    bool useFloat16CoarseQuantizer = false;

    /// for GpuIndexIVFFlat, is storage in float16?
    /// for GpuIndexIVFPQ, are intermediate calculations in float16?
    bool useFloat16 = false;

    /// use precomputed tables?
    bool usePrecomputed = false;

    /// reserve vectors in the invfiles?
    long reserveVecs = 0;

    /// For GpuIndexFlat, store data in transposed layout?
    bool storeTransposed = false;

    /// Set verbose options on the index
    bool verbose = false;

    /// use the cuVS implementation
#if defined USE_NVIDIA_CUVS
    bool use_cuvs = true;
#else
    bool use_cuvs = false;
#endif

    /// This flag controls the CPU fallback logic for coarse quantizer
    /// component of the index. When set to false (default), the cloner will
    /// throw an exception for indices not implemented on GPU. When set to
    /// true, it will fallback to a CPU implementation.
    bool allowCpuCoarseQuantizer = false;
};

struct GpuMultipleClonerOptionsB : public GpuClonerOptionsB {
    /// Whether to shard the index across GPUs, versus replication
    /// across GPUs
    bool shard = false;

    /// IndexIVF::copy_subset_to subset type
    int shard_type = 1;

    /// set to true if an IndexIVF is to be dispatched to multiple GPUs with a
    /// single common IVF quantizer, ie. only the inverted lists are sharded on
    /// the sub-indexes (uses an IndexShardsIVF)
    bool common_ivf_quantizer = false;
};

struct ProgressiveDimIndexFactoryB {
    /// ownership transferred to caller
    virtual void* operator()(int dim) {return nullptr; }

    virtual ~ProgressiveDimIndexFactoryB() {}
};

struct GpuProgressiveDimIndexFactoryB : ProgressiveDimIndexFactoryB {
    GpuMultipleClonerOptionsB options;
    std::vector<void*> vres;
    std::vector<int> devices;
    int ncall;

    explicit GpuProgressiveDimIndexFactoryB(int ngpu) {}

    void* operator()(int dim) override {return nullptr; }

    virtual ~GpuProgressiveDimIndexFactoryB() override {}

};


} // struct_packing_test

} // faiss

// body of function should be 
// int function_name (int q) STRUCT_PACKING_FUNCTION_BODY

#define STRUCT_PACKING_FUNCTION_BODY {  \
    struct_packing_test::StructPackingTestB sb; \
    switch(q) {  \
    case 0:   \
        return sizeof(struct_packing_test::StructPackingTestB);  \
    case 1:  \
        return (char*)&sb.ncall - (char*)&sb;  \
    case 2:  \
        return sizeof(struct_packing_test::StructPackingTestD);  \
    case 3:  \
        return sizeof(struct_packing_test::StructPackingTestC);  \
    case 4:  \
        return sizeof(struct_packing_test::StructPackingTestB);  \
    case 5:  \
        return sizeof(struct_packing_test::StructPackingTestA);  \
    case 6:  \
        return sizeof(struct_packing_test::IndicesOptionsB);  \
    case 7:  \
        return sizeof(struct_packing_test::GpuMultipleClonerOptionsB);  \
    case 8:  \
        return sizeof(struct_packing_test::GpuClonerOptionsB);  \
    case 9:  \
        return sizeof(struct_packing_test::ProgressiveDimIndexFactoryB);  \
    case 10:  \
        return sizeof(struct_packing_test::GpuProgressiveDimIndexFactoryB);  \
    default: \
        return -1; \
    } \
}




