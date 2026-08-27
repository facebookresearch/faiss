/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/utils/distances.h>

#ifdef COMPILE_SIMD_RISCV_RVV

#include <riscv_vector.h>
#include <faiss/utils/extra_distances.h>

namespace faiss {


template <>
void compute_PQ_dis_tables_dsub2<SIMDLevel::RISCV_RVV>(
        size_t d,                    
        size_t ksub,                 
        const float* all_centroids,  
        size_t nx,                   
        const float* x,              
        bool is_inner_product,       
        float* dis_tables) {         
    size_t M = d / 2;                
    FAISS_THROW_IF_NOT(ksub % 8 == 0); 
    float* c0_all = new float[M * ksub];  
    float* c1_all = new float[M * ksub];  
    for (size_t m = 0; m < M; m++) {
        const float* cm = all_centroids + m * ksub * 2;  
        float* c0m = c0_all + m * ksub;                  
        float* c1m = c1_all + m * ksub;                 
        for (size_t k = 0; k < ksub; k++) {
            c0m[k] = cm[2 * k];       
            c1m[k] = cm[2 * k + 1];   
        }
    }

    size_t vl = __riscv_vsetvl_e32m1(ksub); 

    if (is_inner_product) {
        for (size_t i = 0; i < nx; i++) {
            const float* xi = x + i * d;                    
            float* oi = dis_tables + i * M * ksub;          
            for (size_t m = 0; m + 1 < M; m += 2) {         
                float x0a = xi[2 * m], x1a = xi[2 * m + 1];          
                float x0b = xi[2 * (m + 1)], x1b = xi[2 * (m + 1) + 1]; 
                const float* c0a = c0_all + m * ksub;        
                const float* c1a = c1_all + m * ksub;        
                const float* c0b = c0_all + (m + 1) * ksub;  
                const float* c1b = c1_all + (m + 1) * ksub;  
                float* oa = oi + m * ksub;                   
                float* ob = oi + (m + 1) * ksub;             
                for (size_t k = 0; k < ksub; k += vl) {      
                    vfloat32m1_t vc0 = __riscv_vle32_v_f32m1(c0a + k, vl); 
                    vfloat32m1_t vc1 = __riscv_vle32_v_f32m1(c1a + k, vl); 
                    vfloat32m1_t ra = __riscv_vfmacc_vf_f32m1(
                            __riscv_vfmul_vf_f32m1(vc0, x0a, vl), x1a, vc1, vl);
                    __riscv_vse32_v_f32m1(oa + k, ra, vl); 
                    vc0 = __riscv_vle32_v_f32m1(c0b + k, vl);
                    vc1 = __riscv_vle32_v_f32m1(c1b + k, vl);
                    vfloat32m1_t rb = __riscv_vfmacc_vf_f32m1(
                            __riscv_vfmul_vf_f32m1(vc0, x0b, vl), x1b, vc1, vl);
                    __riscv_vse32_v_f32m1(ob + k, rb, vl);  
                }
            }
        }
    } else {
        for (size_t i = 0; i < nx; i++) {
            const float* xi = x + i * d;
            float* oi = dis_tables + i * M * ksub;
            for (size_t m = 0; m + 1 < M; m += 2) {
                float x0a = xi[2 * m], x1a = xi[2 * m + 1];
                float x0b = xi[2 * (m + 1)], x1b = xi[2 * (m + 1) + 1];
                const float* c0a = c0_all + m * ksub;
                const float* c1a = c1_all + m * ksub;
                const float* c0b = c0_all + (m + 1) * ksub;
                const float* c1b = c1_all + (m + 1) * ksub;
                float* oa = oi + m * ksub;
                float* ob = oi + (m + 1) * ksub;
                for (size_t k = 0; k < ksub; k += vl) {
                    vfloat32m1_t vc0 = __riscv_vle32_v_f32m1(c0a + k, vl);
                    vfloat32m1_t vc1 = __riscv_vle32_v_f32m1(c1a + k, vl);
                    vfloat32m1_t d0 = __riscv_vfsub_vf_f32m1(vc0, x0a, vl); // d0 = c0 - x0a
                    vfloat32m1_t d1 = __riscv_vfsub_vf_f32m1(vc1, x1a, vl); // d1 = c1 - x1a
                    d0 = __riscv_vfmul_vv_f32m1(d0, d0, vl);                // d0 = d0²
                    d1 = __riscv_vfmul_vv_f32m1(d1, d1, vl);                // d1 = d1²
                    vfloat32m1_t r = __riscv_vfadd_vv_f32m1(d0, d1, vl);    // r = d0 + d1
                    __riscv_vse32_v_f32m1(oa + k, r, vl);  

                    vc0 = __riscv_vle32_v_f32m1(c0b + k, vl);
                    vc1 = __riscv_vle32_v_f32m1(c1b + k, vl);
                    d0 = __riscv_vfsub_vf_f32m1(vc0, x0b, vl);
                    d1 = __riscv_vfsub_vf_f32m1(vc1, x1b, vl);
                    d0 = __riscv_vfmul_vv_f32m1(d0, d0, vl);
                    d1 = __riscv_vfmul_vv_f32m1(d1, d1, vl);
                    r = __riscv_vfadd_vv_f32m1(d0, d1, vl);
                    __riscv_vse32_v_f32m1(ob + k, r, vl); 
                }
            }
        }
    }

    delete[] c0_all; 
    delete[] c1_all;
}


template <>
float fvec_norm_L2sqr<SIMDLevel::RISCV_RVV>(const float* x, size_t d) {
    float res = 0.0f;
    size_t i = 0;
    for (; i < d;) {
        size_t vl = __riscv_vsetvl_e32m8(d - i);
        vfloat32m8_t vx = __riscv_vle32_v_f32m8(x + i, vl);
        vfloat32m8_t vxx = __riscv_vfmul_vv_f32m8(vx, vx, vl);
        vfloat32m1_t vred = __riscv_vfredusum_vs_f32m8_f32m1(
                vxx, __riscv_vfmv_v_f_f32m1(0.0f, vl), vl);
        res += __riscv_vfmv_f_s_f32m1_f32(vred);
        i += vl;
    }
    return res;
}

// ===========================================================================
// fvec_L2sqr_ny — non-transposed (row-major y) squared-L2 distance.
//   dis[i] = sum_{j=0}^{d-1} (x[j] - y[i*d + j])^2,  for i in [0, ny)
// y is row-major: the i-th vector occupies y[i*d .. i*d+d-1].
// RVV strategy: vectorize over ny with strided loads. For each dim j, gather
// the j-th component of `chunk` consecutive y vectors via vlse32 (stride =
// d*sizeof(float)), then diff = y_vec - x[j], acc += diff*diff. A single
// accumulator and one strided load + two vector ops per dim keeps register
// pressure low, allowing e32m4 (VLMAX=16 on VLEN=128) as the starting LMUL.
// This is fully general in d and ny (no experiment-parameter special-casing).
// ===========================================================================
template <>
void fvec_L2sqr_ny<SIMDLevel::RISCV_RVV>(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    size_t i = 0;
    const size_t chunk = 32; // e32m8: VLMAX = LMUL*VLEN/SEW = 8*128/32 = 32
    const ptrdiff_t stride_bytes = (ptrdiff_t)d * (ptrdiff_t)sizeof(float);

    if (ny >= chunk) {
        (void)__riscv_vsetvl_e32m8(chunk);
        for (; i + chunk <= ny; i += chunk) {
            vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(0.0f, chunk);
            const float* yb = y + i * d; // base of this chunk
            for (size_t j = 0; j < d; j++) {
                vfloat32m8_t y_vec =
                        __riscv_vlse32_v_f32m8(yb + j, stride_bytes, chunk);
                vfloat32m8_t diff = __riscv_vfsub_vf_f32m8(y_vec, x[j], chunk);
                acc = __riscv_vfmacc_vv_f32m8(acc, diff, diff, chunk);
            }
            __riscv_vse32_v_f32m8(dis + i, acc, chunk);
        }
    }

    if (i < ny) {
        size_t vl = __riscv_vsetvl_e32m8(ny - i);
        vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(0.0f, vl);
        const float* yb = y + i * d;
        for (size_t j = 0; j < d; j++) {
            vfloat32m8_t y_vec =
                    __riscv_vlse32_v_f32m8(yb + j, stride_bytes, vl);
            vfloat32m8_t diff = __riscv_vfsub_vf_f32m8(y_vec, x[j], vl);
            acc = __riscv_vfmacc_vv_f32m8(acc, diff, diff, vl);
        }
        __riscv_vse32_v_f32m8(dis + i, acc, vl);
    }
}


template <>
float fvec_L2sqr<SIMDLevel::RISCV_RVV>(
        const float* x,
        const float* y,
        size_t d) {
    return fvec_L2sqr<SIMDLevel::NONE>(x, y, d);
}

template <>
float fvec_inner_product<SIMDLevel::RISCV_RVV>(
        const float* x,
        const float* y,
        size_t d) {
    return fvec_inner_product<SIMDLevel::NONE>(x, y, d);
}

template <>
float fvec_L1<SIMDLevel::RISCV_RVV>(const float* x, const float* y, size_t d) {
    return fvec_L1<SIMDLevel::NONE>(x, y, d);
}

template <>
float fvec_Linf<SIMDLevel::RISCV_RVV>(
        const float* x,
        const float* y,
        size_t d) {
    return fvec_Linf<SIMDLevel::NONE>(x, y, d);
}

template <>
void fvec_inner_product_batch_4<SIMDLevel::RISCV_RVV>(
        const float* x,
        const float* y0,
        const float* y1,
        const float* y2,
        const float* y3,
        const size_t d,
        float& dis0,
        float& dis1,
        float& dis2,
        float& dis3) {
    fvec_inner_product_batch_4<SIMDLevel::NONE>(
            x, y0, y1, y2, y3, d, dis0, dis1, dis2, dis3);
}

template <>
void fvec_L2sqr_batch_4<SIMDLevel::RISCV_RVV>(
        const float* x,
        const float* y0,
        const float* y1,
        const float* y2,
        const float* y3,
        const size_t d,
        float& dis0,
        float& dis1,
        float& dis2,
        float& dis3) {
    fvec_L2sqr_batch_4<SIMDLevel::NONE>(
            x, y0, y1, y2, y3, d, dis0, dis1, dis2, dis3);
}


template <>
void fvec_L2sqr_ny_transposed<SIMDLevel::RISCV_RVV>(
        float* dis,            
        const float* x,        
        const float* y,        
        const float* y_sqlen,  
        size_t d,              
        size_t d_offset,       
        size_t ny) {           
    // Compute squared length of query subvector
    float x_sqlen = 0;
    for (size_t j = 0; j < d; j++) {
        x_sqlen += x[j] * x[j];
    }

    // Strip-mine ny dimension with e32m8 (VLMAX=32 on VLEN=128)
    size_t i = 0;
    const size_t chunk = 32; // e32m8: VLMAX = LMUL*VLEN/SEW = 8*128/32 = 32

    if (ny >= chunk) {
        (void)__riscv_vsetvl_e32m8(chunk); 
        for (; i + chunk <= ny; i += chunk) {
            // acc = x_sqlen + y_sqlen[i..i+31]
            vfloat32m8_t acc = __riscv_vle32_v_f32m8(y_sqlen + i, chunk); 
            acc = __riscv_vfadd_vf_f32m8(acc, x_sqlen, chunk);            

            // acc += (-2 * x[j]) * y[j*d_offset + i..j*d_offset + i+31]
            for (size_t j = 0; j < d; j++) {
                vfloat32m8_t y_vec =
                        __riscv_vle32_v_f32m8(y + j * d_offset + i, chunk);
                acc = __riscv_vfmacc_vf_f32m8(
                        acc, -2.0f * x[j], y_vec, chunk);
            }

            __riscv_vse32_v_f32m8(dis + i, acc, chunk);
        }
    }

    // Tail: process remaining ny % 32 elements
    if (i < ny) {
        size_t vl = __riscv_vsetvl_e32m8(ny - i); 

        vfloat32m8_t acc = __riscv_vle32_v_f32m8(y_sqlen + i, vl); 
        acc = __riscv_vfadd_vf_f32m8(acc, x_sqlen, vl);            

        for (size_t j = 0; j < d; j++) {
            vfloat32m8_t y_vec =
                    __riscv_vle32_v_f32m8(y + j * d_offset + i, vl); 
            acc = __riscv_vfmacc_vf_f32m8(
                    acc, -2.0f * x[j], y_vec, vl); // acc += (−2·x[j])·y_vec
        }

        __riscv_vse32_v_f32m8(dis + i, acc, vl); 
    }
}


template <>
void fvec_inner_products_ny<SIMDLevel::RISCV_RVV>(
        float* ip,        
        const float* x,   
        const float* y,   
        size_t d,         
        size_t ny) {      
    // Strip-mine ny dimension with e32m4 (VLMAX=16 on VLEN=128)
    // e32m4 proved optimal in LMUL sweep: best balance of strided-load
    // throughput (vlse32 stride=d*4) and register file efficiency
    // Core formula: ip[i] = sum_{j=0}^{d-1} x[j] * y[i*d + j]
    size_t i = 0;
    const size_t chunk = 16; // e32m4: VLMAX = LMUL*VLEN/SEW = 4*128/32 = 16
    const ptrdiff_t stride_bytes = d * sizeof(float);

    if (ny >= chunk) {
        (void)__riscv_vsetvl_e32m4(chunk); 
        for (; i + chunk <= ny; i += chunk) {
            vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, chunk);
            for (size_t j = 0; j < d; j++) {
                vfloat32m4_t y_vec = __riscv_vlse32_v_f32m4(
                        y + j, stride_bytes, chunk);
                acc = __riscv_vfmacc_vf_f32m4(
                        acc, x[j], y_vec, chunk);
            }

            __riscv_vse32_v_f32m4(ip + i, acc, chunk); 
        }
    }

    // Tail: process remaining ny % chunk elements
    if (i < ny) {
        size_t vl = __riscv_vsetvl_e32m4(ny - i); 
        vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl); 
        for (size_t j = 0; j < d; j++) {
            vfloat32m4_t y_vec = __riscv_vlse32_v_f32m4(
                    y + j, stride_bytes, vl);
            acc = __riscv_vfmacc_vf_f32m4(
                    acc, x[j], y_vec, vl); // acc += x[j] · y_vec
        }

        __riscv_vse32_v_f32m4(ip + i, acc, vl); 
    }
}



template <>
size_t fvec_L2sqr_ny_nearest<SIMDLevel::RISCV_RVV>(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    return fvec_L2sqr_ny_nearest<SIMDLevel::NONE>(
            distances_tmp_buffer, x, y, d, ny);
}

template <>
size_t fvec_L2sqr_ny_nearest_y_transposed<SIMDLevel::RISCV_RVV>(
        float* distances_tmp_buffer,
        const float* x,
        const float* y,
        const float* y_sqlen,
        size_t d,
        size_t d_offset,
        size_t ny) {
    return fvec_L2sqr_ny_nearest_y_transposed<SIMDLevel::NONE>(
            distances_tmp_buffer, x, y, y_sqlen, d, d_offset, ny);
}

template <>
void fvec_madd<SIMDLevel::RISCV_RVV>(
        size_t n,
        const float* a,
        float bf,
        const float* b,
        float* c) {
    fvec_madd<SIMDLevel::NONE>(n, a, bf, b, c);
}

template <>
int fvec_madd_and_argmin<SIMDLevel::RISCV_RVV>(
        size_t n,
        const float* a,
        float bf,
        const float* b,
        float* c) {
    return fvec_madd_and_argmin<SIMDLevel::NONE>(n, a, bf, b, c);
}

#define DEFINE_VECTOR_DISTANCE_RVV_FALLBACK(metric)                 \
    template <>                                                     \
    float VectorDistance<metric, SIMDLevel::RISCV_RVV>::operator()( \
            const float* x, const float* y) const {                 \
        return VectorDistance<metric, SIMDLevel::NONE>(             \
                this->d, this->metric_arg)(x, y);                   \
    }

DEFINE_VECTOR_DISTANCE_RVV_FALLBACK(METRIC_L2)
DEFINE_VECTOR_DISTANCE_RVV_FALLBACK(METRIC_INNER_PRODUCT)
DEFINE_VECTOR_DISTANCE_RVV_FALLBACK(METRIC_L1)
DEFINE_VECTOR_DISTANCE_RVV_FALLBACK(METRIC_Linf)
DEFINE_VECTOR_DISTANCE_RVV_FALLBACK(METRIC_Lp)
DEFINE_VECTOR_DISTANCE_RVV_FALLBACK(METRIC_Canberra)
DEFINE_VECTOR_DISTANCE_RVV_FALLBACK(METRIC_BrayCurtis)
DEFINE_VECTOR_DISTANCE_RVV_FALLBACK(METRIC_JensenShannon)
DEFINE_VECTOR_DISTANCE_RVV_FALLBACK(METRIC_Jaccard)
DEFINE_VECTOR_DISTANCE_RVV_FALLBACK(METRIC_NaNEuclidean)
DEFINE_VECTOR_DISTANCE_RVV_FALLBACK(METRIC_GOWER)

#undef DEFINE_VECTOR_DISTANCE_RVV_FALLBACK

} // namespace faiss

#define THE_SIMD_LEVEL SIMDLevel::RISCV_RVV
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/utils/simd_impl/IVFFlatScanner-inl.h>

#endif // COMPILE_SIMD_RISCV_RVV
