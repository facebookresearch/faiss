#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include <faiss/utils/distances.h>

#include <faiss/utils/distances_if.h>

/* compute the inner product between x and a subset y of ny vectors,
   whose indices are given by idy.  */
void fvec_inner_products_by_idx_ref(
        float* __restrict ip,
        const float* x,
        const float* y,
        const int64_t* __restrict ids, /* for y vecs */
        size_t d,
        size_t nx,
        size_t ny) {
#pragma omp parallel for
    for (int64_t j = 0; j < nx; j++) {
        const int64_t* __restrict idsj = ids + j * ny;
        const float* xj = x + j * d;
        float* __restrict ipj = ip + j * ny;

        // baseline version
        for (size_t i = 0; i < ny; i++) {
            if (idsj[i] < 0)
                continue;
            ipj[i] = faiss::fvec_inner_product(xj, y + d * idsj[i], d);
        }
    }
}

/* compute the inner product between x and a subset y of ny vectors,
   whose indices are given by idy.  */
void fvec_L2sqr_by_idx_ref(
        float* __restrict dis,
        const float* x,
        const float* y,
        const int64_t* __restrict ids, /* ids of y vecs */
        size_t d,
        size_t nx,
        size_t ny) {
#pragma omp parallel for
    for (int64_t j = 0; j < nx; j++) {
        const int64_t* __restrict idsj = ids + j * ny;
        const float* xj = x + j * d;
        float* __restrict disj = dis + j * ny;

        // baseline version
        for (size_t i = 0; i < ny; i++) {
            if (idsj[i] < 0)
                continue;
            disj[i] = faiss::fvec_L2sqr(xj, y + d * idsj[i], d);
        }
    }
}

TEST(TestDistancesIf, TestNyByIdx) {
    const size_t dim = 16;
    const size_t nx = 32;

    std::default_random_engine rng(123);
    std::uniform_real_distribution<float> u(0, 1);

    std::vector<float> x(nx * dim);
    for (size_t i = 0; i < x.size(); i++) {
        x[i] = u(rng);
    }
    
    std::vector<float> y(64 * dim);
    for (size_t i = 0; i < y.size(); i++) {
        y[i] = u(rng);
    }

    for (size_t attempt = 0; attempt < 5; attempt++) {
        for (const size_t ny : {1, 2, 3, 4, 5, 6, 7, 8, 16, 63, 64}) {
            std::vector<float> dis_IP(nx * ny, 1e20);
            std::vector<float> dis_IP_ref(nx * ny, 1e20);
            std::vector<float> dis_L2(nx * ny, 1e20);
            std::vector<float> dis_L2_ref(nx * ny, 1e20);

            std::uniform_int_distribution<int64_t> ids_u(0, ny - 1);
            std::vector<int64_t> ids(nx * ny);
            for (size_t i = 0; i < nx * ny; i++) {
                if (u(rng) < 0.5) {
                    ids[i] = -1;
                }
                else {
                    ids[i] = ids_u(rng);
                }
            }

            // test IP
            fvec_inner_products_by_idx_ref(
                dis_IP_ref.data(), 
                x.data(), 
                y.data(),
                ids.data(),
                dim,
                nx,
                ny);

            faiss::fvec_inner_products_by_idx(
                dis_IP.data(), 
                x.data(), 
                y.data(),
                ids.data(),
                dim,
                nx,
                ny);

            ASSERT_EQ(dis_IP, dis_IP_ref) << "ny = " << ny;

            // test L2
            fvec_L2sqr_by_idx_ref(
                dis_L2_ref.data(), 
                x.data(), 
                y.data(),
                ids.data(),
                dim,
                nx,
                ny);

            faiss::fvec_L2sqr_by_idx(
                dis_L2.data(), 
                x.data(), 
                y.data(),
                ids.data(),
                dim,
                nx,
                ny);

            ASSERT_EQ(dis_L2, dis_L2_ref) << "ny = " << ny;
        }
    }
}