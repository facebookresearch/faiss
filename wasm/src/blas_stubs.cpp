#include <algorithm>
#include <cctype>
#include <iostream>
#include <vector>

extern "C" {

// Naive implementation of sgemm
// C = alpha * op(A) * op(B) + beta * C
void sgemm_(
        const char* transa,
        const char* transb,
        const int* m,
        const int* n,
        const int* k,
        const float* alpha,
        const float* a,
        const int* lda,
        const float* b,
        const int* ldb,
        const float* beta,
        float* c,
        const int* ldc) {
    bool ta =
            (*transa == 'T' || *transa == 't' || *transa == 'C' ||
             *transa == 'c');
    bool tb =
            (*transb == 'T' || *transb == 't' || *transb == 'C' ||
             *transb == 'c');

    int M = *m;
    int N = *n;
    int K = *k;
    float Alpha = *alpha;
    float Beta = *beta;
    int LDA = *lda;
    int LDB = *ldb;
    int LDC = *ldc;

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int l = 0; l < K; ++l) {
                float A_val = ta
                        ? a[l * LDA + i]
                        : a[i * LDA + l]; // Note: BLAS is column-major usually,
                                          // but FAISS might use row-major?
                // FAISS uses C-style arrays but calls Fortran BLAS.
                // Standard BLAS expects column-major.
                // However, FAISS usually stores vectors row-major.
                // When calling sgemm, FAISS might pass Transpose flags to
                // handle this. Let's assume standard BLAS semantics: Column
                // Major. A(i, l) in column major is A[i + l*LDA]. Wait, C++ is
                // row major. If FAISS passes a float* from C++, it is row
                // major. If it calls sgemm_, it expects sgemm_ to treat it as
                // column major? Usually C libraries calling BLAS pass Transpose
                // arguments to swap dimensions.

                // Let's implement standard Column-Major SGEMM.
                // A is M x K (if not transposed).
                // A[row, col] = A[row + col * LDA]

                float a_val = ta ? a[l + i * LDA] : a[i + l * LDA];
                float b_val = tb ? b[j + l * LDB] : b[l + j * LDB];
                sum += a_val * b_val;
            }

            if (Beta == 0.0f) {
                c[i + j * LDC] = Alpha * sum;
            } else {
                c[i + j * LDC] = Alpha * sum + Beta * c[i + j * LDC];
            }
        }
    }
}

// Stub for other potential missing symbols
void sgemv_(
        const char* trans,
        const int* m,
        const int* n,
        const float* alpha,
        const float* a,
        const int* lda,
        const float* x,
        const int* incx,
        const float* beta,
        float* y,
        const int* incy) {
    // Minimal stub, likely needed
    std::cerr << "Warning: sgemv_ called but not implemented properly"
              << std::endl;
}
}
