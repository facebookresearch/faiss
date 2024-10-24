/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cstring>
#include <functional>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include <faiss/Index.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/kmeans1d.h>

namespace faiss {

using LookUpFunc = std::function<float(idx_t, idx_t)>;

void reduce(
        const std::vector<idx_t>& rows,
        const std::vector<idx_t>& input_cols,
        const LookUpFunc& lookup,
        std::vector<idx_t>& output_cols) {
    for (idx_t col : input_cols) {
        while (!output_cols.empty()) {
            idx_t row = rows[output_cols.size() - 1];
            float a = lookup(row, col);
            float b = lookup(row, output_cols.back());
            if (a >= b) { // defeated
                break;
            }
            output_cols.pop_back();
        }
        if (output_cols.size() < rows.size()) {
            output_cols.push_back(col);
        }
    }
}

void interpolate(
        const std::vector<idx_t>& rows,
        const std::vector<idx_t>& cols,
        const LookUpFunc& lookup,
        idx_t* argmins) {
    std::unordered_map<idx_t, idx_t> idx_to_col;
    for (idx_t idx = 0; idx < cols.size(); ++idx) {
        idx_to_col[cols[idx]] = idx;
    }

    idx_t start = 0;
    for (idx_t r = 0; r < rows.size(); r += 2) {
        idx_t row = rows[r];
        idx_t end = cols.size() - 1;
        if (r < rows.size() - 1) {
            idx_t idx = argmins[rows[r + 1]];
            end = idx_to_col[idx];
        }
        idx_t argmin = cols[start];
        float min = lookup(row, argmin);
        for (idx_t c = start + 1; c <= end; c++) {
            float value = lookup(row, cols[c]);
            if (value < min) {
                argmin = cols[c];
                min = value;
            }
        }
        argmins[row] = argmin;
        start = end;
    }
}

/** SMAWK algo. Find the row minima of a monotone matrix.
 *
 * References:
 *   1. http://web.cs.unlv.edu/larmore/Courses/CSC477/monge.pdf
 *   2. https://gist.github.com/dstein64/8e94a6a25efc1335657e910ff525f405
 *   3. https://github.com/dstein64/kmeans1d
 */
void smawk_impl(
        const std::vector<idx_t>& rows,
        const std::vector<idx_t>& input_cols,
        const LookUpFunc& lookup,
        idx_t* argmins) {
    if (rows.size() == 0) {
        return;
    }

    /**********************************
     * REDUCE
     **********************************/
    auto ptr = &input_cols;
    std::vector<idx_t> survived_cols; // survived columns
    if (rows.size() < input_cols.size()) {
        reduce(rows, input_cols, lookup, survived_cols);
        ptr = &survived_cols;
    }
    auto& cols = *ptr; // avoid memory copy

    /**********************************
     * INTERPOLATE
     **********************************/

    // call recursively on odd-indexed rows
    std::vector<idx_t> odd_rows;
    for (idx_t i = 1; i < rows.size(); i += 2) {
        odd_rows.push_back(rows[i]);
    }
    smawk_impl(odd_rows, cols, lookup, argmins);

    // interpolate the even-indexed rows
    interpolate(rows, cols, lookup, argmins);
}

void smawk(
        const idx_t nrows,
        const idx_t ncols,
        const LookUpFunc& lookup,
        idx_t* argmins) {
    std::vector<idx_t> rows(nrows);
    std::vector<idx_t> cols(ncols);
    std::iota(std::begin(rows), std::end(rows), 0);
    std::iota(std::begin(cols), std::end(cols), 0);

    smawk_impl(rows, cols, lookup, argmins);
}

void smawk(
        const idx_t nrows,
        const idx_t ncols,
        const float* x,
        idx_t* argmins) {
    auto lookup = [&x, &ncols](idx_t i, idx_t j) { return x[i * ncols + j]; };
    smawk(nrows, ncols, lookup, argmins);
}

namespace {

class CostCalculator {
    // The reuslt would be inaccurate if we use float
    std::vector<double> cumsum;
    std::vector<double> cumsum2;

   public:
    CostCalculator(const std::vector<float>& vec, idx_t n) {
        cumsum.push_back(0.0);
        cumsum2.push_back(0.0);
        for (idx_t i = 0; i < n; ++i) {
            float x = vec[i];
            cumsum.push_back(x + cumsum[i]);
            cumsum2.push_back(x * x + cumsum2[i]);
        }
    }

    float operator()(idx_t i, idx_t j) {
        if (j < i) {
            return 0.0f;
        }
        auto mu = (cumsum[j + 1] - cumsum[i]) / (j - i + 1);
        auto result = cumsum2[j + 1] - cumsum2[i];
        result += (j - i + 1) * (mu * mu);
        result -= (2 * mu) * (cumsum[j + 1] - cumsum[i]);
        return float(result);
    }
};

template <class T>
class Matrix {
    std::vector<T> data;
    idx_t nrows;
    idx_t ncols;

   public:
    Matrix(idx_t nrows, idx_t ncols) {
        this->nrows = nrows;
        this->ncols = ncols;
        data.resize(nrows * ncols);
    }

    inline T& at(idx_t i, idx_t j) {
        return data[i * ncols + j];
    }
};

} // anonymous namespace

double kmeans1d(const float* x, size_t n, size_t nclusters, float* centroids) {
    FAISS_THROW_IF_NOT(n >= nclusters);

    // corner case
    if (n == nclusters) {
        memcpy(centroids, x, n * sizeof(*x));
        return 0.0f;
    }

    /***************************************************
     * sort in ascending order, O(NlogN) in time
     ***************************************************/
    std::vector<float> arr(x, x + n);
    std::sort(arr.begin(), arr.end());

    /***************************************************
    dynamic programming algorithm

    Reference: https://arxiv.org/abs/1701.07204
    -------------------------------

    Assume x is already sorted in ascending order.

    N: number of points
    K: number of clusters

    CC(i, j): the cost of grouping xi,...,xj into one cluster
    D[k][m]:  the cost of optimally clustering x1,...,xm into k clusters
    T[k][m]:  the start index of the k-th cluster

    The DP process is as follow:
        D[k][m] = min_i D[k − 1][i − 1] + CC(i, m)
        T[k][m] = argmin_i D[k − 1][i − 1] + CC(i, m)

    This could be solved in O(KN^2) time and O(KN) space.

    To further reduce the time complexity, we use SMAWK algo to
    solve the argmin problem as follow:

    For each k:
        C[m][i] = D[k − 1][i − 1] + CC(i, m)

        Here C is a n x n totally monotone matrix.
        We could find the row minima by SMAWK in O(N) time.

    Now the time complexity is reduced from O(kN^2) to O(KN).
    ****************************************************/

    CostCalculator CC(arr, n);
    Matrix<float> D(nclusters, n);
    Matrix<idx_t> T(nclusters, n);

    for (idx_t m = 0; m < n; m++) {
        D.at(0, m) = CC(0, m);
        T.at(0, m) = 0;
    }

    std::vector<idx_t> indices(nclusters, 0);

    for (idx_t k = 1; k < nclusters; ++k) {
        // we define C here
        auto C = [&D, &CC, &k](idx_t m, idx_t i) {
            if (i == 0) {
                return CC(i, m);
            }
            idx_t col = std::min(m, i - 1);
            return D.at(k - 1, col) + CC(i, m);
        };

        std::vector<idx_t> argmins(n); // argmin of each row
        smawk(n, n, C, argmins.data());
        for (idx_t m = 0; m < argmins.size(); m++) {
            idx_t idx = argmins[m];
            D.at(k, m) = C(m, idx);
            T.at(k, m) = idx;
        }
    }

    /***************************************************
    compute centroids by backtracking

           T[K - 1][T[K][N] - 1]        T[K][N]        N
    --------------|------------------------|-----------|
                  |     cluster K - 1      | cluster K |

    ****************************************************/

    // for imbalance factor
    double tot = 0.0;
    double uf = 0.0;

    idx_t end = n;
    for (idx_t k = nclusters - 1; k >= 0; k--) {
        const idx_t start = T.at(k, end - 1);
        const float sum =
                std::accumulate(arr.data() + start, arr.data() + end, 0.0f);
        const idx_t size = end - start;
        FAISS_THROW_IF_NOT_FMT(
                size > 0, "Cluster %d: size %d", int(k), int(size));
        centroids[k] = sum / size;
        end = start;

        tot += size;
        uf += size * double(size);
    }

    uf = uf * nclusters / (tot * tot);
    return uf;
}

} // namespace faiss
