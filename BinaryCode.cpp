
/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "BinaryCode.h"

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <algorithm>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "FaissAssert.h"
#include "hamming.h"

namespace faiss {

/*****************************************************
 * ExternalTransform
 *****************************************************/

ExternalTransform::ExternalTransform (int d):
    LinearTransform(d, d, true),
    octave_wd ("/experimental/deeplearning/matthijs/"
               "ann_baselines/ckmeans-master"),
    model_type ("okmeans"),
    random_seed (123)
{ is_trained = false; }


void ExternalTransform::reverse_transform (idx_t n, const float * xt,
                                float *x) const
{
    transform_transpose (n, xt, x);
}


void ExternalTransform::train (faiss::Index::idx_t n, const float *x)
{
    const char *tmp_infile = "/tmp/ITQTransform_input.raw";
    const char *tmp_outfile = "/tmp/ITQTransform_output.raw";
    long nmax = 500000;
    if (n > nmax) {
        printf ("ExternalTransform::train: limiting dataset to %ld "
                "points (from %ld)\n", nmax, n);
        n = nmax;
    }


    {
        FILE * f = fopen (tmp_infile, "w");
        assert (f || !"cannot open tmp_infile");
        int ret = fwrite (x, sizeof(*x), n * d_in, f);
        assert (ret == n * d_in || !"could not write whole matrix");
        fclose(f);
    }

    {
        char cmd[256];
        std::string wd = getenv("HOME");
        wd = wd + "/fbsource/fbcode/" + octave_wd;
        snprintf (cmd, 256,
                  "rm -f %s; cd %s; octave -q train_model_script.m %s %d %s %s %d",
                 tmp_outfile,
                 wd.c_str(), model_type.c_str(),
                 d_in, tmp_infile, tmp_outfile, random_seed);

        printf("running \"%s\"\n", cmd);

        int ret = system (cmd);

        if (ret) {
            fprintf (stderr, "command %s returned %d\n", cmd, ret);
            abort ();
        }
    }

    {
        FILE *f = fopen (tmp_outfile, "r");
        assert (f || !"cannot open tmp_outfile");

        {
            A.resize (d_in * d_in);
            int ret = fread (A.data(), sizeof(A[0]), d_in * d_in, f);
            assert (ret == d_in * d_in || !"could not read whole matrix");
        }
        {
            b.resize (d_in);
            int ret = fread (b.data(), sizeof(b[0]), d_in, f);
            assert (ret == d_in || !"could not read bias term");

        }
        fclose (f);
    }
    is_trained = true;
}


/*****************************************************
 * BinaryCode
 *****************************************************/



BinaryCode::BinaryCode (size_t d, int nbits,
                faiss::VectorTransform & vt,
                bool train_thresholds,
                bool train_means):
        d(d), nbits(nbits),
        pca(d, d > nbits ? nbits : d),       // used only if d > nbits
        train_thresholds(train_thresholds),
        train_means(train_means),
        vt(vt),
        is_trained(false)
{
    code_size = (nbits + 7) / 8;
    if (d > nbits) { // PCA
        FAISS_ASSERT (vt.d_in == pca.d_out);
    } else {
        FAISS_ASSERT (vt.d_in == d);
    }
    FAISS_ASSERT (vt.d_out == nbits);
}




void BinaryCode::train (long n, const float *x)
{

    // train PCA
    const float *xp = x;
    if (d > nbits) {
        if (!pca.is_trained)
            pca.train(n, x);
        xp = pca.apply(n, x);
    }

    vt.train(n, xp);

    // Apply the random rotation
    const float *xt = vt.apply(n, xp);
    if (xp != x) delete [] xp;

    // stats on resulting vectors
    float * transposed_x = new float [n * nbits];

    for (long i = 0; i < n; i++)
        for (long j = 0; j < nbits; j++)
            transposed_x [j * n + i] = xt [i * nbits + j];

    if (xt != x)
        delete [] xt;

    thresholds.resize (nbits);
    mean_0.resize(nbits);
    mean_1.resize(nbits);
    for (long i = 0; i < nbits; i++) {
        float *xi = transposed_x + i * n;
        // std::nth_element

        if (train_thresholds) {

            std::sort (xi, xi + n);
            if (n % 2 == 1)
                thresholds [i] = xi [n / 2];
            else
                thresholds [i] = (xi [n / 2 - 1] + xi [n / 2]) / 2;
        }  else {
            thresholds[i] = 0;
        }

        if (train_means) {
            size_t n0 = 0, n1 = 0;
            float sum0 = 0, sum1 = 0;
            for (int j = 0; j < n; j++) {
                if (xi[j] >= thresholds[i]) {sum1 += xi[j]; n1++; }
                else                        {sum0 += xi[j]; n0++; }
            }
            mean_0[i] = sum0 / n0;
            mean_1[i] = sum1 / n1;
        } else {
            mean_0[i] = -1;
            mean_1[i] = 1;
        }

    }
    delete [] transposed_x;
    is_trained = true;
}


static void bitvec2fvec8 (const uint8_t * b, float * x, size_t d)
{
    for (int i = 0; i < d; i += 8) {
        uint8_t w = *b;
        uint8_t mask = 1;
        int nj = i + 8 <= d ? 8 : d - i;
        for (int j = 0; j < nj; j++) {
            if (w & mask)
                x[i + j] = 1;
            else
                x[i + j] = -1;
            mask <<= 1;
        }
        b++;
    }
}



void BinaryCode::encode(size_t n, const float *x, uint8_t *codes) const
{

    // apply PCA & random rotation
    assert (is_trained);
    const float *xp = x;
    if (d > nbits)
        xp = pca.apply(n, x);
    float *xt = vt.apply(n, xp);
    if (xp != x) delete [] xp;

    // binarize vectors
    for (size_t i = 0; i < n; i++) {
        float *xi = xt + i * nbits;
        for (int j = 0; j < nbits; j++)
            xi[j] -= thresholds[j];
        fvec2bitvec (xi, codes + i * code_size, nbits);
    }

    delete [] xt;
}

void BinaryCode::decode(size_t n, const uint8_t *codes, float *x) const
{
    assert (is_trained);

    float *xb = new float[n * nbits];
    for(size_t i = 0; i < n; i++) {
        float *xi = xb + i * nbits;
        bitvec2fvec8(codes + i * code_size, xi, nbits);
        for (int j = 0; j < nbits; j++)
            xi[j] = (xi[j] > 0 ? mean_1[j] : mean_0[j]);
    }

    // revert transformations
    if (d > nbits) {
        float *xp = new float[n * nbits];
        vt.reverse_transform(n, xb, xp);
        pca.reverse_transform(n, xp, x);
        delete [] xp;
    } else {
        vt.reverse_transform(n, xb, x);
    }
    delete [] xb;
}


} // namespace faiss
