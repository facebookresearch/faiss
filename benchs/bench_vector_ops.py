# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD+Patents license found in the
# LICENSE file in the root directory of this source tree.

#! /usr/bin/env python2

import numpy as np
import faiss
import time

xd = 100
yd = 1000000

np.random.seed(1234)

faiss.omp_set_num_threads(1)

print 'xd=%d yd=%d' % (xd, yd)

print 'Running inner products test..'
for d in 3, 4, 12, 36, 64:

    x = faiss.rand(xd * d).reshape(xd, d)
    y = faiss.rand(yd * d).reshape(yd, d)

    distances = np.empty((xd, yd), dtype='float32')

    t0 = time.time()
    for i in xrange(xd):
        faiss.fvec_inner_products_ny(faiss.swig_ptr(distances[i]),
                                     faiss.swig_ptr(x[i]),
                                     faiss.swig_ptr(y),
                                     d, yd)
    t1 = time.time()

    # sparse verification
    ntry = 100
    num, denom = 0, 0
    for t in range(ntry):
        xi = np.random.randint(xd)
        yi = np.random.randint(yd)
        num += abs(distances[xi, yi] - np.dot(x[xi], y[yi]))
        denom += abs(distances[xi, yi])

    print 'd=%d t=%.3f s diff=%g' % (d, t1 - t0, num / denom)


print 'Running L2sqr test..'
for d in 3, 4, 12, 36, 64:

    x = faiss.rand(xd * d).reshape(xd, d)
    y = faiss.rand(yd * d).reshape(yd, d)

    distances = np.empty((xd, yd), dtype='float32')

    t0 = time.time()
    for i in xrange(xd):
        faiss.fvec_L2sqr_ny(faiss.swig_ptr(distances[i]),
                            faiss.swig_ptr(x[i]),
                            faiss.swig_ptr(y),
                            d, yd)
    t1 = time.time()

    # sparse verification
    ntry = 100
    num, denom = 0, 0
    for t in range(ntry):
        xi = np.random.randint(xd)
        yi = np.random.randint(yd)
        num += abs(distances[xi, yi] - np.sum((x[xi] - y[yi]) ** 2))
        denom += abs(distances[xi, yi])

    print 'd=%d t=%.3f s diff=%g' % (d, t1 - t0, num / denom)
