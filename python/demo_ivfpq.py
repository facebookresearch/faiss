
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

#! /usr/bin/env python2


import numpy as np

import faiss


def fvecs_read(filename):
    fv = np.fromfile(filename, dtype = 'float32')
    if fv.size == 0:
        return np.zeros((0, 0), dtype = 'float32')

    dim = fv.view('int32')[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)

    if not all(fv.view('int32')[:,0]==dim):
        raise IOError("non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]

    return fv.copy()   # to make contiguous

rootdir = '/mnt/vol/gfsai-east/ai-group/datasets/simsearch/sift1M'


print "loading database"

xb = fvecs_read(rootdir + '/sift_base.fvecs')
xt = fvecs_read(rootdir + '/sift_learn.fvecs')
xq = fvecs_read(rootdir + '/sift_query.fvecs')

d = xt.shape[1]

gt_index = faiss.IndexFlatL2(d)
gt_index.add(xb)

D, gt_nns = gt_index.search(xq, 1)

coarse_quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFPQ(coarse_quantizer, d, 25, 16, 8)

print "train"
index.train(xt)

print "add"
index.add(xb)

print "search"
index.nprobe = 5
D, nns = index.search(xq, 10)
n_ok = (nns == gt_nns).sum()
nq = xq.shape[0]

print "n_ok=%d/%d" % (n_ok, nq)
