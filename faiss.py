
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python2

# sorry for putting this in the main Faiss directory. This is to avoid
# having to write from python import faiss

import numpy as np
import types
import sys
import pdb

try:
    from swigfaiss_gpu import *
except ImportError as e:
    if e.args[0] != 'ImportError: No module named swigfaiss_gpu':
        # swigfaiss_gpu is there but failed to load: Warn user about it.
        print >> sys.stderr, "Failed to load GPU Faiss: %s" % e.args[0]
        print >> sys.stderr, "Faiss falling back to CPU-only."
    from swigfaiss import *



##################################################################
# The functions below add or replace some methods for classes
# this is to be able to pass in numpy arrays directly


def replace_method(the_class, name, replacement):
    orig_method = getattr(the_class, name)
    if orig_method.__name__ == 'replacement_' + name:
        # replacement was done in parent class
        return
    setattr(the_class, name + '_c', orig_method)
    setattr(the_class, name, replacement)

def handle_Clustering():
    def replacement_train(self, x, index):
        assert x.flags.contiguous
        n, d = x.shape
        assert d == self.d
        self.train_c(n, swig_ptr(x), index)
    replace_method(Clustering, 'train', replacement_train)

handle_Clustering()


def handle_Index(the_class):

    def replacement_add(self, x):
        assert x.flags.contiguous
        n, d = x.shape
        assert d == self.d
        self.add_c(n, swig_ptr(x))

    def replacement_add_with_ids(self, x, ids):
        n, d = x.shape
        assert d == self.d
        assert ids.shape == (n, )
        self.add_with_ids_c(n, swig_ptr(x), swig_ptr(ids))

    def replacement_train(self, x):
        assert x.flags.contiguous
        n, d = x.shape
        assert d == self.d
        self.train_c(n, swig_ptr(x))

    def replacement_search(self, x, k):
        assert x.flags.contiguous
        n, d = x.shape
        assert d == self.d
        distances = np.empty((n, k), dtype=np.float32)
        labels = np.empty((n, k), dtype=np.int64)
        self.search_c(n, swig_ptr(x),
                      k, swig_ptr(distances),
                      swig_ptr(labels))
        return distances, labels

    replace_method(the_class, 'add', replacement_add)
    replace_method(the_class, 'add_with_ids', replacement_add_with_ids)
    replace_method(the_class, 'train', replacement_train)
    replace_method(the_class, 'search', replacement_search)

def handle_VectorTransform(the_class):

    def apply_method(self, x):
        assert x.flags.contiguous
        n, d = x.shape
        assert d == self.d_in
        y = np.empty((n, self.d_out), dtype=np.float32)
        self.apply_noalloc(n, swig_ptr(x), swig_ptr(y))
        return y

    def replacement_vt_train(self, x):
        assert x.flags.contiguous
        n, d = x.shape
        assert d == self.d_in
        self.train_c(n, swig_ptr(x))

    replace_method(the_class, 'train', replacement_vt_train)
    # apply is reserved in Pyton...
    the_class.apply_py = apply_method



def handle_AutoTuneCriterion(the_class):
    def replacement_set_groundtruth(self, D, I):
        if D:
            assert I.shape == D.shape
        self.nq, self.gt_nnn = I.shape
        self.set_groundtruth_c(
            self.gt_nnn, swig_ptr(D) if D else None, swig_ptr(I))

    def replacement_evaluate(self, D, I):
        assert I.shape == D.shape
        assert I.shape == (self.nq, self.nnn)
        return self.evaluate_c(swig_ptr(D), swig_ptr(I))

    replace_method(the_class, 'set_groundtruth', replacement_set_groundtruth)
    replace_method(the_class, 'evaluate', replacement_evaluate)

def handle_ParameterSpace(the_class):
    def replacement_explore(self, index, xq, crit):
        assert xq.shape == (crit.nq, index.d)
        ops = OperatingPoints()
        self.explore_c(index, crit.nq, swig_ptr(xq),
                       crit, ops)
        return ops
    replace_method(the_class, 'explore', replacement_explore)

this_module = sys.modules[__name__]

for symbol in dir(this_module):
    obj = getattr(this_module, symbol)
    # print symbol, isinstance(obj, (type, types.ClassType))
    if isinstance(obj, (type, types.ClassType)):
        the_class = obj
        if issubclass(the_class, Index):
            handle_Index(the_class)

        if issubclass(the_class, VectorTransform):
            handle_VectorTransform(the_class)

        if issubclass(the_class, AutoTuneCriterion):
            handle_AutoTuneCriterion(the_class)

        if issubclass(the_class, ParameterSpace):
            handle_ParameterSpace(the_class)


def vector_float_to_array(v):
    a = np.empty(v.size(), dtype='float32')
    memcpy(swig_ptr(a), v.data(), 4 * v.size())
    return a


class Kmeans:

    def __init__(self, d, k, niter=25, verbose=False):
        self.d = d
        self.k = k
        self.cp = ClusteringParameters()
        self.cp.niter = niter
        self.cp.verbose = verbose
        self.centroids = None

    def train(self, x):
        assert x.flags.contiguous
        n, d = x.shape
        assert d == self.d
        clus = Clustering(d, self.k, self.cp)
        self.index = IndexFlatL2(d)
        clus.train(x, self.index)
        centroids = vector_float_to_array(clus.centroids)
        self.centroids = centroids.reshape(self.k, d)
        self.obj = vector_float_to_array(clus.obj)
        return self.obj[-1]

    def assign(self, x):
        assert self.centroids is not None, "should train before assigning"
        index = IndexFlatL2(self.d)
        index.add(self.centroids)
        D, I = index.search(x, 1)
        return D.ravel(), I.ravel()


def kmin(array, k):
    """ return k smallest values of a float32 array """
    I = np.zeros(k, dtype='int64')
    D = np.zeros(k, dtype='float32')
    ha = float_maxheap_array_t()
    ha.ids = swig_ptr(I)
    ha.val = swig_ptr(D)
    ha.nh = 1
    ha.k = k
    ha.heapify()
    ha.addn(array.size, swig_ptr(array))
    ha.reorder()
    return D, I

def kmax(array, k):
    """ return k largest values of a float32 array """
    I = np.zeros(k, dtype='int64')
    D = np.zeros(k, dtype='float32')
    ha = float_minheap_array_t()
    ha.ids = swig_ptr(I)
    ha.val = swig_ptr(D)
    ha.nh = 1
    ha.k = k
    ha.heapify()
    ha.addn(array.size, swig_ptr(array))
    ha.reorder()
    return D, I


def rand(n, seed=12345):
    res = np.empty(n, dtype='float32')
    float_rand(swig_ptr(res), n, seed)
    return res

def lrand(n, seed=12345):
    res = np.empty(n, dtype='int64')
    long_rand(swig_ptr(res), n, seed)
    return res

def randn(n, seed=12345):
    res = np.empty(n, dtype='float32')
    float_randn(swig_ptr(res), n, seed)
    return res


def eval_intersection(I1, I2):
    """ size of intersection between each line of two result tables"""
    n = I1.shape[0]
    assert I2.shape[0] == n
    k1, k2 = I1.shape[1], I2.shape[1]
    ninter = 0
    for i in range(n):
        ninter += ranklist_intersection_size(k1, swig_ptr(I1[i]), k2, swig_ptr(I2[i]))
    return ninter


def normalize_L2(x):
    fvec_renorm_L2(x.shape[1], x.shape[0], swig_ptr(x))
