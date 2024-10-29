# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import inspect

import faiss
import numpy as np

from faiss.loader import (
    DirectMap,
    IDSelector,
    IDSelectorArray,
    IDSelectorBatch,
    OperatingPoints,
    RangeSearchResult,
    rev_swig_ptr,
    swig_ptr,
    try_extract_index_ivf,
)

##################################################################
# The functions below add or replace some methods for classes
# this is to be able to pass in numpy arrays directly
# The C++ version of the classnames will be suffixed with _c
#
# The docstrings in the wrappers are intended to be similar to numpy
# comments, they will appear with help(Class.method) or ?Class.method
# For methods that are not replaced, the C++ documentation will be used if
# swig 4.x is run with -doxygen.
##################################################################

# For most arrays we force the convesion to the target type with
# np.ascontiguousarray, but for uint8 codes, we raise a type error
# because it is unclear how the conversion should occur: with a view
# (= cast) or conversion?

def _check_dtype_uint8(codes):
    if codes.dtype != 'uint8':
        raise TypeError("Input argument %s must be ndarray of dtype "
                        " uint8, but found %s" % ("codes", codes.dtype))
    return np.ascontiguousarray(codes)


def replace_method(the_class, name, replacement, ignore_missing=False):
    """ Replaces a method in a class with another version. The old method
    is renamed to method_name_c (because presumably it was implemented in C) """
    try:
        orig_method = getattr(the_class, name)
    except AttributeError:
        if ignore_missing:
            return
        raise
    if orig_method.__name__ == 'replacement_' + name:
        # replacement was done in parent class
        return
    setattr(the_class, name + '_c', orig_method)
    setattr(the_class, name, replacement)


def handle_Clustering(the_class):

    def replacement_train(self, x, index, weights=None):
        """Perform clustering on a set of vectors. The index is used for assignment.

        Parameters
        ----------
        x : array_like
            Training vectors, shape (n, self.d). `dtype` must be float32.
        index : faiss.Index
            Index used for assignment. The dimension of the index should be `self.d`.
        weights : array_like, optional
            Per training sample weight (size n) used when computing the weighted
            average to obtain the centroid (default is 1 for all training vectors).
        """
        n, d = x.shape
        x = np.ascontiguousarray(x, dtype='float32')
        assert d == self.d
        if weights is not None:
            weights = np.ascontiguousarray(weights, dtype='float32')
            assert weights.shape == (n, )
            self.train_c(n, swig_ptr(x), index, swig_ptr(weights))
        else:
            self.train_c(n, swig_ptr(x), index)

    def replacement_train_encoded(self, x, codec, index, weights=None):
        """ Perform clustering on a set of compressed vectors. The index is used for assignment.
        The decompression is performed on-the-fly.

        Parameters
        ----------
        x : array_like
            Training vectors, shape (n, codec.code_size()). `dtype` must be `uint8`.
        codec : faiss.Index
            Index used to decode the vectors. Should have dimension `self.d`.
        index : faiss.Index
            Index used for assignment. The dimension of the index should be `self.d`.
        weights : array_like, optional
            Per training sample weight (size n) used when computing the weighted
            average to obtain the centroid (default is 1 for all training vectors).
        """
        n, d = x.shape
        x = _check_dtype_uint8(x)
        assert d == codec.sa_code_size()
        assert codec.d == index.d
        if weights is not None:
            weights = np.ascontiguousarray(weights, dtype='float32')
            assert weights.shape == (n, )
            self.train_encoded_c(n, swig_ptr(x), codec,
                                 index, swig_ptr(weights))
        else:
            self.train_encoded_c(n, swig_ptr(x), codec, index)

    replace_method(the_class, 'train', replacement_train)
    replace_method(the_class, 'train_encoded', replacement_train_encoded)


def handle_Clustering1D(the_class):

    def replacement_train_exact(self, x):
        """Perform clustering on a set of 1D vectors.

        Parameters
        ----------
        x : array_like
            Training vectors, shape (n, 1). `dtype` must be float32.
        """
        n, d = x.shape
        x = np.ascontiguousarray(x, dtype='float32')
        assert d == self.d
        self.train_exact_c(n, swig_ptr(x))

    replace_method(the_class, 'train_exact', replacement_train_exact)


def handle_Quantizer(the_class):

    def replacement_train(self, x):
        """ Train the quantizer on a set of training vectors.

        Parameters
        ----------
        x : array_like
            Training vectors, shape (n, self.d). `dtype` must be float32.
        """
        n, d = x.shape
        x = np.ascontiguousarray(x, dtype='float32')
        assert d == self.d
        self.train_c(n, swig_ptr(x))

    def replacement_compute_codes(self, x):
        """ Compute the codes corresponding to a set of vectors.

        Parameters
        ----------
        x : array_like
            Vectors to encode, shape (n, self.d). `dtype` must be float32.

        Returns
        -------
        codes : array_like
            Corresponding code for each vector, shape (n, self.code_size)
            and `dtype` uint8.
        """
        n, d = x.shape
        x = np.ascontiguousarray(x, dtype='float32')
        assert d == self.d
        codes = np.empty((n, self.code_size), dtype='uint8')
        self.compute_codes_c(swig_ptr(x), swig_ptr(codes), n)
        return codes

    def replacement_decode(self, codes):
        """Reconstruct an approximation of vectors given their codes.

        Parameters
        ----------
        codes : array_like
            Codes to decode, shape (n, self.code_size). `dtype` must be uint8.

        Returns
        -------
            Reconstructed vectors for each code, shape `(n, d)` and `dtype` float32.
        """
        n, cs = codes.shape
        codes = _check_dtype_uint8(codes)
        assert cs == self.code_size
        x = np.empty((n, self.d), dtype='float32')
        self.decode_c(swig_ptr(codes), swig_ptr(x), n)
        return x

    replace_method(the_class, 'train', replacement_train)
    replace_method(the_class, 'compute_codes', replacement_compute_codes)
    replace_method(the_class, 'decode', replacement_decode)


def handle_NSG(the_class):

    def replacement_build(self, x, graph):
        n, d = x.shape
        assert d == self.d
        assert graph.ndim == 2
        assert graph.shape[0] == n
        K = graph.shape[1]
        x = np.ascontiguousarray(x, dtype='float32')
        graph = np.ascontiguousarray(graph, dtype='int64')
        self.build_c(n, swig_ptr(x), swig_ptr(graph), K)

    replace_method(the_class, 'build', replacement_build)


def handle_Index(the_class):

    def replacement_add(self, x):
        """Adds vectors to the index.
        The index must be trained before vectors can be added to it.
        The vectors are implicitly numbered in sequence. When `n` vectors are
        added to the index, they are given ids `ntotal`, `ntotal + 1`, ..., `ntotal + n - 1`.

        Parameters
        ----------
        x : array_like
            Query vectors, shape (n, d) where d is appropriate for the index.
            `dtype` must be float32.
        """

        n, d = x.shape
        assert d == self.d
        x = np.ascontiguousarray(x, dtype='float32')
        self.add_c(n, swig_ptr(x))

    def replacement_add_with_ids(self, x, ids):
        """Adds vectors with arbitrary ids to the index (not all indexes support this).
        The index must be trained before vectors can be added to it.
        Vector `i` is stored in `x[i]` and has id `ids[i]`.

        Parameters
        ----------
        x : array_like
            Query vectors, shape (n, d) where d is appropriate for the index.
            `dtype` must be float32.
        ids : array_like
            Array if ids of size n. The ids must be of type `int64`. Note that `-1` is reserved
            in result lists to mean "not found" so it's better to not use it as an id.
        """
        n, d = x.shape
        assert d == self.d
        x = np.ascontiguousarray(x, dtype='float32')
        ids = np.ascontiguousarray(ids, dtype='int64')
        assert ids.shape == (n, ), 'not same nb of vectors as ids'
        self.add_with_ids_c(n, swig_ptr(x), swig_ptr(ids))

    def replacement_assign(self, x, k, labels=None):
        """Find the k nearest neighbors of the set of vectors x in the index.
        This is the same as the `search` method, but discards the distances.

        Parameters
        ----------
        x : array_like
            Query vectors, shape (n, d) where d is appropriate for the index.
            `dtype` must be float32.
        k : int
            Number of nearest neighbors.
        labels : array_like, optional
            Labels array to store the results.

        Returns
        -------
        labels: array_like
            Labels of the nearest neighbors, shape (n, k).
            When not enough results are found, the label is set to -1
        """
        n, d = x.shape
        assert d == self.d
        x = np.ascontiguousarray(x, dtype='float32')

        if labels is None:
            labels = np.empty((n, k), dtype=np.int64)
        else:
            assert labels.shape == (n, k)

        self.assign_c(n, swig_ptr(x), swig_ptr(labels), k)
        return labels

    def replacement_train(self, x):
        """Trains the index on a representative set of vectors.
        The index must be trained before vectors can be added to it.

        Parameters
        ----------
        x : array_like
            Query vectors, shape (n, d) where d is appropriate for the index.
            `dtype` must be float32.
        """
        n, d = x.shape
        assert d == self.d
        x = np.ascontiguousarray(x, dtype='float32')
        self.train_c(n, swig_ptr(x))

    def replacement_search(self, x, k, *, params=None, D=None, I=None):
        """Find the k nearest neighbors of the set of vectors x in the index.

        Parameters
        ----------
        x : array_like
            Query vectors, shape (n, d) where d is appropriate for the index.
            `dtype` must be float32.
        k : int
            Number of nearest neighbors.
        params : SearchParameters
            Search parameters of the current search (overrides the class-level params)
        D : array_like, optional
            Distance array to store the result.
        I : array_like, optional
            Labels array to store the results.

        Returns
        -------
        D : array_like
            Distances of the nearest neighbors, shape (n, k). When not enough results are found
            the label is set to +Inf or -Inf.
        I : array_like
            Labels of the nearest neighbors, shape (n, k).
            When not enough results are found, the label is set to -1
        """

        n, d = x.shape
        x = np.ascontiguousarray(x, dtype='float32')
        assert d == self.d

        assert k > 0

        if D is None:
            D = np.empty((n, k), dtype=np.float32)
        else:
            assert D.shape == (n, k)

        if I is None:
            I = np.empty((n, k), dtype=np.int64)
        else:
            assert I.shape == (n, k)

        self.search_c(n, swig_ptr(x), k, swig_ptr(D), swig_ptr(I), params)
        return D, I

    def replacement_search_and_reconstruct(self, x, k, *, params=None, D=None, I=None, R=None):
        """Find the k nearest neighbors of the set of vectors x in the index,
        and return an approximation of these vectors.

        Parameters
        ----------
        x : array_like
            Query vectors, shape (n, d) where d is appropriate for the index.
            `dtype` must be float32.
        k : int
            Number of nearest neighbors.
        params : SearchParameters
            Search parameters of the current search (overrides the class-level params)
        D : array_like, optional
            Distance array to store the result.
        I : array_like, optional
            Labels array to store the result.
        R : array_like, optional
            reconstruction array to store

        Returns
        -------
        D : array_like
            Distances of the nearest neighbors, shape (n, k). When not enough results are found
            the label is set to +Inf or -Inf.
        I : array_like
            Labels of the nearest neighbors, shape (n, k). When not enough results are found,
            the label is set to -1
        R : array_like
            Approximate (reconstructed) nearest neighbor vectors, shape (n, k, d).
        """
        n, d = x.shape
        assert d == self.d
        x = np.ascontiguousarray(x, dtype='float32')

        assert k > 0

        if D is None:
            D = np.empty((n, k), dtype=np.float32)
        else:
            assert D.shape == (n, k)

        if I is None:
            I = np.empty((n, k), dtype=np.int64)
        else:
            assert I.shape == (n, k)

        if R is None:
            R = np.empty((n, k, d), dtype=np.float32)
        else:
            assert R.shape == (n, k, d)

        self.search_and_reconstruct_c(
            n, swig_ptr(x),
            k, swig_ptr(D),
            swig_ptr(I), swig_ptr(R), params
        )
        return D, I, R

    def replacement_search_and_return_codes(
            self, x, k, *,
            include_listnos=False, params=None, D=None, I=None, codes=None):
        """Find the k nearest neighbors of the set of vectors x in the index,
        and return the codes stored for these vectors

        Parameters
        ----------
        x : array_like
            Query vectors, shape (n, d) where d is appropriate for the index.
            `dtype` must be float32.
        k : int
            Number of nearest neighbors.
        params : SearchParameters
            Search parameters of the current search (overrides the class-level params)
        include_listnos : bool, optional
            whether to include the list ids in the first bytes of each code
        D : array_like, optional
            Distance array to store the result.
        I : array_like, optional
            Labels array to store the result.
        codes : array_like, optional
            codes array to store

        Returns
        -------
        D : array_like
            Distances of the nearest neighbors, shape (n, k). When not enough results are found
            the label is set to +Inf or -Inf.
        I : array_like
            Labels of the nearest neighbors, shape (n, k). When not enough results are found,
            the label is set to -1
        R : array_like
            Approximate (reconstructed) nearest neighbor vectors, shape (n, k, d).
        """
        n, d = x.shape
        assert d == self.d
        x = np.ascontiguousarray(x, dtype='float32')

        assert k > 0

        if D is None:
            D = np.empty((n, k), dtype=np.float32)
        else:
            assert D.shape == (n, k)

        if I is None:
            I = np.empty((n, k), dtype=np.int64)
        else:
            assert I.shape == (n, k)

        code_size_1 = self.code_size
        if include_listnos:
            code_size_1 += self.coarse_code_size()

        if codes is None:
            codes = np.empty((n, k, code_size_1), dtype=np.uint8)
        else:
            assert codes.shape == (n, k, code_size_1)

        self.search_and_return_codes_c(
            n, swig_ptr(x),
            k, swig_ptr(D),
            swig_ptr(I), swig_ptr(codes), include_listnos,
            params
        )
        return D, I, codes

    def replacement_remove_ids(self, x):
        """Remove some ids from the index.
        This is a O(ntotal) operation by default, so could be expensive.

        Parameters
        ----------
        x : array_like or faiss.IDSelector
            Either an IDSelector that returns True for vectors to remove, or a
            list of ids to reomove (1D array of int64). When `x` is a list,
            it is wrapped into an IDSelector.

        Returns
        -------
        n_remove: int
            number of vectors that were removed
        """
        if isinstance(x, IDSelector):
            sel = x
        else:
            assert x.ndim == 1
            index_ivf = try_extract_index_ivf(self)
            x = np.ascontiguousarray(x, dtype='int64')
            if index_ivf and index_ivf.direct_map.type == DirectMap.Hashtable:
                sel = IDSelectorArray(x.size, swig_ptr(x))
            else:
                sel = IDSelectorBatch(x.size, swig_ptr(x))
        return self.remove_ids_c(sel)

    def replacement_reconstruct(self, key, x=None):
        """Approximate reconstruction of one vector from the index.

        Parameters
        ----------
        key : int
            Id of the vector to reconstruct
        x : array_like, optional
            pre-allocated array to store the results

        Returns
        -------
        x : array_like reconstructed vector, size `self.d`, `dtype`=float32
        """
        if x is None:
            x = np.empty(self.d, dtype=np.float32)
        else:
            assert x.shape == (self.d, )

        self.reconstruct_c(key, swig_ptr(x))
        return x

    def replacement_reconstruct_batch(self, key, x=None):
        """Approximate reconstruction of several vectors from the index.

        Parameters
        ----------
        key : array of ints
            Ids of the vectors to reconstruct
        x : array_like, optional
            pre-allocated array to store the results

        Returns
        -------
        x : array_like
            reconstrcuted vectors, size `len(key), self.d`
        """
        key = np.ascontiguousarray(key, dtype='int64')
        n, = key.shape
        if x is None:
            x = np.empty((n, self.d), dtype=np.float32)
        else:
            assert x.shape == (n, self.d)
        self.reconstruct_batch_c(n, swig_ptr(key), swig_ptr(x))
        return x

    def replacement_reconstruct_n(self, n0=0, ni=-1, x=None):
        """Approximate reconstruction of vectors `n0` ... `n0 + ni - 1` from the index.
        Missing vectors trigger an exception.

        Parameters
        ----------
        n0 : int
            Id of the first vector to reconstruct (default 0)
        ni : int
            Number of vectors to reconstruct (-1 = default = ntotal)
        x : array_like, optional
            pre-allocated array to store the results

        Returns
        -------
        x : array_like
            Reconstructed vectors, size (`ni`, `self.d`), `dtype`=float32
        """
        if ni == -1:
            ni = self.ntotal - n0
        if x is None:
            x = np.empty((ni, self.d), dtype=np.float32)
        else:
            assert x.shape == (ni, self.d)

        self.reconstruct_n_c(n0, ni, swig_ptr(x))
        return x

    def replacement_update_vectors(self, keys, x):
        n = keys.size
        assert keys.shape == (n, )
        assert x.shape == (n, self.d)
        x = np.ascontiguousarray(x, dtype='float32')
        keys = np.ascontiguousarray(keys, dtype='int64')
        self.update_vectors_c(n, swig_ptr(keys), swig_ptr(x))

    # No support passed-in for output buffers
    def replacement_range_search(self, x, thresh, *, params=None):
        """Search vectors that are within a distance of the query vectors.

        Parameters
        ----------
        x : array_like
            Query vectors, shape (n, d) where d is appropriate for the index.
            `dtype` must be float32.
        thresh : float
            Threshold to select neighbors. All elements within this radius are returned,
            except for maximum inner product indexes, where the elements above the
            threshold are returned
        params : SearchParameters
            Search parameters of the current search (overrides the class-level params)


        Returns
        -------
        lims: array_like
            Starting index of the results for each query vector, size n+1.
        D : array_like
            Distances of the nearest neighbors, shape `lims[n]`. The distances for
            query i are in `D[lims[i]:lims[i+1]]`.
        I : array_like
            Labels of nearest neighbors, shape `lims[n]`. The labels for query i
            are in `I[lims[i]:lims[i+1]]`.

        """
        n, d = x.shape
        assert d == self.d
        x = np.ascontiguousarray(x, dtype='float32')
        thresh = float(thresh)

        res = RangeSearchResult(n)
        self.range_search_c(n, swig_ptr(x), thresh, res, params)
        # get pointers and copy them
        lims = rev_swig_ptr(res.lims, n + 1).copy()
        nd = int(lims[-1])
        D = rev_swig_ptr(res.distances, nd).copy()
        I = rev_swig_ptr(res.labels, nd).copy()
        return lims, D, I

    def replacement_search_preassigned(self, x, k, Iq, Dq, *, params=None, D=None, I=None):
        """Find the k nearest neighbors of the set of vectors x in an IVF index,
        with precalculated coarse quantization assignment.

        Parameters
        ----------
        x : array_like
            Query vectors, shape (n, d) where d is appropriate for the index.
            `dtype` must be float32.
        k : int
            Number of nearest neighbors.
        Dq : array_like, optional
            Distance array to the centroids, size (n, nprobe)
        Iq : array_like, optional
            Nearest centroids, size (n, nprobe)

        params : SearchParameters
            Search parameters of the current search (overrides the class-level params)
        D : array_like, optional
            Distance array to store the result.
        I : array_like, optional
            Labels array to store the results.

        Returns
        -------
        D : array_like
            Distances of the nearest neighbors, shape (n, k). When not enough results are found
            the label is set to +Inf or -Inf.
        I : array_like
            Labels of the nearest neighbors, shape (n, k).
            When not enough results are found, the label is set to -1
        """
        n, d = x.shape
        x = np.ascontiguousarray(x, dtype='float32')
        assert d == self.d
        assert k > 0

        if D is None:
            D = np.empty((n, k), dtype=np.float32)
        else:
            assert D.shape == (n, k)

        if I is None:
            I = np.empty((n, k), dtype=np.int64)
        else:
            assert I.shape == (n, k)

        Iq = np.ascontiguousarray(Iq, dtype='int64')
        assert params is None, "params not supported"
        assert Iq.shape == (n, self.nprobe)

        if Dq is not None:
            Dq = np.ascontiguousarray(Dq, dtype='float32')
            assert Dq.shape == Iq.shape

        self.search_preassigned_c(
            n, swig_ptr(x),
            k,
            swig_ptr(Iq), swig_ptr(Dq),
            swig_ptr(D), swig_ptr(I),
            False
        )
        return D, I

    def replacement_range_search_preassigned(self, x, thresh, Iq, Dq, *, params=None):
        """Search vectors that are within a distance of the query vectors.

        Parameters
        ----------
        x : array_like
            Query vectors, shape (n, d) where d is appropriate for the index.
            `dtype` must be float32.
        thresh : float
            Threshold to select neighbors. All elements within this radius are returned,
            except for maximum inner product indexes, where the elements above the
            threshold are returned
        Iq : array_like, optional
            Nearest centroids, size (n, nprobe)
        Dq : array_like, optional
            Distance array to the centroids, size (n, nprobe)
        params : SearchParameters
            Search parameters of the current search (overrides the class-level params)


        Returns
        -------
        lims: array_like
            Starting index of the results for each query vector, size n+1.
        D : array_like
            Distances of the nearest neighbors, shape `lims[n]`. The distances for
            query i are in `D[lims[i]:lims[i+1]]`.
        I : array_like
            Labels of nearest neighbors, shape `lims[n]`. The labels for query i
            are in `I[lims[i]:lims[i+1]]`.

        """
        n, d = x.shape
        assert d == self.d
        x = np.ascontiguousarray(x, dtype='float32')

        Iq = np.ascontiguousarray(Iq, dtype='int64')
        assert params is None, "params not supported"
        assert Iq.shape == (n, self.nprobe)

        if Dq is not None:
            Dq = np.ascontiguousarray(Dq, dtype='float32')
            assert Dq.shape == Iq.shape

        thresh = float(thresh)
        res = RangeSearchResult(n)
        self.range_search_preassigned_c(
            n, swig_ptr(x), thresh,
            swig_ptr(Iq), swig_ptr(Dq),
            res
        )
        # get pointers and copy them
        lims = rev_swig_ptr(res.lims, n + 1).copy()
        nd = int(lims[-1])
        D = rev_swig_ptr(res.distances, nd).copy()
        I = rev_swig_ptr(res.labels, nd).copy()
        return lims, D, I

    def replacement_sa_encode(self, x, codes=None):
        n, d = x.shape
        assert d == self.d
        x = np.ascontiguousarray(x, dtype='float32')

        if codes is None:
            codes = np.empty((n, self.sa_code_size()), dtype=np.uint8)
        else:
            assert codes.shape == (n, self.sa_code_size())

        self.sa_encode_c(n, swig_ptr(x), swig_ptr(codes))
        return codes

    def replacement_sa_decode(self, codes, x=None):
        n, cs = codes.shape
        assert cs == self.sa_code_size()
        codes = _check_dtype_uint8(codes)

        if x is None:
            x = np.empty((n, self.d), dtype=np.float32)
        else:
            assert x.shape == (n, self.d)

        self.sa_decode_c(n, swig_ptr(codes), swig_ptr(x))
        return x

    def replacement_add_sa_codes(self, codes, ids=None):
        n, cs = codes.shape
        assert cs == self.sa_code_size()
        codes = _check_dtype_uint8(codes)

        if ids is not None:
            assert ids.shape == (n,)
            ids = swig_ptr(ids)
        self.add_sa_codes_c(n, swig_ptr(codes), ids)

    def replacement_permute_entries(self, perm):
        n, = perm.shape
        assert n == self.ntotal
        perm = np.ascontiguousarray(perm, dtype='int64')
        self.permute_entries_c(faiss.swig_ptr(perm))

    replace_method(the_class, 'add', replacement_add)
    replace_method(the_class, 'add_with_ids', replacement_add_with_ids)
    replace_method(the_class, 'assign', replacement_assign)
    replace_method(the_class, 'train', replacement_train)
    replace_method(the_class, 'search', replacement_search)
    replace_method(the_class, 'remove_ids', replacement_remove_ids)
    replace_method(the_class, 'reconstruct', replacement_reconstruct)
    replace_method(the_class, 'reconstruct_batch',
                   replacement_reconstruct_batch)
    replace_method(the_class, 'reconstruct_n', replacement_reconstruct_n)
    replace_method(the_class, 'range_search', replacement_range_search)
    replace_method(the_class, 'update_vectors', replacement_update_vectors,
                   ignore_missing=True)
    replace_method(the_class, 'search_and_reconstruct',
                   replacement_search_and_reconstruct, ignore_missing=True)
    replace_method(the_class, 'search_and_return_codes',
                   replacement_search_and_return_codes, ignore_missing=True)

    # these ones are IVF-specific
    replace_method(the_class, 'search_preassigned',
                   replacement_search_preassigned, ignore_missing=True)
    replace_method(the_class, 'range_search_preassigned',
                   replacement_range_search_preassigned, ignore_missing=True)
    replace_method(the_class, 'sa_encode', replacement_sa_encode)
    replace_method(the_class, 'sa_decode', replacement_sa_decode)
    replace_method(the_class, 'add_sa_codes', replacement_add_sa_codes)
    replace_method(the_class, 'permute_entries', replacement_permute_entries,
                   ignore_missing=True)

    # get/set state for pickle
    # the data is serialized to std::vector -> numpy array -> python bytes
    # so not very efficient for now.

    def index_getstate(self):
        return {"this": faiss.serialize_index(self).tobytes()}

    def index_setstate(self, st):
        index2 = faiss.deserialize_index(np.frombuffer(st["this"], dtype="uint8"))
        self.this = index2.this

    the_class.__getstate__ = index_getstate
    the_class.__setstate__ = index_setstate


def handle_IndexBinary(the_class):

    def replacement_add(self, x):
        n, d = x.shape
        x = _check_dtype_uint8(x)
        assert d == self.code_size
        self.add_c(n, swig_ptr(x))

    def replacement_add_with_ids(self, x, ids):
        n, d = x.shape
        x = _check_dtype_uint8(x)
        ids = np.ascontiguousarray(ids, dtype='int64')
        assert d == self.code_size
        assert ids.shape == (n, ), 'not same nb of vectors as ids'
        self.add_with_ids_c(n, swig_ptr(x), swig_ptr(ids))

    def replacement_train(self, x):
        n, d = x.shape
        x = _check_dtype_uint8(x)
        assert d == self.code_size
        self.train_c(n, swig_ptr(x))

    def replacement_reconstruct(self, key):
        x = np.empty(self.code_size, dtype=np.uint8)
        self.reconstruct_c(key, swig_ptr(x))
        return x

    def replacement_reconstruct_n(self, n0=0, ni=-1, x=None):
        if ni == -1:
            ni = self.ntotal - n0
        if x is None:
            x = np.empty((ni, self.code_size), dtype=np.uint8)
        else:
            assert x.shape == (ni, self.code_size)

        self.reconstruct_n_c(n0, ni, swig_ptr(x))
        return x

    def replacement_search(self, x, k):
        x = _check_dtype_uint8(x)
        n, d = x.shape
        assert d == self.code_size
        assert k > 0
        distances = np.empty((n, k), dtype=np.int32)
        labels = np.empty((n, k), dtype=np.int64)
        self.search_c(n, swig_ptr(x),
                      k, swig_ptr(distances),
                      swig_ptr(labels))
        return distances, labels

    def replacement_search_preassigned(self, x, k, Iq, Dq):
        n, d = x.shape
        x = _check_dtype_uint8(x)
        assert d == self.code_size
        assert k > 0

        D = np.empty((n, k), dtype=np.int32)
        I = np.empty((n, k), dtype=np.int64)

        Iq = np.ascontiguousarray(Iq, dtype='int64')
        assert Iq.shape == (n, self.nprobe)

        if Dq is not None:
            Dq = np.ascontiguousarray(Dq, dtype='int32')
            assert Dq.shape == Iq.shape

        self.search_preassigned_c(
            n, swig_ptr(x),
            k,
            swig_ptr(Iq), swig_ptr(Dq),
            swig_ptr(D), swig_ptr(I),
            False
        )
        return D, I

    def replacement_range_search(self, x, thresh):
        n, d = x.shape
        x = _check_dtype_uint8(x)
        assert d == self.code_size
        res = RangeSearchResult(n)
        self.range_search_c(n, swig_ptr(x), thresh, res)
        # get pointers and copy them
        lims = rev_swig_ptr(res.lims, n + 1).copy()
        nd = int(lims[-1])
        D = rev_swig_ptr(res.distances, nd).copy()
        I = rev_swig_ptr(res.labels, nd).copy()
        return lims, D, I

    def replacement_range_search_preassigned(self, x, thresh, Iq, Dq, *, params=None):
        n, d = x.shape
        x = _check_dtype_uint8(x)
        assert d == self.code_size

        Iq = np.ascontiguousarray(Iq, dtype='int64')
        assert params is None, "params not supported"
        assert Iq.shape == (n, self.nprobe)

        if Dq is not None:
            Dq = np.ascontiguousarray(Dq, dtype='int32')
            assert Dq.shape == Iq.shape

        thresh = int(thresh)
        res = RangeSearchResult(n)
        self.range_search_preassigned_c(
            n, swig_ptr(x), thresh,
            swig_ptr(Iq), swig_ptr(Dq),
            res
        )
        # get pointers and copy them
        lims = rev_swig_ptr(res.lims, n + 1).copy()
        nd = int(lims[-1])
        D = rev_swig_ptr(res.distances, nd).copy()
        I = rev_swig_ptr(res.labels, nd).copy()
        return lims, D, I

    def replacement_remove_ids(self, x):
        if isinstance(x, IDSelector):
            sel = x
        else:
            assert x.ndim == 1
            x = np.ascontiguousarray(x, dtype='int64')
            sel = IDSelectorBatch(x.size, swig_ptr(x))
        return self.remove_ids_c(sel)

    def replacement_assign(self, x, k, labels=None):
        """Find the k nearest neighbors of the set of vectors x in the index.
        This is the same as the `search` method, but discards the distances.

        Parameters
        ----------
        x : array_like
            Query vectors, shape (n, d) where d is appropriate for the index.
            `dtype` must be uint8.
        k : int
            Number of nearest neighbors.
        labels : array_like, optional
            Labels array to store the results.

        Returns
        -------
        labels: array_like
            Labels of the nearest neighbors, shape (n, k).
            When not enough results are found, the label is set to -1
        """
        n, d = x.shape
        x = _check_dtype_uint8(x)
        assert d == self.code_size
        assert k > 0

        if labels is None:
            labels = np.empty((n, k), dtype=np.int64)
        else:
            assert labels.shape == (n, k)

        self.assign_c(n, swig_ptr(x), swig_ptr(labels), k)
        return labels

    replace_method(the_class, 'add', replacement_add)
    replace_method(the_class, 'add_with_ids', replacement_add_with_ids)
    replace_method(the_class, 'train', replacement_train)
    replace_method(the_class, 'search', replacement_search)
    replace_method(the_class, 'assign', replacement_assign)
    replace_method(the_class, 'range_search', replacement_range_search)
    replace_method(the_class, 'reconstruct', replacement_reconstruct)
    replace_method(the_class, 'reconstruct_n', replacement_reconstruct_n)
    replace_method(the_class, 'remove_ids', replacement_remove_ids)
    replace_method(the_class, 'search_preassigned',
                   replacement_search_preassigned, ignore_missing=True)
    replace_method(the_class, 'range_search_preassigned',
                   replacement_range_search_preassigned, ignore_missing=True)


def handle_VectorTransform(the_class):

    def apply_method(self, x):
        n, d = x.shape
        x = np.ascontiguousarray(x, dtype='float32')
        assert d == self.d_in
        y = np.empty((n, self.d_out), dtype=np.float32)
        self.apply_noalloc(n, swig_ptr(x), swig_ptr(y))
        return y

    def replacement_reverse_transform(self, x):
        n, d = x.shape
        x = np.ascontiguousarray(x, dtype='float32')
        assert d == self.d_out
        y = np.empty((n, self.d_in), dtype=np.float32)
        self.reverse_transform_c(n, swig_ptr(x), swig_ptr(y))
        return y

    def replacement_vt_train(self, x):
        n, d = x.shape
        x = np.ascontiguousarray(x, dtype='float32')
        assert d == self.d_in
        self.train_c(n, swig_ptr(x))

    replace_method(the_class, 'train', replacement_vt_train)
    # apply is reserved in Pyton...
    the_class.apply_py = apply_method
    the_class.apply = apply_method
    replace_method(the_class, 'reverse_transform',
                   replacement_reverse_transform)


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
        xq = np.ascontiguousarray(xq, dtype='float32')
        ops = OperatingPoints()
        self.explore_c(index, crit.nq, swig_ptr(xq),
                       crit, ops)
        return ops
    replace_method(the_class, 'explore', replacement_explore)


def handle_MatrixStats(the_class):
    original_init = the_class.__init__

    def replacement_init(self, m):
        assert len(m.shape) == 2
        m = np.ascontiguousarray(m, dtype='float32')
        original_init(self, m.shape[0], m.shape[1], swig_ptr(m))

    the_class.__init__ = replacement_init


def handle_IOWriter(the_class):
    """ add a write_bytes method """
    def write_bytes(self, b):
        return self(swig_ptr(b), 1, len(b))

    the_class.write_bytes = write_bytes


def handle_IOReader(the_class):
    """ add a read_bytes method """

    def read_bytes(self, totsz):
        buf = bytearray(totsz)
        was_read = self(swig_ptr(buf), 1, len(buf))
        return bytes(buf[:was_read])

    the_class.read_bytes = read_bytes


def handle_IndexRowwiseMinMax(the_class):
    def replacement_train_inplace(self, x):
        """Trains the index on a representative set of vectors inplace.
        The index must be trained before vectors can be added to it.

        This call WILL change the values in the input array, because
        of two scaling proceduces being performed inplace.

        Parameters
        ----------
        x : array_like
            Query vectors, shape (n, d) where d is appropriate for the index.
            `dtype` must be float32.
        """
        n, d = x.shape
        assert d == self.d
        x = np.ascontiguousarray(x, dtype='float32')
        self.train_inplace_c(n, swig_ptr(x))

    replace_method(the_class, 'train_inplace', replacement_train_inplace)


def handle_CodePacker(the_class):

    def replacement_pack_1(self, x, offset, block):
        assert x.shape == (self.code_size,)
        nblock, block_size = block.shape
        assert block_size == self.block_size
        assert 0 <= offset < block_size * self.nvec
        self.pack_1_c(swig_ptr(x), offset, faiss.swig_ptr(block))

    def replacement_unpack_1(self, block, offset):
        nblock, block_size = block.shape
        assert block_size == self.block_size
        assert 0 <= offset < block_size * self.nvec
        x = np.zeros(self.code_size, dtype='uint8')
        self.unpack_1_c(faiss.swig_ptr(block), offset, swig_ptr(x))
        return x

    replace_method(the_class, 'pack_1', replacement_pack_1)
    replace_method(the_class, 'unpack_1', replacement_unpack_1)

######################################################
# MapLong2Long interface
######################################################


def handle_MapLong2Long(the_class):

    def replacement_map_add(self, keys, vals):
        n, = keys.shape
        assert (n,) == vals.shape
        self.add_c(n, swig_ptr(keys), swig_ptr(vals))

    def replacement_map_search_multiple(self, keys):
        n, = keys.shape
        vals = np.empty(n, dtype='int64')
        self.search_multiple_c(n, swig_ptr(keys), swig_ptr(vals))
        return vals

    replace_method(the_class, 'add', replacement_map_add)
    replace_method(the_class, 'search_multiple',
                   replacement_map_search_multiple)


######################################################
# SearchParameters and related interface
######################################################


def add_to_referenced_objects(self, ref):
    if not hasattr(self, 'referenced_objects'):
        self.referenced_objects = [ref]
    else:
        self.referenced_objects.append(ref)

class RememberSwigOwnership:
    """
    SWIG's seattr transfers ownership of SWIG wrapped objects to the class
    (btw this seems to contradict https://www.swig.org/Doc1.3/Python.html#Python_nn22
    31.4.2)
    This interferes with how we manage ownership: with the referenced_objects
    table. Therefore, we reset the thisown field in this context manager.
    """

    def __init__(self, obj):
        self.obj = obj

    def __enter__(self):
        if hasattr(self.obj, "thisown"):
            self.old_thisown = self.obj.thisown
        else:
            self.old_thisown = None

    def __exit__(self, *ignored):
        if self.old_thisown is not None:
            self.obj.thisown = self.old_thisown


def handle_SearchParameters(the_class):
    """ this wrapper is to enable initializations of the form
    SearchParametersXX(a=3, b=SearchParamsYY)
    This also requires the enclosing class to keep a reference on the
    sub-object, since the C++ code assumes the object ownwership is
    handled externally.
    """
    the_class.original_init = the_class.__init__

    def replacement_init(self, **args):
        self.original_init()
        for k, v in args.items():
            assert hasattr(self, k)
            with RememberSwigOwnership(v):
                setattr(self, k, v)
            if type(v) not in (int, float, bool, str):
                add_to_referenced_objects(self, v)

    the_class.__init__ = replacement_init


def handle_IDSelectorSubset(the_class, class_owns, force_int64=True):
    the_class.original_init = the_class.__init__

    def replacement_init(self, *args):
        if len(args) == 1:
            # assume it's an array
            subset, = args
            if force_int64:
                subset = np.ascontiguousarray(subset, dtype='int64')
            args = (len(subset), faiss.swig_ptr(subset))
            if not class_owns:
                add_to_referenced_objects(self, subset)
        self.original_init(*args)

    the_class.__init__ = replacement_init


def handle_CodeSet(the_class):

    def replacement_insert(self, codes, inserted=None):
        n, d = codes.shape
        assert d == self.d
        codes = np.ascontiguousarray(codes, dtype=np.uint8)

        if inserted is None:
            inserted = np.empty(n, dtype=bool)
        else:
            assert inserted.shape == (n, )

        self.insert_c(n, swig_ptr(codes), swig_ptr(inserted))
        return inserted

    replace_method(the_class, 'insert', replacement_insert)

######################################################
# Syntatic sugar for NeuralNet classes
######################################################


def handle_Tensor2D(the_class):
    the_class.original_init = the_class.__init__

    def replacement_init(self, *args):
        if len(args) == 1:
            array, = args
            n, d = array.shape
            self.original_init(n, d)
            faiss.copy_array_to_vector(
                np.ascontiguousarray(array).ravel(), self.v)
        else:
            self.original_init(*args)

    def numpy(self):
        shape = np.zeros(2, dtype=np.int64)
        faiss.memcpy(faiss.swig_ptr(shape), self.shape, shape.nbytes)
        return faiss.vector_to_array(self.v).reshape(shape[0], shape[1])

    the_class.__init__ = replacement_init
    the_class.numpy = numpy


def handle_Embedding(the_class):
    the_class.original_init = the_class.__init__

    def replacement_init(self, *args):
        if len(args) != 1 or args[0].__class__ == the_class:
            self.original_init(*args)
            return
        # assume it's a torch.Embedding
        emb = args[0]
        self.original_init(emb.num_embeddings, emb.embedding_dim)
        self.from_torch(emb)

    def from_torch(self, emb):
        """ copy weights from torch.Embedding """
        assert emb.weight.shape == (self.num_embeddings, self.embedding_dim)
        faiss.copy_array_to_vector(
            np.ascontiguousarray(emb.weight.data).ravel(), self.weight)

    def from_array(self, array):
        """ copy weights from numpy array """
        assert array.shape == (self.num_embeddings, self.embedding_dim)
        faiss.copy_array_to_vector(
            np.ascontiguousarray(array).ravel(), self.weight)

    the_class.from_array = from_array
    the_class.from_torch = from_torch
    the_class.__init__ = replacement_init


def handle_Linear(the_class):
    the_class.original_init = the_class.__init__

    def replacement_init(self, *args):
        if len(args) != 1 or args[0].__class__ == the_class:
            self.original_init(*args)
            return
        # assume it's a torch.Linear
        linear = args[0]
        bias = linear.bias is not None
        self.original_init(linear.in_features, linear.out_features, bias)
        self.from_torch(linear)

    def from_torch(self, linear):
        """ copy weights from torch.Linear """
        assert linear.weight.shape == (self.out_features, self.in_features)
        faiss.copy_array_to_vector(
            linear.weight.data.numpy().ravel(), self.weight)
        if linear.bias is not None:
            assert linear.bias.shape == (self.out_features,)
            faiss.copy_array_to_vector(linear.bias.data.numpy(), self.bias)

    def from_array(self, array, bias=None):
        """ copy weights from numpy array """
        assert array.shape == (self.out_features, self.in_features)
        faiss.copy_array_to_vector(
            np.ascontiguousarray(array).ravel(), self.weight)
        if bias is not None:
            assert bias.shape == (self.out_features,)
            faiss.copy_array_to_vector(bias, self.bias)

    the_class.__init__ = replacement_init
    the_class.from_array = from_array
    the_class.from_torch = from_torch

######################################################
# Syntatic sugar for QINCo and QINCoStep
######################################################

def handle_QINCoStep(the_class):
    the_class.original_init = the_class.__init__

    def replacement_init(self, *args):
        if len(args) != 1 or args[0].__class__ == the_class:
            self.original_init(*args)
            return
        step = args[0]
        # assume it's a Torch QINCoStep
        self.original_init(step.d, step.K, step.L, step.h)
        self.from_torch(step)

    def from_torch(self, step):
        """ copy weights from torch.QINCoStep """
        assert (step.d, step.K, step.L, step.h) == (self.d, self.K, self.L, self.h)
        self.codebook.from_torch(step.codebook)
        self.MLPconcat.from_torch(step.MLPconcat)

        for l in range(step.L):
            src = step.residual_blocks[l]
            dest = self.get_residual_block(l)
            dest.linear1.from_torch(src[0])
            dest.linear2.from_torch(src[2])

    the_class.__init__ = replacement_init
    the_class.from_torch = from_torch


def handle_QINCo(the_class):
    the_class.original_init = the_class.__init__

    def replacement_init(self, *args):
        if len(args) != 1 or args[0].__class__ == the_class:
            self.original_init(*args)
            return

        # assume it's a Torch QINCo
        qinco = args[0]
        self.original_init(qinco.d, qinco.K, qinco.L, qinco.M, qinco.h)
        self.from_torch(qinco)

    def from_torch(self, qinco):
        """ copy weights from torch.QINCo """
        assert (
            (qinco.d, qinco.K, qinco.L, qinco.M, qinco.h) ==
            (self.d, self.K, self.L, self.M, self.h)
        )
        self.codebook0.from_torch(qinco.codebook0)
        for m in range(qinco.M - 1):
            self.get_step(m).from_torch(qinco.steps[m])

    the_class.__init__ = replacement_init
    the_class.from_torch = from_torch
