# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This contrib module contains Pytorch code for quantization.
"""

import torch
import faiss
import math
from faiss.contrib.torch import clustering
# the kmeans can produce both torch and numpy centroids


class Quantizer:

    def __init__(self, d, code_size):
        """
        d: dimension of vectors
        code_size: nb of bytes of the code (per vector)
        """
        self.d = d
        self.code_size = code_size

    def train(self, x):
        """
        takes a n-by-d array and performs training
        """
        pass

    def encode(self, x):
        """
        takes a n-by-d float array, encodes to an n-by-code_size uint8 array
        """
        pass

    def decode(self, codes):
        """
        takes a n-by-code_size uint8 array, returns a n-by-d array
        """
        pass


class VectorQuantizer(Quantizer):

    def __init__(self, d, k):

        code_size = int(math.ceil(torch.log2(k) / 8))
        Quantizer.__init__(d, code_size)
        self.k = k

    def train(self, x):
        pass


class ProductQuantizer(Quantizer):
    def __init__(self, d, M, nbits):
        """ M: number of subvectors, d%M == 0
        nbits: number of bits that each vector is encoded into
        """
        assert d % M == 0
        assert nbits == 8  # todo: implement other nbits values
        code_size = int(math.ceil(M * nbits / 8))
        Quantizer.__init__(self, d, code_size)
        self.M = M
        self.nbits = nbits
        self.code_size = code_size

    def train(self, x):
        nc = 2 ** self.nbits
        sd = self.d // self.M
        dev = x.device
        dtype = x.dtype
        self.codebook = torch.zeros((self.M, nc, sd), device=dev, dtype=dtype)
        for m in range(self.M):
            xsub = x[:, m * self.d // self.M: (m + 1) * self.d // self.M]
            data = clustering.DatasetAssign(xsub.contiguous())
            self.codebook[m] = clustering.kmeans(2 ** self.nbits, data)

    def encode(self, x):
        codes = torch.zeros((x.shape[0], self.code_size), dtype=torch.uint8)
        for m in range(self.M):
            xsub = x[:, m * self.d // self.M:(m + 1) * self.d // self.M]
            _, I = faiss.knn(xsub.contiguous(), self.codebook[m], 1)
            codes[:, m] = I.ravel()
        return codes

    def decode(self, codes):
        idxs = [codes[:, m].long() for m in range(self.M)]
        vectors = [self.codebook[m, idxs[m], :] for m in range(self.M)]
        stacked_vectors = torch.stack(vectors, dim=1)
        cbd = self.codebook.shape[-1]
        x_rec = stacked_vectors.reshape(-1, cbd * self.M)
        return x_rec
