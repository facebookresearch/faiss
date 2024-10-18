# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This contrib module contains Pytorch code for quantization.
"""

import numpy as np
import torch
import faiss

from faiss.contrib import torch_utils


class Quantizer:

    def __init__(self, d, code_size):
        self.d = d
        self.code_size = code_size

    def train(self, x):
        pass

    def encode(self, x):
        pass

    def decode(self, x):
        pass


class VectorQuantizer(Quantizer):

    def __init__(self, d, k):
        code_size = int(torch.ceil(torch.log2(k) / 8))
        Quantizer.__init__(d, code_size)
        self.k = k

    def train(self, x):
        pass


class ProductQuantizer(Quantizer):

    def __init__(self, d, M, nbits):
        code_size = int(torch.ceil(M * nbits / 8))
        Quantizer.__init__(d, code_size)
        self.M = M
        self.nbits = nbits

    def train(self, x):
        pass
