# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This demonstrates how to reproduce the QINCo paper results using the Faiss
QINCo implementation. The code loads the reference model because training 
is not implemented in Faiss.

Prepare the data with

cd /tmp

# get the reference qinco code
git clone https://github.com/facebookresearch/Qinco.git

# get the data
wget https://dl.fbaipublicfiles.com/QINCo/datasets/bigann/bigann1M.bvecs

# get the model
wget https://dl.fbaipublicfiles.com/QINCo/models/bigann_8x8_L2.pt

"""

import numpy as np
from faiss.contrib.vecs_io import bvecs_mmap
import sys
import time
import torch
import faiss

# make sure pickle deserialization will work
sys.path.append("/tmp/Qinco")
import model_qinco

with torch.no_grad():

    qinco = torch.load("/tmp/bigann_8x8_L2.pt", weights_only=False)
    qinco.eval()
    # print(qinco)
    if True:
        torch.set_num_threads(1)
        faiss.omp_set_num_threads(1)

    x_base = bvecs_mmap("/tmp/bigann1M.bvecs")[:1000].astype('float32')
    x_scaled = torch.from_numpy(x_base) / qinco.db_scale

    t0 = time.time()
    codes, _ = qinco.encode(x_scaled)
    x_decoded_scaled = qinco.decode(codes)
    print(f"Pytorch encode {time.time() - t0:.3f} s")
    # multi-thread: 1.13s, single-thread: 7.744

    x_decoded = x_decoded_scaled.numpy() * qinco.db_scale

    err = ((x_decoded - x_base) ** 2).sum(1).mean()
    print("MSE=", err)  # = 14211.956, near the L=2 result in Fig 4 of the paper

    qinco2 = faiss.QINCo(qinco)
    t0 = time.time()
    codes2 = qinco2.encode(faiss.Tensor2D(x_scaled))
    x_decoded2 = qinco2.decode(codes2).numpy() * qinco.db_scale
    print(f"Faiss encode {time.time() - t0:.3f} s")
    # multi-thread: 3.2s, single thread: 7.019

    # these tests don't work because there are outlier encodings
    # np.testing.assert_array_equal(codes.numpy(), codes2.numpy())
    # np.testing.assert_allclose(x_decoded, x_decoded2)

    ndiff = (codes.numpy() != codes2.numpy()).sum() / codes.numel()
    assert ndiff < 0.01
    ndiff = (((x_decoded - x_decoded2) ** 2).sum(1) > 1e-5).sum()
    assert ndiff / len(x_base) < 0.01

    err = ((x_decoded2 - x_base) ** 2).sum(1).mean()
    print("MSE=", err)  # = 14213.551
