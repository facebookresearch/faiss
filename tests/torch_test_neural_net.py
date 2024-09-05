# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch  # usort: skip
from torch import nn  # usort: skip
import unittest  # usort: skip
import numpy as np  # usort: skip

import faiss  # usort: skip

from faiss.contrib import datasets  # usort: skip
from faiss.contrib.inspect_tools import get_additive_quantizer_codebooks  # usort: skip


class TestLayer(unittest.TestCase):

    @torch.no_grad()
    def test_Embedding(self):
        """ verify that the Faiss Embedding works the same as in Pytorch """
        torch.manual_seed(123)

        emb = nn.Embedding(40, 50)
        idx = torch.randint(40, (25, ))
        ref_batch = emb(idx)

        emb2 = faiss.Embedding(emb)
        idx2 = faiss.Int32Tensor2D(idx[:, None].to(dtype=torch.int32))
        new_batch = emb2(idx2)

        new_batch = new_batch.numpy()
        np.testing.assert_allclose(ref_batch.numpy(), new_batch, atol=2e-6)

    @torch.no_grad()
    def do_test_Linear(self, bias):
        """ verify that the Faiss Linear works the same as in Pytorch """
        torch.manual_seed(123)
        linear = nn.Linear(50, 40, bias=bias)
        x = torch.randn(25, 50)
        ref_y = linear(x)

        linear2 = faiss.Linear(linear)
        x2 = faiss.Tensor2D(x)
        y = linear2(x2)
        np.testing.assert_allclose(ref_y.numpy(), y.numpy(), atol=2e-6)

    def test_Linear(self):
        self.do_test_Linear(True)

    def test_Linear_nobias(self):
        self.do_test_Linear(False)

######################################################
# QINCo Pytorch implementation copied from
# https://github.com/facebookresearch/Qinco/blob/main/model_qinco.py
#
# The implementation is copied here to avoid introducting an additional
# dependency.
######################################################


def pairwise_distances(a, b):
    anorms = (a**2).sum(-1)
    bnorms = (b**2).sum(-1)
    return anorms[:, None] + bnorms - 2 * a @ b.T


def compute_batch_distances(a, b):
    anorms = (a**2).sum(-1)
    bnorms = (b**2).sum(-1)
    return (
        anorms.unsqueeze(-1) + bnorms.unsqueeze(1) - 2 * torch.bmm(a, b.transpose(2, 1))
    )


def assign_batch_multiple(x, zqs):
    bs, d = x.shape
    bs, K, d = zqs.shape

    L2distances = compute_batch_distances(x.unsqueeze(1), zqs).squeeze(1)  # [bs x ksq]
    idx = torch.argmin(L2distances, dim=1).unsqueeze(1)  # [bsx1]
    quantized = torch.gather(zqs, dim=1, index=idx.unsqueeze(-1).repeat(1, 1, d))
    return idx.squeeze(1), quantized.squeeze(1)


def assign_to_codebook(x, c, bs=16384):
    nq, d = x.shape
    nb, d2 = c.shape
    assert d == d2
    if nq * nb < bs * bs:
        # small enough to represent the whole distance table
        dis = pairwise_distances(x, c)
        return dis.argmin(1)

    # otherwise tile computation to avoid OOM
    res = torch.empty((nq,), dtype=torch.int64, device=x.device)
    cnorms = (c**2).sum(1)
    for i in range(0, nq, bs):
        xnorms = (x[i : i + bs] ** 2).sum(1, keepdim=True)
        for j in range(0, nb, bs):
            dis = xnorms + cnorms[j : j + bs] - 2 * x[i : i + bs] @ c[j : j + bs].T
            dmini, imini = dis.min(1)
            if j == 0:
                dmin = dmini
                imin = imini
            else:
                (mask,) = torch.where(dmini < dmin)
                dmin[mask] = dmini[mask]
                imin[mask] = imini[mask] + j
        res[i : i + bs] = imin
    return res


class QINCoStep(nn.Module):
    """
    One quantization step for QINCo.
    Contains the codebook, concatenation block, and residual blocks
    """

    def __init__(self, d, K, L, h):
        nn.Module.__init__(self)

        self.d, self.K, self.L, self.h = d, K, L, h

        self.codebook = nn.Embedding(K, d)
        self.MLPconcat = nn.Linear(2 * d, d)

        self.residual_blocks = []
        for l in range(L):
            residual_block = nn.Sequential(
                nn.Linear(d, h, bias=False), nn.ReLU(), nn.Linear(h, d, bias=False)
            )
            self.add_module(f"residual_block{l}", residual_block)
            self.residual_blocks.append(residual_block)

    def decode(self, xhat, codes):
        zqs = self.codebook(codes)
        cc = torch.concatenate((zqs, xhat), 1)
        zqs = zqs + self.MLPconcat(cc)

        for residual_block in self.residual_blocks:
            zqs = zqs + residual_block(zqs)

        return zqs

    def encode(self, xhat, x):
        # we are trying out the whole codebook
        zqs = self.codebook.weight
        K, d = zqs.shape
        bs, d = xhat.shape

        # repeat so that they are of size bs * K
        zqs_r = zqs.repeat(bs, 1, 1).reshape(bs * K, d)
        xhat_r = xhat.reshape(bs, 1, d).repeat(1, K, 1).reshape(bs * K, d)

        # pass on batch of size bs * K
        cc = torch.concatenate((zqs_r, xhat_r), 1)
        zqs_r = zqs_r + self.MLPconcat(cc)

        for residual_block in self.residual_blocks:
            zqs_r = zqs_r + residual_block(zqs_r)

        # possible next steps
        zqs_r = zqs_r.reshape(bs, K, d) + xhat.reshape(bs, 1, d)
        codes, xhat_next = assign_batch_multiple(x, zqs_r)

        return codes, xhat_next - xhat


class QINCo(nn.Module):
    """
    QINCo quantizer, built from a chain of residual quantization steps
    """

    def __init__(self, d, K, L, M, h):
        nn.Module.__init__(self)

        self.d, self.K, self.L, self.M, self.h = d, K, L, M, h

        self.codebook0 = nn.Embedding(K, d)

        self.steps = []
        for m in range(1, M):
            step = QINCoStep(d, K, L, h)
            self.add_module(f"step{m}", step)
            self.steps.append(step)

    def decode(self, codes):
        xhat = self.codebook0(codes[:, 0])
        for i, step in enumerate(self.steps):
            xhat = xhat + step.decode(xhat, codes[:, i + 1])
        return xhat

    def encode(self, x, code0=None):
        """
        Encode a batch of vectors x to codes of length M.
        If this function is called from IVF-QINCo, codes are 1 index longer,
        due to the first index being the IVF index, and codebook0 is the IVF codebook.
        """
        M = len(self.steps) + 1
        bs, d = x.shape
        codes = torch.zeros(bs, M, dtype=int, device=x.device)

        if code0 is None:
            # at IVF training time, the code0 is fixed (and precomputed)
            code0 = assign_to_codebook(x, self.codebook0.weight)

        codes[:, 0] = code0
        xhat = self.codebook0.weight[code0]

        for i, step in enumerate(self.steps):
            codes[:, i + 1], toadd = step.encode(xhat, x)
            xhat = xhat + toadd

        return codes, xhat


######################################################
# QINCo tests
######################################################

def copy_QINCoStep(step):
    step2 = faiss.QINCoStep(step.d, step.K, step.L, step.h)
    step2.codebook.from_torch(step.codebook)
    step2.MLPconcat.from_torch(step.MLPconcat)

    for l in range(step.L):
        src = step.residual_blocks[l]
        dest = step2.get_residual_block(l)
        dest.linear1.from_torch(src[0])
        dest.linear2.from_torch(src[2])
    return step2


class TestQINCoStep(unittest.TestCase):
    @torch.no_grad()
    def test_decode(self):
        torch.manual_seed(123)
        step = QINCoStep(d=16, K=20, L=2, h=8)

        codes = torch.randint(0, 20, (10, ))
        xhat = torch.randn(10, 16)
        ref_decode = step.decode(xhat, codes)

        # step2 = copy_QINCoStep(step)
        step2 = faiss.QINCoStep(step)
        codes2 = faiss.Int32Tensor2D(codes[:, None].to(dtype=torch.int32))

        np.testing.assert_array_equal(
            step.codebook(codes).numpy(),
            step2.codebook(codes2).numpy()
        )

        xhat2 = faiss.Tensor2D(xhat)
        # xhat2 = faiss.Tensor2D(len(codes), step2.d)

        new_decode = step2.decode(xhat2, codes2)

        np.testing.assert_allclose(
            ref_decode.numpy(),
            new_decode.numpy(),
            atol=2e-6
        )

    @torch.no_grad()
    def test_encode(self):
        torch.manual_seed(123)
        step = QINCoStep(d=16, K=20, L=2, h=8)

        # create plausible x for testing starting from actual codes
        codes = torch.randint(0, 20, (10, ))
        xhat = torch.zeros(10, 16)
        x = step.decode(xhat, codes)
        del codes
        ref_codes, toadd = step.encode(xhat, x)

        step2 = copy_QINCoStep(step)
        xhat2 = faiss.Tensor2D(xhat)
        x2 = faiss.Tensor2D(x)
        toadd2 = faiss.Tensor2D(10, 16)

        new_codes = step2.encode(xhat2, x2, toadd2)

        np.testing.assert_allclose(
            ref_codes.numpy(),
            new_codes.numpy().ravel(),
            atol=2e-6
        )
        np.testing.assert_allclose(toadd.numpy(), toadd2.numpy(), atol=2e-6)



class TestQINCo(unittest.TestCase):

    @torch.no_grad()
    def test_decode(self):
        torch.manual_seed(123)
        qinco = QINCo(d=16, K=20, L=2, M=3, h=8)
        codes = torch.randint(0, 20, (10, 3))
        x_ref = qinco.decode(codes)

        qinco2 = faiss.QINCo(qinco)
        codes2 = faiss.Int32Tensor2D(codes.to(dtype=torch.int32))
        x_new = qinco2.decode(codes2)

        np.testing.assert_allclose(x_ref.numpy(), x_new.numpy(), atol=2e-6)

    @torch.no_grad()
    def test_encode(self):
        torch.manual_seed(123)
        qinco = QINCo(d=16, K=20, L=2, M=3, h=8)
        codes = torch.randint(0, 20, (10, 3))
        x = qinco.decode(codes)
        del codes

        ref_codes, _ = qinco.encode(x)

        qinco2 = faiss.QINCo(qinco)
        x2 = faiss.Tensor2D(x)

        new_codes = qinco2.encode(x2)

        np.testing.assert_allclose(ref_codes.numpy(), new_codes.numpy(), atol=2e-6)


######################################################
# Test index
######################################################

class TestIndexQINCo(unittest.TestCase):

    def test_search(self):
        """
        We can't train qinco with just Faiss so we just train a RQ and use the 
        codebooks in QINCo with L = 0 residual blocks
        """
        ds = datasets.SyntheticDataset(32, 1000, 100, 0)

        # prepare reference quantizer
        M = 5
        index_ref = faiss.index_factory(ds.d, "RQ5x4")
        rq = index_ref.rq
        # rq = faiss.ResidualQuantizer(ds.d, M, 4)
        rq.train_type = faiss.ResidualQuantizer.Train_default
        rq.max_beam_size = 1    # beam search not implemented for QINCo (yet)
        index_ref.train(ds.get_train())
        codebooks = get_additive_quantizer_codebooks(rq)

        # convert to QINCo index
        qinco_index = faiss.IndexQINCo(ds.d, M, 4, 0, ds.d)
        qinco = qinco_index.qinco
        qinco.codebook0.from_array(codebooks[0])
        for i in range(1, qinco.M):
            step = qinco.get_step(i - 1)
            step.codebook.from_array(codebooks[i])
            # MLPConcat left at zero -- it's added to the backbone
        qinco_index.is_trained = True

        # verify that the encoding gives the same results
        ref_codes = rq.compute_codes(ds.get_database())
        ref_decoded = rq.decode(ref_codes)
        new_decoded = qinco_index.sa_decode(ref_codes)
        np.testing.assert_allclose(ref_decoded, new_decoded, atol=2e-6)

        new_codes = qinco_index.sa_encode(ds.get_database())
        np.testing.assert_array_equal(ref_codes, new_codes)

        # verify that search gives the same results
        Dref, Iref = index_ref.search(ds.get_queries(), 5)
        Dnew, Inew = qinco_index.search(ds.get_queries(), 5)

        np.testing.assert_array_equal(Iref, Inew)
        np.testing.assert_allclose(Dref, Dnew, atol=2e-6)
