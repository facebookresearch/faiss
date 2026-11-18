# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""IVF fine search and exact probe selection.

The fine search is checked against an exact FP64 top-3 over the probed lists of the represented operands. Scores
must lie within gamma_{2d+3} * (|c|^2 + 2 |q| . |c|) of the FP64 distance, the a-priori FP32 bound the IVF tests
also use; labels and second IDs must match wherever no other candidate's error interval reaches theirs. Probe
selection must return the nprobe smallest coarse scores, ordered by score, lower list IDs first among equal scores.
"""

import unittest

import torch

from faiss.contrib.e_means import ivf_fine
from faiss.contrib.e_means.ivf_flat import IndexIVFFlat, _dot_fp32


def _data(n, k, nlist, d, seed, ties=False, long_list=False):
    g = torch.Generator().manual_seed(seed)
    coarse = torch.randn(nlist, d, generator=g)
    candidates = torch.randn(k, d, generator=g)
    if long_list:  # half of the candidates land in one list
        candidates[: k // 2] = coarse[0] + 0.05 * torch.randn(k // 2, d, generator=g)
    if ties:  # duplicated candidates: exact score ties between distinct IDs
        candidates[1::2] = candidates[::2]
    rows = torch.randn(n, d, generator=g)
    return coarse, candidates, rows


def _index(coarse, candidates, nprobe, device="cuda", dtype="bf16", implem=4):
    index = IndexIVFFlat(coarse.to(device), nprobe=nprobe)
    index.implem, index.dtype = implem, dtype
    index.add(candidates.to(device))
    return index


def _topk_route(index, rows):
    """The nprobe smallest coarse scores (route's scores, same query chunks), lower list IDs first among equals."""
    dt, xb_t = index._lowp_operands(index.centroids, "coarse")
    chunk = index.query_chunk if index.implem >= 2 else len(rows)
    out = []
    for i0 in range(0, len(rows), max(1, chunk)):
        part = rows[i0:i0 + chunk]
        dis = index._current_routing_norms() - 2 * _dot_fp32(part if dt is None else part.to(dt), xb_t)
        out.append(torch.sort(dis, dim=1, stable=True).indices[:, :index.nprobe])
    return torch.cat(out) if out else torch.empty((0, index.nprobe), dtype=torch.int64, device=rows.device)


def _exact(index, rows, probes):
    """FP64 distances to every stored candidate (inf outside the probed lists), their FP32 error bounds, and the IDs."""
    fine = index._mm_dtype("fine")
    vecs = index.vectors_lowp if index.vectors_lowp is not None else index.vectors
    c = vecs.double()
    q = (rows if fine is None else rows.to(fine)).double()
    lims = index.xb_lims.long()
    owner = torch.repeat_interleave(torch.arange(index.nlist, device=c.device), lims[1:] - lims[:-1])
    probed = (owner[None, None, :] == probes.long()[:, :, None]).any(1)
    norms = c.square().sum(1)
    dis = torch.where(probed, norms[None] - 2 * q @ c.T, torch.inf)
    envelope = norms[None] + 2 * q.abs() @ c.abs().T
    unit = torch.finfo(torch.float32).eps / 2
    operations = 2 * c.shape[1] + 3
    # Relative FP32 rounding plus an absolute term for products that underflow below the normal range.
    bound = operations * unit / (1 - operations * unit) * envelope + operations * 2.0 ** -149
    return dis, bound, index.ids.long()


class _Checks(unittest.TestCase):
    def _assert_exact(self, index, rows, probes, result):
        scores, ids = result
        self.assertEqual(scores.dtype, torch.float32)
        self.assertEqual(ids.dtype, torch.int32)
        dis, bound, cid = _exact(index, rows, probes)
        order = torch.argsort(cid)  # columns by ID, then a stable sort by distance: ties go to the lower ID
        dis, bound, cid = dis[:, order], bound[:, order], cid[order]
        top = torch.sort(dis, dim=1, stable=True).indices[:, :3]
        s, b, i = dis.gather(1, top), bound.gather(1, top), cid[top]
        found = torch.isfinite(s)
        # Missing entries: inf and -1 exactly where the probed lists hold fewer than one / two candidates.
        for col in range(2):
            self.assertTrue(torch.equal(ids[:, col] < 0, ~found[:, col]))
            self.assertTrue(torch.isinf(scores[~found[:, col], col]).all())
        # Every returned score is within the FP32 bound of its own candidate's FP64 distance.
        pos = torch.searchsorted(cid, ids.long().clamp(min=0))
        for col in range(2):
            ok = found[:, col]
            exact = dis.gather(1, pos[:, col:col + 1])[:, 0]
            limit = bound.gather(1, pos[:, col:col + 1])[:, 0]
            self.assertTrue(((scores[ok, col].double() - exact[ok]).abs() <= limit[ok]).all())
        both = (ids[:, 0] >= 0) & (ids[:, 1] >= 0)
        self.assertTrue((ids[both, 0] != ids[both, 1]).all())  # two distinct candidates
        # Labels and second IDs wherever no other candidate's interval [dis - bound, dis + bound] reaches theirs.
        low = dis - bound
        rest = low.scatter(1, top[:, :1], torch.inf).min(1).values
        sure1 = found[:, 0] & (rest > s[:, 0] + b[:, 0])
        self.assertTrue(torch.equal(ids[sure1, 0].long(), i[sure1, 0]))
        rest2 = low.scatter(1, top[:, :2], torch.inf).min(1).values
        sure2 = sure1 & found[:, 1] & (rest2 > s[:, 1] + b[:, 1])
        self.assertTrue(torch.equal(ids[sure2, 1].long(), i[sure2, 1]))
        return sure1.float().mean().item()

    def _case(self, n, k, nlist, d, nprobe, seed=0, device="cuda", dtype="bf16", implem=4, **kwargs):
        coarse, candidates, rows = _data(n, k, nlist, d, seed, **kwargs)
        index = _index(coarse, candidates, nprobe, device, dtype, implem)
        rows = rows.to(device)
        probes = _topk_route(index, rows)
        return index, rows, probes, index.search_2nn(rows, coarse_idx=probes)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestFineSearchCUDA(_Checks):
    def test_exact_reference(self):
        for dtype, implem, d in (("bf16", 4, 16), ("bf16", 4, 64), ("bf16", 4, 96), ("bf16", 4, 128), ("bf16", 4, 1024),
                                 ("fp16", 4, 64), ("bf16", 2, 64), ("bf16", 4, 3), ("bf16", 4, 17), ("bf16", 4, 100)):
            for nprobe in (1, 4, 16):
                with self.subTest(dtype=dtype, implem=implem, d=d, nprobe=nprobe):
                    index, rows, probes, result = self._case(2048, 4096, 64, d, nprobe, seed=d + nprobe,
                                                             dtype=dtype, implem=implem)
                    # The bounds leave few rows undecided (about 2% at d 1024): the label check is not vacuous.
                    self.assertGreater(self._assert_exact(index, rows, probes, result), 0.95)
                    self.assertTrue(torch.equal(index.search_1nn(rows, coarse_idx=probes)[1], result[1][:, 0]))

    def test_kernel_paths(self):
        # The Triton kernel and every CuTe candidate-tile width meet the exact reference.
        for d in (64, 100):
            with self.subTest(d=d):
                index, rows, probes, _ = self._case(4096, 8192, 32, d, 4, seed=d)
                xq, vecs = ivf_fine._operands(index, rows)
                self._assert_exact(index, rows, probes, ivf_fine._search_triton(index, xq, vecs, probes))
                if ivf_fine._cute_ready(torch.cuda.current_device()):
                    for tile in (128, 256):
                        self._assert_exact(index, rows, probes, ivf_fine.search_2nn(index, rows, probes, tile=tile))

    def test_ties_long_lists_missing_and_repeated_probes(self):
        index, rows, probes, result = self._case(2048, 4096, 32, 64, 4, ties=True)
        self._assert_exact(index, rows, probes, result)
        twins = (result[1][:, 0] % 2 == 1) & (result[0][:, 0] == result[0][:, 1])
        self.assertFalse(twins.any())  # equal scores of duplicated candidates: the lower (even) ID first
        index, rows, probes, result = self._case(2048, 8192, 16, 128, 2, long_list=True)
        self._assert_exact(index, rows, probes, result)
        index, rows, probes, result = self._case(512, 48, 32, 32, 1)  # lists with fewer than two candidates
        self._assert_exact(index, rows, probes, result)
        self.assertTrue((result[1][:, 1] == -1).any())
        index, rows, probes, _ = self._case(1024, 2048, 16, 64, 4, seed=5)
        probes[:, 1] = probes[:, 0]  # a list probed twice
        self._assert_exact(index, rows, probes, index.search_2nn(rows, coarse_idx=probes))
        index, rows, probes, result = self._case(2048, 20000, 9000, 32, 8, seed=9)  # several routing tiles
        self._assert_exact(index, rows, probes, result)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires 2 GPUs")
    def test_second_device(self):
        coarse, candidates, rows = _data(2048, 4096, 32, 64, 3)
        index = _index(coarse, candidates, 4, device="cuda:1")
        rows = rows.to("cuda:1")
        probes = _topk_route(index, rows)
        for current in (0, 1):  # launched while either device is current
            with torch.cuda.device(current):
                self.assertTrue(torch.equal(index.route(rows), probes))
                self._assert_exact(index, rows, probes, index.search_2nn(rows))

    def test_empty_inputs_alignment_and_shape_errors(self):
        coarse, candidates, rows = _data(64, 512, 16, 32, 0)
        index = _index(coarse, candidates, 4)
        scores, ids = index.search_2nn(torch.empty((0, 32), device="cuda"))
        self.assertEqual(tuple(scores.shape), (0, 2))
        with self.assertRaises(ValueError):
            index.search_2nn(torch.randn(8, 32, device="cuda"),
                             coarse_idx=torch.zeros((8, 3), dtype=torch.int64, device="cuda"))
        with self.assertRaises(ValueError):  # query width differs from the index
            index.search_2nn(torch.randn(8, 33, device="cuda"),
                             coarse_idx=torch.zeros((8, 4), dtype=torch.int64, device="cuda"))
        # A BF16 view that starts mid-allocation (2-byte offset).
        buffer = torch.empty(64 * 32 + 1, dtype=torch.bfloat16, device="cuda")
        shifted = buffer[1:].view(64, 32)
        shifted.copy_(rows.cuda())
        probes = _topk_route(index, shifted.float())
        self._assert_exact(index, shifted.float(), probes, index.search_2nn(shifted, coarse_idx=probes))
        empty = _index(coarse, torch.empty((0, 32)), 4)  # no candidates anywhere
        scores, ids = empty.search_2nn(rows.cuda(), coarse_idx=probes)
        self.assertTrue(torch.isinf(scores).all() and (ids == -1).all())


class TestFineSearchCPU(_Checks):
    def test_blocks(self):
        # Small blocks split rows and candidates; the running top-2 meets the same exact reference (CPU matmuls of
        # other shapes may round the last bit differently).
        index, rows, probes, result = self._case(256, 2048, 4, 16, 2, device="cpu", seed=8)
        xq, vecs = ivf_fine._operands(index, rows)
        blocked = ivf_fine._search_torch(index, xq, vecs, probes, block=64)
        self._assert_exact(index, rows, probes, blocked)
        torch.testing.assert_close(blocked[0], result[0], rtol=1e-6, atol=1e-4)

    def test_exact_reference(self):
        for dtype, implem in (("bf16", 4), ("fp16", 4), ("bf16", 2)):
            with self.subTest(dtype=dtype, implem=implem):
                index, rows, probes, result = self._case(512, 1024, 16, 24, 3, device="cpu", dtype=dtype, implem=implem)
                self._assert_exact(index, rows, probes, result)
        index, rows, probes, result = self._case(256, 512, 8, 16, 2, device="cpu", ties=True)
        self._assert_exact(index, rows, probes, result)
        probes[:, 1] = probes[:, 0]  # a list probed twice: the second candidate is still another ID
        self._assert_exact(index, rows, probes, index.search_2nn(rows, coarse_idx=probes))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestExactRouting(unittest.TestCase):
    def test_any_nlist_and_nprobe(self):
        for d, nlist, nprobe in ((64, 64, 1), (96, 512, 24), (128, 1024, 16), (1024, 256, 4), (32, 20000, 64),
                                 (16, 70000, 7), (32, 2048, 16), (32, 2049, 16), (16, 4097, 2049), (16, 9000, 5000),
                                 (16, 9000, 9000)):
            with self.subTest(d=d, nlist=nlist, nprobe=nprobe):
                coarse, candidates, rows = _data(8192 if nprobe < 1024 else 256, 4 * nlist, nlist, d, nlist)
                index = _index(coarse, candidates, nprobe)
                rows = rows.cuda()
                self.assertTrue(torch.equal(index.route(rows), _topk_route(index, rows)))

    def test_boundary_ties_take_lower_ids(self):
        coarse = torch.randn(64, 32)
        coarse[32:] = coarse[:32]  # every score appears twice
        index = _index(coarse, torch.randn(256, 32), 5)
        rows = torch.randn(1024, 32, device="cuda")
        self.assertTrue(torch.equal(index.route(rows), _topk_route(index, rows)))  # lower ID first among equals


if __name__ == "__main__":
    unittest.main()
