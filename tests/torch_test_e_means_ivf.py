# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Represented-score oracle and centroid-owner IVF lifecycle checks."""

import datetime
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist

from faiss.contrib.e_means.ivf_flat import IndexIVFFlat, _bucket_sort, _check_int32_count
from faiss.contrib.e_means.sharded_centroids import EmeansShardedCentroids
from faiss.contrib.e_means.sharded_centroids_ivf import EmeansShardedCentroidsIVF


def represented_oracle(queries, candidates):
    operands = candidates.half()
    scores = (operands.float().square().sum(1)[None]
              - 2 * (queries.half().float() @ operands.float().T))
    return scores.min(1)


def source_score_bound(test, scores, queries, candidates):
    # Independent FP64 operands; bound norm, dot and epilogue FP32 rounding.
    q, c = queries.double(), candidates.double()
    reference = c.square().sum(1)[None] - 2 * (q @ c.T)
    labels = reference.argmin(1)
    envelope = c.square().sum(1)[None] + 2 * (q.abs() @ c.abs().T)
    unit = torch.finfo(torch.float32).eps / 2
    operations = 2 * c.shape[1] + 3
    gamma = operations * unit / (1 - operations * unit)
    row = torch.arange(len(q))
    error = (scores.double() - reference[row, labels]).abs()
    test.assertTrue((error <= gamma * envelope[row, labels]).all(),
                    f'FP32 source bound violated: {error}')


class TestIVFFlat(unittest.TestCase):
    def test_finite_1024d_inputs_do_not_overflow_dot_output(self):
        finite_output_regression('cpu')

    def test_all_probes_split_tiles_empty_lists_and_ties(self):
        gen = torch.Generator().manual_seed(4)
        candidates = torch.randn((128, 8), generator=gen)
        candidates[1] = candidates[0]
        queries = torch.cat((candidates[:3], torch.randn((17, 8), generator=gen)))
        index = IndexIVFFlat(torch.zeros((4, 8)), nprobe=4,
                             chunk_budget=32, search_workspace_bytes=4096)
        index.add(candidates)
        self.assertEqual(int((index.list_sizes == 0).sum()), 3)
        scores, ids = index.search_1nn(queries)
        expected_scores, expected_ids = represented_oracle(queries, candidates)
        torch.testing.assert_close(scores, expected_scores, rtol=0, atol=0)
        torch.testing.assert_close(ids.long(), expected_ids)
        self.assertEqual(ids[0].item(), 0)
        self.assertEqual(ids[1].item(), 0)

    def test_representation_and_routing_can_change_winners(self):
        candidates = torch.tensor([[1.0002, 0], [1.0, 0]])
        query = torch.tensor([[1.0, 0]])
        index = IndexIVFFlat(torch.zeros((2, 2)), nprobe=2)
        index.add(candidates)
        self.assertEqual(index.search_1nn(query)[1].item(), 0)
        self.assertEqual((query[:, None] - candidates).square().sum(2).argmin(1).item(), 1)
        index = IndexIVFFlat(torch.tensor([[0.0, 0], [100.0, 0]]), nprobe=1)
        candidates = torch.tensor([[40.0, 0], [100.0, 0]])
        query = torch.tensor([[60.0, 0]])
        index.add(candidates)
        self.assertEqual(index.search_1nn(query)[1].item(), 1)
        self.assertEqual(represented_oracle(query, candidates)[1].item(), 0)

    def test_global_route_can_have_no_local_candidate(self):
        index = IndexIVFFlat(torch.tensor([[0.0, 0], [100.0, 0]]), nprobe=1)
        index.add(torch.tensor([[0.0, 0]]))
        index.set_routing_nonempty(torch.ones(2, dtype=torch.bool))
        query = torch.tensor([[100.0, 0]])
        probes = index.route(query)
        self.assertEqual(probes.item(), 1)
        scores, ids = index.search_1nn(query, coarse_idx=probes)
        self.assertTrue(torch.isinf(scores).all())
        self.assertEqual(ids.item(), -1)

    def test_original_alias_empty_queries_and_workspace_failure(self):
        # Reject unsupported composition before superclass/group work, even
        # when no process group exists, in both accepted core call forms.
        positional = (1, 1, 1, None, None, 'throughput', 1234, False,
                      None, 4096, None, 'native', False, True)
        with patch.object(EmeansShardedCentroids, '__init__',
                          side_effect=AssertionError('entered group constructor')) as base:
            for args, kwargs in (((2, 4), dict(spherical=True)), (positional, {})):
                with self.subTest(positional=len(args) == 14):
                    with self.assertRaisesRegex(ValueError, 'spherical=True'):
                        EmeansShardedCentroidsIVF(*args, group=None, **kwargs)
            base.assert_not_called()
        ordinary = positional[:-1] + (False,)
        with patch.object(EmeansShardedCentroids, '__init__',
                          side_effect=RuntimeError('ordinary forwarding')) as base:
            with self.assertRaisesRegex(RuntimeError, 'ordinary forwarding'):
                EmeansShardedCentroidsIVF(*ordinary, group=None)
            base.assert_called_once_with(*ordinary, group=None)
        coarse = torch.zeros((1, 2))
        index = IndexIVFFlat(coarse, nprobe=1)
        coarse.fill_(100)
        index.add(torch.tensor([[0.0, 0], [1.0, 0]], requires_grad=True))
        scores, ids = index.search_1nn(torch.tensor([[0.0, 0]], requires_grad=True))
        self.assertTrue(scores.requires_grad)
        self.assertEqual(ids.item(), 0)
        self.assertEqual(index.centroids.data_ptr(), coarse.data_ptr())
        self.assertEqual(index.centroids.sum().item(), 200)
        self.assertEqual(len(index.search_1nn(torch.empty((0, 2)))[1]), 0)
        index.search_workspace_bytes = 1
        with self.assertRaisesRegex(ValueError, 'one candidate'):
            index.search_1nn(torch.zeros((1, 2)))
        index.search_workspace_bytes = 4096
        index.add(torch.zeros((1, 2), dtype=torch.uint8))
        self.assertEqual(index.search_1nn(torch.zeros((1, 2)))[1].item(), 0)

    def test_all_original_precision_storage_and_entrypoint_branches(self):
        gen = torch.Generator().manual_seed(18)
        candidates = torch.randn((19, 8), generator=gen)
        queries = torch.randn((7, 8), generator=gen)
        for implem in (1, 2, 3, 4, 5):
            for dtype_name, dtype in (('fp16', torch.float16), ('bf16', torch.bfloat16)):
                for scope in ('all', 'fine'):
                    with self.subTest(implem=implem, dtype=dtype_name, scope=scope):
                        index = IndexIVFFlat(torch.zeros((2, 8)), nprobe=2, query_chunk=2)
                        index.implem, index.dtype, index.lowp_scope = implem, dtype_name, scope
                        fine = None if implem < 3 else dtype
                        coarse = None if implem < 3 or scope != 'all' else dtype
                        self.assertEqual(index._mm_dtype('fine'), fine)
                        self.assertEqual(index._mm_dtype('coarse'), coarse)
                        represented = candidates if fine is None else candidates.to(fine).float()
                        represented_q = queries if fine is None else queries.to(fine).float()
                        expected = (represented.double().square().sum(1)[None]
                                    - 2 * (represented_q.double() @ represented.double().T))
                        direct = index._exact_1nn(queries, candidates)
                        topk = index._exact_topk(queries, candidates, 2)
                        torch.testing.assert_close(direct[1], expected.argmin(1))
                        source_score_bound(self, direct[0], represented_q, represented)
                        torch.testing.assert_close(topk[1], expected.topk(2, largest=False).indices)
                        coarse_c = candidates if coarse is None else candidates.to(coarse).float()
                        coarse_q = queries if coarse is None else queries.to(coarse).float()
                        coarse_expected = (coarse_c.double().square().sum(1)[None]
                                           - 2 * (coarse_q.double() @ coarse_c.double().T))
                        coarse_result = index._exact_1nn(queries, candidates, stage='coarse')
                        coarse_topk = index._exact_topk(queries, candidates, 2, stage='coarse')
                        torch.testing.assert_close(coarse_result[1], coarse_expected.argmin(1))
                        source_score_bound(self, coarse_result[0], coarse_q, coarse_c)
                        torch.testing.assert_close(coarse_topk[1], coarse_expected.topk(2, largest=False).indices)
                        index.add(candidates)
                        self.assertEqual(index.vectors is None, implem >= 4 and fine is not None)
                        self.assertEqual(index.vectors_lowp is None, fine is None)
                        actual = index.search_1nn_batched(queries, False)
                        torch.testing.assert_close(actual[1].long(), expected.argmin(1))
                        source_score_bound(self, actual[0], represented_q, represented)
                        torch.testing.assert_close(index._sq_norms(candidates, 'fine').double(), represented.double().square().sum(1), rtol=2e-6, atol=3e-6)

    def test_precision_selectors_reject_unsupported_values(self):
        index = IndexIVFFlat(torch.zeros((2, 2)), nprobe=2)
        index.implem = np.int64(3)
        self.assertIs(type(index.implem), int)
        index.implem = 4
        invalid = {
            'implem': (0, 6, True, 3.0, '4', None),
            'dtype': ('native', 'float16', torch.float16, None, 1),
            'lowp_scope': ('coarse', 'both', '', None, 1),
        }
        for name, values in invalid.items():
            original = getattr(index, name)
            for value in values:
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        setattr(index, name, value)
                    self.assertEqual(getattr(index, name), original)

    def test_fp16_to_bf16_coarse_norm_counterexample(self):
        coarse = torch.tensor([[1.001], [1.008]])
        query = torch.tensor([[1.]])
        index = IndexIVFFlat(coarse, nprobe=1)
        fixed_membership_norms = index.centroids_norms.clone()
        index.dtype = 'bf16'
        index.set_routing_nonempty(torch.ones(2, dtype=torch.bool))

        represented = coarse.bfloat16().float()
        current_norms = represented.square().sum(1)
        dot = query.bfloat16().float() @ represented.T
        stale_scores = fixed_membership_norms[None] - 2 * dot
        coherent_scores = current_norms[None] - 2 * dot
        self.assertEqual(stale_scores.argmin(1).item(), 1)
        self.assertEqual(coherent_scores.argmin(1).item(), 0)
        self.assertEqual(index.route(query).item(), 0)
        torch.testing.assert_close(index._query_centroids_norms, current_norms,
                                   rtol=0, atol=0)
        torch.testing.assert_close(index.centroids_norms, fixed_membership_norms,
                                   rtol=0, atol=0)

    def test_cpu_add_policy_and_occupancy_survive_mutation(self):
        coarse = torch.tensor([[1.001], [1.008]])
        index = IndexIVFFlat(coarse, nprobe=1)
        index.dtype = 'bf16'
        index.add(torch.tensor([[1.]]))
        torch.testing.assert_close(index.list_sizes,
                                   torch.tensor([1, 0], dtype=torch.int32))
        original_ids = index.ids.clone()
        original_sizes = index.list_sizes.clone()

        index.dtype = 'fp16'
        torch.testing.assert_close(index.ids, original_ids)
        torch.testing.assert_close(index.list_sizes, original_sizes)
        self.assertEqual(index.route(torch.tensor([[1.008]])).item(), 0)
        self.assertTrue(torch.isinf(index.routing_norms[1]))

    def test_every_supported_precision_transition_and_rebuild(self):
        states = [(implem, dtype, scope)
                  for implem in range(1, 6)
                  for dtype in ('fp16', 'bf16')
                  for scope in ('all', 'fine')]
        coarse = torch.tensor([[0., 0.], [4., 0.]])
        candidates = torch.tensor([[0., 0.], [4., 0.], [2., 1.]])
        queries = torch.tensor([[.25, 0.], [3.75, 0.]])
        probes = torch.tensor([[0, 1], [0, 1]])

        def policy_dtype(state, stage):
            implem, dtype, scope = state
            if implem < 3 or (stage == 'coarse' and scope != 'all'):
                return None
            return torch.float16 if dtype == 'fp16' else torch.bfloat16

        def represented(value, dtype):
            return value if dtype is None else value.to(dtype).float()

        for before in states:
            for after in states:
                with self.subTest(before=before, after=after):
                    index = IndexIVFFlat(coarse, nprobe=2, query_chunk=1)
                    index.implem, index.dtype, index.lowp_scope = before
                    index.add(candidates)
                    original_ids = index.ids.clone()
                    original_sizes = index.list_sizes.clone()

                    index.implem, index.dtype, index.lowp_scope = after
                    torch.testing.assert_close(index.ids, original_ids)
                    torch.testing.assert_close(index.list_sizes, original_sizes)
                    coarse_dtype = policy_dtype(after, 'coarse')
                    cq = represented(queries, coarse_dtype)
                    cc = represented(coarse, coarse_dtype)
                    coarse_scores = (cc.double().square().sum(1)[None]
                                     - 2 * (cq.double() @ cc.double().T))
                    torch.testing.assert_close(index.route(queries),
                                               coarse_scores.topk(2, largest=False).indices)

                    fine_dtype = policy_dtype(after, 'fine')
                    fq = represented(queries, fine_dtype)
                    fc = represented(candidates, fine_dtype)
                    fine_scores = (fc.double().square().sum(1)[None]
                                   - 2 * (fq.double() @ fc.double().T))
                    compatible = (policy_dtype(before, 'fine') == fine_dtype)
                    if not compatible:
                        with self.assertRaisesRegex(RuntimeError, 'call add again'):
                            index.search_1nn(queries, coarse_idx=probes)
                        index.add(candidates)
                    scores, ids = index.search_1nn(queries, coarse_idx=probes)
                    torch.testing.assert_close(ids.long(), fine_scores.argmin(1))
                    source_score_bound(self, scores, fq, fc)

    def test_original_stage_policy_supplied_norms_and_cache_timing(self):
        coarse = torch.tensor([[1.003, .127], [2.009, -.211]])
        query = torch.tensor([[1.019, .193]])
        index = IndexIVFFlat(coarse, nprobe=2)
        cached = index.centroids_norms.clone()
        index.dtype, index.lowp_scope = 'bf16', 'fine'
        torch.testing.assert_close(index.centroids_norms, cached, rtol=0, atol=0)
        self.assertIs(index._as_computed(coarse, 'coarse'), coarse)
        torch.testing.assert_close(index._as_computed(coarse, 'fine'), coarse.bfloat16().float(), rtol=0, atol=0)
        index.lowp_scope = 'all'
        supplied = torch.tensor([8., 1.])
        scores, ids = index._exact_1nn(query, coarse, supplied, 'coarse')
        expected = (supplied.double()
                    - 2 * (query.bfloat16().double()
                           @ coarse.bfloat16().double().T)[0])
        self.assertEqual(ids.item(), expected.argmin().item())
        torch.testing.assert_close(scores.double(), expected.min()[None], rtol=2e-6, atol=3e-6)

    def test_original_floating_training_inputs_reach_inherited_preparation(self):
        model = object.__new__(EmeansShardedCentroidsIVF)
        model.d, model.quantizer, model.assignment_backend = 8, None, 'native'
        model.assignment_chunk_size = None
        model._centroids, model.centroids = torch.zeros((2, 8)), None
        model._batch_stats = torch.zeros((2, 9))
        model._fp8_quantizer = model._int8_quantizer = model.spherical = False
        model.recipe = {}
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            model._input_dtype = None
            prepared = model._prepare_input(torch.ones((3, 8), dtype=dtype))
            torch.testing.assert_close(prepared, torch.ones((3, 8)))

    def test_original_hooks_and_full_vs_chunked_queries(self):
        from contextlib import contextmanager
        class Observed(IndexIVFFlat):
            events = []
            @contextmanager
            def _timed(self, name):
                self.events.append(name)
                yield
            def _chunk_body(self, kind):
                body = super()._chunk_body(kind)
                def observed(rows, *args):
                    self.events.append((kind, len(rows)))
                    return body(rows, *args)
                return observed
        index = Observed(torch.zeros((2, 8)), nprobe=2, query_chunk=2)
        rows = torch.zeros((5, 8))
        index.add(rows)
        self.assertIn('add', index.events)
        self.assertEqual(index.events.count(('assign', 2)), 2)
        index.events.clear()
        index.implem = 1
        index.route(rows)
        self.assertEqual(index.events, ['coarse', ('topk', 5)])
        index.events.clear()
        index.implem = 2
        index.route(rows)
        self.assertEqual(index.events, ['coarse', ('topk', 2), ('topk', 2), ('topk', 1)])

    def test_original_empty_add_conversions_and_real_budgets(self):
        index = IndexIVFFlat(torch.zeros((2, 8), dtype=torch.bfloat16), nprobe=2,
                             group_size=2.5, chunk_budget=4096.5,
                             search_workspace_bytes=8192.5)
        self.assertEqual(index.centroids.dtype, torch.float32)
        index.add(torch.empty((0, 8)))
        score, labels = index.search_1nn(torch.zeros((3, 8), dtype=torch.float16))
        self.assertTrue(torch.isinf(score).all())
        self.assertEqual(labels.tolist(), [-1, -1, -1])
        self.assertEqual(index.coarse_ndis, 6)
        index.add(torch.zeros((17, 8), dtype=torch.bfloat16))
        probes = torch.tensor([[0, 1]], dtype=torch.int32)
        score, labels = index.search_1nn(torch.zeros((1, 8)), coarse_idx=probes)
        self.assertEqual(labels.item(), 0)
        self.assertEqual(index.coarse_ndis, 40)
        self.assertEqual(index.ndis.item(), 17)
        index.chunk_budget = 0
        index.search_workspace_bytes = None
        self.assertEqual(index.search_1nn(torch.zeros((1, 8)))[1].item(), 0)
        self.assertEqual(index.coarse_ndis, 42)

    def test_grouped_planner_padding_and_int32_boundaries(self):
        index = IndexIVFFlat(torch.zeros((1, 8)), nprobe=1,
                             chunk_budget=32, search_workspace_bytes=4096)
        index.add(torch.zeros((33, 8)))
        for grouped in (False, True):
            chunks, _, _ = index.list_chunks(torch.tensor([0, 7]).numpy(), grouped=grouped)
            coverage = torch.zeros((7, 33), dtype=torch.int32)
            for _group, qstart, qlength, bstart, blength in chunks:
                for q, nq, b, nb in zip(qstart, qlength, bstart, blength):
                    coverage[q:q+nq, b:b+nb] += 1
                    padded = (nb + 15) // 16 * 16 if grouped else nb
                    self.assertLessEqual(padded * nq, index.chunk_budget)
            self.assertTrue((coverage == 1).all())
        _check_int32_count(2**31 - 1, 'fixture')
        with self.assertRaisesRegex(ValueError, 'INT32'):
            _check_int32_count(2**31, 'fixture')
        class TooMany:
            ndim = 1
            def numel(self):
                return 2**31
        with self.assertRaisesRegex(ValueError, 'INT32'):
            _bucket_sort(TooMany(), 1)

    def test_global_occupancy_over_int32_keeps_a_live_list(self):
        class Prepared:
            list_sizes = torch.tensor([2**31 - 1, 0], dtype=torch.int32)
            def add(self, rows):
                pass
            def set_routing_nonempty(self, nonempty):
                self.nonempty = nonempty
        model = object.__new__(EmeansShardedCentroidsIVF)
        model._ivf, model._centroids, model._group = Prepared(), torch.zeros((1, 8)), None
        observed = []
        def reduce(occupancy, group):
            observed.append(occupancy.dtype)
            occupancy.add_(torch.tensor([1, 0], dtype=torch.int64))
        with patch.object(dist, 'all_reduce', side_effect=reduce):
            result = model._prepare_assignment()
        self.assertEqual(observed, [torch.int64])
        self.assertEqual(result.nonempty.tolist(), [True, False])

    def test_bf16_dot_rounding_and_native_float64_computation(self):
        index = IndexIVFFlat(torch.zeros((1, 8)), nprobe=1)
        index.dtype = 'bf16'
        candidates = torch.full((2, 32), 1.0078125)
        candidates[1, 0] = 1.015625
        query = torch.full((1, 32), 1.0078125)
        expected = (candidates.double().square().sum(1)
                    - 2 * (query.double() @ candidates.double().T)[0])
        self.assertFalse(torch.equal((query.bfloat16() @ candidates.bfloat16().T).float(),
                                     (query.double() @ candidates.double().T).float()))
        scores, ids = index._exact_1nn(query, candidates)
        self.assertEqual(ids.item(), expected.argmin().item())
        torch.testing.assert_close(scores.double(), expected.min()[None], rtol=0, atol=0)
        index.implem = 1
        native_q = torch.tensor([[1. + 2**-30, 2.]], dtype=torch.float64)
        native_c = torch.tensor([[3., 4.], [5., 6.]], dtype=torch.float64)
        supplied = torch.tensor([.1, .2], dtype=torch.float64)
        expected_native = supplied - 2 * (native_q @ native_c.T).float()[0]
        scores, ids = index._exact_1nn(native_q, native_c, supplied)
        self.assertEqual(scores.dtype, torch.float64)
        self.assertEqual(ids.item(), expected_native.argmin().item())
        torch.testing.assert_close(scores, expected_native.min()[None], rtol=0, atol=0)

    def test_original_package_and_module_import_identity_optional_torch(self):
        from pathlib import Path
        source = Path(__file__).resolve().parents[1] / 'contrib'
        script = """
import importlib, sys
import faiss.contrib
faiss.contrib.__path__.insert(0, sys.argv[1])
order = sys.argv[2].split(',')
modules = [importlib.import_module('faiss.contrib.e_means.' + name) for name in order]
from faiss.contrib.e_means import EmeansShardedCentroids, EmeansShardedCentroidsIVF
from faiss.contrib.e_means.sharded_centroids import EmeansShardedCentroidsIVF as old
from faiss.contrib.e_means.sharded_centroids_ivf import EmeansShardedCentroidsIVF as new
assert old is new is EmeansShardedCentroidsIVF
assert EmeansShardedCentroids is modules[order.index('sharded_centroids')].EmeansShardedCentroids
"""
        for order in ('sharded_centroids,sharded_centroids_ivf', 'sharded_centroids_ivf,sharded_centroids'):
            subprocess.run([sys.executable, '-c', script, str(source), order], check=True, timeout=30)
        optional = """
import importlib.abc, sys
import faiss.contrib
faiss.contrib.__path__.insert(0, sys.argv[1])
class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ImportError('intentional optional-Torch fixture')
sys.meta_path.insert(0, BlockTorch())
import faiss.contrib.e_means as package
assert 'torch' not in sys.modules
assert set(package.__all__) == {'Emeans', 'DistributedEmeans', 'EmeansShardedCentroids', 'EmeansShardedCentroidsIVF'}
try:
    package.EmeansShardedCentroidsIVF
except ImportError:
    pass
else:
    raise AssertionError('missing optional Torch was hidden')
"""
        subprocess.run([sys.executable, '-c', optional, str(source)], check=True, timeout=30)

    def test_auto_keeps_ivf_auxiliary_backend_and_chunk_bound(self):
        # Metadata only: IVF keeps its own engine, without group/GPU execution.
        from faiss.contrib.e_means import Emeans

        native = Emeans(128, 4096, assignment_backend='native')
        model = EmeansShardedCentroidsIVF.__new__(EmeansShardedCentroidsIVF)
        model.d, model.k, model.centroid_chunk_size = native.d, native.k, native.centroid_chunk_size
        model.assignment_chunk_size = native.assignment_chunk_size
        for requested, backend in (('auto', 'native'), ('native', 'native'), ('triton', 'triton')):
            control = Emeans(native.d, native.k, assignment_backend=backend,
                             centroid_chunk_size=native.centroid_chunk_size)
            model.assignment_backend = requested
            model._resolved_assignment_backend = None if requested == 'auto' else requested
            model._resolve_assignment_backend(None)
            self.assertEqual(model.assignment_backend, requested)
            self.assertEqual(model._assignment_backend, backend)
            self.assertEqual(model._get_assignment_chunk_size(True, True),
                             control._get_assignment_chunk_size(True, True))
        model.assignment_backend, model.assignment_chunk_size = 'auto', 23
        self.assertEqual(model._get_assignment_chunk_size(True, True), 23)


def finite_output_regression(device):
    candidates = torch.stack((torch.full((1024,), 31.), torch.full((1024,), 32.))).to(device)
    query = candidates[1:]
    # One list reproduces Root's all-probe wrong-ID witness; two real coarse
    # rows also exercise the 2D assignment/top-k call sites, not only BMM.
    for coarse in (torch.zeros((1, 1024), device=device), candidates.clone()):
        index = IndexIVFFlat(coarse, nprobe=len(coarse), query_chunk=1)
        index.add(candidates)
        scores, ids = index.search_1nn(query)
        assert torch.isfinite(scores).all()
        torch.testing.assert_close(ids.long(), torch.tensor([1], device=device))
        torch.testing.assert_close(scores, torch.tensor([-1048576.], device=device), rtol=0, atol=0)
        direct = (query.double()[:, None] - candidates.double()).square().sum(2)
        torch.testing.assert_close(ids.long(), direct.argmin(1))
        if len(coarse) == 2:
            torch.testing.assert_close(index.list_sizes.long(), torch.ones(2, dtype=torch.int64, device=device))
            assert index.route(query)[0, 0].item() == 1


class TestIVFFlatCUDA(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec('triton'),
                         'requires CUDA and Triton')
    def test_cuda_fp32_mm_and_bmm_outputs(self):
        finite_output_regression('cuda:0')

    @unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec('triton'),
                         'requires CUDA and Triton')
    def test_original_grouped_benign_ragged_empty_and_ties(self):
        # Exactly represented small coordinates: independent CPU FP64 oracle.
        candidates_cpu = torch.zeros((5, 16))
        candidates_cpu[:, 0] = torch.tensor([0., 0., 8., 9., 16.])
        coarse_cpu = torch.zeros((4, 16))
        coarse_cpu[:, 0] = torch.tensor([0., 8., 16., 24.])
        queries_cpu = torch.zeros((7, 16))
        queries_cpu[:, 0] = torch.tensor([0., .25, -.25, 8., 9., 16., 24.])
        probes_cpu = torch.tensor([[0], [0], [0], [1], [1], [2], [3]])
        reference_scores = torch.full((7,), torch.inf, dtype=torch.float64)
        reference_ids = torch.full((7,), -1, dtype=torch.int64)
        candidate_lists = ((0, 1), (2, 3), (4,), ())
        for row, probe in enumerate(probes_cpu[:, 0].tolist()):
            ids = candidate_lists[probe]
            if ids:
                c = candidates_cpu[list(ids)].double()
                score = c.square().sum(1) - 2 * (c * queries_cpu[row].double()).sum(1)
                winner = score.argmin().item()
                reference_scores[row], reference_ids[row] = score[winner], ids[winner]
        for label, dtype, implem in (('fp16', 'fp16', 4),
                                     ('bf16', 'bf16', 4),
                                     ('native', 'fp16', 2)):
            if dtype == 'bf16' and torch.cuda.get_device_capability() < (8, 0):
                continue
            with self.subTest(dtype=label):
                index = IndexIVFFlat(coarse_cpu.cuda(), nprobe=1,
                                     query_chunk=8, search_workspace_bytes=1 << 20)
                index.dtype, index.implem = dtype, implem
                index.add(candidates_cpu.cuda())
                self.assertEqual(index.list_sizes.tolist(), [2, 2, 1, 0])
                query, probes = queries_cpu.cuda(), probes_cpu.cuda()
                for grouped in (False, True):
                    scores, ids = index.search_1nn_batched(query, grouped, probes)
                    torch.testing.assert_close(scores.double().cpu(), reference_scores,
                                               rtol=0, atol=0)
                    torch.testing.assert_close(ids.long().cpu(), reference_ids,
                                               rtol=0, atol=0)


def sharded_contract(rank, rendezvous):
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=90))
    try:
        device = torch.device('cuda', rank)
        initial = torch.tensor([[0., 0], [0., 0], [8., 0], [16., 0]], device=device)
        rows = torch.tensor([[0., 0], [1., 0], [8., 0], [9., 0], [16., 0],
                             [0., 0], [2., 0], [7., 0], [15., 0], [16., 0]], device=device)
        local = rows[rank * 5:(rank + 1) * 5]
        model = EmeansShardedCentroidsIVF(
            2, 4, group=dist.group.WORLD, niter=1, batch_size=8, alpha0=.25,
            nlist=3, nprobe=3, coarse_niter=1, assignment_chunk_size=2,
            search_workspace_bytes=4096, capture_assignments=True, seed=7)
        model.begin(10, initial[rank * 2:(rank + 1) * 2])
        for start, end in ((0, 4), (4, 5)):
            before = model.current_centroids().clone()
            shards = [torch.empty_like(before) for _ in range(2)]
            dist.all_gather(shards, before)
            full = torch.cat(shards)
            all_rows = torch.cat((rows[start:end], rows[5 + start:5 + end]))
            winners = represented_oracle(all_rows, full)[1]
            expected_sums = torch.zeros_like(full)
            expected_counts = torch.zeros(4, device=device)
            expected_sums.index_add_(0, winners, all_rows)
            expected_counts.index_add_(0, winners, torch.ones(len(winners), device=device))
            expected_objective = (all_rows - full[winners]).square().sum(dtype=torch.float64)
            snapshot = None
            for offset in range(start, end, 2):
                stats = model.accumulate(local[offset:min(offset + 2, end)])
                if snapshot is None:
                    snapshot = model._ivf.vectors_lowp
                assert model._ivf.vectors_lowp is snapshot
                torch.testing.assert_close(model.current_centroids(), before, rtol=0, atol=0)
                chunk_labels = winners[rank * (end - start) + offset - start:
                                       rank * (end - start) + min(offset + 2, end) - start]
                torch.testing.assert_close(stats['labels'], chunk_labels)
            torch.testing.assert_close(stats['sums'], expected_sums[rank * 2:(rank + 1) * 2], rtol=0, atol=0)
            torch.testing.assert_close(stats['counts'], expected_counts[rank * 2:(rank + 1) * 2], rtol=0, atol=0)
            objective = stats['objective'].clone()
            dist.all_reduce(objective)
            torch.testing.assert_close(objective, expected_objective, rtol=0, atol=0)
            model.update(stats)
        model.end_epoch()
        result = model.finalize()
        assert torch.isfinite(result).all() and model._ivf is not None
        model.reset()
        assert model._ivf is None
    finally:
        dist.destroy_process_group()


class TestIVFShardedCUDA(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.device_count() >= 2 and dist.is_nccl_available()
                         and importlib.util.find_spec('triton'), 'requires two CUDA GPUs, NCCL and Triton')
    def test_original_fp32_statistics_and_frozen_transfer_snapshot(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(sharded_contract, args=('file://' + directory + '/group',),
                     nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
