# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Focused optional-import, grouping, capture and storage-ownership checks."""

import gc
import importlib.util
import importlib.metadata
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock
import weakref
import warnings

import torch

PATH = Path(importlib.util.find_spec(
    'faiss.contrib.e_means.e_means_grouping').origin).resolve()
spec = importlib.util.spec_from_file_location('owned_e_means_grouping', PATH)
grouping = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grouping)


class TestGroupingCPU(unittest.TestCase):
    def test_import_without_torch_numpy_or_cuda(self):
        code = '''import builtins, importlib.util, sys, warnings
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'numpy', 'cuda'):
        raise AssertionError('eager optional import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
with warnings.catch_warnings(record=True) as seen:
    warnings.simplefilter('always')
    spec = importlib.util.spec_from_file_location('isolated_grouping', sys.argv[1])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
assert not seen
'''
        subprocess.run([sys.executable, '-c', code, str(PATH)], check=True)

    def test_provider_present_absent_warn_once_and_broken_propagates(self):
        def reset():
            grouping._grouping_provider.cache_clear()
            grouping._provider_warning_emitted = False

        try:
            reset()
            present = mock.Mock(CountingIterator=object(), SortOrder=object(),
                                make_lower_bound=object(), make_radix_sort=object())
            with mock.patch.object(grouping.importlib, 'import_module', return_value=present) as imported, \
                 mock.patch.object(grouping.importlib.metadata, 'version') as version, \
                 warnings.catch_warnings(record=True) as seen:
                warnings.simplefilter('always')
                self.assertEqual(grouping.selected_grouping_provider(warn=True), 'cuda.compute')
                self.assertEqual(grouping.selected_grouping_provider(warn=True), 'cuda.compute')
                imported.assert_called_once_with('cuda.compute')
                version.assert_not_called()
                self.assertFalse(seen)

            reset()
            with mock.patch.object(grouping.importlib, 'import_module', return_value=object()), \
                 self.assertRaisesRegex(ImportError, 'missing required grouping API.*tested 1.2.0'):
                grouping.selected_grouping_provider(warn=True)

            reset()
            missing = ModuleNotFoundError('no cuda.compute', name='cuda.compute')
            with mock.patch.object(grouping.importlib, 'import_module', side_effect=missing), \
                 mock.patch.object(grouping.importlib.metadata, 'version',
                                   side_effect=importlib.metadata.PackageNotFoundError), \
                 warnings.catch_warnings(record=True) as seen:
                warnings.simplefilter('always')
                self.assertEqual(grouping.selected_grouping_provider(warn=True), 'torch')
                self.assertEqual(grouping.selected_grouping_provider(warn=True), 'torch')
                self.assertEqual(len(seen), 1)
                self.assertIs(seen[0].category, RuntimeWarning)
                self.assertIn('~0.6–2.4%', str(seen[0].message))
                self.assertIn('Install cuda-cccl', str(seen[0].message))

            reset()
            with mock.patch.object(grouping.importlib, 'import_module', side_effect=missing), \
                 mock.patch.object(grouping.importlib.metadata, 'version', return_value='1.2.0'), \
                 self.assertRaisesRegex(ImportError, 'cuda-cccl 1.2.0.*tested 1.2.0') as caught:
                grouping.selected_grouping_provider(warn=True)
            self.assertIs(caught.exception.__cause__, missing)

            reset()
            broken = ModuleNotFoundError('broken CCCL child import', name='cccl.bindings')
            with mock.patch.object(grouping.importlib, 'import_module', side_effect=broken), \
                 mock.patch.object(grouping.importlib.metadata, 'version') as version, \
                 self.assertRaises(ModuleNotFoundError) as caught:
                grouping.selected_grouping_provider(warn=True)
            self.assertIs(caught.exception, broken)
            version.assert_not_called()

            reset()
            broken = ImportError('installed CCCL ABI failure')
            with mock.patch.object(grouping.importlib, 'import_module', side_effect=broken), \
                 self.assertRaises(ImportError) as caught:
                grouping.selected_grouping_provider(warn=True)
            self.assertIs(caught.exception, broken)
        finally:
            reset()

    def test_host_group_count_and_cpu_input_guards(self):
        keys = torch.zeros(4, dtype=torch.int32)
        with mock.patch.dict(sys.modules, {'cuda.compute': None}):
            for count in (True, 1.5, torch.tensor(2)):
                with self.assertRaises(TypeError):
                    grouping.prepare_grouping(keys, count)
            for count in (0, -1, 2**31):
                with self.assertRaises(ValueError):
                    grouping.prepare_grouping(keys, count)
            for values in (keys, keys.long(), keys.reshape(2, 2), keys[:0]):
                with self.assertRaises(ValueError):
                    grouping.prepare_grouping(values, 2)
        # A cached descriptor must expose the live buffer without owning it.
        tensor = torch.zeros(4, dtype=torch.int32)
        owners = (tensor,)
        ref = weakref.ref(tensor)
        descriptor = grouping._array(tensor)
        pointer = tensor.data_ptr()
        self.assertEqual(descriptor.data_ptr(), pointer)
        self.assertEqual(descriptor.__cuda_array_interface__['data'][0], pointer)
        self.assertEqual(descriptor.__cuda_array_interface__['shape'], tuple(tensor.shape))
        del tensor
        self.assertIsNotNone(ref())
        del owners
        gc.collect()
        self.assertIsNone(ref())


@unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
class TestGroupingCUDA(unittest.TestCase):
    def check_reference(self, handle, values, groups, device):
        order = handle.order.cpu().tolist()
        expected_offsets = [sum(value < group for value in values) for group in range(groups + 1)]
        self.assertEqual(sorted(order), list(range(len(values))))
        ordered_keys = [values[index] for index in order]
        self.assertTrue(all(left <= right for left, right in zip(ordered_keys, ordered_keys[1:])))
        self.assertEqual(handle.offsets.cpu().tolist(), expected_offsets)
        self.assertEqual(handle.order.dtype, torch.int32)
        self.assertEqual(handle.offsets.dtype, torch.uint64)
        self.assertEqual(tuple(handle.order.shape), (len(values),))
        self.assertEqual(tuple(handle.offsets.shape), (groups + 1,))
        self.assertEqual(handle.order.device, device)
        self.assertEqual(handle.offsets.device, device)
        self.assertTrue(handle.order.is_contiguous() and handle.offsets.is_contiguous())

    def test_ties_empty_groups_current_stream_and_accounting(self):
        values, groups = [2, 0, 2, 3, 0], 5
        backing = torch.tensor([99] + values + [99], dtype=torch.int32, device='cuda')
        keys = backing[1:-1]
        handle = grouping.prepare_grouping(keys, groups)
        self.assertEqual(handle.provider, grouping.selected_grouping_provider())
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            order, offsets = grouping.launch_grouping(handle)
        stream.synchronize()
        for actual, expected in ((order, handle.order), (offsets, handle.offsets)):
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertEqual(actual.device, keys.device)
            self.assertEqual(tuple(actual.shape), tuple(expected.shape))
            self.assertTrue(actual.is_contiguous())
            self.assertEqual(actual.data_ptr(), expected.data_ptr())
            self.assertEqual(actual.untyped_storage().data_ptr(), expected.untyped_storage().data_ptr())
        self.check_reference(handle, values, groups, keys.device)
        borrowed_storage = (keys.device, keys.untyped_storage().data_ptr())
        storages = {(storage.device, storage.data_ptr()): storage.nbytes()
                    for storage in (tensor.untyped_storage() for tensor in handle.tensors)}
        owned_storages = {identity: size for identity, size in storages.items()
                          if identity != borrowed_storage}
        self.assertEqual(handle.workspace_bytes, sum(owned_storages.values()))
        for output in (handle.order, handle.offsets):
            self.assertIn((output.device, output.untyped_storage().data_ptr()), owned_storages)
        self.assertEqual(handle.retained_bytes, sum(storages.values()))
        # Current UINT8 buffers are queried scratch; order/prefix roles are I32/U64.
        scratch = {(tensor.device, tensor.untyped_storage().data_ptr()): tensor.numel() * tensor.element_size()
                   for tensor in handle.tensors if tensor.dtype is torch.uint8
                   and (tensor.device, tensor.untyped_storage().data_ptr()) != borrowed_storage}
        if handle.provider == 'cuda.compute':
            self.assertEqual(handle.scratch_bytes, sum(scratch.values()))
        else:
            self.assertIsNone(handle.scratch_bytes)
            self.assertFalse(scratch)

    def test_mutated_capture_fixed_pointers_and_retirement(self):
        groups, count = 17, 257
        values = [i % groups for i in range(count)]
        keys = torch.tensor(values, dtype=torch.int32, device='cuda')
        handle = grouping.prepare_grouping(keys, groups)
        pointers = {tensor.data_ptr() for tensor in handle.tensors}
        grouping.launch_grouping(handle)
        self.check_reference(handle, values, groups, keys.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            grouping.launch_grouping(handle)
        for values in ([groups - 1] * count, [(3 * i + 2) % groups for i in range(count)]):
            keys.copy_(torch.tensor(values, dtype=torch.int32, device='cuda'))
            graph.replay()
            self.check_reference(handle, values, groups, keys.device)
            self.assertEqual(pointers, {tensor.data_ptr() for tensor in handle.tensors})
        refs = [weakref.ref(tensor) for tensor in handle.tensors]
        graph.reset()
        del graph, keys, handle
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))

    def test_torch_fallback_stable_stream_capture_failure_and_retirement(self):
        groups, count = 17, 257
        values = [(3 * i + 2) % groups for i in range(count)]
        keys = torch.empty(count, dtype=torch.int32, device='cuda')
        producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
        producer.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(producer):
            keys.copy_(torch.tensor(values, dtype=torch.int32, device='cuda'))
        consumer.wait_stream(producer)
        with mock.patch.object(grouping, '_grouping_provider', return_value='torch'):
            handle = grouping.prepare_grouping(keys, groups)
            with torch.cuda.stream(consumer):
                grouping.launch_grouping(handle)
            consumer.synchronize()
            self.assertEqual(handle.provider, 'torch')
            self.assertEqual(handle.order.cpu().tolist(),
                             sorted(range(count), key=values.__getitem__))
            self.check_reference(handle, values, groups, keys.device)
            pointers = [tensor.data_ptr() for tensor in handle.tensors]
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                grouping.launch_grouping(handle)
            for values in ([groups - 1] * count, [(5 * i + 1) % groups for i in range(count)]):
                keys.copy_(torch.tensor(values, dtype=torch.int32, device='cuda'))
                graph.replay()
                self.assertEqual(handle.order.cpu().tolist(),
                                 sorted(range(count), key=values.__getitem__))
                self.check_reference(handle, values, groups, keys.device)
                self.assertEqual(pointers, [tensor.data_ptr() for tensor in handle.tensors])
            refs = [weakref.ref(tensor) for tensor in handle.tensors]
            graph.reset()
            del graph, keys, handle
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))

        keys = torch.zeros(33, dtype=torch.int32, device='cuda')
        with mock.patch.object(grouping, '_grouping_provider', return_value='torch'):
            handle = grouping.prepare_grouping(keys, 3)
        handle.order.resize_(34)
        with self.assertRaisesRegex(ValueError, 'resized or rebound'):
            grouping.launch_grouping(handle)

        broken = ImportError('installed CCCL failure at grouped preparation')
        keys = torch.zeros(33, dtype=torch.int32, device='cuda')
        with mock.patch.object(grouping, '_grouping_provider', side_effect=broken), \
             self.assertRaises(ImportError) as caught:
            grouping.prepare_grouping(keys, 3)
        self.assertIs(caught.exception, broken)


if __name__ == '__main__':
    unittest.main()
