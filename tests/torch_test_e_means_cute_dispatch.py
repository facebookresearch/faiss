# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from faiss.contrib.e_means import e_means_cute as backend


GROUPED_READY = backend._stack_status()[0]


class TestCuteDispatch(unittest.TestCase):
    @staticmethod
    def prepared(k, d, dtype=torch.bfloat16, spherical=False, transform=None):
        return SimpleNamespace(num_centroids=k, dim=d, spherical=spherical,
                               transform=transform, centroids=SimpleNamespace(dtype=dtype, device='cuda:0'))

    @staticmethod
    def properties():
        return SimpleNamespace(major=9, minor=0, multi_processor_count=132,
                               shared_memory_per_block_optin=232448)

    @unittest.skipUnless(GROUPED_READY, "optional CuTe stack required")
    def test_any_selected_plan_is_legal(self):
        from faiss.contrib.e_means.e_means_cute_grouped import _validate_plan

        # Probe metadata properties; benchmark validation decides route performance.
        cells = [(3, 128, 7), (70001, 32768, 113),
                 (70001, 32768, 1024), (70001, 32769, 113)]
        with patch.object(torch.cuda, 'get_device_properties', return_value=self.properties()):
            for b, k, d in cells:
                with self.subTest(b=b, k=k, d=d):
                    plan = backend._grouped_plan(b, self.prepared(k, d))
                    if plan is not None:
                        _validate_plan(b, k, d, torch.bfloat16, plan,
                                       self.properties().shared_memory_per_block_optin)

    @unittest.skipUnless(GROUPED_READY, "optional CuTe stack required")
    def test_identical_operation_metadata_has_identical_choice(self):
        with patch.object(torch.cuda, 'get_device_properties', return_value=self.properties()):
            plain = backend._grouped_plan(70001, self.prepared(32768, 113))
            centered = backend._grouped_plan(70001, self.prepared(32768, 113,
                                             transform=object()))
        self.assertEqual(plain, centered)

    @unittest.skipUnless(GROUPED_READY, "optional CuTe stack required")
    def test_validation_errors_remain_errors(self):
        from faiss.contrib.e_means.e_means_cute_grouped import _validate_plan

        with self.assertRaisesRegex(ValueError, 'explicit'):
            _validate_plan(3, 128, 7, torch.bfloat16, 'invalid', 232448)
        with self.assertRaisesRegex(ValueError, 'legal operands'):
            _validate_plan(0, 128, 7, torch.bfloat16, (32, 128, 2), 232448)
        with self.assertRaisesRegex(ValueError, 'shared memory'):
            _validate_plan(3, 128, 7, torch.bfloat16, (32, 128, 2), 0)

    def test_selected_failure_propagates_without_installing_binding(self):
        prepared = SimpleNamespace(bindings=OrderedDict())
        failure = ImportError('required grouping dependency is unavailable')
        with patch.object(backend, '_grouped_plan', return_value=object()), patch.object(
                backend, '_new_grouped_binding', side_effect=failure) as factory:
            with self.assertRaises(ImportError) as caught:
                backend._binding_for(17, prepared)
        self.assertIs(caught.exception, failure)
        self.assertEqual(factory.call_count, 1)
        self.assertFalse(prepared.bindings)


if __name__ == '__main__':
    unittest.main()
