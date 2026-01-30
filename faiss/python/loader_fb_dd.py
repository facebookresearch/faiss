# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Loader for Dynamic Dispatch (DD) variant of pyfaiss.

This loader imports from swigfaiss_dd which links against faiss_dd,
providing runtime SIMD detection and dispatch.
"""

import logging


logger = logging.getLogger(__name__)

# DD variant uses swigfaiss_dd module
from swigfaiss_dd import *  # noqa: E402, F401, F403
