# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from distutils.version import LooseVersion
import platform
import subprocess
import logging


logger = logging.getLogger(__name__)

# we import * so that the symbol X can be accessed as faiss.X
logger.info("Loading faiss.")
from .swigfaiss import *
logger.info("Successfully loaded faiss.")
