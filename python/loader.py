# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import platform
import subprocess
import logging


def instruction_set():
    if platform.system() == "Darwin":
        if subprocess.check_output(["/usr/sbin/sysctl", "hw.optional.avx2_0"])[-1] == '1':
            return "AVX2"
        else:
            return "default"
    elif platform.system() == "Linux":
        import numpy.distutils.cpuinfo
        if "avx2" in numpy.distutils.cpuinfo.cpu.info[0].get('flags', ""):
            return "AVX2"
        else:
            return "default"


logger = logging.getLogger(__name__)

try:
    instr_set = instruction_set()
    if instr_set == "AVX2":
        logger.info("Loading faiss with AVX2 support.")
        from .swigfaiss_avx2 import *
    else:
        logger.info("Loading faiss.")
        from .swigfaiss import *

except ImportError:
    # we import * so that the symbol X can be accessed as faiss.X
    logger.info("Loading faiss.")
    from .swigfaiss import *
