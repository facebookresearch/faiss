# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from distutils.version import LooseVersion
import platform
import subprocess
import logging


def supported_instruction_sets():
    """
    Returns the set of supported CPU features, see
    https://github.com/numpy/numpy/blob/master/numpy/core/src/common/npy_cpu_features.h
    for the list of features that this set may contain per architecture.

    Example:
    >>> supported_instruction_sets()  # for x86
    {"SSE2", "AVX2", ...}
    >>> supported_instruction_sets()  # for PPC
    {"VSX", "VSX2", ...}
    >>> supported_instruction_sets()  # for ARM
    {"NEON", "ASIMD", ...}
    """
    import numpy
    if LooseVersion(numpy.__version__) >= "1.19":
        # use private API as next-best thing until numpy/numpy#18058 is solved
        from numpy.core._multiarray_umath import __cpu_features__
        # __cpu_features__ is a dictionary with CPU features
        # as keys, and True / False as values
        supported = {k for k, v in __cpu_features__.items() if v}
        return supported

    # platform-dependent legacy fallback before numpy 1.19, no windows
    if platform.system() == "Darwin":
        if subprocess.check_output(["/usr/sbin/sysctl", "hw.optional.avx2_0"])[-1] == '1':
            return {"AVX2"}
    elif platform.system() == "Linux":
        import numpy.distutils.cpuinfo
        if "avx2" in numpy.distutils.cpuinfo.cpu.info[0].get('flags', ""):
            return {"AVX2"}
    return set()


logger = logging.getLogger(__name__)

try:
    has_AVX2 = "AVX2" in supported_instruction_sets()
    if has_AVX2:
        logger.info("Loading faiss with AVX2 support.")
        from .swigfaiss_avx2 import *
    else:
        logger.info("Loading faiss.")
        from .swigfaiss import *

except ImportError:
    # we import * so that the symbol X can be accessed as faiss.X
    logger.info("Loading faiss.")
    from .swigfaiss import *
