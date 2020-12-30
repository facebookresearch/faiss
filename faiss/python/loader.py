# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from distutils.version import LooseVersion
import platform
import subprocess
import logging


def supports_AVX2():
    import numpy
    if LooseVersion(numpy.__version__) >= "1.19":
        # use private API as next-best thing until numpy/numpy#18058 is solved
        from numpy.core._multiarray_umath import __cpu_features__
        return __cpu_features__["AVX2"]

    # platform-dependent fallback before numpy 1.19, no windows
    if platform.system() == "Darwin":
        if subprocess.check_output(["/usr/sbin/sysctl", "hw.optional.avx2_0"])[-1] == '1':
            return True
        else:
            return False
    elif platform.system() == "Linux":
        import numpy.distutils.cpuinfo
        if "avx2" in numpy.distutils.cpuinfo.cpu.info[0].get('flags', ""):
            return True
        else:
            return False
    return False


logger = logging.getLogger(__name__)

try:
    has_AVX2 = supports_AVX2()
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
