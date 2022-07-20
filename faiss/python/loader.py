# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from packaging.version import Version
import platform
import subprocess
import logging
import os


def supported_instruction_sets():
    """
    Returns the set of supported CPU features, see
    https://github.com/numpy/numpy/blob/master/numpy/core/src/common/npy_cpu_features.h
    for the list of features that this set may contain per architecture.

    Example:
    >>> supported_instruction_sets()  # for x86
    {"SSE2", "AVX2", "AVX512", ...}
    >>> supported_instruction_sets()  # for PPC
    {"VSX", "VSX2", ...}
    >>> supported_instruction_sets()  # for ARM
    {"NEON", "ASIMD", ...}
    """
    import numpy
    if Version(numpy.__version__) >= Version("1.19"):
        # use private API as next-best thing until numpy/numpy#18058 is solved
        from numpy.core._multiarray_umath import __cpu_features__
        # __cpu_features__ is a dictionary with CPU features
        # as keys, and True / False as values
        supported = {k for k, v in __cpu_features__.items() if v}
        for f in os.getenv("FAISS_DISABLE_CPU_FEATURES", "").split(", \t\n\r"):
            supported.discard(f)
        return supported

    # platform-dependent legacy fallback before numpy 1.19, no windows
    if platform.system() == "Darwin":
        if subprocess.check_output(["/usr/sbin/sysctl", "hw.optional.avx2_0"])[-1] == '1':
            return {"AVX2"}
    elif platform.system() == "Linux":
        import numpy.distutils.cpuinfo
        result = set()
        if "avx2" in numpy.distutils.cpuinfo.cpu.info[0].get('flags', ""):
            result.add("AVX2")
        if "avx512" in numpy.distutils.cpuinfo.cpu.info[0].get('flags', ""):
            result.add("AVX512")
        return result
    return set()

# Currently numpy.core._multiarray_umath.__cpu_features__ doesn't support Arm SVE,
# so let's read Features in /proc/cpuinfo and search 'sve' entry
def is_sve_supported():
    if platform.machine() != "aarch64":
        return False
    # Currently SVE is only supported on Linux
    if platform.system() != "Linux":
        return False
    if not os.path.exists('/proc/cpuinfo'):
        return False
    proc = subprocess.Popen(['cat', '/proc/cpuinfo'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
    so, _se = proc.communicate()
    if proc.returncode != 0:
        return False
    for line in so.decode(encoding='UTF-8').splitlines():
        if ':' in line:
            l, r = line.split(':', 1)
            if l.strip() == 'Features' and "sve" in r.strip().split():
                return True
    return False

logger = logging.getLogger(__name__)

instruction_sets = None

# try to load optimization level from env variable
opt_env_variable_name = "FAISS_OPT_LEVEL"
opt_level = os.environ.get(opt_env_variable_name, None)

if opt_level is None:
    logger.debug(f"Environment variable {opt_env_variable_name} is not set, " \
                "so let's pick the instruction set according to the current CPU")
    instruction_sets = supported_instruction_sets()
else:
    logger.debug(f"Using {opt_level} as an instruction set.")
    instruction_sets = set()
    instruction_sets.add(opt_level)

loaded = False
has_AVX512 = any("AVX512" in x.upper() for x in instruction_sets)
if has_AVX512:
    try:
        logger.info("Loading faiss with AVX512 support.")
        from .swigfaiss_avx512 import *
        logger.info("Successfully loaded faiss with AVX512 support.")
        loaded = True
    except ImportError as e:
        logger.info(f"Could not load library with AVX512 support due to:\n{e!r}")
        # reset so that we load without AVX512 below
        loaded = False

has_AVX2 = "AVX2" in instruction_sets
if has_AVX2 and not loaded:
    try:
        logger.info("Loading faiss with AVX2 support.")
        from .swigfaiss_avx2 import *
        logger.info("Successfully loaded faiss with AVX2 support.")
        loaded = True
    except ImportError as e:
        logger.info(f"Could not load library with AVX2 support due to:\n{e!r}")
        # reset so that we load without AVX2 below
        loaded = False

has_SVE = is_sve_supported() and not "SVE" in os.getenv("FAISS_DISABLE_CPU_FEATURES", "").split(", \t\n\r")
if has_SVE and not loaded:
    try:
        logger.info("Loading faiss with SVE support.")
        from .swigfaiss_sve import *
        logger.info("Successfully loaded faiss with SVE support.")
        loaded = True
    except ImportError as e:
        logger.info(f"Could not load library with SVE support due to:\n{e!r}")
        # reset so that we load without SVE below
        loaded = False

if not loaded:
    # we import * so that the symbol X can be accessed as faiss.X
    logger.info("Loading faiss.")
    from .swigfaiss import *
    logger.info("Successfully loaded faiss.")
