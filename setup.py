# -*- coding: utf-8 -*-
"""faiss build for python.

Build
-----

On Linux:

GPU support is automatically built when nvcc compiler is available. Set
`CUDA_HOME` environment variable to specify where CUDA is installed.

    apt-get install swig libblas-dev liblapack-dev
    pip install numpy setuptools
    python setup.py bdist_wheel

On macOS:

    brew install llvm swig openblas
    pip install numpy setuptools
    python setup.py bdist_wheel

"""
from setuptools import setup
from setuptools.extension import Extension
from distutils.command.build import build
from distutils.command.build_ext import build_ext
from distutils.util import get_platform
import numpy as np
import os
import subprocess

SOURCES = [
    'python/swigfaiss.i',
    'AutoTune.cpp',
    'AuxIndexStructures.cpp',
    'Clustering.cpp',
    'FaissException.cpp',
    'Heap.cpp',
    'Index.cpp',
    'IndexBinary.cpp',
    'IndexBinaryFlat.cpp',
    'IndexBinaryIVF.cpp',
    'IndexFlat.cpp',
    'IndexHNSW.cpp',
    'IndexIVF.cpp',
    'IndexIVFFlat.cpp',
    'IndexIVFPQ.cpp',
    'IndexLSH.cpp',
    'IndexPQ.cpp',
    'IndexScalarQuantizer.cpp',
    'MetaIndexes.cpp',
    'OnDiskInvertedLists.cpp',
    'PolysemousTraining.cpp',
    'ProductQuantizer.cpp',
    'VectorTransform.cpp',
    'hamming.cpp',
    'index_io.cpp',
    'utils.cpp',
]

GPU_SOURCES = [
    'gpu/GpuResources.cpp',
    'gpu/IndexProxy.cpp',
    'gpu/StandardGpuResources.cpp',
    'gpu/GpuAutoTune.cpp',
    'gpu/GpuClonerOptions.cpp',
    'gpu/impl/RemapIndices.cpp',
    'gpu/utils/DeviceMemory.cpp',
    'gpu/utils/StackDeviceMemory.cpp',
    'gpu/utils/DeviceUtils.cpp',
    'gpu/utils/Timer.cpp',
    'gpu/utils/MemorySpace.cpp',
    'gpu/utils/WorkerThread.cpp',
    'gpu/GpuIndexFlat.cu',
    'gpu/GpuIndexIVFFlat.cu',
    'gpu/GpuIndexIVFPQ.cu',
    'gpu/GpuIndex.cu',
    'gpu/GpuIndexIVF.cu',
    'gpu/impl/L2Select.cu',
    'gpu/impl/InvertedListAppend.cu',
    'gpu/impl/IVFFlat.cu',
    'gpu/impl/Distance.cu',
    'gpu/impl/IVFPQ.cu',
    'gpu/impl/PQScanMultiPassPrecomputed.cu',
    'gpu/impl/IVFUtils.cu',
    'gpu/impl/VectorResidual.cu',
    'gpu/impl/FlatIndex.cu',
    'gpu/impl/L2Norm.cu',
    'gpu/impl/IVFUtilsSelect1.cu',
    'gpu/impl/PQCodeDistances.cu',
    'gpu/impl/PQScanMultiPassNoPrecomputed.cu',
    'gpu/impl/IVFFlatScan.cu',
    'gpu/impl/IVFUtilsSelect2.cu',
    'gpu/impl/IVFBase.cu',
    'gpu/impl/BroadcastSum.cu',
    'gpu/utils/BlockSelectHalf.cu',
    'gpu/utils/Float16.cu',
    'gpu/utils/nvidia/fp16_emu.cu',
    'gpu/utils/BlockSelectFloat.cu',
    'gpu/utils/warpselect/WarpSelectFloat1.cu',
    'gpu/utils/warpselect/WarpSelectFloat128.cu',
    'gpu/utils/warpselect/WarpSelectHalf256.cu',
    'gpu/utils/warpselect/WarpSelectFloatF1024.cu',
    'gpu/utils/warpselect/WarpSelectHalfT512.cu',
    'gpu/utils/warpselect/WarpSelectHalfF1024.cu',
    'gpu/utils/warpselect/WarpSelectFloat64.cu',
    'gpu/utils/warpselect/WarpSelectFloat32.cu',
    'gpu/utils/warpselect/WarpSelectHalfF512.cu',
    'gpu/utils/warpselect/WarpSelectHalf128.cu',
    'gpu/utils/warpselect/WarpSelectFloatT1024.cu',
    'gpu/utils/warpselect/WarpSelectHalf64.cu',
    'gpu/utils/warpselect/WarpSelectFloatF512.cu',
    'gpu/utils/warpselect/WarpSelectFloatT512.cu',
    'gpu/utils/warpselect/WarpSelectFloat256.cu',
    'gpu/utils/warpselect/WarpSelectHalf1.cu',
    'gpu/utils/warpselect/WarpSelectHalfT1024.cu',
    'gpu/utils/warpselect/WarpSelectHalf32.cu',
    'gpu/utils/blockselect/BlockSelectFloatT512.cu',
    'gpu/utils/blockselect/BlockSelectHalfF512.cu',
    'gpu/utils/blockselect/BlockSelectFloat256.cu',
    'gpu/utils/blockselect/BlockSelectFloat32.cu',
    'gpu/utils/blockselect/BlockSelectHalfT1024.cu',
    'gpu/utils/blockselect/BlockSelectHalf128.cu',
    'gpu/utils/blockselect/BlockSelectHalf32.cu',
    'gpu/utils/blockselect/BlockSelectFloatF1024.cu',
    'gpu/utils/blockselect/BlockSelectFloat128.cu',
    'gpu/utils/blockselect/BlockSelectHalfT512.cu',
    'gpu/utils/blockselect/BlockSelectFloat64.cu',
    'gpu/utils/blockselect/BlockSelectHalf64.cu',
    'gpu/utils/blockselect/BlockSelectFloatF512.cu',
    'gpu/utils/blockselect/BlockSelectFloat1.cu',
    'gpu/utils/blockselect/BlockSelectHalf256.cu',
    'gpu/utils/blockselect/BlockSelectFloatT1024.cu',
    'gpu/utils/blockselect/BlockSelectHalfF1024.cu',
    'gpu/utils/blockselect/BlockSelectHalf1.cu',
    'gpu/utils/WarpSelectHalf.cu',
    'gpu/utils/WarpSelectFloat.cu',
    'gpu/utils/MatrixMult.cu',
]

HEADERS = [
    'AutoTune.h',
    'AuxIndexStructures.h',
    'Clustering.h',
    'FaissException.h',
    'Heap.h',
    'Index.h',
    'IndexBinary.h',
    'IndexBinaryFlat.h',
    'IndexBinaryIVF.h',
    'IndexFlat.h',
    'IndexHNSW.h',
    'IndexIVF.h',
    'IndexIVFFlat.h',
    'IndexIVFPQ.h',
    'IndexLSH.h',
    'IndexPQ.h',
    'IndexScalarQuantizer.h',
    'MetaIndexes.h',
    'OnDiskInvertedLists.h',
    'PolysemousTraining.h',
    'ProductQuantizer.h',
    'VectorTransform.h',
    'hamming.h',
    'index_io.h',
    'utils.h',
]

GPU_HEADERS = [
    'gpu/GpuFaissAssert.h',
    'gpu/GpuIndicesOptions.h',
    'gpu/GpuResources.h',
    'gpu/IndexProxy.h',
    'gpu/StandardGpuResources.h',
    'gpu/GpuAutoTune.h',
    'gpu/GpuClonerOptions.h',
    'gpu/impl/RemapIndices.h',
    'gpu/utils/DeviceMemory.h',
    'gpu/utils/StackDeviceMemory.h',
    'gpu/utils/DeviceUtils.h',
    'gpu/utils/Timer.h',
    'gpu/utils/MemorySpace.h',
    'gpu/utils/WorkerThread.h',
    'gpu/GpuIndex.h',
    'gpu/GpuIndexFlat.h',
    'gpu/GpuIndexIVF.h',
    'gpu/GpuIndexIVFFlat.h',
    'gpu/GpuIndexIVFPQ.h',
    'gpu/utils/BlockSelectKernel.cuh',
    'gpu/utils/Comparators.cuh',
    'gpu/utils/ConversionOperators.cuh',
    'gpu/utils/CopyUtils.cuh',
    'gpu/utils/DeviceDefs.cuh',
    'gpu/utils/DeviceTensor.cuh',
    'gpu/utils/DeviceTensor-inl.cuh',
    'gpu/utils/DeviceVector.cuh',
    'gpu/utils/Float16.cuh',
    'gpu/utils/HostTensor.cuh',
    'gpu/utils/HostTensor-inl.cuh',
    'gpu/utils/Limits.cuh',
    'gpu/utils/LoadStoreOperators.cuh',
    'gpu/utils/MathOperators.cuh',
    'gpu/utils/MatrixMult.cuh',
    'gpu/utils/MergeNetworkBlock.cuh',
    'gpu/utils/MergeNetworkUtils.cuh',
    'gpu/utils/MergeNetworkWarp.cuh',
    'gpu/utils/NoTypeTensor.cuh',
    'gpu/utils/Pair.cuh',
    'gpu/utils/PtxUtils.cuh',
    'gpu/utils/ReductionOperators.cuh',
    'gpu/utils/Reductions.cuh',
    'gpu/utils/Select.cuh',
    'gpu/utils/Tensor.cuh',
    'gpu/utils/Tensor-inl.cuh',
    'gpu/utils/ThrustAllocator.cuh',
    'gpu/utils/Transpose.cuh',
    'gpu/utils/WarpSelectKernel.cuh',
    'gpu/utils/WarpShuffles.cuh',
    'gpu/utils/blockselect/BlockSelectImpl.cuh',
    'gpu/utils/warpselect/WarpSelectImpl.cuh',
    'gpu/utils/nvidia/fp16_emu.cuh',
]


def locate_cuda():
    """Locate the CUDA environment on the system. Returns a dict with keys
    'home', 'nvcc', 'include', and 'lib' and values giving the absolute path
    to each directory. Starts by looking for the CUDAHOME env variable. If not
    found, everything is based on finding 'nvcc' in the PATH.
    """
    # adapted from
    # https://stackoverflow.com/questions/10034325/
    nvcc = None
    envs = ['CUDA_HOME', 'CUDA_ROOT', 'CUDAHOME', 'CUDAROOT']
    for env in envs:
        if env in os.environ:
            nvcc = os.path.join(os.environ[env], 'bin', 'nvcc')
            break
    else:
        # otherwise, search PATH for NVCC
        nvcc = find_in_path('nvcc')

    home = os.path.dirname(os.path.dirname(nvcc)) if nvcc else None
    return {
        'home': home,
        'nvcc': nvcc,
        'include': os.path.join(home, 'include') if nvcc else None,
        'lib': os.path.join(home, 'lib64') if nvcc else None,
    }


def find_in_path(filename):
    """Find file on system path."""
    # http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/52224
    paths = os.getenv('PATH').split(os.pathsep)
    for path in paths:
        if os.path.exists(os.path.join(path, filename)):
            return os.path.abspath(os.path.join(path, filename))


class CustomBuild(build):
    """Build ext first so that swig-generated file is packaged.
    """
    sub_commands = [
        ('build_ext', build.has_ext_modules),
        ('build_py', build.has_pure_modules),
        ('build_clib', build.has_c_libraries),
        ('build_scripts', build.has_scripts),
    ]


class CustomBuildExt(build_ext):
    """Customize extension build by injecting nvcc.
    """
    def build_extensions(self):
        # Suppress -Wstrict-prototypes bug in python.
        # https://stackoverflow.com/questions/8106258/
        try:
            self.compiler.compiler_so.remove("-Wstrict-prototypes")
        except (AttributeError, ValueError):
            pass

        customize_compiler_for_nvcc(self.compiler)
        self.swig = self.swig or os.getenv('SWIG')
        build_ext.build_extensions(self)


def customize_compiler_for_nvcc(self):
    """Inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.

    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on.

    Taken from https://github.com/rmcgibbo/npcuda-example/
    """

    self.src_extensions.append('.cu')
    default_compiler_so = self.compiler_so
    default_compile = self._compile

    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            self.set_executable('compiler_so', CUDA['nvcc'])
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        default_compile(obj, src, ext, cc_args, postargs, pp_opts)
        self.compiler_so = default_compiler_so

    self._compile = _compile


def get_config():
    config = {'include_dirs': ['.', np.get_include()]}
    if CUDA.get('nvcc'):
        config['include_dirs'] += [CUDA['include']]
        config['library_dirs'] = [CUDA['lib']]
        config['libraries'] = ['cuda', 'cudart', 'cublas']

    platform = get_platform()
    if platform.startswith('linux'):
        try:
            config = pkgconfig('blas', 'lapack', config=config)
        except subprocess.CalledProcessError:
            config['libraries'] = (
                config.get('libraries', []) + ['blas', 'lapack'])
    elif platform.startswith('macosx'):
        # Only Homebrew environment is supported.
        if 'CC' not in os.environ and 'CXX' not in os.environ:
            llvm_home = subprocess.check_output([
                'brew', '--prefix', 'llvm'
            ]).decode('utf8').strip()
            clang = os.path.join(llvm_home, 'bin', 'clang++')
            if str == bytes:
                os.environ['CC'] = clang.encode('ascii')
                os.environ['CXX'] = clang.encode('ascii')
            else:
                os.environ['CC'] = clang
                os.environ['CXX'] = clang
            config['runtime_library_dirs'] = [os.path.join(llvm_home, 'lib')]
            config['extra_compile_args'] = ['-stdlib=libc++']
            config['extra_link_args'] = ['-stdlib=libc++']
        try:
            config = pkgconfig('openblas', config=config)
        except subprocess.CalledProcessError:
            config['libraries'] = (
                config.get('libraries', []) + ['blas', 'lapack'])

    config['extra_link_args'] = (
        config.get('extra_link_args', []) + ['-fopenmp'])

    config['extra_compile_args'] = {
        'gcc': config.get('extra_compile_args', []) + [
            '-fPIC', '-fopenmp', '-m64', '-g', '-O3', '-Wno-sign-compare',
            '-msse4', '-mpopcnt', '-std=c++11',
        ],
        'nvcc': [
            '-g', '-O3', '-Xcompiler', '-fPIC', '-Xcudafe',
            '--diag_suppress=unrecognized_attribute',
            '-gencode', 'arch=compute_35,code="compute_35"',
            '-gencode', 'arch=compute_52,code="compute_52"',
            '-gencode', 'arch=compute_60,code="compute_60"',
            '-lineinfo', '-std=c++11',
        ],
    }
    return config


def pkgconfig(*packages, **kw):
    """
    Query pkg-config for library compile and linking options. Return
    configuration in distutils Extension format.

    Usage:

    pkgconfig('opencv')
    pkgconfig('opencv', 'libavformat')
    pkgconfig('opencv', optional='--static')
    pkgconfig('opencv', config=c)

    returns e.g.

    {'extra_compile_args': [],
     'extra_link_args': [],
     'include_dirs': ['/usr/include/ffmpeg'],
     'libraries': ['avformat'],
     'library_dirs': []}

    Intended use:

    Extension('pyextension', sources=['source.cpp'], **c)

    Set PKG_CONFIG_PATH environment variable for nonstandard library
    locations.

    based on work of Micah Dowty
    http://code.activestate.com/recipes/502261-python-distutils-pkg-config/
    """
    config = kw.setdefault('config', {})
    optional_args = kw.setdefault('optional', '')

    # { <Extension arg>: [<pkg config option>, <prefix length to strip>], }
    flag_map = {
        'include_dirs': ['--cflags-only-I', 2],
        'library_dirs': ['--libs-only-L', 2],
        'libraries': ['--libs-only-l', 2],
        'extra_compile_args': ['--cflags-only-other', 0],
        'extra_link_args': ['--libs-only-other', 0],
    }
    for package in packages:
        for distutils_key, (pkg_option, n) in flag_map.items():
            items = subprocess.check_output([
                'pkg-config',
                optional_args,
                pkg_option,
                package
            ]).decode('utf8').split()
            config.setdefault(distutils_key, []).extend(
                [i[n:] for i in items])
    return config


CUDA = locate_cuda()

_swigfaiss = Extension(
    '_swigfaiss',
    sources=SOURCES,
    depends=HEADERS,
    define_macros=[('FINTEGER', 'int'),],
    language='c++',
    swig_opts=['-c++'],
    **get_config()
)

_swigfaiss_gpu = Extension(
    '_swigfaiss_gpu',
    sources=SOURCES + GPU_SOURCES,
    depends=HEADERS + GPU_HEADERS,
    define_macros=[('FINTEGER', 'int'), ('FAISS_USE_FLOAT16',)],
    language='c++',
    swig_opts=['-c++', '-DGPU_WRAPPER'],
    **get_config()
)

LONG_DESCRIPTION = """
Faiss is a library for efficient similarity search and clustering of dense
vectors. It contains algorithms that search in sets of vectors of any size,
 up to ones that possibly do not fit in RAM. It also contains supporting
code for evaluation and parameter tuning. Faiss is written in C++ with
complete wrappers for Python/numpy. Some of the most useful algorithms
are implemented on the GPU. It is developed by Facebook AI Research.
"""

setup(
    name='faiss',
    version='0.1',
    description=(
        'A library for efficient similarity search and clustering of dense '
        'vectors'
    ),
    long_description=LONG_DESCRIPTION,
    url='https://github.com/facebookresearch/faiss',
    author='Matthijs Douze, Jeff Johnson, Herve Jegou',
    author_email='matthijs@fb.com',
    license='BSD',
    keywords='search nearest neighbors',
    cmdclass={
        'build': CustomBuild,
        'build_ext': CustomBuildExt,
    },
    install_requires=['numpy'],
    package_dir={'faiss': 'python'},
    packages=['faiss'],
    ext_modules=[_swigfaiss] + ([_swigfaiss_gpu] if CUDA.get('nvcc') else [])
)
