from setuptools import setup
from setuptools.extension import Extension
from distutils.command.build import build
from distutils.util import get_platform
import numpy as np
import subprocess

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


class CustomBuild(build):
    # Build ext first so that swig-generated file is packaged.
    sub_commands = [
        ('build_ext', build.has_ext_modules),
        ('build_py', build.has_pure_modules),
        ('build_clib', build.has_c_libraries),
        ('build_scripts', build.has_scripts),
    ]


def get_config():
    config = {'extra_compile_args': [], 'include_dirs': []}
    platform = get_platform()
    if platform.startswith('linux'):
        config = pkgconfig('blas', 'lapack', config=config)
    elif platform.startswith('macosx'):
        config = pkgconfig('openblas', config=config)
    config['extra_compile_args'] += ['-fPIC', '-fopenmp']
    config['include_dirs'] += ['.', np.get_include()]
    return config


_swigfaiss = Extension(
    '_swigfaiss',
    sources=[
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
    ],
    depends=[
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
    ],
    define_macros=[
        ('FINTEGER','int'),
    ],
    language='c++',
    swig_opts=['-c++'],
    **get_config()
)

long_description="""
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
    long_description=long_description,
    url='https://github.com/facebookresearch/faiss',
    author='Matthijs Douze, Jeff Johnson, Herve Jegou',
    author_email='matthijs@fb.com',
    license='BSD',
    keywords='search nearest neighbors',
    cmdclass={'build': CustomBuild},
    install_requires=['numpy'],
    package_dir={'faiss': 'python'},
    packages=['faiss'],
    ext_modules=[_swigfaiss]
)
