# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function

import os
import platform
import shutil

from setuptools import find_packages, setup

# make the faiss python package dir
shutil.rmtree("faiss", ignore_errors=True)
os.mkdir("faiss")
shutil.copytree("contrib", "faiss/contrib")
shutil.copyfile("__init__.py", "faiss/__init__.py")
shutil.copyfile("loader.py", "faiss/loader.py")
shutil.copyfile("class_wrappers.py", "faiss/class_wrappers.py")
shutil.copyfile("gpu_wrappers.py", "faiss/gpu_wrappers.py")
shutil.copyfile("extra_wrappers.py", "faiss/extra_wrappers.py")
shutil.copyfile("array_conversions.py", "faiss/array_conversions.py")

ext = ".pyd" if platform.system() == "Windows" else ".so"
prefix = "Release/" * (platform.system() == "Windows")

swigfaiss_generic_lib = f"{prefix}_swigfaiss{ext}"
swigfaiss_avx2_lib = f"{prefix}_swigfaiss_avx2{ext}"
swigfaiss_avx512_lib = f"{prefix}_swigfaiss_avx512{ext}"
swigfaiss_avx512_spr_lib = f"{prefix}_swigfaiss_avx512_spr{ext}"
callbacks_lib = f"{prefix}libfaiss_python_callbacks{ext}"
swigfaiss_sve_lib = f"{prefix}_swigfaiss_sve{ext}"
faiss_example_external_module_lib = f"_faiss_example_external_module{ext}"

found_swigfaiss_generic = os.path.exists(swigfaiss_generic_lib)
found_swigfaiss_avx2 = os.path.exists(swigfaiss_avx2_lib)
found_swigfaiss_avx512 = os.path.exists(swigfaiss_avx512_lib)
found_swigfaiss_avx512_spr = os.path.exists(swigfaiss_avx512_spr_lib)
found_callbacks = os.path.exists(callbacks_lib)
found_swigfaiss_sve = os.path.exists(swigfaiss_sve_lib)
found_faiss_example_external_module_lib = os.path.exists(
    faiss_example_external_module_lib
)

assert (
    found_swigfaiss_generic
    or found_swigfaiss_avx2
    or found_swigfaiss_avx512
    or found_swigfaiss_avx512_spr
    or found_swigfaiss_sve
    or found_faiss_example_external_module_lib
), (
    f"Could not find {swigfaiss_generic_lib} or "
    f"{swigfaiss_avx2_lib} or {swigfaiss_avx512_lib} or {swigfaiss_avx512_spr_lib} or {swigfaiss_sve_lib} or {faiss_example_external_module_lib}. "
    f"Faiss may not be compiled yet."
)

if found_swigfaiss_generic:
    print(f"Copying {swigfaiss_generic_lib}")
    shutil.copyfile("swigfaiss.py", "faiss/swigfaiss.py")
    shutil.copyfile(swigfaiss_generic_lib, f"faiss/_swigfaiss{ext}")

if found_swigfaiss_avx2:
    print(f"Copying {swigfaiss_avx2_lib}")
    shutil.copyfile("swigfaiss_avx2.py", "faiss/swigfaiss_avx2.py")
    shutil.copyfile(swigfaiss_avx2_lib, f"faiss/_swigfaiss_avx2{ext}")

if found_swigfaiss_avx512:
    print(f"Copying {swigfaiss_avx512_lib}")
    shutil.copyfile("swigfaiss_avx512.py", "faiss/swigfaiss_avx512.py")
    shutil.copyfile(swigfaiss_avx512_lib, f"faiss/_swigfaiss_avx512{ext}")

if found_swigfaiss_avx512_spr:
    print(f"Copying {swigfaiss_avx512_spr_lib}")
    shutil.copyfile("swigfaiss_avx512_spr.py", "faiss/swigfaiss_avx512_spr.py")
    shutil.copyfile(swigfaiss_avx512_spr_lib, f"faiss/_swigfaiss_avx512_spr{ext}")

if found_callbacks:
    print(f"Copying {callbacks_lib}")
    shutil.copyfile(callbacks_lib, f"faiss/{callbacks_lib}")

if found_swigfaiss_sve:
    print(f"Copying {swigfaiss_sve_lib}")
    shutil.copyfile("swigfaiss_sve.py", "faiss/swigfaiss_sve.py")
    shutil.copyfile(swigfaiss_sve_lib, f"faiss/_swigfaiss_sve{ext}")

if found_faiss_example_external_module_lib:
    print(f"Copying {faiss_example_external_module_lib}")
    shutil.copyfile(
        "faiss_example_external_module.py", "faiss/faiss_example_external_module.py"
    )
    shutil.copyfile(
        faiss_example_external_module_lib,
        f"faiss/_faiss_example_external_module{ext}",
    )

long_description = """
Faiss is a library for efficient similarity search and clustering of dense
vectors. It contains algorithms that search in sets of vectors of any size,
 up to ones that possibly do not fit in RAM. It also contains supporting
code for evaluation and parameter tuning. Faiss is written in C++ with
complete wrappers for Python/numpy. Some of the most useful algorithms
are implemented on the GPU. It is developed by Facebook AI Research.
"""
setup(
    name="faiss",
    version="1.10.0",
    description="A library for efficient similarity search and clustering of dense vectors",
    long_description=long_description,
    url="https://github.com/facebookresearch/faiss",
    author="Matthijs Douze, Jeff Johnson, Herve Jegou, Lucas Hosseini",
    author_email="faiss@meta.com",
    license="MIT",
    keywords="search nearest neighbors",
    install_requires=["numpy", "packaging"],
    packages=["faiss", "faiss.contrib", "faiss.contrib.torch"],
    package_data={
        "faiss": ["*.so", "*.pyd"],
    },
    zip_safe=False,
)
