########################################################################################################################
## Setup file for faiss by Amro Tork
########################################################################################################################
from __future__ import print_function
from setuptools import setup, find_packages
from distutils.spawn import find_executable
from distutils.sysconfig import get_python_inc
from numpy import get_include

from distutils.command.clean import clean
from distutils.command.install import install

import os
import shutil
import platform


platform_type = platform.system()
if platform_type == "Darwin":
    lib_extension="so"
elif platform_type == "Linux":
    lib_extension="so"
else:
    print("Unknown platform type: {}".format(platform_type))
    exit(1)
here = os.path.abspath(os.path.dirname(__file__))
makefile = os.path.join(here,"makefile.inc")

class FaissInstall(install):
    def run(self):
        ## Setup files.
        python_path = os.path.abspath(find_executable("python"))
        python_bin_folder = os.path.dirname(python_path)
        python_env = os.path.dirname(python_bin_folder)
        python_inc = get_python_inc()
        numpy_inc = get_include()

        if platform_type == "Darwin":
            lib_extension = "so"
            path_makefile_inc = os.path.join(here,"example_makefiles","makefile.inc.Mac.brew")
            shutil.copy(path_makefile_inc,makefile)
        elif platform_type == "Linux":
            gpp_compiler = os.path.abspath(find_executable("g++"))
            gcc_compiler = os.path.abspath(find_executable("gcc"))
            lib_extension = "so"

            ## Creating include file.
            with open(makefile, 'w') as outfile:
                makefile_str = """
                ### Automatically created make include file.
                CC={}
                CXX={}
                
                CFLAGS=-fPIC -m64 -Wall -g -O3 -mavx -msse4 -mpopcnt -fopenmp -Wno-sign-compare -fopenmp
                CXXFLAGS=$(CFLAGS) -std=c++11
                LDFLAGS=-g -fPIC  -fopenmp
                
                # common linux flags
                SHAREDEXT=so
                SHAREDFLAGS=-shared
                FAISSSHAREDFLAGS=-shared

                """.format(gcc_compiler,gpp_compiler)
                outfile.write(makefile_str)
                outfile.write("MKLROOT={}\n".format(python_env))
                outfile.write("BLASLDFLAGS=-Wl,--no-as-needed -L$(MKLROOT)/lib   -lmkl_intel_ilp64 -lmkl_core -lmkl_gnu_thread -ldl -lpthread\n")
                outfile.write("BLASCFLAGS=-DFINTEGER=long\n\n\n")
                outfile.write("SWIGEXEC=swig\n")
                outfile.write("PYTHONCFLAGS=-I{} -I{}\n".format(python_inc,numpy_inc))
                outfile.close()
        else:
            print("Unknown platform {}".format(platform_type))


        ## Run make
        os.system("make")
        os.system("make py")


        ## Checking compilation.
        check_fpath = os.path.join("python", "_swigfaiss.{}".format(lib_extension))
        if not os.path.exists(check_fpath):
            print("Could not find {}".format(check_fpath))
            print("Have you run `make` and `make py` "
                  "(and optionally `cd gpu && make && make py && cd ..`)?")

        # make the faiss python package dir
        shutil.rmtree("faiss", ignore_errors=True)
        os.mkdir("faiss")
        shutil.copyfile("python/__init__.py", "faiss/__init__.py")
        shutil.copyfile("faiss.py", "faiss/faiss.py")
        shutil.copyfile("python/swigfaiss.py", "faiss/swigfaiss.py")
        shutil.copyfile(os.path.join("python","_swigfaiss.{}".format(lib_extension)), os.path.join("faiss","_swigfaiss.{}".format(lib_extension)))
        try:
            shutil.copyfile("python/_swigfaiss_gpu.so", "faiss/_swigfaiss_gpu.so")
            shutil.copyfile("python/swigfaiss_gpu.py", "faiss/swigfaiss_gpu.py")
        except:
            pass
        
        install.run(self)
        c = clean(self.distribution)
        c.all = True
        c.finalize_options()
        c.run()
        
class FaissClean(clean):
    def run(self):
        ## Setup files.
        platform_type = platform.system()
        if platform_type == "Darwin":
            lib_extension="dylib"
        elif platform_type == "Linux":
            lib_extension="so"
        else:
            print("Unknown platform type: {}".format(platform_type))
            exit(1)

        os.remove(makefile)

        ## Run make
        os.system("make clean")

        # make the faiss python package dir
        shutil.rmtree("faiss", ignore_errors=True)

        # Run base clean.
        clean.run(self)
        

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
    description='A library for efficient similarity search and clustering of dense vectors',
    long_description=long_description,
    url='https://github.com/facebookresearch/faiss',
    author='Matthijs Douze, Jeff Johnson, Herve Jegou',
    author_email='matthijs@fb.com',
    license='BSD',
    keywords='search nearest neighbors',
    cmdclass={'install': FaissInstall,"clean": FaissClean},
    install_requires=['numpy'],
    packages=['faiss'],
    package_data={
        'faiss': ['*.{}'.format(lib_extension)],
    },

)
