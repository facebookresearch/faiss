# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD+Patents license found in the
# LICENSE file in the root directory of this source tree.

-include ../makefile.inc

HEADERS = $(wildcard ../*.h)

all: cpu build

#############################
# CPU

cpu: _swigfaiss.so

# Also silently generates swigfaiss.py.
swigfaiss_wrap.cpp: swigfaiss.swig $(HEADERS)
	$(SWIG) -python -c++ -Doverride= -I../ -o $@ $<

swigfaiss_wrap.o: swigfaiss_wrap.cpp
	$(CXX) $(CXXFLAGS) $(CPUFLAGS) $(PYTHONCFLAGS) -I../ -c $< -o $@

# Extension is .so even on OSX.
_swigfaiss.so: swigfaiss_wrap.o ../libfaiss.a
	$(CXX) $(SHAREDFLAGS) $(LDFLAGS) -o $@ $^  $(LIBS)


#############################
# GPU

gpu: _swigfaiss_gpu.so

# Also silently generates swigfaiss.py.
swigfaiss_gpu_wrap.cpp: swigfaiss.swig
	$(SWIG) -python -c++ -Doverride= -I../ -DGPU_WRAPPER -o $@ $<

swigfaiss_gpu_wrap.o: swigfaiss_gpu_wrap.cpp
	$(NVCC) $(NVCCFLAGS) $(PYTHONCFLAGS) -I../ -c $< -o $@

_swigfaiss_gpu.so: swigfaiss_gpu_wrap.o ../gpu/libgpufaiss.a ../libfaiss.a
	$(CXX) $(SHAREDFLAGS) $(NVCCLDFLAGS) $(LDFLAGS) -o $@ $^ $(NVCCLIBS) $(LIBS)


build:
	$(PYTHON) setup.py build

install: build
	$(PYTHON) setup.py install

clean:
	rm -f swigfaiss_wrap.cpp swigfaiss_gpu_wrap.cpp
	rm -f swigfaiss.py swigfaiss_gpu.py
	rm -f _swigfaiss.so _swigfaiss_gpu.so
	rm -rf build/

.PHONY: all build clean cpu gpu install
