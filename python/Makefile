# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD+Patents license found in the
# LICENSE file in the root directory of this source tree.

-include ../makefile.inc

HEADERS=$(wildcard ../*.h)

ifeq ($(OS),Darwin)
	SHAREDFLAGS += -Wl,-F. -undefined dynamic_lookup
endif

all: _swigfaiss.so

#############################
# CPU

# Also silently generates swigfaiss.py.
swigfaiss_wrap.cpp: swigfaiss.swig $(HEADERS)
	$(SWIG) -python -c++ -Doverride= -I../ -o $@ $<

swigfaiss_wrap.o: swigfaiss_wrap.cpp
	$(CXX) -I../ $(CXXFLAGS) $(CPUFLAGS) -c $< -o $@

# Extension is .so even on OSX.
_swigfaiss.so: swigfaiss_wrap.cpp ../libfaiss.a
	$(CXX) -I../ $(CXXFLAGS) $(CPUFLAGS) $(LDFLAGS) $(SHAREDFLAGS) \
	$(PYTHONCFLAGS) -o $@ $^ $(LIBS)


#############################
# GPU

# Also silently generates swigfaiss.py.
swigfaiss_gpu_wrap.cpp: swigfaiss.swig
	$(SWIG) -python -c++ -Doverride= -I../ -DGPU_WRAPPER -o $@ $<

_swigfaiss_gpu.so: swigfaiss_gpu_wrap.o ../gpu/libgpufaiss.a ../libfaiss.a
	$(NVCC) -I../ $(NVCCFLAGS) $(NVCCLIBS) $(PYTHONCFLAGS) -o $@ $^ \
	$(SHAREDFLAGS)


clean:
	rm -f swigfaiss_wrap.cxx swigfaiss_gpu_wrap.cxx swigfaiss.py
	rm -f _swigfaiss.so _swigfaiss_gpu.so

test: _swigfaiss.so
	PYTHONPATH=./ $(PYTHON) -m unittest discover tests/ -v

.PHONY: all clean test
