# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

-include ../makefile.inc

ifneq ($(strip $(NVCC)),)
	SWIGFLAGS += -DGPU_WRAPPER
endif

all: build

# Also silently generates swigfaiss.py.
swigfaiss.cpp: swigfaiss.swig ../libfaiss.a
	$(SWIG) -java -c++ -Doverride= -I../ $(SWIGFLAGS) -package com.gameofdimension.faiss.swig -outdir src/main/java/com/gameofdimension/faiss/swig -o $@ $<

swigfaiss_avx2.cpp: swigfaiss.swig ../libfaiss.a
	$(SWIG) -java -c++ -Doverride= -module swigfaiss_avx2 -I../ $(SWIGFLAGS) -package com.gameofdimension.faiss.swig -outdir src/main/java/com/gameofdimension/faiss/swig -o $@ $<

%.o: %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(CPUFLAGS) $(PYTHONCFLAGS) $(JAVACFLAGS)\
               -I../ -c $< -o $@

# Extension is .so even on OSX.
_%.so: %.o ../libfaiss.a
	$(CXX) $(SHAREDFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS)

build: _swigfaiss.so

# install: build
# 	$(PYTHON) setup.py install
# 
# clean:
# 	rm -f swigfaiss*.cpp swigfaiss*.o swigfaiss*.py _swigfaiss*.so
# 	rm -rf build/
# 
.PHONY: all build clean install
