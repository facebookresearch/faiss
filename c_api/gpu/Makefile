# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

.SUFFIXES: .cpp .o

# C API with GPU support

include ../../makefile.inc
DEBUGFLAG=-DNDEBUG # no debugging

LIBNAME=libfaiss
CLIBNAME=libgpufaiss_c
LIBGPUCOBJ=GpuAutoTune_c.o GpuClonerOptions_c.o GpuIndex_c.o GpuResources_c.o \
	StandardGpuResources_c.o DeviceUtils_c.o
LIBCOBJ=../libfaiss_c.a
CFLAGS=-fPIC -m64 -Wno-sign-compare -g -O3 -Wall -Wextra
CUDACFLAGS=-I$(CUDA_ROOT)/include

# Build shared object file by default
all: $(CLIBNAME).$(SHAREDEXT)

# Build static object file containing the wrapper implementation only.
# Consumers are required to link with the C++ standard library and remaining
# portions of this library: libfaiss_c.a, libfaiss.a, and libstdc++.
$(CLIBNAME).a: $(LIBGPUCOBJ) ../../$(LIBNAME).a
	ar r $@ $^

# Build dynamic library
$(CLIBNAME).$(SHAREDEXT): $(LIBCOBJ) $(LIBGPUCOBJ) ../../libfaiss.a
	$(CXX) $(LDFLAGS) $(SHAREDFLAGS) $(CUDACFLAGS) -o $@ \
	-Wl,--whole-archive $(LIBCOBJ) ../../libfaiss.a \
	-Wl,--no-whole-archive -static-libstdc++ $(LIBGPUCOBJ) $(LIBS) \
	$(NVCCLDFLAGS) $(NVCCLIBS)

# Build GPU example
bin/example_gpu_c: example_gpu_c.c $(CLIBNAME).$(SHAREDEXT)
	$(CC) $(CFLAGS) $(CUDACFLAGS) $(NVCCLIBS) -std=c99 -I. -I.. -o $@ example_gpu_c.c \
	-L. -lgpufaiss_c 

clean:
	rm -f $(CLIBNAME).a $(CLIBNAME).$(SHAREDEXT)* *.o bin/example_gpu_c

%.o: %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(CPUFLAGS) -c $< -o $@

# Dependencies

GpuAutoTune_c.o: CXXFLAGS += -I.. -I../.. -I../../gpu -I../../impl $(CUDACFLAGS) $(DEBUGFLAG)
GpuAutoTune_c.o: GpuAutoTune_c.cpp GpuAutoTune_c.h ../../gpu/GpuCloner.h ../../gpu/GpuAutoTune.h ../Index_c.h ../macros_impl.h

GpuClonerOptions_c.o: CXXFLAGS += -I.. -I../.. -I../../gpu -I../../impl $(CUDACFLAGS) $(DEBUGFLAG)
GpuClonerOptions_c.o: GpuClonerOptions_c.cpp GpuClonerOptions_c.h GpuIndicesOptions_c.h ../../gpu/GpuClonerOptions.h ../macros_impl.h

GpuIndex_c.o: CXXFLAGS += -I.. -I../.. -I../../gpu -I../../impl $(CUDACFLAGS) $(DEBUGFLAG)
GpuIndex_c.o: GpuIndex_c.cpp GpuIndex_c.h ../../gpu/GpuIndex.h ../macros_impl.h

GpuResources_c.o: CXXFLAGS += -I.. -I../.. -I../../gpu -I../../impl $(CUDACFLAGS) $(DEBUGFLAG)
GpuResources_c.o: GpuResources_c.cpp GpuResources_c.h ../../gpu/GpuResources.h ../macros_impl.h

StandardGpuResources_c.o: CXXFLAGS += -I.. -I../.. -I../../gpu -I../../impl $(CUDACFLAGS) $(DEBUGFLAG)
StandardGpuResources_c.o: StandardGpuResources_c.cpp StandardGpuResources_c.h ../../gpu/StandardGpuResources.h ../macros_impl.h

DeviceUtils_c.o: CXXFLAGS += -I.. -I../.. -I../../gpu -I../../impl $(CUDACFLAGS) $(DEBUGFLAG)
DeviceUtils_c.o: DeviceUtils_c.cpp DeviceUtils_c.h ../../gpu/utils/DeviceUtils.h ../macros_impl.h
