# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD+Patents license found in the
# LICENSE file in the root directory of this source tree.

.SUFFIXES: .cpp .o

# C API with GPU support

include ../../makefile.inc

LIBNAME=libgpufaiss
CLIBNAME=libgpufaiss_c
LIBOBJ=GpuAutoTune_c.o GpuClonerOptions_c.o GpuIndex_c.o GpuResources_c.o \
	StandardGpuResources_c.o

# Build shared object file by default
all: $(CLIBNAME).$(SHAREDEXT)

# Build static library (requires consumers to link with libstdc++)
$(CLIBNAME).a: $(LIBCOBJ) ../../gpu/$(LIBNAME).a
	ar r $@ $^

# Build dynamic library
$(CLIBNAME).$(SHAREDEXT): $(LIBOBJ) ../../gpu/$(LIBNAME).a
	$(CXX) $(LDFLAGS) -static-libstdc++ $(FAISSSHAREDFLAGS) -o $@ $^ \
	$(BLASLDFLAGS) -L.. -lfaiss_c

# Build GPU example
bin/example_gpu_c: example_gpu_c.c $(CLIBNAME).$(SHAREDEXT)
	$(CC) $(CFLAGS) $(CUDACFLAGS) -std=c99 -I. -I.. $(LDFLAGS) -o $@ example_gpu_c.c \
	-L$(CUDAROOT)/lib64 -L. -L.. -lgpufaiss_c -lfaiss_c -lcublas -lcudart

clean:
	rm -f $(CLIBNAME).a $(CLIBNAME).$(SHAREDEXT)* *.o

# Dependencies

GpuAutoTune_c.o: CFLAGS += -I.. -I../.. $(CUDACFLAGS)
GpuAutoTune_c.o: GpuAutoTune_c.cpp GpuAutoTune_c.h ../../gpu/GpuAutoTune.h ../Index_c.h ../macros_impl.h

GpuClonerOptions_c.o: CFLAGS += -I.. -I../.. $(CUDACFLAGS)
GpuClonerOptions_c.o: GpuClonerOptions_c.cpp GpuClonerOptions_c.h GpuIndicesOptions_c.h ../../gpu/GpuClonerOptions.h ../macros_impl.h

GpuIndex_c.o: CFLAGS += -I.. -I../.. $(CUDACFLAGS)
GpuIndex_c.o: GpuIndex_c.cpp GpuIndex_c.h ../../gpu/GpuIndex.h ../macros_impl.h

GpuResources_c.o: CFLAGS += -I.. -I../.. $(CUDACFLAGS)
GpuResources_c.o: GpuResources_c.cpp GpuResources_c.h ../../gpu/GpuResources.h ../macros_impl.h

StandardGpuResources_c.o: CFLAGS += -I.. -I../.. $(CUDACFLAGS)
StandardGpuResources_c.o: StandardGpuResources_c.cpp StandardGpuResources_c.h ../../gpu/StandardGpuResources.h ../macros_impl.h
