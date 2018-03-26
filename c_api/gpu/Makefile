# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD+Patents license found in the
# LICENSE file in the root directory of this source tree.

.SUFFIXES: .cpp .o

# C API with GPU support

include ../../makefile.inc
DEBUGFLAG=-DNDEBUG # no debugging

LIBNAME=libgpufaiss
CLIBNAME=libgpufaiss_c
LIBGPUCOBJ=GpuAutoTune_c.o GpuClonerOptions_c.o GpuIndex_c.o GpuResources_c.o \
	StandardGpuResources_c.o
LIBCOBJ=../error_impl.o ../Index_c.o ../IndexFlat_c.o ../Clustering_c.o \
	../AuxIndexStructures_c.o ../AutoTune_c.o ../IndexIVF_c.o

# Build shared object file by default
all: $(CLIBNAME).$(SHAREDEXT) bin/example_gpu_c

# Build static library (requires consumers to link with libstdc++)
$(CLIBNAME).a: $(LIBGPUCOBJ) ../../gpu/$(LIBNAME).a
	ar r $@ $^

# Build dynamic library
$(CLIBNAME).$(SHAREDEXT): $(LIBCOBJ) $(LIBGPUCOBJ) ../../libfaiss.a ../../gpu/$(LIBNAME).a
	$(CXX) $(LDFLAGS) $(FAISSSHAREDFLAGS) $(CUDACFLAGS) -o $@ \
	-Wl,--whole-archive $(LIBCOBJ) ../../libfaiss.a $(BLASLDFLAGS) \
	-Wl,--no-whole-archive -static-libstdc++ $(LIBGPUCOBJ) ../../gpu/$(LIBNAME).a \
	-L$(CUDAROOT)/lib64 -lcublas -lcudart

# Build GPU example
bin/example_gpu_c: example_gpu_c.c $(CLIBNAME).$(SHAREDEXT)
	$(CC) $(CFLAGS) $(CUDACFLAGS) -std=c99 -I. -I.. -o $@ example_gpu_c.c \
	-L. -lgpufaiss_c 

clean:
	rm -f $(CLIBNAME).a $(CLIBNAME).$(SHAREDEXT)* *.o bin/example_gpu_c

# Dependencies

GpuAutoTune_c.o: CFLAGS += -I.. -I../.. $(CUDACFLAGS) $(DEBUGFLAG)
GpuAutoTune_c.o: GpuAutoTune_c.cpp GpuAutoTune_c.h ../../gpu/GpuAutoTune.h ../Index_c.h ../macros_impl.h

GpuClonerOptions_c.o: CFLAGS += -I.. -I../.. $(CUDACFLAGS) $(DEBUGFLAG)
GpuClonerOptions_c.o: GpuClonerOptions_c.cpp GpuClonerOptions_c.h GpuIndicesOptions_c.h ../../gpu/GpuClonerOptions.h ../macros_impl.h

GpuIndex_c.o: CFLAGS += -I.. -I../.. $(CUDACFLAGS) $(DEBUGFLAG)
GpuIndex_c.o: GpuIndex_c.cpp GpuIndex_c.h ../../gpu/GpuIndex.h ../macros_impl.h

GpuResources_c.o: CFLAGS += -I.. -I../.. $(CUDACFLAGS) $(DEBUGFLAG)
GpuResources_c.o: GpuResources_c.cpp GpuResources_c.h ../../gpu/GpuResources.h ../macros_impl.h

StandardGpuResources_c.o: CFLAGS += -I.. -I../.. $(CUDACFLAGS) $(DEBUGFLAG)
StandardGpuResources_c.o: StandardGpuResources_c.cpp StandardGpuResources_c.h ../../gpu/StandardGpuResources.h ../macros_impl.h
