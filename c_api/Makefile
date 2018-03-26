# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD+Patents license found in the
# LICENSE file in the root directory of this source tree.

.SUFFIXES: .cpp .o

# C API

include ../makefile.inc
DEBUGFLAG=-DNDEBUG # no debugging

LIBNAME=libfaiss
CLIBNAME=libfaiss_c
LIBCOBJ=error_impl.o Index_c.o IndexFlat_c.o Clustering_c.o \
	AuxIndexStructures_c.o AutoTune_c.o IndexIVF_c.o IndexIVFFlat_c.o

# Build shared object file by default
all: $(CLIBNAME).$(SHAREDEXT)

# Build static library (requires consumers to link with libstdc++)
$(CLIBNAME).a: $(LIBCOBJ) ../$(LIBNAME).a
	ar r $@ $^

# Build dynamic library
$(CLIBNAME).$(SHAREDEXT): $(LIBCOBJ) ../$(LIBNAME).a
	$(CXX) $(LDFLAGS) $(FAISSSHAREDFLAGS) -o $@ \
	-Wl,--whole-archive $^ $(BLASLDFLAGS) -Wl,--no-whole-archive -static-libstdc++

bin/example_c: example_c.c $(CLIBNAME).$(SHAREDEXT)
	$(CC) $(CFLAGS) -std=c99 -I. -I.. -L. -o $@ example_c.c \
	$(LDFLAGS) -lm $(BLASLDFLAGS) -lfaiss_c

clean:
	rm -f $(CLIBNAME).a $(CLIBNAME).$(SHAREDEXT)* *.o bin/example_c

# Dependencies

error_impl.o: CFLAGS += -I.. $(DEBUGFLAG)
error_impl.o: error_impl.cpp error_c.h error_impl.h macros_impl.h

Index_c.o: CFLAGS += -I.. $(DEBUGFLAG)
Index_c.o: Index_c.cpp Index_c.h ../Index.h macros_impl.h

IndexFlat_c.o: CFLAGS += -I.. $(DEBUGFLAG)
IndexFlat_c.o: IndexFlat_c.cpp IndexFlat_c.h ../IndexFlat.h macros_impl.h

IndexIVF_c.o: CFLAGS += -I.. $(DEBUGFLAG)
IndexIVF_c.o: IndexIVF_c.cpp IndexIVF_c.h ../IndexIVF.h macros_impl.h

IndexIVFFlat_c.o: CFLAGS += -I.. $(DEBUGFLAG)
IndexIVFFlat_c.o: IndexIVFFlat_c.cpp IndexIVFFlat_c.h ../IndexIVFFlat.h macros_impl.h

Clustering_c.o: CFLAGS += -I.. $(DEBUGFLAG)
Clustering_c.o: Clustering_c.cpp Clustering_c.h ../Clustering.h macros_impl.h

AutoTune_c.o: CFLAGS += -I.. $(DEBUGFLAG)
AutoTune_c.o: AutoTune_c.cpp AutoTune_c.h ../AutoTune.h macros_impl.h

AuxIndexStructures_c.o: CFLAGS += -I.. $(DEBUGFLAG)
AuxIndexStructures_c.o: AuxIndexStructures_c.cpp AuxIndexStructures_c.h ../AuxIndexStructures.h macros_impl.h
