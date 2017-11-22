# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD+Patents license found in the
# LICENSE file in the root directory of this source tree.

.SUFFIXES: .cpp .o


MAKEFILE_INC=makefile.inc

-include $(MAKEFILE_INC)

LIBNAME=libfaiss

all: .env_ok $(LIBNAME).a tests/demo_ivfpq_indexing

py: _swigfaiss.so



#############################
# Various


LIBOBJ=hamming.o  utils.o \
       IndexFlat.o IndexIVF.o IndexLSH.o IndexPQ.o  \
       IndexIVFPQ.o   \
       Clustering.o Heap.o VectorTransform.o index_io.o \
       PolysemousTraining.o MetaIndexes.o Index.o \
       ProductQuantizer.o AutoTune.o AuxIndexStructures.o \
       IndexScalarQuantizer.o FaissException.o


$(LIBNAME).a: $(LIBOBJ)
	ar r $(LIBNAME).a $^

$(LIBNAME).$(SHAREDEXT): $(LIBOBJ)
	$(CC) $(LDFLAGS) $(FAISSSHAREDFLAGS) -o $(LIBNAME).$(SHAREDEXT) $^ $(BLASLDFLAGS)

.cpp.o:
	$(CC) $(CFLAGS) -c $< -o $@ $(FLAGS) $(EXTRAFLAGS)

utils.o:             EXTRAFLAGS=$(BLASCFLAGS)
VectorTransform.o:   EXTRAFLAGS=$(BLASCFLAGS)
ProductQuantizer.o:  EXTRAFLAGS=$(BLASCFLAGS)

# for MKL, the flags when generating a dynamic lib are different from
# the ones when making an executable, but by default they are the same

BLASLDFLAGSSO ?= $(BLASLDFLAGS)


#############################
# pure C++ test in the test directory

tests/test_blas: tests/test_blas.cpp
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(BLASLDFLAGS) $(BLASCFLAGS)


tests/demo_ivfpq_indexing: tests/demo_ivfpq_indexing.cpp $(LIBNAME).a
	$(CC) -o $@ $(CFLAGS) $< $(LIBNAME).a $(LDFLAGS) $(BLASLDFLAGS)

tests/demo_sift1M: tests/demo_sift1M.cpp $(LIBNAME).a
	$(CC) -o $@ $(CFLAGS) $< $(LIBNAME).a $(LDFLAGS) $(BLASLDFLAGS)


#############################
# SWIG interfaces

HFILES = IndexFlat.h Index.h IndexLSH.h IndexPQ.h IndexIVF.h \
    IndexIVFPQ.h VectorTransform.h index_io.h utils.h \
    PolysemousTraining.h Heap.h MetaIndexes.h AuxIndexStructures.h \
    Clustering.h hamming.h AutoTune.h IndexScalarQuantizer.h FaissException.h

# also silently generates python/swigfaiss.py
python/swigfaiss_wrap.cxx: swigfaiss.swig $(HFILES)
	$(SWIGEXEC) -python -c++ -Doverride= -o $@ $<


# extension is .so even on the mac
python/_swigfaiss.so: python/swigfaiss_wrap.cxx $(LIBNAME).a
	$(CC) -I. $(CFLAGS) $(LDFLAGS) $(PYTHONCFLAGS) $(SHAREDFLAGS) \
	-o $@ $^ $(BLASLDFLAGSSO)

_swigfaiss.so: python/_swigfaiss.so
	cp python/_swigfaiss.so python/swigfaiss.py .

#############################
# Dependencies.
# make dep > x
# then copy/paste from x by hand below

dep:
	for i in $(patsubst %.o,%.cpp,$(LIBOBJ)) ; do \
	   cpp -MM -std=gnu++0x $$i ; \
	done

hamming.o: hamming.cpp hamming.h Heap.h FaissAssert.h FaissException.h
utils.o: utils.cpp utils.h Heap.h AuxIndexStructures.h Index.h \
 FaissAssert.h FaissException.h
IndexFlat.o: IndexFlat.cpp IndexFlat.h Index.h utils.h Heap.h \
 FaissAssert.h FaissException.h AuxIndexStructures.h
IndexIVF.o: IndexIVF.cpp IndexIVF.h Index.h Clustering.h Heap.h utils.h \
 hamming.h FaissAssert.h FaissException.h IndexFlat.h \
 AuxIndexStructures.h
IndexLSH.o: IndexLSH.cpp IndexLSH.h Index.h VectorTransform.h utils.h \
 Heap.h hamming.h FaissAssert.h FaissException.h
IndexPQ.o: IndexPQ.cpp IndexPQ.h Index.h ProductQuantizer.h Clustering.h \
 Heap.h PolysemousTraining.h FaissAssert.h FaissException.h hamming.h
IndexIVFPQ.o: IndexIVFPQ.cpp IndexIVFPQ.h IndexIVF.h Index.h Clustering.h \
 Heap.h IndexPQ.h ProductQuantizer.h PolysemousTraining.h utils.h \
 IndexFlat.h hamming.h FaissAssert.h FaissException.h \
 AuxIndexStructures.h
Clustering.o: Clustering.cpp Clustering.h Index.h utils.h Heap.h \
 FaissAssert.h FaissException.h IndexFlat.h
Heap.o: Heap.cpp Heap.h
VectorTransform.o: VectorTransform.cpp VectorTransform.h Index.h utils.h \
 Heap.h FaissAssert.h FaissException.h IndexPQ.h ProductQuantizer.h \
 Clustering.h PolysemousTraining.h
index_io.o: index_io.cpp index_io.h FaissAssert.h FaissException.h \
 IndexFlat.h Index.h VectorTransform.h IndexLSH.h IndexPQ.h \
 ProductQuantizer.h Clustering.h Heap.h PolysemousTraining.h IndexIVF.h \
 IndexIVFPQ.h MetaIndexes.h IndexScalarQuantizer.h
PolysemousTraining.o: PolysemousTraining.cpp PolysemousTraining.h \
 ProductQuantizer.h Clustering.h Index.h Heap.h utils.h hamming.h \
 FaissAssert.h FaissException.h
MetaIndexes.o: MetaIndexes.cpp MetaIndexes.h Index.h FaissAssert.h \
 FaissException.h Heap.h AuxIndexStructures.h
Index.o: Index.cpp IndexFlat.h Index.h FaissAssert.h FaissException.h
ProductQuantizer.o: ProductQuantizer.cpp ProductQuantizer.h Clustering.h \
 Index.h Heap.h FaissAssert.h FaissException.h VectorTransform.h \
 IndexFlat.h utils.h
AutoTune.o: AutoTune.cpp AutoTune.h Index.h FaissAssert.h \
 FaissException.h utils.h Heap.h IndexFlat.h VectorTransform.h IndexLSH.h \
 IndexPQ.h ProductQuantizer.h Clustering.h PolysemousTraining.h \
 IndexIVF.h IndexIVFPQ.h MetaIndexes.h IndexScalarQuantizer.h
AuxIndexStructures.o: AuxIndexStructures.cpp AuxIndexStructures.h Index.h
IndexScalarQuantizer.o: IndexScalarQuantizer.cpp IndexScalarQuantizer.h \
 IndexIVF.h Index.h Clustering.h Heap.h utils.h FaissAssert.h \
 FaissException.h
FaissException.o: FaissException.cpp FaissException.h


clean:
	rm -f $(LIBNAME).a $(LIBNAME).$(SHAREDEXT)* *.o \
	   	lua/swigfaiss.so lua/swigfaiss_wrap.cxx \
		python/_swigfaiss.so python/swigfaiss_wrap.cxx \
		python/swigfaiss.py _swigfaiss.so swigfaiss.py

.env_ok:
ifeq ($(wildcard $(MAKEFILE_INC)),)
	$(error Cannot find $(MAKEFILE_INC). Did you forget to copy the relevant file from ./example_makefiles?)
endif
ifeq ($(shell command -v $(CC) 2>/dev/null),)
	$(error Cannot find $(CC), please refer to $(CURDIR)/makefile.inc to set up your environment)
endif

.swig_ok: .env_ok
ifeq ($(shell command -v $(SWIGEXEC) 2>/dev/null),)
	$(error Cannot find $(SWIGEXEC), please refer to $(CURDIR)/makefile.inc to set up your environment)
endif
