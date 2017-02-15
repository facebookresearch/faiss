
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

.SUFFIXES: .cpp .o

MAKEFILE_INC=makefile.inc
-include $(MAKEFILE_INC)

LIBNAME=libfaiss

all: .env_ok $(LIBNAME).a tests/demo_ivfpq_indexing
lua: .swig_ok lua/swigfaiss.$(SHAREDEXT)
py: .swig_ok _swigfaiss.so


#############################
# Various


LIBOBJ=hamming.o  utils.o \
       IndexFlat.o IndexIVF.o IndexLSH.o IndexPQ.o  \
       IndexIVFPQ.o   \
       Clustering.o Heap.o VectorTransform.o index_io.o \
       PolysemousTraining.o MetaIndexes.o Index.o \
       ProductQuantizer.o AutoTune.o AuxIndexStructures.o


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
	$(CC) $(CFLAGS) $< -o $@ $(BLASLDFLAGS) $(BLASCFLAGS)


tests/demo_ivfpq_indexing: tests/demo_ivfpq_indexing.cpp $(LIBNAME).a
	$(CC) -o $@ $(CFLAGS) -g -O3 $< $(LIBNAME).a  $(BLASLDFLAGS)

tests/demo_sift1M: tests/demo_sift1M.cpp $(LIBNAME).a
	$(CC) -o $@ $(CFLAGS) $< $(LIBNAME).a  $(BLASLDFLAGS)


#############################
# SWIG interfaces

HFILES = IndexFlat.h Index.h IndexLSH.h IndexPQ.h IndexIVF.h \
    IndexIVFPQ.h VectorTransform.h index_io.h utils.h \
    PolysemousTraining.h Heap.h MetaIndexes.h AuxIndexStructures.h \
    Clustering.h hamming.h AutoTune.h

# also silently generates python/swigfaiss.py
python/swigfaiss_wrap.cxx: swigfaiss.swig $(HFILES)
	$(SWIGEXEC) -python -c++ -o $@ $<


# extension is .so even on the mac
python/_swigfaiss.so: python/swigfaiss_wrap.cxx $(LIBNAME).a
	$(CC) -I. $(CFLAGS) $(LDFLAGS) $(PYTHONCFLAGS) $(SHAREDFLAGS) \
	-o $@ $^ $(BLASLDFLAGSSO)

_swigfaiss.so: python/_swigfaiss.so
	cp python/_swigfaiss.so python/swigfaiss.py .

#############################
# Dependencies

# for i in *.cpp ; do gcc -I.. -MM $i -msse4; done
AutoTune.o: AutoTune.cpp AutoTune.h Index.h FaissAssert.h utils.h Heap.h \
 IndexFlat.h VectorTransform.h IndexLSH.h IndexPQ.h ProductQuantizer.h \
 Clustering.h PolysemousTraining.h IndexIVF.h IndexIVFPQ.h MetaIndexes.h
AuxIndexStructures.o: AuxIndexStructures.cpp AuxIndexStructures.h Index.h
BinaryCode.o: BinaryCode.cpp BinaryCode.h VectorTransform.h Index.h \
 FaissAssert.h hamming.h Heap.h
Clustering.o: Clustering.cpp Clustering.h Index.h utils.h Heap.h \
 FaissAssert.h IndexFlat.h
hamming.o: hamming.cpp hamming.h Heap.h FaissAssert.h
Heap.o: Heap.cpp Heap.h
Index.o: Index.cpp IndexFlat.h Index.h FaissAssert.h
IndexFlat.o: IndexFlat.cpp IndexFlat.h Index.h utils.h Heap.h \
 FaissAssert.h
index_io.o: index_io.cpp index_io.h FaissAssert.h IndexFlat.h Index.h \
 VectorTransform.h IndexLSH.h IndexPQ.h ProductQuantizer.h Clustering.h \
 Heap.h PolysemousTraining.h IndexIVF.h IndexIVFPQ.h
IndexIVF.o: IndexIVF.cpp IndexIVF.h Index.h Clustering.h Heap.h utils.h \
 hamming.h FaissAssert.h IndexFlat.h AuxIndexStructures.h
IndexIVFPQ.o: IndexIVFPQ.cpp IndexIVFPQ.h IndexIVF.h Index.h Clustering.h \
 Heap.h IndexPQ.h ProductQuantizer.h PolysemousTraining.h utils.h \
 IndexFlat.h hamming.h FaissAssert.h AuxIndexStructures.h
IndexLSH.o: IndexLSH.cpp IndexLSH.h Index.h VectorTransform.h utils.h \
 Heap.h hamming.h FaissAssert.h
IndexNested.o: IndexNested.cpp IndexNested.h IndexIVF.h Index.h \
 Clustering.h Heap.h IndexIVFPQ.h IndexPQ.h ProductQuantizer.h \
 PolysemousTraining.h IndexFlat.h FaissAssert.h
IndexPQ.o: IndexPQ.cpp IndexPQ.h Index.h ProductQuantizer.h Clustering.h \
 Heap.h PolysemousTraining.h FaissAssert.h hamming.h

MetaIndexes.o: MetaIndexes.cpp MetaIndexes.h Index.h FaissAssert.h Heap.h
PolysemousTraining.o: PolysemousTraining.cpp PolysemousTraining.h \
 ProductQuantizer.h Clustering.h Index.h Heap.h utils.h hamming.h \
 FaissAssert.h
ProductQuantizer.o: ProductQuantizer.cpp ProductQuantizer.h Clustering.h \
 Index.h Heap.h FaissAssert.h VectorTransform.h IndexFlat.h utils.h
utils.o: utils.cpp utils.h Heap.h AuxIndexStructures.h Index.h \
 FaissAssert.h
VectorTransform.o: VectorTransform.cpp VectorTransform.h Index.h utils.h \
 Heap.h FaissAssert.h IndexPQ.h ProductQuantizer.h Clustering.h \
 PolysemousTraining.h

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
