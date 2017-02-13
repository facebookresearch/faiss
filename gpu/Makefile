# Copyright (c) 2015-present, Facebook, Inc. # All rights reserved. # # This source code is licensed under the CC-by-NC license found in the # LICENSE file in the root directory of this source tree. An additional grant # of patent rights can be found in the PATENTS file in the same directory.
.SUFFIXES: .cpp .o .cu

include ../makefile.inc


CC11=$(TP2)/gcc/4.9.x/centos6-native/108cf83/bin/g++

LIBNAME=libgpufaiss

all: $(LIBNAME).a test/demo_ivfpq_indexing_gpu

py:../_swigfaiss_gpu.so

CPPOBJ=     GpuResources.o \
            IndexProxy.o \
            StandardGpuResources.o \
	    GpuAutoTune.o \
            impl/RemapIndices.o

CUOBJ=      impl/BroadcastSum.o \
            impl/Distance.o \
            impl/FlatIndex.o \
            impl/InvertedListAppend.o \
            impl/IVFFlat.o \
            impl/IVFFlatScan.o \
            impl/IVFPQ.o \
            impl/IVFUtils.o \
            impl/IVFUtilsSelect1.o \
            impl/IVFUtilsSelect2.o \
            impl/L2Norm.o \
            impl/PQCodeDistances.o \
            impl/PQScanMultiPassNoPrecomputed.o \
            impl/PQScanMultiPassPrecomputed.o \
            impl/VectorResidual.o \
            GpuIndex.o \
	    GpuAutoTune.o \
            GpuIndexFlat.o \
            GpuIndexIVF.o \
            GpuIndexIVFFlat.o \
            GpuIndexIVFPQ.o


CUDAROOT=/usr/local/cuda-7.5/
CUDACFLAGS=-I$(CUDAROOT)/include

NVCC=$(CUDAROOT)/bin/nvcc


NVCCFLAGS= $(CUDAFLAGS) \
   -I $(CUDAROOT)/targets/x86_64-linux/include/ \
   -Xcompiler -fPIC \
   -Xcudafe --diag_suppress=unrecognized_attribute \
   -gencode arch=compute_35,code="compute_35" \
   -gencode arch=compute_52,code="compute_52" \
   --std c++11 -lineinfo \
   -ccbin $(CC11) -DFAISS_USE_FLOAT16


.cpp.o:
	$(CC11) -std=c++11 -fPIC -m64 -Wall -g -Wno-sign-compare -O3 -fopenmp \
         -c $< -o $@ $(EXTRAFLAGS) $(CUDACFLAGS)

.cu.o:
	$(NVCC) $(NVCCFLAGS) -g -O3 \
         -c $< -o $@ $(EXTRAFLAGS)


$(LIBNAME).a: $(CPPOBJ) $(CUOBJ)
	ar r $@ $^


# here we replace -Wl, with -Xlinker for nvcc

BLASLDFLAGSNVCC= -Xlinker --no-as-needed,-rpath,$(MKLROOT)/lib/intel64 \
           -L$(MKLROOT)/lib/intel64 \
	   -lmkl_intel_ilp64 -lmkl_core -lmkl_gnu_thread \
           -ldl -lpthread     -lm

# set
# LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64/:/home/matthijs/fbsource/fbcode/third-party2/gcc/4.9.x/centos6-native/108cf83/lib64/

test/demo_ivfpq_indexing_gpu: test/demo_ivfpq_indexing_gpu.cpp $(LIBNAME).a ../libfaiss.a
	$(NVCC) $(NVCCFLAGS) -o $@ $^ -Xcompiler -fopenmp -lcublas \
	$(BLASLDFLAGSNVCC)

#############################################################
# Python interface
#############################################################


# also silently generates python/swigfaiss.py
../python/swigfaiss_gpu_wrap.cxx: ../python/swigfaiss.swig
	cd ../python; $(SWIGEXEC) -python -c++ -DGPU_WRAPPER -o $@ $<

BLASLDFLAGSSONVCC=\
        -Xlinker $(MKLROOT)/lib/intel64/libmkl_intel_ilp64.a \
	-Xlinker $(MKLROOT)/lib/intel64/libmkl_core.a \
        -Xlinker $(MKLROOT)/lib/intel64/libmkl_gnu_thread.a



# extension is .so even on the mac
../python/_swigfaiss_gpu.so: ../python/swigfaiss_gpu_wrap.cxx $(LIBNAME).a ../libfaiss.a
	$(NVCC) $(NVCCFLAGS) $(PYTHONCFLAGS) -shared -o $@ $^ -Xcompiler -fopenmp -lcublas \
	$(BLASLDFLAGSSONVCC)

../_swigfaiss_gpu.so: ../python/_swigfaiss_gpu.so
	cp ../python/_swigfaiss_gpu.so ../python/swigfaiss_gpu.py ..



clean:
	rm -rf *.o impl/*.o utils/*.o test/*.o $(LIBNAME).a \
                 ../python/*swigfaiss_gpu* ../*swigfaiss_gpu*



# for i in *.cu */*.cu; do echo -n ${i%%/*}/;  cpp -MM $i; done
dep:
	for i in $(patsubst %.o,%.cpp,$(CPPOBJ)) $(patsubst %.o,%.cu,$(CUOBJ)); do \
	     echo -n $${i%/*}/ ; \
             cpp -MM -std=gnu++0x $$i; \
	done




GpuResources.cpp/GpuResources.o: GpuResources.cpp GpuResources.h utils/DeviceMemory.h \
 utils/DeviceUtils.h
IndexProxy.cpp/IndexProxy.o: IndexProxy.cpp IndexProxy.h ../Index.h utils/WorkerThread.h \
 ../FaissAssert.h ../Clustering.h ../Index.h GpuIndexFlat.h GpuIndex.h \
 StandardGpuResources.h GpuResources.h utils/DeviceMemory.h \
 utils/StackDeviceMemory.h utils/DeviceUtils.h
StandardGpuResources.cpp/StandardGpuResources.o: StandardGpuResources.cpp StandardGpuResources.h \
 GpuResources.h utils/DeviceMemory.h utils/StackDeviceMemory.h \
 utils/DeviceUtils.h ../FaissAssert.h
impl/BroadcastSum.o: impl/BroadcastSum.cu impl/../../FaissAssert.h \
 impl/../utils/DeviceUtils.h impl/../utils/MathOperators.cuh \
 impl/../utils/Float16.cuh impl/../utils/../GpuResources.h \
 impl/../utils/../utils/DeviceMemory.h impl/../utils/DeviceTensor.cuh \
 impl/../utils/Tensor.cuh impl/../utils/Tensor-inl.cuh \
 impl/../utils/../../FaissAssert.h impl/../utils/DeviceTensor-inl.cuh \
 impl/../utils/StaticUtils.h
impl/Distance.o: impl/Distance.cu impl/Distance.cuh \
 impl/../utils/DeviceTensor.cuh impl/../utils/Tensor.cuh \
 impl/../utils/Tensor-inl.cuh impl/../utils/../../FaissAssert.h \
 impl/../utils/DeviceUtils.h impl/../utils/DeviceMemory.h \
 impl/../utils/DeviceTensor-inl.cuh impl/../utils/Float16.cuh \
 impl/../utils/../GpuResources.h impl/BroadcastSum.cuh impl/L2Norm.cuh \
 impl/../../FaissAssert.h impl/../utils/Limits.cuh impl/../utils/Pair.cuh \
 impl/../utils/MathOperators.cuh impl/../utils/WarpShuffles.cuh \
 impl/../utils/DeviceDefs.cuh impl/../utils/MatrixMult.cuh \
 impl/../utils/BlockSelectKernel.cuh impl/../utils/Select.cuh \
 impl/../utils/Comparators.cuh impl/../utils/MergeNetworkBlock.cuh \
 impl/../utils/PtxUtils.cuh impl/../utils/StaticUtils.h \
 impl/../utils/MergeNetworkWarp.cuh impl/../utils/Reductions.cuh \
 impl/../utils/ReductionOperators.cuh
impl/FlatIndex.o: impl/FlatIndex.cu impl/FlatIndex.cuh \
 impl/../utils/DeviceTensor.cuh impl/../utils/Tensor.cuh \
 impl/../utils/Tensor-inl.cuh impl/../utils/../../FaissAssert.h \
 impl/../utils/DeviceUtils.h impl/../utils/DeviceMemory.h \
 impl/../utils/DeviceTensor-inl.cuh impl/../utils/DeviceVector.cuh \
 impl/../utils/StaticUtils.h impl/../utils/Float16.cuh \
 impl/../utils/../GpuResources.h impl/Distance.cuh impl/L2Norm.cuh \
 impl/../utils/CopyUtils.cuh impl/../utils/HostTensor.cuh \
 impl/../utils/HostTensor-inl.cuh
impl/InvertedListAppend.o: impl/InvertedListAppend.cu \
 impl/InvertedListAppend.cuh impl/../GpuIndicesOptions.h \
 impl/../utils/Tensor.cuh impl/../utils/Tensor-inl.cuh \
 impl/../utils/../../FaissAssert.h impl/../utils/DeviceUtils.h \
 impl/../../FaissAssert.h impl/../utils/Float16.cuh \
 impl/../utils/../GpuResources.h impl/../utils/../utils/DeviceMemory.h \
 impl/../utils/DeviceTensor.cuh impl/../utils/DeviceTensor-inl.cuh \
 impl/../utils/StaticUtils.h
impl/IVFFlat.o: impl/IVFFlat.cu impl/IVFFlat.cuh impl/../GpuIndicesOptions.h \
 impl/../utils/DeviceVector.cuh impl/../utils/../../FaissAssert.h \
 impl/../utils/DeviceUtils.h impl/../utils/StaticUtils.h \
 impl/../utils/DeviceTensor.cuh impl/../utils/Tensor.cuh \
 impl/../utils/Tensor-inl.cuh impl/../utils/DeviceMemory.h \
 impl/../utils/DeviceTensor-inl.cuh impl/../GpuResources.h \
 impl/FlatIndex.cuh impl/../utils/Float16.cuh impl/InvertedListAppend.cuh \
 impl/IVFFlatScan.cuh impl/RemapIndices.h impl/../utils/CopyUtils.cuh \
 impl/../utils/HostTensor.cuh impl/../utils/HostTensor-inl.cuh \
 impl/../utils/DeviceDefs.cuh impl/../utils/Transpose.cuh
impl/IVFFlatScan.o: impl/IVFFlatScan.cu impl/IVFFlatScan.cuh \
 impl/../GpuIndicesOptions.h impl/../utils/Tensor.cuh \
 impl/../utils/Tensor-inl.cuh impl/../utils/../../FaissAssert.h \
 impl/../utils/DeviceUtils.h impl/../GpuResources.h \
 impl/../utils/DeviceMemory.h impl/IVFUtils.cuh \
 impl/../utils/ConversionOperators.cuh impl/../utils/Float16.cuh \
 impl/../utils/DeviceTensor.cuh impl/../utils/DeviceTensor-inl.cuh \
 impl/../utils/DeviceDefs.cuh impl/../utils/MathOperators.cuh \
 impl/../utils/LoadStoreOperators.cuh impl/../utils/PtxUtils.cuh \
 impl/../utils/Reductions.cuh impl/../utils/ReductionOperators.cuh \
 impl/../utils/Limits.cuh impl/../utils/Pair.cuh \
 impl/../utils/WarpShuffles.cuh impl/../utils/StaticUtils.h
impl/IVFPQ.o: impl/IVFPQ.cu impl/IVFPQ.cuh impl/../GpuIndicesOptions.h \
 impl/../utils/DeviceVector.cuh impl/../utils/../../FaissAssert.h \
 impl/../utils/DeviceUtils.h impl/../utils/StaticUtils.h \
 impl/../utils/DeviceTensor.cuh impl/../utils/Tensor.cuh \
 impl/../utils/Tensor-inl.cuh impl/../utils/DeviceMemory.h \
 impl/../utils/DeviceTensor-inl.cuh impl/../utils/Float16.cuh \
 impl/../utils/../GpuResources.h impl/BroadcastSum.cuh impl/Distance.cuh \
 impl/FlatIndex.cuh impl/InvertedListAppend.cuh impl/L2Norm.cuh \
 impl/PQCodeDistances.cuh impl/../utils/NoTypeTensor.cuh \
 impl/PQScanMultiPassNoPrecomputed.cuh \
 impl/PQScanMultiPassPrecomputed.cuh impl/RemapIndices.h \
 impl/VectorResidual.cuh impl/../utils/DeviceDefs.cuh \
 impl/../utils/HostTensor.cuh impl/../utils/HostTensor-inl.cuh \
 impl/../utils/MatrixMult.cuh impl/../utils/Transpose.cuh
impl/IVFUtils.o: impl/IVFUtils.cu impl/IVFUtils.cuh \
 impl/../GpuIndicesOptions.h impl/../utils/Tensor.cuh \
 impl/../utils/Tensor-inl.cuh impl/../utils/../../FaissAssert.h \
 impl/../utils/DeviceUtils.h impl/../utils/StaticUtils.h \
 impl/../utils/ThrustAllocator.cuh
impl/IVFUtilsSelect1.o: impl/IVFUtilsSelect1.cu impl/IVFUtils.cuh \
 impl/../GpuIndicesOptions.h impl/../utils/Tensor.cuh \
 impl/../utils/Tensor-inl.cuh impl/../utils/../../FaissAssert.h \
 impl/../utils/DeviceUtils.h impl/../utils/Select.cuh \
 impl/../utils/Comparators.cuh impl/../utils/Float16.cuh \
 impl/../utils/../GpuResources.h impl/../utils/../utils/DeviceMemory.h \
 impl/../utils/DeviceTensor.cuh impl/../utils/DeviceTensor-inl.cuh \
 impl/../utils/DeviceDefs.cuh impl/../utils/MergeNetworkBlock.cuh \
 impl/../utils/PtxUtils.cuh impl/../utils/StaticUtils.h \
 impl/../utils/WarpShuffles.cuh impl/../utils/MergeNetworkWarp.cuh \
 impl/../utils/Reductions.cuh impl/../utils/ReductionOperators.cuh \
 impl/../utils/Limits.cuh impl/../utils/Pair.cuh \
 impl/../utils/MathOperators.cuh
impl/IVFUtilsSelect2.o: impl/IVFUtilsSelect2.cu impl/IVFUtils.cuh \
 impl/../GpuIndicesOptions.h impl/../utils/Tensor.cuh \
 impl/../utils/Tensor-inl.cuh impl/../utils/../../FaissAssert.h \
 impl/../utils/DeviceUtils.h impl/../utils/Select.cuh \
 impl/../utils/Comparators.cuh impl/../utils/Float16.cuh \
 impl/../utils/../GpuResources.h impl/../utils/../utils/DeviceMemory.h \
 impl/../utils/DeviceTensor.cuh impl/../utils/DeviceTensor-inl.cuh \
 impl/../utils/DeviceDefs.cuh impl/../utils/MergeNetworkBlock.cuh \
 impl/../utils/PtxUtils.cuh impl/../utils/StaticUtils.h \
 impl/../utils/WarpShuffles.cuh impl/../utils/MergeNetworkWarp.cuh \
 impl/../utils/Reductions.cuh impl/../utils/ReductionOperators.cuh \
 impl/../utils/Limits.cuh impl/../utils/Pair.cuh \
 impl/../utils/MathOperators.cuh
impl/L2Norm.o: impl/L2Norm.cu impl/L2Norm.cuh impl/../utils/Float16.cuh \
 impl/../utils/../GpuResources.h impl/../utils/../utils/DeviceMemory.h \
 impl/../utils/DeviceTensor.cuh impl/../utils/Tensor.cuh \
 impl/../utils/Tensor-inl.cuh impl/../utils/../../FaissAssert.h \
 impl/../utils/DeviceUtils.h impl/../utils/DeviceTensor-inl.cuh \
 impl/../../FaissAssert.h impl/../utils/ConversionOperators.cuh \
 impl/../utils/DeviceDefs.cuh impl/../utils/MathOperators.cuh \
 impl/../utils/PtxUtils.cuh impl/../utils/StaticUtils.h \
 impl/../utils/Reductions.cuh impl/../utils/ReductionOperators.cuh \
 impl/../utils/Limits.cuh impl/../utils/Pair.cuh \
 impl/../utils/WarpShuffles.cuh
impl/PQCodeDistances.o: impl/PQCodeDistances.cu impl/PQCodeDistances.cuh \
 impl/../utils/Tensor.cuh impl/../utils/Tensor-inl.cuh \
 impl/../utils/../../FaissAssert.h impl/../utils/DeviceUtils.h \
 impl/../utils/NoTypeTensor.cuh impl/BroadcastSum.cuh \
 impl/../utils/Float16.cuh impl/../utils/../GpuResources.h \
 impl/../utils/../utils/DeviceMemory.h impl/../utils/DeviceTensor.cuh \
 impl/../utils/DeviceTensor-inl.cuh impl/Distance.cuh impl/L2Norm.cuh \
 impl/../utils/DeviceDefs.cuh impl/../utils/MatrixMult.cuh \
 impl/../utils/PtxUtils.cuh impl/../utils/StaticUtils.h \
 impl/../utils/Transpose.cuh
impl/PQScanMultiPassNoPrecomputed.o: impl/PQScanMultiPassNoPrecomputed.cu \
 impl/PQScanMultiPassNoPrecomputed.cuh impl/../GpuIndicesOptions.h \
 impl/../utils/Tensor.cuh impl/../utils/Tensor-inl.cuh \
 impl/../utils/../../FaissAssert.h impl/../utils/DeviceUtils.h \
 impl/../GpuResources.h impl/../utils/DeviceMemory.h \
 impl/PQCodeDistances.cuh impl/../utils/NoTypeTensor.cuh \
 impl/PQCodeLoad.cuh impl/../utils/PtxUtils.cuh impl/IVFUtils.cuh \
 impl/../utils/ConversionOperators.cuh impl/../utils/Float16.cuh \
 impl/../utils/DeviceTensor.cuh impl/../utils/DeviceTensor-inl.cuh \
 impl/../utils/LoadStoreOperators.cuh impl/../utils/StaticUtils.h \
 impl/../utils/HostTensor.cuh impl/../utils/HostTensor-inl.cuh
impl/PQScanMultiPassPrecomputed.o: impl/PQScanMultiPassPrecomputed.cu \
 impl/PQScanMultiPassPrecomputed.cuh impl/../GpuIndicesOptions.h \
 impl/../utils/Tensor.cuh impl/../utils/Tensor-inl.cuh \
 impl/../utils/../../FaissAssert.h impl/../utils/DeviceUtils.h \
 impl/../utils/NoTypeTensor.cuh impl/../GpuResources.h \
 impl/../utils/DeviceMemory.h impl/PQCodeLoad.cuh \
 impl/../utils/PtxUtils.cuh impl/IVFUtils.cuh \
 impl/../utils/ConversionOperators.cuh impl/../utils/Float16.cuh \
 impl/../utils/DeviceTensor.cuh impl/../utils/DeviceTensor-inl.cuh \
 impl/../utils/LoadStoreOperators.cuh impl/../utils/MathOperators.cuh \
 impl/../utils/StaticUtils.h
impl/VectorResidual.o: impl/VectorResidual.cu impl/VectorResidual.cuh \
 impl/../utils/Tensor.cuh impl/../utils/Tensor-inl.cuh \
 impl/../utils/../../FaissAssert.h impl/../utils/DeviceUtils.h \
 impl/../utils/Float16.cuh impl/../utils/../GpuResources.h \
 impl/../utils/../utils/DeviceMemory.h impl/../utils/DeviceTensor.cuh \
 impl/../utils/DeviceTensor-inl.cuh impl/../../FaissAssert.h \
 impl/../utils/ConversionOperators.cuh impl/../utils/StaticUtils.h
GpuIndex.cu/GpuIndex.o: GpuIndex.cu GpuIndex.h ../Index.h ../FaissAssert.h \
 GpuResources.h utils/DeviceMemory.h utils/DeviceUtils.h
GpuIndexFlat.cu/GpuIndexFlat.o: GpuIndexFlat.cu GpuIndexFlat.h GpuIndex.h ../Index.h \
 ../IndexFlat.h ../Index.h GpuResources.h utils/DeviceMemory.h \
 impl/FlatIndex.cuh impl/../utils/DeviceTensor.cuh \
 impl/../utils/Tensor.cuh impl/../utils/Tensor-inl.cuh \
 impl/../utils/../../FaissAssert.h impl/../utils/DeviceUtils.h \
 impl/../utils/DeviceTensor-inl.cuh impl/../utils/DeviceVector.cuh \
 impl/../utils/StaticUtils.h impl/../utils/Float16.cuh \
 utils/CopyUtils.cuh utils/HostTensor.cuh utils/HostTensor-inl.cuh
GpuIndexIVF.cu/GpuIndexIVF.o: GpuIndexIVF.cu GpuIndexIVF.h GpuIndex.h ../Index.h \
 GpuIndicesOptions.h ../Clustering.h ../Index.h ../FaissAssert.h \
 ../IndexFlat.h ../IndexIVF.h ../Clustering.h ../Heap.h GpuIndexFlat.h \
 utils/DeviceUtils.h utils/Float16.cuh utils/../GpuResources.h \
 utils/../utils/DeviceMemory.h utils/DeviceTensor.cuh utils/Tensor.cuh \
 utils/Tensor-inl.cuh utils/../../FaissAssert.h \
 utils/DeviceTensor-inl.cuh
GpuIndexIVFFlat.cu/GpuIndexIVFFlat.o: GpuIndexIVFFlat.cu GpuIndexIVFFlat.h GpuIndexIVF.h \
 GpuIndex.h ../Index.h GpuIndicesOptions.h ../Clustering.h ../Index.h \
 ../IndexFlat.h ../IndexIVF.h ../Clustering.h ../Heap.h GpuIndexFlat.h \
 GpuResources.h utils/DeviceMemory.h impl/IVFFlat.cuh \
 impl/../utils/DeviceVector.cuh impl/../utils/../../FaissAssert.h \
 impl/../utils/DeviceUtils.h impl/../utils/StaticUtils.h \
 impl/../utils/DeviceTensor.cuh impl/../utils/Tensor.cuh \
 impl/../utils/Tensor-inl.cuh impl/../utils/DeviceTensor-inl.cuh \
 utils/CopyUtils.cuh utils/HostTensor.cuh utils/HostTensor-inl.cuh \
 utils/Float16.cuh
GpuIndexIVFPQ.cu/GpuIndexIVFPQ.o: GpuIndexIVFPQ.cu GpuIndexIVFPQ.h GpuIndexIVF.h \
 GpuIndex.h ../Index.h GpuIndicesOptions.h ../Clustering.h ../Index.h \
 ../ProductQuantizer.h ../Clustering.h ../Heap.h GpuIndexFlat.h \
 GpuResources.h utils/DeviceMemory.h impl/IVFPQ.cuh \
 impl/../utils/DeviceVector.cuh impl/../utils/../../FaissAssert.h \
 impl/../utils/DeviceUtils.h impl/../utils/StaticUtils.h \
 impl/../utils/DeviceTensor.cuh impl/../utils/Tensor.cuh \
 impl/../utils/Tensor-inl.cuh impl/../utils/DeviceTensor-inl.cuh \
 impl/../utils/Float16.cuh utils/CopyUtils.cuh utils/HostTensor.cuh \
 utils/HostTensor-inl.cuh ../IndexFlat.h ../IndexIVFPQ.h ../IndexIVF.h \
 ../IndexPQ.h ../ProductQuantizer.h ../PolysemousTraining.h
