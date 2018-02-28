# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD+Patents license found in the
# LICENSE file in the root directory of this source tree.

.SUFFIXES: .cpp .o .cu

include ../makefile.inc






LIBNAME=libgpufaiss

all: $(LIBNAME).a test/demo_ivfpq_indexing_gpu

py:../_swigfaiss_gpu.so

CPPOBJ=     GpuResources.o \
            IndexProxy.o \
            StandardGpuResources.o \
	    GpuAutoTune.o \
	    GpuClonerOptions.o \
            impl/RemapIndices.o \
	    utils/DeviceMemory.o \
            utils/StackDeviceMemory.o \
            utils/DeviceUtils.o \
            utils/Timer.o \
            utils/MemorySpace.o \
            utils/WorkerThread.o


INS=1 32 64 128 256 F512 T512 F1024 T1024


CUOBJ=      impl/BroadcastSum.o \
            impl/Distance.o \
            impl/FlatIndex.o \
            impl/InvertedListAppend.o \
            impl/IVFBase.o \
            impl/IVFFlat.o \
            impl/IVFFlatScan.o \
            impl/IVFPQ.o \
            impl/IVFUtils.o \
            impl/IVFUtilsSelect1.o \
            impl/IVFUtilsSelect2.o \
            impl/L2Norm.o \
            impl/L2Select.o \
            impl/PQCodeDistances.o \
            impl/PQScanMultiPassNoPrecomputed.o \
            impl/PQScanMultiPassPrecomputed.o \
            impl/VectorResidual.o \
            GpuIndex.o \
            GpuIndexFlat.o \
            GpuIndexIVF.o \
            GpuIndexIVFFlat.o \
            GpuIndexIVFPQ.o  \
            utils/Float16.o \
            utils/MatrixMult.o \
            utils/BlockSelectFloat.o \
            utils/BlockSelectHalf.o \
            utils/WarpSelectFloat.o \
            utils/WarpSelectHalf.o \
	    utils/nvidia/fp16_emu.o \
	    $(foreach v,$(INS),   \
	        utils/blockselect/BlockSelectHalf$(v).o \
                utils/blockselect/BlockSelectFloat$(v).o \
                utils/warpselect/WarpSelectHalf$(v).o \
                utils/warpselect/WarpSelectFloat$(v).o \
             )


.cpp.o:
	$(CXX) $(CXXFLAGS) -fPIC -m64 -Wall -g -Wno-sign-compare -O3 -fopenmp \
         -c $< -o $@ $(EXTRAFLAGS) $(CUDACFLAGS)

.cu.o:
	$(NVCC) $(NVCCFLAGS) -g -O3 \
         -c $< -o $@ $(EXTRAFLAGS)


$(LIBNAME).a: $(CPPOBJ) $(CUOBJ)
	ar r $@ $^

$(LIBNAME).$(SHAREDEXT): $(CPPOBJ) $(CUOBJ)
	$(CXX) -L$(CUDAROOT)/lib64 -lcublas -lcudart $(LDFLAGS) \
	$(FAISSSHAREDFLAGS) -o $(LIBNAME).$(SHAREDEXT) $^


test/demo_ivfpq_indexing_gpu: test/demo_ivfpq_indexing_gpu.cpp \
	$(LIBNAME).a ../libfaiss.a
	$(NVCC) $(NVCCFLAGS) -o $@ $^ -Xcompiler -fopenmp -lcublas \
	$(BLASLDFLAGSNVCC)

#############################################################
# Python interface
#############################################################


# also silently generates python/swigfaiss.py
../python/swigfaiss_gpu_wrap.cxx: ../swigfaiss.swig
	cd ../python; $(SWIGEXEC) -Doverride= -python -c++ -DGPU_WRAPPER -o $@ $<


# extension is .so even on the mac
../python/_swigfaiss_gpu.so: ../python/swigfaiss_gpu_wrap.cxx \
             $(LIBNAME).a ../libfaiss.a
	$(NVCC) -I.. $(NVCCFLAGS) $(PYTHONCFLAGS) -shared -o $@ $^ \
	-Xcompiler -fopenmp -lcublas \
	$(BLASLDFLAGSSONVCC)

../_swigfaiss_gpu.so: ../python/_swigfaiss_gpu.so
	cp ../python/_swigfaiss_gpu.so ../python/swigfaiss_gpu.py ..



clean:
	rm -rf *.o impl/*.o utils/*.o test/*.o $(LIBNAME).a \
		 $(LIBNAME).$(SHAREDEXT) \
                 ../python/*swigfaiss_gpu* ../*swigfaiss_gpu*



# make dep > x
# then copy x below
dep:
	for i in $(patsubst %.o,%.cpp,$(CPPOBJ)) \
                 $(patsubst %.o,%.cu,$(CUOBJ)); do \
	     echo -n $$( dirname $$i )/ ; \
             cpp -MM -std=gnu++0x $$i; \
	done


./GpuResources.o: GpuResources.cpp GpuResources.h utils/DeviceMemory.h \
 utils/DeviceUtils.h utils/../../FaissAssert.h \
 utils/../../FaissException.h
./IndexProxy.o: IndexProxy.cpp IndexProxy.h ../Index.h utils/WorkerThread.h \
 ../FaissAssert.h ../FaissException.h ../Clustering.h ../Index.h \
 GpuIndexFlat.h GpuIndex.h utils/MemorySpace.h utils/../../FaissAssert.h \
 StandardGpuResources.h GpuResources.h utils/DeviceMemory.h \
 utils/StackDeviceMemory.h utils/DeviceUtils.h
./StandardGpuResources.o: StandardGpuResources.cpp StandardGpuResources.h \
 GpuResources.h utils/DeviceMemory.h utils/StackDeviceMemory.h \
 utils/DeviceUtils.h utils/../../FaissAssert.h \
 utils/../../FaissException.h ../FaissAssert.h
./GpuAutoTune.o: GpuAutoTune.cpp GpuAutoTune.h ../Index.h ../AutoTune.h \
 ../Index.h GpuClonerOptions.h GpuIndicesOptions.h GpuIndex.h \
 utils/MemorySpace.h utils/../../FaissAssert.h \
 utils/../../FaissException.h ../FaissAssert.h ../index_io.h \
 ../IndexFlat.h ../IndexIVF.h ../Clustering.h ../Heap.h ../IndexIVFPQ.h \
 ../IndexIVF.h ../IndexPQ.h ../ProductQuantizer.h ../PolysemousTraining.h \
 ../VectorTransform.h ../MetaIndexes.h GpuIndexFlat.h GpuIndexIVFFlat.h \
 GpuIndexIVF.h ../Clustering.h GpuIndexIVFPQ.h IndexProxy.h \
 utils/WorkerThread.h
./GpuClonerOptions.o: GpuClonerOptions.cpp GpuClonerOptions.h \
 GpuIndicesOptions.h
impl/RemapIndices.o: impl/RemapIndices.cpp impl/RemapIndices.h \
 impl/../../FaissAssert.h impl/../../FaissException.h
utils/DeviceMemory.o: utils/DeviceMemory.cpp utils/DeviceMemory.h \
 utils/DeviceUtils.h utils/../../FaissAssert.h \
 utils/../../FaissException.h
utils/StackDeviceMemory.o: utils/StackDeviceMemory.cpp \
 utils/StackDeviceMemory.h utils/DeviceMemory.h utils/DeviceUtils.h \
 utils/../../FaissAssert.h utils/../../FaissException.h \
 utils/StaticUtils.h
utils/DeviceUtils.o: utils/DeviceUtils.cpp utils/DeviceUtils.h \
 utils/../../FaissAssert.h utils/../../FaissException.h
utils/Timer.o: utils/Timer.cpp utils/Timer.h utils/DeviceUtils.h \
 utils/../../FaissAssert.h utils/../../FaissException.h
utils/MemorySpace.o: utils/MemorySpace.cpp utils/MemorySpace.h \
 utils/../../FaissAssert.h utils/../../FaissException.h
utils/WorkerThread.o: utils/WorkerThread.cpp utils/WorkerThread.h \
 utils/../../FaissAssert.h utils/../../FaissException.h
impl/BroadcastSum.o: impl/BroadcastSum.cu impl/../../FaissAssert.h \
 impl/../../FaissException.h impl/../utils/DeviceUtils.h \
 impl/../utils/../../FaissAssert.h impl/../utils/MathOperators.cuh \
 impl/../utils/Float16.cuh impl/../utils/../GpuResources.h \
 impl/../utils/../utils/DeviceMemory.h impl/../utils/DeviceTensor.cuh \
 impl/../utils/Tensor.cuh impl/../utils/Tensor-inl.cuh \
 impl/../utils/../GpuFaissAssert.h impl/../utils/../../FaissAssert.h \
 impl/../utils/MemorySpace.h impl/../utils/DeviceTensor-inl.cuh \
 impl/../utils/StaticUtils.h
impl/Distance.o: impl/Distance.cu impl/Distance.cuh \
 impl/../utils/DeviceTensor.cuh impl/../utils/Tensor.cuh \
 impl/../utils/Tensor-inl.cuh impl/../utils/../GpuFaissAssert.h \
 impl/../utils/../../FaissAssert.h impl/../utils/../../FaissException.h \
 impl/../utils/DeviceUtils.h impl/../utils/../../FaissAssert.h \
 impl/../utils/DeviceMemory.h impl/../utils/MemorySpace.h \
 impl/../utils/DeviceTensor-inl.cuh impl/../utils/Float16.cuh \
 impl/../utils/../GpuResources.h impl/BroadcastSum.cuh impl/L2Norm.cuh \
 impl/L2Select.cuh impl/../../FaissAssert.h impl/../utils/Limits.cuh \
 impl/../utils/Pair.cuh impl/../utils/MathOperators.cuh \
 impl/../utils/WarpShuffles.cuh impl/../utils/DeviceDefs.cuh \
 impl/../utils/MatrixMult.cuh impl/../utils/BlockSelectKernel.cuh \
 impl/../utils/Select.cuh impl/../utils/Comparators.cuh \
 impl/../utils/MergeNetworkBlock.cuh impl/../utils/MergeNetworkUtils.cuh \
 impl/../utils/PtxUtils.cuh impl/../utils/StaticUtils.h \
 impl/../utils/MergeNetworkWarp.cuh impl/../utils/Reductions.cuh \
 impl/../utils/ReductionOperators.cuh
impl/FlatIndex.o: impl/FlatIndex.cu impl/FlatIndex.cuh \
 impl/../utils/DeviceTensor.cuh impl/../utils/Tensor.cuh \
 impl/../utils/Tensor-inl.cuh impl/../utils/../GpuFaissAssert.h \
 impl/../utils/../../FaissAssert.h impl/../utils/../../FaissException.h \
 impl/../utils/DeviceUtils.h impl/../utils/../../FaissAssert.h \
 impl/../utils/DeviceMemory.h impl/../utils/MemorySpace.h \
 impl/../utils/DeviceTensor-inl.cuh impl/../utils/DeviceVector.cuh \
 impl/../utils/StaticUtils.h impl/../utils/Float16.cuh \
 impl/../utils/../GpuResources.h impl/Distance.cuh impl/L2Norm.cuh \
 impl/../utils/CopyUtils.cuh impl/../utils/HostTensor.cuh \
 impl/../utils/HostTensor-inl.cuh impl/../utils/Transpose.cuh
impl/InvertedListAppend.o: impl/InvertedListAppend.cu \
 impl/InvertedListAppend.cuh impl/../GpuIndicesOptions.h \
 impl/../utils/Tensor.cuh impl/../utils/Tensor-inl.cuh \
 impl/../utils/../GpuFaissAssert.h impl/../utils/../../FaissAssert.h \
 impl/../utils/../../FaissException.h impl/../utils/DeviceUtils.h \
 impl/../utils/../../FaissAssert.h impl/../../FaissAssert.h \
 impl/../utils/Float16.cuh impl/../utils/../GpuResources.h \
 impl/../utils/../utils/DeviceMemory.h impl/../utils/DeviceTensor.cuh \
 impl/../utils/MemorySpace.h impl/../utils/DeviceTensor-inl.cuh \
 impl/../utils/StaticUtils.h
impl/IVFBase.o: impl/IVFBase.cu impl/IVFBase.cuh impl/../GpuIndicesOptions.h \
 impl/../utils/DeviceVector.cuh impl/../utils/../../FaissAssert.h \
 impl/../utils/../../FaissException.h impl/../utils/DeviceUtils.h \
 impl/../utils/MemorySpace.h impl/../utils/StaticUtils.h \
 impl/../utils/DeviceTensor.cuh impl/../utils/Tensor.cuh \
 impl/../utils/Tensor-inl.cuh impl/../utils/../GpuFaissAssert.h \
 impl/../utils/../../FaissAssert.h impl/../utils/DeviceMemory.h \
 impl/../utils/DeviceTensor-inl.cuh impl/../GpuResources.h \
 impl/FlatIndex.cuh impl/../utils/Float16.cuh impl/InvertedListAppend.cuh \
 impl/RemapIndices.h impl/../utils/DeviceDefs.cuh \
 impl/../utils/HostTensor.cuh impl/../utils/HostTensor-inl.cuh
impl/IVFFlat.o: impl/IVFFlat.cu impl/IVFFlat.cuh impl/IVFBase.cuh \
 impl/../GpuIndicesOptions.h impl/../utils/DeviceVector.cuh \
 impl/../utils/../../FaissAssert.h impl/../utils/../../FaissException.h \
 impl/../utils/DeviceUtils.h impl/../utils/MemorySpace.h \
 impl/../utils/StaticUtils.h impl/../utils/DeviceTensor.cuh \
 impl/../utils/Tensor.cuh impl/../utils/Tensor-inl.cuh \
 impl/../utils/../GpuFaissAssert.h impl/../utils/../../FaissAssert.h \
 impl/../utils/DeviceMemory.h impl/../utils/DeviceTensor-inl.cuh \
 impl/../GpuResources.h impl/FlatIndex.cuh impl/../utils/Float16.cuh \
 impl/InvertedListAppend.cuh impl/IVFFlatScan.cuh impl/RemapIndices.h \
 impl/../utils/CopyUtils.cuh impl/../utils/HostTensor.cuh \
 impl/../utils/HostTensor-inl.cuh impl/../utils/DeviceDefs.cuh \
 impl/../utils/Transpose.cuh
impl/IVFFlatScan.o: impl/IVFFlatScan.cu impl/IVFFlatScan.cuh \
 impl/../GpuIndicesOptions.h impl/../utils/Tensor.cuh \
 impl/../utils/Tensor-inl.cuh impl/../utils/../GpuFaissAssert.h \
 impl/../utils/../../FaissAssert.h impl/../utils/../../FaissException.h \
 impl/../utils/DeviceUtils.h impl/../utils/../../FaissAssert.h \
 impl/../GpuResources.h impl/../utils/DeviceMemory.h impl/IVFUtils.cuh \
 impl/../utils/ConversionOperators.cuh impl/../utils/Float16.cuh \
 impl/../utils/DeviceTensor.cuh impl/../utils/MemorySpace.h \
 impl/../utils/DeviceTensor-inl.cuh impl/../utils/DeviceDefs.cuh \
 impl/../utils/MathOperators.cuh impl/../utils/LoadStoreOperators.cuh \
 impl/../utils/PtxUtils.cuh impl/../utils/Reductions.cuh \
 impl/../utils/ReductionOperators.cuh impl/../utils/Limits.cuh \
 impl/../utils/Pair.cuh impl/../utils/WarpShuffles.cuh \
 impl/../utils/StaticUtils.h
impl/IVFPQ.o: impl/IVFPQ.cu impl/IVFPQ.cuh impl/IVFBase.cuh \
 impl/../GpuIndicesOptions.h impl/../utils/DeviceVector.cuh \
 impl/../utils/../../FaissAssert.h impl/../utils/../../FaissException.h \
 impl/../utils/DeviceUtils.h impl/../utils/MemorySpace.h \
 impl/../utils/StaticUtils.h impl/../utils/DeviceTensor.cuh \
 impl/../utils/Tensor.cuh impl/../utils/Tensor-inl.cuh \
 impl/../utils/../GpuFaissAssert.h impl/../utils/../../FaissAssert.h \
 impl/../utils/DeviceMemory.h impl/../utils/DeviceTensor-inl.cuh \
 impl/../utils/Float16.cuh impl/../utils/../GpuResources.h \
 impl/BroadcastSum.cuh impl/Distance.cuh impl/FlatIndex.cuh \
 impl/InvertedListAppend.cuh impl/L2Norm.cuh impl/PQCodeDistances.cuh \
 impl/../utils/NoTypeTensor.cuh impl/PQScanMultiPassNoPrecomputed.cuh \
 impl/PQScanMultiPassPrecomputed.cuh impl/RemapIndices.h \
 impl/VectorResidual.cuh impl/../utils/DeviceDefs.cuh \
 impl/../utils/HostTensor.cuh impl/../utils/HostTensor-inl.cuh \
 impl/../utils/MatrixMult.cuh impl/../utils/Transpose.cuh
impl/IVFUtils.o: impl/IVFUtils.cu impl/IVFUtils.cuh \
 impl/../GpuIndicesOptions.h impl/../utils/Tensor.cuh \
 impl/../utils/Tensor-inl.cuh impl/../utils/../GpuFaissAssert.h \
 impl/../utils/../../FaissAssert.h impl/../utils/../../FaissException.h \
 impl/../utils/DeviceUtils.h impl/../utils/../../FaissAssert.h \
 impl/../utils/StaticUtils.h impl/../utils/ThrustAllocator.cuh
impl/IVFUtilsSelect1.o: impl/IVFUtilsSelect1.cu impl/IVFUtils.cuh \
 impl/../GpuIndicesOptions.h impl/../utils/Tensor.cuh \
 impl/../utils/Tensor-inl.cuh impl/../utils/../GpuFaissAssert.h \
 impl/../utils/../../FaissAssert.h impl/../utils/../../FaissException.h \
 impl/../utils/DeviceUtils.h impl/../utils/../../FaissAssert.h \
 impl/../utils/Limits.cuh impl/../utils/Float16.cuh \
 impl/../utils/../GpuResources.h impl/../utils/../utils/DeviceMemory.h \
 impl/../utils/DeviceTensor.cuh impl/../utils/MemorySpace.h \
 impl/../utils/DeviceTensor-inl.cuh impl/../utils/Pair.cuh \
 impl/../utils/MathOperators.cuh impl/../utils/WarpShuffles.cuh \
 impl/../utils/DeviceDefs.cuh impl/../utils/Select.cuh \
 impl/../utils/Comparators.cuh impl/../utils/MergeNetworkBlock.cuh \
 impl/../utils/MergeNetworkUtils.cuh impl/../utils/PtxUtils.cuh \
 impl/../utils/StaticUtils.h impl/../utils/MergeNetworkWarp.cuh \
 impl/../utils/Reductions.cuh impl/../utils/ReductionOperators.cuh
impl/IVFUtilsSelect2.o: impl/IVFUtilsSelect2.cu impl/IVFUtils.cuh \
 impl/../GpuIndicesOptions.h impl/../utils/Tensor.cuh \
 impl/../utils/Tensor-inl.cuh impl/../utils/../GpuFaissAssert.h \
 impl/../utils/../../FaissAssert.h impl/../utils/../../FaissException.h \
 impl/../utils/DeviceUtils.h impl/../utils/../../FaissAssert.h \
 impl/../utils/Limits.cuh impl/../utils/Float16.cuh \
 impl/../utils/../GpuResources.h impl/../utils/../utils/DeviceMemory.h \
 impl/../utils/DeviceTensor.cuh impl/../utils/MemorySpace.h \
 impl/../utils/DeviceTensor-inl.cuh impl/../utils/Pair.cuh \
 impl/../utils/MathOperators.cuh impl/../utils/WarpShuffles.cuh \
 impl/../utils/DeviceDefs.cuh impl/../utils/Select.cuh \
 impl/../utils/Comparators.cuh impl/../utils/MergeNetworkBlock.cuh \
 impl/../utils/MergeNetworkUtils.cuh impl/../utils/PtxUtils.cuh \
 impl/../utils/StaticUtils.h impl/../utils/MergeNetworkWarp.cuh \
 impl/../utils/Reductions.cuh impl/../utils/ReductionOperators.cuh
impl/L2Norm.o: impl/L2Norm.cu impl/L2Norm.cuh impl/../utils/Float16.cuh \
 impl/../utils/../GpuResources.h impl/../utils/../utils/DeviceMemory.h \
 impl/../utils/DeviceTensor.cuh impl/../utils/Tensor.cuh \
 impl/../utils/Tensor-inl.cuh impl/../utils/../GpuFaissAssert.h \
 impl/../utils/../../FaissAssert.h impl/../utils/../../FaissException.h \
 impl/../utils/DeviceUtils.h impl/../utils/../../FaissAssert.h \
 impl/../utils/MemorySpace.h impl/../utils/DeviceTensor-inl.cuh \
 impl/../../FaissAssert.h impl/../utils/ConversionOperators.cuh \
 impl/../utils/DeviceDefs.cuh impl/../utils/MathOperators.cuh \
 impl/../utils/PtxUtils.cuh impl/../utils/StaticUtils.h \
 impl/../utils/Reductions.cuh impl/../utils/ReductionOperators.cuh \
 impl/../utils/Limits.cuh impl/../utils/Pair.cuh \
 impl/../utils/WarpShuffles.cuh
impl/L2Select.o: impl/L2Select.cu impl/L2Select.cuh impl/../utils/Float16.cuh \
 impl/../utils/../GpuResources.h impl/../utils/../utils/DeviceMemory.h \
 impl/../utils/DeviceTensor.cuh impl/../utils/Tensor.cuh \
 impl/../utils/Tensor-inl.cuh impl/../utils/../GpuFaissAssert.h \
 impl/../utils/../../FaissAssert.h impl/../utils/../../FaissException.h \
 impl/../utils/DeviceUtils.h impl/../utils/../../FaissAssert.h \
 impl/../utils/MemorySpace.h impl/../utils/DeviceTensor-inl.cuh \
 impl/../../FaissAssert.h impl/../utils/MathOperators.cuh \
 impl/../utils/Pair.cuh impl/../utils/WarpShuffles.cuh \
 impl/../utils/DeviceDefs.cuh impl/../utils/Reductions.cuh \
 impl/../utils/PtxUtils.cuh impl/../utils/ReductionOperators.cuh \
 impl/../utils/Limits.cuh impl/../utils/StaticUtils.h \
 impl/../utils/Select.cuh impl/../utils/Comparators.cuh \
 impl/../utils/MergeNetworkBlock.cuh impl/../utils/MergeNetworkUtils.cuh \
 impl/../utils/MergeNetworkWarp.cuh
impl/PQCodeDistances.o: impl/PQCodeDistances.cu impl/PQCodeDistances.cuh \
 impl/../utils/Tensor.cuh impl/../utils/Tensor-inl.cuh \
 impl/../utils/../GpuFaissAssert.h impl/../utils/../../FaissAssert.h \
 impl/../utils/../../FaissException.h impl/../utils/DeviceUtils.h \
 impl/../utils/../../FaissAssert.h impl/../utils/NoTypeTensor.cuh \
 impl/BroadcastSum.cuh impl/../utils/Float16.cuh \
 impl/../utils/../GpuResources.h impl/../utils/../utils/DeviceMemory.h \
 impl/../utils/DeviceTensor.cuh impl/../utils/MemorySpace.h \
 impl/../utils/DeviceTensor-inl.cuh impl/Distance.cuh impl/L2Norm.cuh \
 impl/../utils/DeviceDefs.cuh impl/../utils/MatrixMult.cuh \
 impl/../utils/PtxUtils.cuh impl/../utils/StaticUtils.h \
 impl/../utils/Transpose.cuh
impl/PQScanMultiPassNoPrecomputed.o: impl/PQScanMultiPassNoPrecomputed.cu \
 impl/PQScanMultiPassNoPrecomputed.cuh impl/../GpuIndicesOptions.h \
 impl/../utils/Tensor.cuh impl/../utils/Tensor-inl.cuh \
 impl/../utils/../GpuFaissAssert.h impl/../utils/../../FaissAssert.h \
 impl/../utils/../../FaissException.h impl/../utils/DeviceUtils.h \
 impl/../utils/../../FaissAssert.h impl/../GpuResources.h \
 impl/../utils/DeviceMemory.h impl/PQCodeDistances.cuh \
 impl/../utils/NoTypeTensor.cuh impl/PQCodeLoad.cuh \
 impl/../utils/PtxUtils.cuh impl/IVFUtils.cuh \
 impl/../utils/ConversionOperators.cuh impl/../utils/Float16.cuh \
 impl/../utils/DeviceTensor.cuh impl/../utils/MemorySpace.h \
 impl/../utils/DeviceTensor-inl.cuh impl/../utils/LoadStoreOperators.cuh \
 impl/../utils/StaticUtils.h impl/../utils/HostTensor.cuh \
 impl/../utils/HostTensor-inl.cuh
impl/PQScanMultiPassPrecomputed.o: impl/PQScanMultiPassPrecomputed.cu \
 impl/PQScanMultiPassPrecomputed.cuh impl/../GpuIndicesOptions.h \
 impl/../utils/Tensor.cuh impl/../utils/Tensor-inl.cuh \
 impl/../utils/../GpuFaissAssert.h impl/../utils/../../FaissAssert.h \
 impl/../utils/../../FaissException.h impl/../utils/DeviceUtils.h \
 impl/../utils/../../FaissAssert.h impl/../utils/NoTypeTensor.cuh \
 impl/../GpuResources.h impl/../utils/DeviceMemory.h impl/PQCodeLoad.cuh \
 impl/../utils/PtxUtils.cuh impl/IVFUtils.cuh \
 impl/../utils/ConversionOperators.cuh impl/../utils/Float16.cuh \
 impl/../utils/DeviceTensor.cuh impl/../utils/MemorySpace.h \
 impl/../utils/DeviceTensor-inl.cuh impl/../utils/LoadStoreOperators.cuh \
 impl/../utils/MathOperators.cuh impl/../utils/StaticUtils.h
impl/VectorResidual.o: impl/VectorResidual.cu impl/VectorResidual.cuh \
 impl/../utils/Tensor.cuh impl/../utils/Tensor-inl.cuh \
 impl/../utils/../GpuFaissAssert.h impl/../utils/../../FaissAssert.h \
 impl/../utils/../../FaissException.h impl/../utils/DeviceUtils.h \
 impl/../utils/../../FaissAssert.h impl/../utils/Float16.cuh \
 impl/../utils/../GpuResources.h impl/../utils/../utils/DeviceMemory.h \
 impl/../utils/DeviceTensor.cuh impl/../utils/MemorySpace.h \
 impl/../utils/DeviceTensor-inl.cuh impl/../../FaissAssert.h \
 impl/../utils/ConversionOperators.cuh impl/../utils/StaticUtils.h
./GpuIndex.o: GpuIndex.cu GpuIndex.h ../Index.h utils/MemorySpace.h \
 utils/../../FaissAssert.h utils/../../FaissException.h ../FaissAssert.h \
 GpuResources.h utils/DeviceMemory.h utils/DeviceUtils.h
./GpuIndexFlat.o: GpuIndexFlat.cu GpuIndexFlat.h GpuIndex.h ../Index.h \
 utils/MemorySpace.h utils/../../FaissAssert.h \
 utils/../../FaissException.h ../IndexFlat.h ../Index.h GpuResources.h \
 utils/DeviceMemory.h impl/FlatIndex.cuh impl/../utils/DeviceTensor.cuh \
 impl/../utils/Tensor.cuh impl/../utils/Tensor-inl.cuh \
 impl/../utils/../GpuFaissAssert.h impl/../utils/../../FaissAssert.h \
 impl/../utils/DeviceUtils.h impl/../utils/../../FaissAssert.h \
 impl/../utils/DeviceTensor-inl.cuh impl/../utils/DeviceVector.cuh \
 impl/../utils/StaticUtils.h impl/../utils/Float16.cuh \
 utils/CopyUtils.cuh utils/HostTensor.cuh utils/HostTensor-inl.cuh
./GpuIndexIVF.o: GpuIndexIVF.cu GpuIndexIVF.h GpuIndex.h ../Index.h \
 utils/MemorySpace.h utils/../../FaissAssert.h \
 utils/../../FaissException.h GpuIndexFlat.h GpuIndicesOptions.h \
 ../Clustering.h ../Index.h ../FaissAssert.h ../IndexFlat.h ../IndexIVF.h \
 ../Clustering.h ../Heap.h utils/DeviceUtils.h utils/Float16.cuh \
 utils/../GpuResources.h utils/../utils/DeviceMemory.h \
 utils/DeviceTensor.cuh utils/Tensor.cuh utils/Tensor-inl.cuh \
 utils/../GpuFaissAssert.h utils/../../FaissAssert.h \
 utils/DeviceTensor-inl.cuh
./GpuIndexIVFFlat.o: GpuIndexIVFFlat.cu GpuIndexIVFFlat.h GpuIndexIVF.h \
 GpuIndex.h ../Index.h utils/MemorySpace.h utils/../../FaissAssert.h \
 utils/../../FaissException.h GpuIndexFlat.h GpuIndicesOptions.h \
 ../Clustering.h ../Index.h ../IndexFlat.h ../IndexIVF.h ../Clustering.h \
 ../Heap.h GpuResources.h utils/DeviceMemory.h impl/IVFFlat.cuh \
 impl/IVFBase.cuh impl/../utils/DeviceVector.cuh \
 impl/../utils/../../FaissAssert.h impl/../utils/DeviceUtils.h \
 impl/../utils/StaticUtils.h impl/../utils/DeviceTensor.cuh \
 impl/../utils/Tensor.cuh impl/../utils/Tensor-inl.cuh \
 impl/../utils/../GpuFaissAssert.h impl/../utils/../../FaissAssert.h \
 impl/../utils/DeviceTensor-inl.cuh utils/CopyUtils.cuh \
 utils/HostTensor.cuh utils/HostTensor-inl.cuh utils/Float16.cuh
./GpuIndexIVFPQ.o: GpuIndexIVFPQ.cu GpuIndexIVFPQ.h GpuIndexIVF.h \
 GpuIndex.h ../Index.h utils/MemorySpace.h utils/../../FaissAssert.h \
 utils/../../FaissException.h GpuIndexFlat.h GpuIndicesOptions.h \
 ../Clustering.h ../Index.h ../IndexFlat.h ../IndexIVFPQ.h ../IndexIVF.h \
 ../Clustering.h ../Heap.h ../IndexPQ.h ../ProductQuantizer.h \
 ../PolysemousTraining.h ../ProductQuantizer.h GpuResources.h \
 utils/DeviceMemory.h impl/IVFPQ.cuh impl/IVFBase.cuh \
 impl/../utils/DeviceVector.cuh impl/../utils/../../FaissAssert.h \
 impl/../utils/DeviceUtils.h impl/../utils/StaticUtils.h \
 impl/../utils/DeviceTensor.cuh impl/../utils/Tensor.cuh \
 impl/../utils/Tensor-inl.cuh impl/../utils/../GpuFaissAssert.h \
 impl/../utils/../../FaissAssert.h impl/../utils/DeviceTensor-inl.cuh \
 impl/../utils/Float16.cuh utils/CopyUtils.cuh utils/HostTensor.cuh \
 utils/HostTensor-inl.cuh
utils/Float16.o: utils/Float16.cu utils/Float16.cuh utils/../GpuResources.h \
 utils/../utils/DeviceMemory.h utils/DeviceTensor.cuh utils/Tensor.cuh \
 utils/Tensor-inl.cuh utils/../GpuFaissAssert.h utils/../../FaissAssert.h \
 utils/../../FaissException.h utils/DeviceUtils.h \
 utils/../../FaissAssert.h utils/MemorySpace.h utils/DeviceTensor-inl.cuh \
 utils/nvidia/fp16_emu.cuh
utils/MatrixMult.o: utils/MatrixMult.cu utils/MatrixMult.cuh utils/Float16.cuh \
 utils/../GpuResources.h utils/../utils/DeviceMemory.h \
 utils/DeviceTensor.cuh utils/Tensor.cuh utils/Tensor-inl.cuh \
 utils/../GpuFaissAssert.h utils/../../FaissAssert.h \
 utils/../../FaissException.h utils/DeviceUtils.h \
 utils/../../FaissAssert.h utils/MemorySpace.h utils/DeviceTensor-inl.cuh \
 utils/HostTensor.cuh utils/HostTensor-inl.cuh
utils/BlockSelectFloat.o: utils/BlockSelectFloat.cu \
 utils/blockselect/BlockSelectImpl.cuh \
 utils/blockselect/../BlockSelectKernel.cuh \
 utils/blockselect/../Float16.cuh utils/blockselect/../../GpuResources.h \
 utils/blockselect/../../utils/DeviceMemory.h \
 utils/blockselect/../DeviceTensor.cuh utils/blockselect/../Tensor.cuh \
 utils/blockselect/../Tensor-inl.cuh \
 utils/blockselect/../../GpuFaissAssert.h \
 utils/blockselect/../../../FaissAssert.h \
 utils/blockselect/../../../FaissException.h \
 utils/blockselect/../DeviceUtils.h \
 utils/blockselect/../../../FaissAssert.h \
 utils/blockselect/../MemorySpace.h \
 utils/blockselect/../DeviceTensor-inl.cuh \
 utils/blockselect/../Select.cuh utils/blockselect/../Comparators.cuh \
 utils/blockselect/../DeviceDefs.cuh \
 utils/blockselect/../MergeNetworkBlock.cuh \
 utils/blockselect/../MergeNetworkUtils.cuh \
 utils/blockselect/../PtxUtils.cuh utils/blockselect/../StaticUtils.h \
 utils/blockselect/../WarpShuffles.cuh \
 utils/blockselect/../MergeNetworkWarp.cuh \
 utils/blockselect/../Reductions.cuh \
 utils/blockselect/../ReductionOperators.cuh \
 utils/blockselect/../Limits.cuh utils/blockselect/../Pair.cuh \
 utils/blockselect/../MathOperators.cuh
utils/BlockSelectHalf.o: utils/BlockSelectHalf.cu \
 utils/blockselect/BlockSelectImpl.cuh \
 utils/blockselect/../BlockSelectKernel.cuh \
 utils/blockselect/../Float16.cuh utils/blockselect/../../GpuResources.h \
 utils/blockselect/../../utils/DeviceMemory.h \
 utils/blockselect/../DeviceTensor.cuh utils/blockselect/../Tensor.cuh \
 utils/blockselect/../Tensor-inl.cuh \
 utils/blockselect/../../GpuFaissAssert.h \
 utils/blockselect/../../../FaissAssert.h \
 utils/blockselect/../../../FaissException.h \
 utils/blockselect/../DeviceUtils.h \
 utils/blockselect/../../../FaissAssert.h \
 utils/blockselect/../MemorySpace.h \
 utils/blockselect/../DeviceTensor-inl.cuh \
 utils/blockselect/../Select.cuh utils/blockselect/../Comparators.cuh \
 utils/blockselect/../DeviceDefs.cuh \
 utils/blockselect/../MergeNetworkBlock.cuh \
 utils/blockselect/../MergeNetworkUtils.cuh \
 utils/blockselect/../PtxUtils.cuh utils/blockselect/../StaticUtils.h \
 utils/blockselect/../WarpShuffles.cuh \
 utils/blockselect/../MergeNetworkWarp.cuh \
 utils/blockselect/../Reductions.cuh \
 utils/blockselect/../ReductionOperators.cuh \
 utils/blockselect/../Limits.cuh utils/blockselect/../Pair.cuh \
 utils/blockselect/../MathOperators.cuh
utils/WarpSelectFloat.o: utils/WarpSelectFloat.cu \
 utils/warpselect/WarpSelectImpl.cuh \
 utils/warpselect/../WarpSelectKernel.cuh utils/warpselect/../Float16.cuh \
 utils/warpselect/../../GpuResources.h \
 utils/warpselect/../../utils/DeviceMemory.h \
 utils/warpselect/../DeviceTensor.cuh utils/warpselect/../Tensor.cuh \
 utils/warpselect/../Tensor-inl.cuh \
 utils/warpselect/../../GpuFaissAssert.h \
 utils/warpselect/../../../FaissAssert.h \
 utils/warpselect/../../../FaissException.h \
 utils/warpselect/../DeviceUtils.h \
 utils/warpselect/../../../FaissAssert.h \
 utils/warpselect/../MemorySpace.h \
 utils/warpselect/../DeviceTensor-inl.cuh utils/warpselect/../Select.cuh \
 utils/warpselect/../Comparators.cuh utils/warpselect/../DeviceDefs.cuh \
 utils/warpselect/../MergeNetworkBlock.cuh \
 utils/warpselect/../MergeNetworkUtils.cuh \
 utils/warpselect/../PtxUtils.cuh utils/warpselect/../StaticUtils.h \
 utils/warpselect/../WarpShuffles.cuh \
 utils/warpselect/../MergeNetworkWarp.cuh \
 utils/warpselect/../Reductions.cuh \
 utils/warpselect/../ReductionOperators.cuh \
 utils/warpselect/../Limits.cuh utils/warpselect/../Pair.cuh \
 utils/warpselect/../MathOperators.cuh
utils/WarpSelectHalf.o: utils/WarpSelectHalf.cu \
 utils/warpselect/WarpSelectImpl.cuh \
 utils/warpselect/../WarpSelectKernel.cuh utils/warpselect/../Float16.cuh \
 utils/warpselect/../../GpuResources.h \
 utils/warpselect/../../utils/DeviceMemory.h \
 utils/warpselect/../DeviceTensor.cuh utils/warpselect/../Tensor.cuh \
 utils/warpselect/../Tensor-inl.cuh \
 utils/warpselect/../../GpuFaissAssert.h \
 utils/warpselect/../../../FaissAssert.h \
 utils/warpselect/../../../FaissException.h \
 utils/warpselect/../DeviceUtils.h \
 utils/warpselect/../../../FaissAssert.h \
 utils/warpselect/../MemorySpace.h \
 utils/warpselect/../DeviceTensor-inl.cuh utils/warpselect/../Select.cuh \
 utils/warpselect/../Comparators.cuh utils/warpselect/../DeviceDefs.cuh \
 utils/warpselect/../MergeNetworkBlock.cuh \
 utils/warpselect/../MergeNetworkUtils.cuh \
 utils/warpselect/../PtxUtils.cuh utils/warpselect/../StaticUtils.h \
 utils/warpselect/../WarpShuffles.cuh \
 utils/warpselect/../MergeNetworkWarp.cuh \
 utils/warpselect/../Reductions.cuh \
 utils/warpselect/../ReductionOperators.cuh \
 utils/warpselect/../Limits.cuh utils/warpselect/../Pair.cuh \
 utils/warpselect/../MathOperators.cuh
utils/nvidia/fp16_emu.o: utils/nvidia/fp16_emu.cu utils/nvidia/fp16_emu.cuh
utils/blockselect/BlockSelectHalf1.o: utils/blockselect/BlockSelectHalf1.cu \
 utils/blockselect/BlockSelectImpl.cuh \
 utils/blockselect/../BlockSelectKernel.cuh \
 utils/blockselect/../Float16.cuh utils/blockselect/../../GpuResources.h \
 utils/blockselect/../../utils/DeviceMemory.h \
 utils/blockselect/../DeviceTensor.cuh utils/blockselect/../Tensor.cuh \
 utils/blockselect/../Tensor-inl.cuh \
 utils/blockselect/../../GpuFaissAssert.h \
 utils/blockselect/../../../FaissAssert.h \
 utils/blockselect/../../../FaissException.h \
 utils/blockselect/../DeviceUtils.h \
 utils/blockselect/../../../FaissAssert.h \
 utils/blockselect/../MemorySpace.h \
 utils/blockselect/../DeviceTensor-inl.cuh \
 utils/blockselect/../Select.cuh utils/blockselect/../Comparators.cuh \
 utils/blockselect/../DeviceDefs.cuh \
 utils/blockselect/../MergeNetworkBlock.cuh \
 utils/blockselect/../MergeNetworkUtils.cuh \
 utils/blockselect/../PtxUtils.cuh utils/blockselect/../StaticUtils.h \
 utils/blockselect/../WarpShuffles.cuh \
 utils/blockselect/../MergeNetworkWarp.cuh \
 utils/blockselect/../Reductions.cuh \
 utils/blockselect/../ReductionOperators.cuh \
 utils/blockselect/../Limits.cuh utils/blockselect/../Pair.cuh \
 utils/blockselect/../MathOperators.cuh
utils/blockselect/BlockSelectFloat1.o: utils/blockselect/BlockSelectFloat1.cu \
 utils/blockselect/BlockSelectImpl.cuh \
 utils/blockselect/../BlockSelectKernel.cuh \
 utils/blockselect/../Float16.cuh utils/blockselect/../../GpuResources.h \
 utils/blockselect/../../utils/DeviceMemory.h \
 utils/blockselect/../DeviceTensor.cuh utils/blockselect/../Tensor.cuh \
 utils/blockselect/../Tensor-inl.cuh \
 utils/blockselect/../../GpuFaissAssert.h \
 utils/blockselect/../../../FaissAssert.h \
 utils/blockselect/../../../FaissException.h \
 utils/blockselect/../DeviceUtils.h \
 utils/blockselect/../../../FaissAssert.h \
 utils/blockselect/../MemorySpace.h \
 utils/blockselect/../DeviceTensor-inl.cuh \
 utils/blockselect/../Select.cuh utils/blockselect/../Comparators.cuh \
 utils/blockselect/../DeviceDefs.cuh \
 utils/blockselect/../MergeNetworkBlock.cuh \
 utils/blockselect/../MergeNetworkUtils.cuh \
 utils/blockselect/../PtxUtils.cuh utils/blockselect/../StaticUtils.h \
 utils/blockselect/../WarpShuffles.cuh \
 utils/blockselect/../MergeNetworkWarp.cuh \
 utils/blockselect/../Reductions.cuh \
 utils/blockselect/../ReductionOperators.cuh \
 utils/blockselect/../Limits.cuh utils/blockselect/../Pair.cuh \
 utils/blockselect/../MathOperators.cuh
utils/warpselect/WarpSelectHalf1.o: utils/warpselect/WarpSelectHalf1.cu \
 utils/warpselect/WarpSelectImpl.cuh \
 utils/warpselect/../WarpSelectKernel.cuh utils/warpselect/../Float16.cuh \
 utils/warpselect/../../GpuResources.h \
 utils/warpselect/../../utils/DeviceMemory.h \
 utils/warpselect/../DeviceTensor.cuh utils/warpselect/../Tensor.cuh \
 utils/warpselect/../Tensor-inl.cuh \
 utils/warpselect/../../GpuFaissAssert.h \
 utils/warpselect/../../../FaissAssert.h \
 utils/warpselect/../../../FaissException.h \
 utils/warpselect/../DeviceUtils.h \
 utils/warpselect/../../../FaissAssert.h \
 utils/warpselect/../MemorySpace.h \
 utils/warpselect/../DeviceTensor-inl.cuh utils/warpselect/../Select.cuh \
 utils/warpselect/../Comparators.cuh utils/warpselect/../DeviceDefs.cuh \
 utils/warpselect/../MergeNetworkBlock.cuh \
 utils/warpselect/../MergeNetworkUtils.cuh \
 utils/warpselect/../PtxUtils.cuh utils/warpselect/../StaticUtils.h \
 utils/warpselect/../WarpShuffles.cuh \
 utils/warpselect/../MergeNetworkWarp.cuh \
 utils/warpselect/../Reductions.cuh \
 utils/warpselect/../ReductionOperators.cuh \
 utils/warpselect/../Limits.cuh utils/warpselect/../Pair.cuh \
 utils/warpselect/../MathOperators.cuh
utils/warpselect/WarpSelectFloat1.o: utils/warpselect/WarpSelectFloat1.cu \
 utils/warpselect/WarpSelectImpl.cuh \
 utils/warpselect/../WarpSelectKernel.cuh utils/warpselect/../Float16.cuh \
 utils/warpselect/../../GpuResources.h \
 utils/warpselect/../../utils/DeviceMemory.h \
 utils/warpselect/../DeviceTensor.cuh utils/warpselect/../Tensor.cuh \
 utils/warpselect/../Tensor-inl.cuh \
 utils/warpselect/../../GpuFaissAssert.h \
 utils/warpselect/../../../FaissAssert.h \
 utils/warpselect/../../../FaissException.h \
 utils/warpselect/../DeviceUtils.h \
 utils/warpselect/../../../FaissAssert.h \
 utils/warpselect/../MemorySpace.h \
 utils/warpselect/../DeviceTensor-inl.cuh utils/warpselect/../Select.cuh \
 utils/warpselect/../Comparators.cuh utils/warpselect/../DeviceDefs.cuh \
 utils/warpselect/../MergeNetworkBlock.cuh \
 utils/warpselect/../MergeNetworkUtils.cuh \
 utils/warpselect/../PtxUtils.cuh utils/warpselect/../StaticUtils.h \
 utils/warpselect/../WarpShuffles.cuh \
 utils/warpselect/../MergeNetworkWarp.cuh \
 utils/warpselect/../Reductions.cuh \
 utils/warpselect/../ReductionOperators.cuh \
 utils/warpselect/../Limits.cuh utils/warpselect/../Pair.cuh \
 utils/warpselect/../MathOperators.cuh
utils/blockselect/BlockSelectHalf32.o: utils/blockselect/BlockSelectHalf32.cu \
 utils/blockselect/BlockSelectImpl.cuh \
 utils/blockselect/../BlockSelectKernel.cuh \
 utils/blockselect/../Float16.cuh utils/blockselect/../../GpuResources.h \
 utils/blockselect/../../utils/DeviceMemory.h \
 utils/blockselect/../DeviceTensor.cuh utils/blockselect/../Tensor.cuh \
 utils/blockselect/../Tensor-inl.cuh \
 utils/blockselect/../../GpuFaissAssert.h \
 utils/blockselect/../../../FaissAssert.h \
 utils/blockselect/../../../FaissException.h \
 utils/blockselect/../DeviceUtils.h \
 utils/blockselect/../../../FaissAssert.h \
 utils/blockselect/../MemorySpace.h \
 utils/blockselect/../DeviceTensor-inl.cuh \
 utils/blockselect/../Select.cuh utils/blockselect/../Comparators.cuh \
 utils/blockselect/../DeviceDefs.cuh \
 utils/blockselect/../MergeNetworkBlock.cuh \
 utils/blockselect/../MergeNetworkUtils.cuh \
 utils/blockselect/../PtxUtils.cuh utils/blockselect/../StaticUtils.h \
 utils/blockselect/../WarpShuffles.cuh \
 utils/blockselect/../MergeNetworkWarp.cuh \
 utils/blockselect/../Reductions.cuh \
 utils/blockselect/../ReductionOperators.cuh \
 utils/blockselect/../Limits.cuh utils/blockselect/../Pair.cuh \
 utils/blockselect/../MathOperators.cuh
utils/blockselect/BlockSelectFloat32.o: utils/blockselect/BlockSelectFloat32.cu \
 utils/blockselect/BlockSelectImpl.cuh \
 utils/blockselect/../BlockSelectKernel.cuh \
 utils/blockselect/../Float16.cuh utils/blockselect/../../GpuResources.h \
 utils/blockselect/../../utils/DeviceMemory.h \
 utils/blockselect/../DeviceTensor.cuh utils/blockselect/../Tensor.cuh \
 utils/blockselect/../Tensor-inl.cuh \
 utils/blockselect/../../GpuFaissAssert.h \
 utils/blockselect/../../../FaissAssert.h \
 utils/blockselect/../../../FaissException.h \
 utils/blockselect/../DeviceUtils.h \
 utils/blockselect/../../../FaissAssert.h \
 utils/blockselect/../MemorySpace.h \
 utils/blockselect/../DeviceTensor-inl.cuh \
 utils/blockselect/../Select.cuh utils/blockselect/../Comparators.cuh \
 utils/blockselect/../DeviceDefs.cuh \
 utils/blockselect/../MergeNetworkBlock.cuh \
 utils/blockselect/../MergeNetworkUtils.cuh \
 utils/blockselect/../PtxUtils.cuh utils/blockselect/../StaticUtils.h \
 utils/blockselect/../WarpShuffles.cuh \
 utils/blockselect/../MergeNetworkWarp.cuh \
 utils/blockselect/../Reductions.cuh \
 utils/blockselect/../ReductionOperators.cuh \
 utils/blockselect/../Limits.cuh utils/blockselect/../Pair.cuh \
 utils/blockselect/../MathOperators.cuh
utils/warpselect/WarpSelectHalf32.o: utils/warpselect/WarpSelectHalf32.cu \
 utils/warpselect/WarpSelectImpl.cuh \
 utils/warpselect/../WarpSelectKernel.cuh utils/warpselect/../Float16.cuh \
 utils/warpselect/../../GpuResources.h \
 utils/warpselect/../../utils/DeviceMemory.h \
 utils/warpselect/../DeviceTensor.cuh utils/warpselect/../Tensor.cuh \
 utils/warpselect/../Tensor-inl.cuh \
 utils/warpselect/../../GpuFaissAssert.h \
 utils/warpselect/../../../FaissAssert.h \
 utils/warpselect/../../../FaissException.h \
 utils/warpselect/../DeviceUtils.h \
 utils/warpselect/../../../FaissAssert.h \
 utils/warpselect/../MemorySpace.h \
 utils/warpselect/../DeviceTensor-inl.cuh utils/warpselect/../Select.cuh \
 utils/warpselect/../Comparators.cuh utils/warpselect/../DeviceDefs.cuh \
 utils/warpselect/../MergeNetworkBlock.cuh \
 utils/warpselect/../MergeNetworkUtils.cuh \
 utils/warpselect/../PtxUtils.cuh utils/warpselect/../StaticUtils.h \
 utils/warpselect/../WarpShuffles.cuh \
 utils/warpselect/../MergeNetworkWarp.cuh \
 utils/warpselect/../Reductions.cuh \
 utils/warpselect/../ReductionOperators.cuh \
 utils/warpselect/../Limits.cuh utils/warpselect/../Pair.cuh \
 utils/warpselect/../MathOperators.cuh
utils/warpselect/WarpSelectFloat32.o: utils/warpselect/WarpSelectFloat32.cu \
 utils/warpselect/WarpSelectImpl.cuh \
 utils/warpselect/../WarpSelectKernel.cuh utils/warpselect/../Float16.cuh \
 utils/warpselect/../../GpuResources.h \
 utils/warpselect/../../utils/DeviceMemory.h \
 utils/warpselect/../DeviceTensor.cuh utils/warpselect/../Tensor.cuh \
 utils/warpselect/../Tensor-inl.cuh \
 utils/warpselect/../../GpuFaissAssert.h \
 utils/warpselect/../../../FaissAssert.h \
 utils/warpselect/../../../FaissException.h \
 utils/warpselect/../DeviceUtils.h \
 utils/warpselect/../../../FaissAssert.h \
 utils/warpselect/../MemorySpace.h \
 utils/warpselect/../DeviceTensor-inl.cuh utils/warpselect/../Select.cuh \
 utils/warpselect/../Comparators.cuh utils/warpselect/../DeviceDefs.cuh \
 utils/warpselect/../MergeNetworkBlock.cuh \
 utils/warpselect/../MergeNetworkUtils.cuh \
 utils/warpselect/../PtxUtils.cuh utils/warpselect/../StaticUtils.h \
 utils/warpselect/../WarpShuffles.cuh \
 utils/warpselect/../MergeNetworkWarp.cuh \
 utils/warpselect/../Reductions.cuh \
 utils/warpselect/../ReductionOperators.cuh \
 utils/warpselect/../Limits.cuh utils/warpselect/../Pair.cuh \
 utils/warpselect/../MathOperators.cuh
utils/blockselect/BlockSelectHalf64.o: utils/blockselect/BlockSelectHalf64.cu \
 utils/blockselect/BlockSelectImpl.cuh \
 utils/blockselect/../BlockSelectKernel.cuh \
 utils/blockselect/../Float16.cuh utils/blockselect/../../GpuResources.h \
 utils/blockselect/../../utils/DeviceMemory.h \
 utils/blockselect/../DeviceTensor.cuh utils/blockselect/../Tensor.cuh \
 utils/blockselect/../Tensor-inl.cuh \
 utils/blockselect/../../GpuFaissAssert.h \
 utils/blockselect/../../../FaissAssert.h \
 utils/blockselect/../../../FaissException.h \
 utils/blockselect/../DeviceUtils.h \
 utils/blockselect/../../../FaissAssert.h \
 utils/blockselect/../MemorySpace.h \
 utils/blockselect/../DeviceTensor-inl.cuh \
 utils/blockselect/../Select.cuh utils/blockselect/../Comparators.cuh \
 utils/blockselect/../DeviceDefs.cuh \
 utils/blockselect/../MergeNetworkBlock.cuh \
 utils/blockselect/../MergeNetworkUtils.cuh \
 utils/blockselect/../PtxUtils.cuh utils/blockselect/../StaticUtils.h \
 utils/blockselect/../WarpShuffles.cuh \
 utils/blockselect/../MergeNetworkWarp.cuh \
 utils/blockselect/../Reductions.cuh \
 utils/blockselect/../ReductionOperators.cuh \
 utils/blockselect/../Limits.cuh utils/blockselect/../Pair.cuh \
 utils/blockselect/../MathOperators.cuh
utils/blockselect/BlockSelectFloat64.o: utils/blockselect/BlockSelectFloat64.cu \
 utils/blockselect/BlockSelectImpl.cuh \
 utils/blockselect/../BlockSelectKernel.cuh \
 utils/blockselect/../Float16.cuh utils/blockselect/../../GpuResources.h \
 utils/blockselect/../../utils/DeviceMemory.h \
 utils/blockselect/../DeviceTensor.cuh utils/blockselect/../Tensor.cuh \
 utils/blockselect/../Tensor-inl.cuh \
 utils/blockselect/../../GpuFaissAssert.h \
 utils/blockselect/../../../FaissAssert.h \
 utils/blockselect/../../../FaissException.h \
 utils/blockselect/../DeviceUtils.h \
 utils/blockselect/../../../FaissAssert.h \
 utils/blockselect/../MemorySpace.h \
 utils/blockselect/../DeviceTensor-inl.cuh \
 utils/blockselect/../Select.cuh utils/blockselect/../Comparators.cuh \
 utils/blockselect/../DeviceDefs.cuh \
 utils/blockselect/../MergeNetworkBlock.cuh \
 utils/blockselect/../MergeNetworkUtils.cuh \
 utils/blockselect/../PtxUtils.cuh utils/blockselect/../StaticUtils.h \
 utils/blockselect/../WarpShuffles.cuh \
 utils/blockselect/../MergeNetworkWarp.cuh \
 utils/blockselect/../Reductions.cuh \
 utils/blockselect/../ReductionOperators.cuh \
 utils/blockselect/../Limits.cuh utils/blockselect/../Pair.cuh \
 utils/blockselect/../MathOperators.cuh
utils/warpselect/WarpSelectHalf64.o: utils/warpselect/WarpSelectHalf64.cu \
 utils/warpselect/WarpSelectImpl.cuh \
 utils/warpselect/../WarpSelectKernel.cuh utils/warpselect/../Float16.cuh \
 utils/warpselect/../../GpuResources.h \
 utils/warpselect/../../utils/DeviceMemory.h \
 utils/warpselect/../DeviceTensor.cuh utils/warpselect/../Tensor.cuh \
 utils/warpselect/../Tensor-inl.cuh \
 utils/warpselect/../../GpuFaissAssert.h \
 utils/warpselect/../../../FaissAssert.h \
 utils/warpselect/../../../FaissException.h \
 utils/warpselect/../DeviceUtils.h \
 utils/warpselect/../../../FaissAssert.h \
 utils/warpselect/../MemorySpace.h \
 utils/warpselect/../DeviceTensor-inl.cuh utils/warpselect/../Select.cuh \
 utils/warpselect/../Comparators.cuh utils/warpselect/../DeviceDefs.cuh \
 utils/warpselect/../MergeNetworkBlock.cuh \
 utils/warpselect/../MergeNetworkUtils.cuh \
 utils/warpselect/../PtxUtils.cuh utils/warpselect/../StaticUtils.h \
 utils/warpselect/../WarpShuffles.cuh \
 utils/warpselect/../MergeNetworkWarp.cuh \
 utils/warpselect/../Reductions.cuh \
 utils/warpselect/../ReductionOperators.cuh \
 utils/warpselect/../Limits.cuh utils/warpselect/../Pair.cuh \
 utils/warpselect/../MathOperators.cuh
utils/warpselect/WarpSelectFloat64.o: utils/warpselect/WarpSelectFloat64.cu \
 utils/warpselect/WarpSelectImpl.cuh \
 utils/warpselect/../WarpSelectKernel.cuh utils/warpselect/../Float16.cuh \
 utils/warpselect/../../GpuResources.h \
 utils/warpselect/../../utils/DeviceMemory.h \
 utils/warpselect/../DeviceTensor.cuh utils/warpselect/../Tensor.cuh \
 utils/warpselect/../Tensor-inl.cuh \
 utils/warpselect/../../GpuFaissAssert.h \
 utils/warpselect/../../../FaissAssert.h \
 utils/warpselect/../../../FaissException.h \
 utils/warpselect/../DeviceUtils.h \
 utils/warpselect/../../../FaissAssert.h \
 utils/warpselect/../MemorySpace.h \
 utils/warpselect/../DeviceTensor-inl.cuh utils/warpselect/../Select.cuh \
 utils/warpselect/../Comparators.cuh utils/warpselect/../DeviceDefs.cuh \
 utils/warpselect/../MergeNetworkBlock.cuh \
 utils/warpselect/../MergeNetworkUtils.cuh \
 utils/warpselect/../PtxUtils.cuh utils/warpselect/../StaticUtils.h \
 utils/warpselect/../WarpShuffles.cuh \
 utils/warpselect/../MergeNetworkWarp.cuh \
 utils/warpselect/../Reductions.cuh \
 utils/warpselect/../ReductionOperators.cuh \
 utils/warpselect/../Limits.cuh utils/warpselect/../Pair.cuh \
 utils/warpselect/../MathOperators.cuh
utils/blockselect/BlockSelectHalf128.o: utils/blockselect/BlockSelectHalf128.cu \
 utils/blockselect/BlockSelectImpl.cuh \
 utils/blockselect/../BlockSelectKernel.cuh \
 utils/blockselect/../Float16.cuh utils/blockselect/../../GpuResources.h \
 utils/blockselect/../../utils/DeviceMemory.h \
 utils/blockselect/../DeviceTensor.cuh utils/blockselect/../Tensor.cuh \
 utils/blockselect/../Tensor-inl.cuh \
 utils/blockselect/../../GpuFaissAssert.h \
 utils/blockselect/../../../FaissAssert.h \
 utils/blockselect/../../../FaissException.h \
 utils/blockselect/../DeviceUtils.h \
 utils/blockselect/../../../FaissAssert.h \
 utils/blockselect/../MemorySpace.h \
 utils/blockselect/../DeviceTensor-inl.cuh \
 utils/blockselect/../Select.cuh utils/blockselect/../Comparators.cuh \
 utils/blockselect/../DeviceDefs.cuh \
 utils/blockselect/../MergeNetworkBlock.cuh \
 utils/blockselect/../MergeNetworkUtils.cuh \
 utils/blockselect/../PtxUtils.cuh utils/blockselect/../StaticUtils.h \
 utils/blockselect/../WarpShuffles.cuh \
 utils/blockselect/../MergeNetworkWarp.cuh \
 utils/blockselect/../Reductions.cuh \
 utils/blockselect/../ReductionOperators.cuh \
 utils/blockselect/../Limits.cuh utils/blockselect/../Pair.cuh \
 utils/blockselect/../MathOperators.cuh
utils/blockselect/BlockSelectFloat128.o: utils/blockselect/BlockSelectFloat128.cu \
 utils/blockselect/BlockSelectImpl.cuh \
 utils/blockselect/../BlockSelectKernel.cuh \
 utils/blockselect/../Float16.cuh utils/blockselect/../../GpuResources.h \
 utils/blockselect/../../utils/DeviceMemory.h \
 utils/blockselect/../DeviceTensor.cuh utils/blockselect/../Tensor.cuh \
 utils/blockselect/../Tensor-inl.cuh \
 utils/blockselect/../../GpuFaissAssert.h \
 utils/blockselect/../../../FaissAssert.h \
 utils/blockselect/../../../FaissException.h \
 utils/blockselect/../DeviceUtils.h \
 utils/blockselect/../../../FaissAssert.h \
 utils/blockselect/../MemorySpace.h \
 utils/blockselect/../DeviceTensor-inl.cuh \
 utils/blockselect/../Select.cuh utils/blockselect/../Comparators.cuh \
 utils/blockselect/../DeviceDefs.cuh \
 utils/blockselect/../MergeNetworkBlock.cuh \
 utils/blockselect/../MergeNetworkUtils.cuh \
 utils/blockselect/../PtxUtils.cuh utils/blockselect/../StaticUtils.h \
 utils/blockselect/../WarpShuffles.cuh \
 utils/blockselect/../MergeNetworkWarp.cuh \
 utils/blockselect/../Reductions.cuh \
 utils/blockselect/../ReductionOperators.cuh \
 utils/blockselect/../Limits.cuh utils/blockselect/../Pair.cuh \
 utils/blockselect/../MathOperators.cuh
utils/warpselect/WarpSelectHalf128.o: utils/warpselect/WarpSelectHalf128.cu \
 utils/warpselect/WarpSelectImpl.cuh \
 utils/warpselect/../WarpSelectKernel.cuh utils/warpselect/../Float16.cuh \
 utils/warpselect/../../GpuResources.h \
 utils/warpselect/../../utils/DeviceMemory.h \
 utils/warpselect/../DeviceTensor.cuh utils/warpselect/../Tensor.cuh \
 utils/warpselect/../Tensor-inl.cuh \
 utils/warpselect/../../GpuFaissAssert.h \
 utils/warpselect/../../../FaissAssert.h \
 utils/warpselect/../../../FaissException.h \
 utils/warpselect/../DeviceUtils.h \
 utils/warpselect/../../../FaissAssert.h \
 utils/warpselect/../MemorySpace.h \
 utils/warpselect/../DeviceTensor-inl.cuh utils/warpselect/../Select.cuh \
 utils/warpselect/../Comparators.cuh utils/warpselect/../DeviceDefs.cuh \
 utils/warpselect/../MergeNetworkBlock.cuh \
 utils/warpselect/../MergeNetworkUtils.cuh \
 utils/warpselect/../PtxUtils.cuh utils/warpselect/../StaticUtils.h \
 utils/warpselect/../WarpShuffles.cuh \
 utils/warpselect/../MergeNetworkWarp.cuh \
 utils/warpselect/../Reductions.cuh \
 utils/warpselect/../ReductionOperators.cuh \
 utils/warpselect/../Limits.cuh utils/warpselect/../Pair.cuh \
 utils/warpselect/../MathOperators.cuh
utils/warpselect/WarpSelectFloat128.o: utils/warpselect/WarpSelectFloat128.cu \
 utils/warpselect/WarpSelectImpl.cuh \
 utils/warpselect/../WarpSelectKernel.cuh utils/warpselect/../Float16.cuh \
 utils/warpselect/../../GpuResources.h \
 utils/warpselect/../../utils/DeviceMemory.h \
 utils/warpselect/../DeviceTensor.cuh utils/warpselect/../Tensor.cuh \
 utils/warpselect/../Tensor-inl.cuh \
 utils/warpselect/../../GpuFaissAssert.h \
 utils/warpselect/../../../FaissAssert.h \
 utils/warpselect/../../../FaissException.h \
 utils/warpselect/../DeviceUtils.h \
 utils/warpselect/../../../FaissAssert.h \
 utils/warpselect/../MemorySpace.h \
 utils/warpselect/../DeviceTensor-inl.cuh utils/warpselect/../Select.cuh \
 utils/warpselect/../Comparators.cuh utils/warpselect/../DeviceDefs.cuh \
 utils/warpselect/../MergeNetworkBlock.cuh \
 utils/warpselect/../MergeNetworkUtils.cuh \
 utils/warpselect/../PtxUtils.cuh utils/warpselect/../StaticUtils.h \
 utils/warpselect/../WarpShuffles.cuh \
 utils/warpselect/../MergeNetworkWarp.cuh \
 utils/warpselect/../Reductions.cuh \
 utils/warpselect/../ReductionOperators.cuh \
 utils/warpselect/../Limits.cuh utils/warpselect/../Pair.cuh \
 utils/warpselect/../MathOperators.cuh
utils/blockselect/BlockSelectHalf256.o: utils/blockselect/BlockSelectHalf256.cu \
 utils/blockselect/BlockSelectImpl.cuh \
 utils/blockselect/../BlockSelectKernel.cuh \
 utils/blockselect/../Float16.cuh utils/blockselect/../../GpuResources.h \
 utils/blockselect/../../utils/DeviceMemory.h \
 utils/blockselect/../DeviceTensor.cuh utils/blockselect/../Tensor.cuh \
 utils/blockselect/../Tensor-inl.cuh \
 utils/blockselect/../../GpuFaissAssert.h \
 utils/blockselect/../../../FaissAssert.h \
 utils/blockselect/../../../FaissException.h \
 utils/blockselect/../DeviceUtils.h \
 utils/blockselect/../../../FaissAssert.h \
 utils/blockselect/../MemorySpace.h \
 utils/blockselect/../DeviceTensor-inl.cuh \
 utils/blockselect/../Select.cuh utils/blockselect/../Comparators.cuh \
 utils/blockselect/../DeviceDefs.cuh \
 utils/blockselect/../MergeNetworkBlock.cuh \
 utils/blockselect/../MergeNetworkUtils.cuh \
 utils/blockselect/../PtxUtils.cuh utils/blockselect/../StaticUtils.h \
 utils/blockselect/../WarpShuffles.cuh \
 utils/blockselect/../MergeNetworkWarp.cuh \
 utils/blockselect/../Reductions.cuh \
 utils/blockselect/../ReductionOperators.cuh \
 utils/blockselect/../Limits.cuh utils/blockselect/../Pair.cuh \
 utils/blockselect/../MathOperators.cuh
utils/blockselect/BlockSelectFloat256.o: utils/blockselect/BlockSelectFloat256.cu \
 utils/blockselect/BlockSelectImpl.cuh \
 utils/blockselect/../BlockSelectKernel.cuh \
 utils/blockselect/../Float16.cuh utils/blockselect/../../GpuResources.h \
 utils/blockselect/../../utils/DeviceMemory.h \
 utils/blockselect/../DeviceTensor.cuh utils/blockselect/../Tensor.cuh \
 utils/blockselect/../Tensor-inl.cuh \
 utils/blockselect/../../GpuFaissAssert.h \
 utils/blockselect/../../../FaissAssert.h \
 utils/blockselect/../../../FaissException.h \
 utils/blockselect/../DeviceUtils.h \
 utils/blockselect/../../../FaissAssert.h \
 utils/blockselect/../MemorySpace.h \
 utils/blockselect/../DeviceTensor-inl.cuh \
 utils/blockselect/../Select.cuh utils/blockselect/../Comparators.cuh \
 utils/blockselect/../DeviceDefs.cuh \
 utils/blockselect/../MergeNetworkBlock.cuh \
 utils/blockselect/../MergeNetworkUtils.cuh \
 utils/blockselect/../PtxUtils.cuh utils/blockselect/../StaticUtils.h \
 utils/blockselect/../WarpShuffles.cuh \
 utils/blockselect/../MergeNetworkWarp.cuh \
 utils/blockselect/../Reductions.cuh \
 utils/blockselect/../ReductionOperators.cuh \
 utils/blockselect/../Limits.cuh utils/blockselect/../Pair.cuh \
 utils/blockselect/../MathOperators.cuh
utils/warpselect/WarpSelectHalf256.o: utils/warpselect/WarpSelectHalf256.cu \
 utils/warpselect/WarpSelectImpl.cuh \
 utils/warpselect/../WarpSelectKernel.cuh utils/warpselect/../Float16.cuh \
 utils/warpselect/../../GpuResources.h \
 utils/warpselect/../../utils/DeviceMemory.h \
 utils/warpselect/../DeviceTensor.cuh utils/warpselect/../Tensor.cuh \
 utils/warpselect/../Tensor-inl.cuh \
 utils/warpselect/../../GpuFaissAssert.h \
 utils/warpselect/../../../FaissAssert.h \
 utils/warpselect/../../../FaissException.h \
 utils/warpselect/../DeviceUtils.h \
 utils/warpselect/../../../FaissAssert.h \
 utils/warpselect/../MemorySpace.h \
 utils/warpselect/../DeviceTensor-inl.cuh utils/warpselect/../Select.cuh \
 utils/warpselect/../Comparators.cuh utils/warpselect/../DeviceDefs.cuh \
 utils/warpselect/../MergeNetworkBlock.cuh \
 utils/warpselect/../MergeNetworkUtils.cuh \
 utils/warpselect/../PtxUtils.cuh utils/warpselect/../StaticUtils.h \
 utils/warpselect/../WarpShuffles.cuh \
 utils/warpselect/../MergeNetworkWarp.cuh \
 utils/warpselect/../Reductions.cuh \
 utils/warpselect/../ReductionOperators.cuh \
 utils/warpselect/../Limits.cuh utils/warpselect/../Pair.cuh \
 utils/warpselect/../MathOperators.cuh
utils/warpselect/WarpSelectFloat256.o: utils/warpselect/WarpSelectFloat256.cu \
 utils/warpselect/WarpSelectImpl.cuh \
 utils/warpselect/../WarpSelectKernel.cuh utils/warpselect/../Float16.cuh \
 utils/warpselect/../../GpuResources.h \
 utils/warpselect/../../utils/DeviceMemory.h \
 utils/warpselect/../DeviceTensor.cuh utils/warpselect/../Tensor.cuh \
 utils/warpselect/../Tensor-inl.cuh \
 utils/warpselect/../../GpuFaissAssert.h \
 utils/warpselect/../../../FaissAssert.h \
 utils/warpselect/../../../FaissException.h \
 utils/warpselect/../DeviceUtils.h \
 utils/warpselect/../../../FaissAssert.h \
 utils/warpselect/../MemorySpace.h \
 utils/warpselect/../DeviceTensor-inl.cuh utils/warpselect/../Select.cuh \
 utils/warpselect/../Comparators.cuh utils/warpselect/../DeviceDefs.cuh \
 utils/warpselect/../MergeNetworkBlock.cuh \
 utils/warpselect/../MergeNetworkUtils.cuh \
 utils/warpselect/../PtxUtils.cuh utils/warpselect/../StaticUtils.h \
 utils/warpselect/../WarpShuffles.cuh \
 utils/warpselect/../MergeNetworkWarp.cuh \
 utils/warpselect/../Reductions.cuh \
 utils/warpselect/../ReductionOperators.cuh \
 utils/warpselect/../Limits.cuh utils/warpselect/../Pair.cuh \
 utils/warpselect/../MathOperators.cuh
utils/blockselect/BlockSelectHalfF512.o: utils/blockselect/BlockSelectHalfF512.cu \
 utils/blockselect/BlockSelectImpl.cuh \
 utils/blockselect/../BlockSelectKernel.cuh \
 utils/blockselect/../Float16.cuh utils/blockselect/../../GpuResources.h \
 utils/blockselect/../../utils/DeviceMemory.h \
 utils/blockselect/../DeviceTensor.cuh utils/blockselect/../Tensor.cuh \
 utils/blockselect/../Tensor-inl.cuh \
 utils/blockselect/../../GpuFaissAssert.h \
 utils/blockselect/../../../FaissAssert.h \
 utils/blockselect/../../../FaissException.h \
 utils/blockselect/../DeviceUtils.h \
 utils/blockselect/../../../FaissAssert.h \
 utils/blockselect/../MemorySpace.h \
 utils/blockselect/../DeviceTensor-inl.cuh \
 utils/blockselect/../Select.cuh utils/blockselect/../Comparators.cuh \
 utils/blockselect/../DeviceDefs.cuh \
 utils/blockselect/../MergeNetworkBlock.cuh \
 utils/blockselect/../MergeNetworkUtils.cuh \
 utils/blockselect/../PtxUtils.cuh utils/blockselect/../StaticUtils.h \
 utils/blockselect/../WarpShuffles.cuh \
 utils/blockselect/../MergeNetworkWarp.cuh \
 utils/blockselect/../Reductions.cuh \
 utils/blockselect/../ReductionOperators.cuh \
 utils/blockselect/../Limits.cuh utils/blockselect/../Pair.cuh \
 utils/blockselect/../MathOperators.cuh
utils/blockselect/BlockSelectFloatF512.o: utils/blockselect/BlockSelectFloatF512.cu \
 utils/blockselect/BlockSelectImpl.cuh \
 utils/blockselect/../BlockSelectKernel.cuh \
 utils/blockselect/../Float16.cuh utils/blockselect/../../GpuResources.h \
 utils/blockselect/../../utils/DeviceMemory.h \
 utils/blockselect/../DeviceTensor.cuh utils/blockselect/../Tensor.cuh \
 utils/blockselect/../Tensor-inl.cuh \
 utils/blockselect/../../GpuFaissAssert.h \
 utils/blockselect/../../../FaissAssert.h \
 utils/blockselect/../../../FaissException.h \
 utils/blockselect/../DeviceUtils.h \
 utils/blockselect/../../../FaissAssert.h \
 utils/blockselect/../MemorySpace.h \
 utils/blockselect/../DeviceTensor-inl.cuh \
 utils/blockselect/../Select.cuh utils/blockselect/../Comparators.cuh \
 utils/blockselect/../DeviceDefs.cuh \
 utils/blockselect/../MergeNetworkBlock.cuh \
 utils/blockselect/../MergeNetworkUtils.cuh \
 utils/blockselect/../PtxUtils.cuh utils/blockselect/../StaticUtils.h \
 utils/blockselect/../WarpShuffles.cuh \
 utils/blockselect/../MergeNetworkWarp.cuh \
 utils/blockselect/../Reductions.cuh \
 utils/blockselect/../ReductionOperators.cuh \
 utils/blockselect/../Limits.cuh utils/blockselect/../Pair.cuh \
 utils/blockselect/../MathOperators.cuh
utils/warpselect/WarpSelectHalfF512.o: utils/warpselect/WarpSelectHalfF512.cu \
 utils/warpselect/WarpSelectImpl.cuh \
 utils/warpselect/../WarpSelectKernel.cuh utils/warpselect/../Float16.cuh \
 utils/warpselect/../../GpuResources.h \
 utils/warpselect/../../utils/DeviceMemory.h \
 utils/warpselect/../DeviceTensor.cuh utils/warpselect/../Tensor.cuh \
 utils/warpselect/../Tensor-inl.cuh \
 utils/warpselect/../../GpuFaissAssert.h \
 utils/warpselect/../../../FaissAssert.h \
 utils/warpselect/../../../FaissException.h \
 utils/warpselect/../DeviceUtils.h \
 utils/warpselect/../../../FaissAssert.h \
 utils/warpselect/../MemorySpace.h \
 utils/warpselect/../DeviceTensor-inl.cuh utils/warpselect/../Select.cuh \
 utils/warpselect/../Comparators.cuh utils/warpselect/../DeviceDefs.cuh \
 utils/warpselect/../MergeNetworkBlock.cuh \
 utils/warpselect/../MergeNetworkUtils.cuh \
 utils/warpselect/../PtxUtils.cuh utils/warpselect/../StaticUtils.h \
 utils/warpselect/../WarpShuffles.cuh \
 utils/warpselect/../MergeNetworkWarp.cuh \
 utils/warpselect/../Reductions.cuh \
 utils/warpselect/../ReductionOperators.cuh \
 utils/warpselect/../Limits.cuh utils/warpselect/../Pair.cuh \
 utils/warpselect/../MathOperators.cuh
utils/warpselect/WarpSelectFloatF512.o: utils/warpselect/WarpSelectFloatF512.cu \
 utils/warpselect/WarpSelectImpl.cuh \
 utils/warpselect/../WarpSelectKernel.cuh utils/warpselect/../Float16.cuh \
 utils/warpselect/../../GpuResources.h \
 utils/warpselect/../../utils/DeviceMemory.h \
 utils/warpselect/../DeviceTensor.cuh utils/warpselect/../Tensor.cuh \
 utils/warpselect/../Tensor-inl.cuh \
 utils/warpselect/../../GpuFaissAssert.h \
 utils/warpselect/../../../FaissAssert.h \
 utils/warpselect/../../../FaissException.h \
 utils/warpselect/../DeviceUtils.h \
 utils/warpselect/../../../FaissAssert.h \
 utils/warpselect/../MemorySpace.h \
 utils/warpselect/../DeviceTensor-inl.cuh utils/warpselect/../Select.cuh \
 utils/warpselect/../Comparators.cuh utils/warpselect/../DeviceDefs.cuh \
 utils/warpselect/../MergeNetworkBlock.cuh \
 utils/warpselect/../MergeNetworkUtils.cuh \
 utils/warpselect/../PtxUtils.cuh utils/warpselect/../StaticUtils.h \
 utils/warpselect/../WarpShuffles.cuh \
 utils/warpselect/../MergeNetworkWarp.cuh \
 utils/warpselect/../Reductions.cuh \
 utils/warpselect/../ReductionOperators.cuh \
 utils/warpselect/../Limits.cuh utils/warpselect/../Pair.cuh \
 utils/warpselect/../MathOperators.cuh
utils/blockselect/BlockSelectHalfT512.o: utils/blockselect/BlockSelectHalfT512.cu \
 utils/blockselect/BlockSelectImpl.cuh \
 utils/blockselect/../BlockSelectKernel.cuh \
 utils/blockselect/../Float16.cuh utils/blockselect/../../GpuResources.h \
 utils/blockselect/../../utils/DeviceMemory.h \
 utils/blockselect/../DeviceTensor.cuh utils/blockselect/../Tensor.cuh \
 utils/blockselect/../Tensor-inl.cuh \
 utils/blockselect/../../GpuFaissAssert.h \
 utils/blockselect/../../../FaissAssert.h \
 utils/blockselect/../../../FaissException.h \
 utils/blockselect/../DeviceUtils.h \
 utils/blockselect/../../../FaissAssert.h \
 utils/blockselect/../MemorySpace.h \
 utils/blockselect/../DeviceTensor-inl.cuh \
 utils/blockselect/../Select.cuh utils/blockselect/../Comparators.cuh \
 utils/blockselect/../DeviceDefs.cuh \
 utils/blockselect/../MergeNetworkBlock.cuh \
 utils/blockselect/../MergeNetworkUtils.cuh \
 utils/blockselect/../PtxUtils.cuh utils/blockselect/../StaticUtils.h \
 utils/blockselect/../WarpShuffles.cuh \
 utils/blockselect/../MergeNetworkWarp.cuh \
 utils/blockselect/../Reductions.cuh \
 utils/blockselect/../ReductionOperators.cuh \
 utils/blockselect/../Limits.cuh utils/blockselect/../Pair.cuh \
 utils/blockselect/../MathOperators.cuh
utils/blockselect/BlockSelectFloatT512.o: utils/blockselect/BlockSelectFloatT512.cu \
 utils/blockselect/BlockSelectImpl.cuh \
 utils/blockselect/../BlockSelectKernel.cuh \
 utils/blockselect/../Float16.cuh utils/blockselect/../../GpuResources.h \
 utils/blockselect/../../utils/DeviceMemory.h \
 utils/blockselect/../DeviceTensor.cuh utils/blockselect/../Tensor.cuh \
 utils/blockselect/../Tensor-inl.cuh \
 utils/blockselect/../../GpuFaissAssert.h \
 utils/blockselect/../../../FaissAssert.h \
 utils/blockselect/../../../FaissException.h \
 utils/blockselect/../DeviceUtils.h \
 utils/blockselect/../../../FaissAssert.h \
 utils/blockselect/../MemorySpace.h \
 utils/blockselect/../DeviceTensor-inl.cuh \
 utils/blockselect/../Select.cuh utils/blockselect/../Comparators.cuh \
 utils/blockselect/../DeviceDefs.cuh \
 utils/blockselect/../MergeNetworkBlock.cuh \
 utils/blockselect/../MergeNetworkUtils.cuh \
 utils/blockselect/../PtxUtils.cuh utils/blockselect/../StaticUtils.h \
 utils/blockselect/../WarpShuffles.cuh \
 utils/blockselect/../MergeNetworkWarp.cuh \
 utils/blockselect/../Reductions.cuh \
 utils/blockselect/../ReductionOperators.cuh \
 utils/blockselect/../Limits.cuh utils/blockselect/../Pair.cuh \
 utils/blockselect/../MathOperators.cuh
utils/warpselect/WarpSelectHalfT512.o: utils/warpselect/WarpSelectHalfT512.cu \
 utils/warpselect/WarpSelectImpl.cuh \
 utils/warpselect/../WarpSelectKernel.cuh utils/warpselect/../Float16.cuh \
 utils/warpselect/../../GpuResources.h \
 utils/warpselect/../../utils/DeviceMemory.h \
 utils/warpselect/../DeviceTensor.cuh utils/warpselect/../Tensor.cuh \
 utils/warpselect/../Tensor-inl.cuh \
 utils/warpselect/../../GpuFaissAssert.h \
 utils/warpselect/../../../FaissAssert.h \
 utils/warpselect/../../../FaissException.h \
 utils/warpselect/../DeviceUtils.h \
 utils/warpselect/../../../FaissAssert.h \
 utils/warpselect/../MemorySpace.h \
 utils/warpselect/../DeviceTensor-inl.cuh utils/warpselect/../Select.cuh \
 utils/warpselect/../Comparators.cuh utils/warpselect/../DeviceDefs.cuh \
 utils/warpselect/../MergeNetworkBlock.cuh \
 utils/warpselect/../MergeNetworkUtils.cuh \
 utils/warpselect/../PtxUtils.cuh utils/warpselect/../StaticUtils.h \
 utils/warpselect/../WarpShuffles.cuh \
 utils/warpselect/../MergeNetworkWarp.cuh \
 utils/warpselect/../Reductions.cuh \
 utils/warpselect/../ReductionOperators.cuh \
 utils/warpselect/../Limits.cuh utils/warpselect/../Pair.cuh \
 utils/warpselect/../MathOperators.cuh
utils/warpselect/WarpSelectFloatT512.o: utils/warpselect/WarpSelectFloatT512.cu \
 utils/warpselect/WarpSelectImpl.cuh \
 utils/warpselect/../WarpSelectKernel.cuh utils/warpselect/../Float16.cuh \
 utils/warpselect/../../GpuResources.h \
 utils/warpselect/../../utils/DeviceMemory.h \
 utils/warpselect/../DeviceTensor.cuh utils/warpselect/../Tensor.cuh \
 utils/warpselect/../Tensor-inl.cuh \
 utils/warpselect/../../GpuFaissAssert.h \
 utils/warpselect/../../../FaissAssert.h \
 utils/warpselect/../../../FaissException.h \
 utils/warpselect/../DeviceUtils.h \
 utils/warpselect/../../../FaissAssert.h \
 utils/warpselect/../MemorySpace.h \
 utils/warpselect/../DeviceTensor-inl.cuh utils/warpselect/../Select.cuh \
 utils/warpselect/../Comparators.cuh utils/warpselect/../DeviceDefs.cuh \
 utils/warpselect/../MergeNetworkBlock.cuh \
 utils/warpselect/../MergeNetworkUtils.cuh \
 utils/warpselect/../PtxUtils.cuh utils/warpselect/../StaticUtils.h \
 utils/warpselect/../WarpShuffles.cuh \
 utils/warpselect/../MergeNetworkWarp.cuh \
 utils/warpselect/../Reductions.cuh \
 utils/warpselect/../ReductionOperators.cuh \
 utils/warpselect/../Limits.cuh utils/warpselect/../Pair.cuh \
 utils/warpselect/../MathOperators.cuh
utils/blockselect/BlockSelectHalfF1024.o: utils/blockselect/BlockSelectHalfF1024.cu \
 utils/blockselect/BlockSelectImpl.cuh \
 utils/blockselect/../BlockSelectKernel.cuh \
 utils/blockselect/../Float16.cuh utils/blockselect/../../GpuResources.h \
 utils/blockselect/../../utils/DeviceMemory.h \
 utils/blockselect/../DeviceTensor.cuh utils/blockselect/../Tensor.cuh \
 utils/blockselect/../Tensor-inl.cuh \
 utils/blockselect/../../GpuFaissAssert.h \
 utils/blockselect/../../../FaissAssert.h \
 utils/blockselect/../../../FaissException.h \
 utils/blockselect/../DeviceUtils.h \
 utils/blockselect/../../../FaissAssert.h \
 utils/blockselect/../MemorySpace.h \
 utils/blockselect/../DeviceTensor-inl.cuh \
 utils/blockselect/../Select.cuh utils/blockselect/../Comparators.cuh \
 utils/blockselect/../DeviceDefs.cuh \
 utils/blockselect/../MergeNetworkBlock.cuh \
 utils/blockselect/../MergeNetworkUtils.cuh \
 utils/blockselect/../PtxUtils.cuh utils/blockselect/../StaticUtils.h \
 utils/blockselect/../WarpShuffles.cuh \
 utils/blockselect/../MergeNetworkWarp.cuh \
 utils/blockselect/../Reductions.cuh \
 utils/blockselect/../ReductionOperators.cuh \
 utils/blockselect/../Limits.cuh utils/blockselect/../Pair.cuh \
 utils/blockselect/../MathOperators.cuh
utils/blockselect/BlockSelectFloatF1024.o: utils/blockselect/BlockSelectFloatF1024.cu \
 utils/blockselect/BlockSelectImpl.cuh \
 utils/blockselect/../BlockSelectKernel.cuh \
 utils/blockselect/../Float16.cuh utils/blockselect/../../GpuResources.h \
 utils/blockselect/../../utils/DeviceMemory.h \
 utils/blockselect/../DeviceTensor.cuh utils/blockselect/../Tensor.cuh \
 utils/blockselect/../Tensor-inl.cuh \
 utils/blockselect/../../GpuFaissAssert.h \
 utils/blockselect/../../../FaissAssert.h \
 utils/blockselect/../../../FaissException.h \
 utils/blockselect/../DeviceUtils.h \
 utils/blockselect/../../../FaissAssert.h \
 utils/blockselect/../MemorySpace.h \
 utils/blockselect/../DeviceTensor-inl.cuh \
 utils/blockselect/../Select.cuh utils/blockselect/../Comparators.cuh \
 utils/blockselect/../DeviceDefs.cuh \
 utils/blockselect/../MergeNetworkBlock.cuh \
 utils/blockselect/../MergeNetworkUtils.cuh \
 utils/blockselect/../PtxUtils.cuh utils/blockselect/../StaticUtils.h \
 utils/blockselect/../WarpShuffles.cuh \
 utils/blockselect/../MergeNetworkWarp.cuh \
 utils/blockselect/../Reductions.cuh \
 utils/blockselect/../ReductionOperators.cuh \
 utils/blockselect/../Limits.cuh utils/blockselect/../Pair.cuh \
 utils/blockselect/../MathOperators.cuh
utils/warpselect/WarpSelectHalfF1024.o: utils/warpselect/WarpSelectHalfF1024.cu \
 utils/warpselect/WarpSelectImpl.cuh \
 utils/warpselect/../WarpSelectKernel.cuh utils/warpselect/../Float16.cuh \
 utils/warpselect/../../GpuResources.h \
 utils/warpselect/../../utils/DeviceMemory.h \
 utils/warpselect/../DeviceTensor.cuh utils/warpselect/../Tensor.cuh \
 utils/warpselect/../Tensor-inl.cuh \
 utils/warpselect/../../GpuFaissAssert.h \
 utils/warpselect/../../../FaissAssert.h \
 utils/warpselect/../../../FaissException.h \
 utils/warpselect/../DeviceUtils.h \
 utils/warpselect/../../../FaissAssert.h \
 utils/warpselect/../MemorySpace.h \
 utils/warpselect/../DeviceTensor-inl.cuh utils/warpselect/../Select.cuh \
 utils/warpselect/../Comparators.cuh utils/warpselect/../DeviceDefs.cuh \
 utils/warpselect/../MergeNetworkBlock.cuh \
 utils/warpselect/../MergeNetworkUtils.cuh \
 utils/warpselect/../PtxUtils.cuh utils/warpselect/../StaticUtils.h \
 utils/warpselect/../WarpShuffles.cuh \
 utils/warpselect/../MergeNetworkWarp.cuh \
 utils/warpselect/../Reductions.cuh \
 utils/warpselect/../ReductionOperators.cuh \
 utils/warpselect/../Limits.cuh utils/warpselect/../Pair.cuh \
 utils/warpselect/../MathOperators.cuh
utils/warpselect/WarpSelectFloatF1024.o: utils/warpselect/WarpSelectFloatF1024.cu \
 utils/warpselect/WarpSelectImpl.cuh \
 utils/warpselect/../WarpSelectKernel.cuh utils/warpselect/../Float16.cuh \
 utils/warpselect/../../GpuResources.h \
 utils/warpselect/../../utils/DeviceMemory.h \
 utils/warpselect/../DeviceTensor.cuh utils/warpselect/../Tensor.cuh \
 utils/warpselect/../Tensor-inl.cuh \
 utils/warpselect/../../GpuFaissAssert.h \
 utils/warpselect/../../../FaissAssert.h \
 utils/warpselect/../../../FaissException.h \
 utils/warpselect/../DeviceUtils.h \
 utils/warpselect/../../../FaissAssert.h \
 utils/warpselect/../MemorySpace.h \
 utils/warpselect/../DeviceTensor-inl.cuh utils/warpselect/../Select.cuh \
 utils/warpselect/../Comparators.cuh utils/warpselect/../DeviceDefs.cuh \
 utils/warpselect/../MergeNetworkBlock.cuh \
 utils/warpselect/../MergeNetworkUtils.cuh \
 utils/warpselect/../PtxUtils.cuh utils/warpselect/../StaticUtils.h \
 utils/warpselect/../WarpShuffles.cuh \
 utils/warpselect/../MergeNetworkWarp.cuh \
 utils/warpselect/../Reductions.cuh \
 utils/warpselect/../ReductionOperators.cuh \
 utils/warpselect/../Limits.cuh utils/warpselect/../Pair.cuh \
 utils/warpselect/../MathOperators.cuh
utils/blockselect/BlockSelectHalfT1024.o: utils/blockselect/BlockSelectHalfT1024.cu \
 utils/blockselect/BlockSelectImpl.cuh \
 utils/blockselect/../BlockSelectKernel.cuh \
 utils/blockselect/../Float16.cuh utils/blockselect/../../GpuResources.h \
 utils/blockselect/../../utils/DeviceMemory.h \
 utils/blockselect/../DeviceTensor.cuh utils/blockselect/../Tensor.cuh \
 utils/blockselect/../Tensor-inl.cuh \
 utils/blockselect/../../GpuFaissAssert.h \
 utils/blockselect/../../../FaissAssert.h \
 utils/blockselect/../../../FaissException.h \
 utils/blockselect/../DeviceUtils.h \
 utils/blockselect/../../../FaissAssert.h \
 utils/blockselect/../MemorySpace.h \
 utils/blockselect/../DeviceTensor-inl.cuh \
 utils/blockselect/../Select.cuh utils/blockselect/../Comparators.cuh \
 utils/blockselect/../DeviceDefs.cuh \
 utils/blockselect/../MergeNetworkBlock.cuh \
 utils/blockselect/../MergeNetworkUtils.cuh \
 utils/blockselect/../PtxUtils.cuh utils/blockselect/../StaticUtils.h \
 utils/blockselect/../WarpShuffles.cuh \
 utils/blockselect/../MergeNetworkWarp.cuh \
 utils/blockselect/../Reductions.cuh \
 utils/blockselect/../ReductionOperators.cuh \
 utils/blockselect/../Limits.cuh utils/blockselect/../Pair.cuh \
 utils/blockselect/../MathOperators.cuh
utils/blockselect/BlockSelectFloatT1024.o: utils/blockselect/BlockSelectFloatT1024.cu \
 utils/blockselect/BlockSelectImpl.cuh \
 utils/blockselect/../BlockSelectKernel.cuh \
 utils/blockselect/../Float16.cuh utils/blockselect/../../GpuResources.h \
 utils/blockselect/../../utils/DeviceMemory.h \
 utils/blockselect/../DeviceTensor.cuh utils/blockselect/../Tensor.cuh \
 utils/blockselect/../Tensor-inl.cuh \
 utils/blockselect/../../GpuFaissAssert.h \
 utils/blockselect/../../../FaissAssert.h \
 utils/blockselect/../../../FaissException.h \
 utils/blockselect/../DeviceUtils.h \
 utils/blockselect/../../../FaissAssert.h \
 utils/blockselect/../MemorySpace.h \
 utils/blockselect/../DeviceTensor-inl.cuh \
 utils/blockselect/../Select.cuh utils/blockselect/../Comparators.cuh \
 utils/blockselect/../DeviceDefs.cuh \
 utils/blockselect/../MergeNetworkBlock.cuh \
 utils/blockselect/../MergeNetworkUtils.cuh \
 utils/blockselect/../PtxUtils.cuh utils/blockselect/../StaticUtils.h \
 utils/blockselect/../WarpShuffles.cuh \
 utils/blockselect/../MergeNetworkWarp.cuh \
 utils/blockselect/../Reductions.cuh \
 utils/blockselect/../ReductionOperators.cuh \
 utils/blockselect/../Limits.cuh utils/blockselect/../Pair.cuh \
 utils/blockselect/../MathOperators.cuh
utils/warpselect/WarpSelectHalfT1024.o: utils/warpselect/WarpSelectHalfT1024.cu \
 utils/warpselect/WarpSelectImpl.cuh \
 utils/warpselect/../WarpSelectKernel.cuh utils/warpselect/../Float16.cuh \
 utils/warpselect/../../GpuResources.h \
 utils/warpselect/../../utils/DeviceMemory.h \
 utils/warpselect/../DeviceTensor.cuh utils/warpselect/../Tensor.cuh \
 utils/warpselect/../Tensor-inl.cuh \
 utils/warpselect/../../GpuFaissAssert.h \
 utils/warpselect/../../../FaissAssert.h \
 utils/warpselect/../../../FaissException.h \
 utils/warpselect/../DeviceUtils.h \
 utils/warpselect/../../../FaissAssert.h \
 utils/warpselect/../MemorySpace.h \
 utils/warpselect/../DeviceTensor-inl.cuh utils/warpselect/../Select.cuh \
 utils/warpselect/../Comparators.cuh utils/warpselect/../DeviceDefs.cuh \
 utils/warpselect/../MergeNetworkBlock.cuh \
 utils/warpselect/../MergeNetworkUtils.cuh \
 utils/warpselect/../PtxUtils.cuh utils/warpselect/../StaticUtils.h \
 utils/warpselect/../WarpShuffles.cuh \
 utils/warpselect/../MergeNetworkWarp.cuh \
 utils/warpselect/../Reductions.cuh \
 utils/warpselect/../ReductionOperators.cuh \
 utils/warpselect/../Limits.cuh utils/warpselect/../Pair.cuh \
 utils/warpselect/../MathOperators.cuh
utils/warpselect/WarpSelectFloatT1024.o: utils/warpselect/WarpSelectFloatT1024.cu \
 utils/warpselect/WarpSelectImpl.cuh \
 utils/warpselect/../WarpSelectKernel.cuh utils/warpselect/../Float16.cuh \
 utils/warpselect/../../GpuResources.h \
 utils/warpselect/../../utils/DeviceMemory.h \
 utils/warpselect/../DeviceTensor.cuh utils/warpselect/../Tensor.cuh \
 utils/warpselect/../Tensor-inl.cuh \
 utils/warpselect/../../GpuFaissAssert.h \
 utils/warpselect/../../../FaissAssert.h \
 utils/warpselect/../../../FaissException.h \
 utils/warpselect/../DeviceUtils.h \
 utils/warpselect/../../../FaissAssert.h \
 utils/warpselect/../MemorySpace.h \
 utils/warpselect/../DeviceTensor-inl.cuh utils/warpselect/../Select.cuh \
 utils/warpselect/../Comparators.cuh utils/warpselect/../DeviceDefs.cuh \
 utils/warpselect/../MergeNetworkBlock.cuh \
 utils/warpselect/../MergeNetworkUtils.cuh \
 utils/warpselect/../PtxUtils.cuh utils/warpselect/../StaticUtils.h \
 utils/warpselect/../WarpShuffles.cuh \
 utils/warpselect/../MergeNetworkWarp.cuh \
 utils/warpselect/../Reductions.cuh \
 utils/warpselect/../ReductionOperators.cuh \
 utils/warpselect/../Limits.cuh utils/warpselect/../Pair.cuh \
 utils/warpselect/../MathOperators.cuh
