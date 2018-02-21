MAKEFILE_INC=../../makefile.inc

-include $(MAKEFILE_INC)

NVCCLDFLAGS =	-Xcompiler \"-Wl,-rpath=../../:../../gpu/\" \
		-L../.. -L../../gpu -lfaiss -lgpufaiss


all: cpu gpu

cpu: 1-Flat 2-IVFFlat 3-IVFPQ

gpu: 4-GPU 5-Multiple-GPUs

1-Flat: 1-Flat.cpp ../../libfaiss.$(SHAREDEXT)
	$(CC) -o $@ $(CFLAGS) $< -L../../ -lfaiss -I../../../ $(LDFLAGS) \
	$(BLASLDFLAGS)

2-IVFFlat: 2-IVFFlat.cpp ../../libfaiss.$(SHAREDEXT)
	$(CC) -o $@ $(CFLAGS) $< -L../../ -lfaiss -I../../../ $(LDFLAGS) \
	$(BLASLDFLAGS)

3-IVFPQ: 3-IVFPQ.cpp ../../libfaiss.$(SHAREDEXT)
	$(CC) -o $@ $(CFLAGS) $< -L../../ -lfaiss -I../../../ $(LDFLAGS) \
	$(BLASLDFLAGS)

4-GPU: 4-GPU.cpp ../../libfaiss.$(SHAREDEXT) ../../gpu/libgpufaiss.$(SHAREDEXT)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(NVCCLDFLAGS) \
	-I../../../ -Xcompiler -fopenmp -lcublas $(BLASLDFLAGSNVCC) \

5-Multiple-GPUs: 5-Multiple-GPUs.cpp ../../libfaiss.$(SHAREDEXT) \
	../../gpu/libgpufaiss.$(SHAREDEXT)
	$(NVCC) $(NVCCFLAGS) -o $@ $<  $(NVCCLDFLAGS) \
	-I../../../ -Xcompiler -fopenmp -lcublas $(BLASLDFLAGSNVCC)

../../libfaiss.$(SHAREDEXT):
	cd ../../ && make libfaiss.$(SHAREDEXT)

../../gpu/libgpufaiss.$(SHAREDEXT):
	cd ../../gpu/ && make libgpufaiss.$(SHAREDEXT)
