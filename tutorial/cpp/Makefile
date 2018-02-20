MAKEFILE_INC=../../makefile.inc

-include $(MAKEFILE_INC)

all: cpu gpu

cpu: 1-Flat 2-IVFFlat 3-IVFPQ

gpu: 4-GPU 5-Multiple-GPUs

1-Flat: 1-Flat.cpp ../../libfaiss.a
	$(CC) -o $@ $(CFLAGS) $^ -I../../../ $(LDFLAGS) $(BLASLDFLAGS)

2-IVFFlat: 2-IVFFlat.cpp ../../libfaiss.a
	$(CC) -o $@ $(CFLAGS) $^ -I../../../ $(LDFLAGS) $(BLASLDFLAGS)

3-IVFPQ: 3-IVFPQ.cpp ../../libfaiss.a
	$(CC) -o $@ $(CFLAGS) $^ -I../../../ $(LDFLAGS) $(BLASLDFLAGS)

4-GPU: 4-GPU.cpp ../../libfaiss.a ../../gpu/libgpufaiss.a
	$(NVCC) $(NVCCFLAGS) -o $@ $^ -I../../../ -Xcompiler -fopenmp -lcublas \
	$(BLASLDFLAGSNVCC)

5-Multiple-GPUs: 5-Multiple-GPUs.cpp ../../libfaiss.a ../../gpu/libgpufaiss.a
	$(NVCC) $(NVCCFLAGS) -o $@ $^ -I../../../ -Xcompiler -fopenmp -lcublas \
	$(BLASLDFLAGSNVCC)
