#!/bin/bash
BIGANN_DIR=${BASE_DIR}/sift1bi
FAISS_DIR=.
DEMO_DIR=${FAISS_DIR}/build/faiss/gpu/demos
RESULT_DIR=${FAISS_DIR}/demo_out
PROF_SCRIPT_DIR=${FAISS_DIR}/faiss/gpu/demos/scripts/mpi/sift

N_THREADS=48
USE_SHARD=1
MAX_MEM=28991029248
USE_PRECOMP=1
USE_IMI=1
USE_GPU=1
PRINT_GPU_MEM=0

# IMIPQ single process
N_GPUS=1
## Indexing
SEARCH=0
mpirun -np 1 ${DEMO_DIR}/demo_imipq_gpu_sift_m_mpi 128 15560 8 8 ${BIGANN_DIR}/bigann_learn.bvecs 32381696 ${BIGANN_DIR}/bigann_base.bvecs 1000000000 ${BIGANN_DIR}/bigann_query.bvecs 0 "" 3 4 6 7 6 7 0 ${N_THREADS} ${N_GPUS} ${USE_SHARD} 0 1 ${MAX_MEM} ${RESULT_DIR}/coarse/coarse_imipq15560_32M_m_replica_full.bin ${RESULT_DIR}/index/index_imipq15560_32M_m_replica.bin ${SEARCH} 0 0 1 0 ${USE_PRECOMP} ${USE_IMI} ${USE_GPU} ${PRINT_GPU_MEM} 0 | tee ${RESULT_DIR}/outs/mpi_1_1gpu_imipq1000M_15560_32M_m_27GB_10Kqueries_search0.txt
## Searching
SEARCH=1
### PROFILER OFF
mpirun -np 1 ${DEMO_DIR}/demo_imipq_gpu_sift_m_mpi 128 15560 8 8 ${BIGANN_DIR}/bigann_learn.bvecs 32381696 ${BIGANN_DIR}/bigann_base.bvecs 250000000 ${BIGANN_DIR}/bigann_query.bvecs 0 "" 11 12 6 7 6 7 0 ${N_THREADS} ${N_GPUS} ${USE_SHARD} 0 1 ${MAX_MEM} ${RESULT_DIR}/coarse/coarse_imipq15560_32M_m_replica_full.bin ${RESULT_DIR}/index/index_imipq15560_32M_m_replica_250M.bin ${SEARCH} 0 0 1 0 ${USE_PRECOMP} ${USE_IMI} ${USE_GPU} ${PRINT_GPU_MEM} 0 | tee ${RESULT_DIR}/outs/mpi_1_1gpu_imipq250M_15560_32M_m_27GB_1Mqueries_search1_noprof.txt
### PROFILER ON (Disabled)
# mpirun -np 1 ${PROF_SCRIPT_DIR}/nvprof_wrapper.sh ${RESULT_DIR}/outs/mpi_1_1gpu_imipq1000M_15560_32M_m_27GB_10Kqueries_prof ${DEMO_DIR}/demo_imipq_gpu_sift_m_mpi 128 15560 8 8 ${BIGANN_DIR}/bigann_learn.bvecs 32381696 ${BIGANN_DIR}/bigann_base.bvecs 1000000000 ${BIGANN_DIR}/bigann_query.bvecs 0 "" 3 4 6 7 6 7 0 ${N_THREADS} ${N_GPUS} ${USE_SHARD} 0 1 ${MAX_MEM} ${RESULT_DIR}/coarse/coarse_imipq15560_32M_m_replica_full.bin ${RESULT_DIR}/index/index_imipq15560_32M_m_replica.bin ${SEARCH} 0 0 1 0 ${USE_PRECOMP} ${USE_IMI} ${USE_GPU} ${PRINT_GPU_MEM} 0 | tee ${RESULT_DIR}/outs/mpi_1_1gpu_imipq1000M_15560_32M_m_27GB_10Kqueries_search1.txt

# IMIPQ multi process
N_GPUS=2
## Indexing
SEARCH=0
mpirun -np 2 ${DEMO_DIR}/demo_imipq_gpu_sift_m_mpi 128 15560 8 8 ${BIGANN_DIR}/bigann_learn.bvecs 32381696 ${BIGANN_DIR}/bigann_base.bvecs 500000000 ${BIGANN_DIR}/bigann_query.bvecs 0 "" 11 12 6 7 6 7 0 ${N_THREADS} ${N_GPUS} ${USE_SHARD} 0 1 ${MAX_MEM} ${RESULT_DIR}/coarse/coarse_imipq15560_32M_m_replica_full.bin ${RESULT_DIR}/index/index_imipq15560_32M_m_replica_500M.bin ${SEARCH} 0 0 1 0 ${USE_PRECOMP} ${USE_IMI} ${USE_GPU} ${PRINT_GPU_MEM} 0 | tee ${RESULT_DIR}/outs/mpi_2_1gpu_imipq500M_15560_32M_m_27GB_1Mqueries_search0.txt
## Searching
SEARCH=1
### PROFILER OFF
mpirun -np 2 ${DEMO_DIR}/demo_imipq_gpu_sift_m_mpi 128 15560 8 8 ${BIGANN_DIR}/bigann_learn.bvecs 32381696 ${BIGANN_DIR}/bigann_base.bvecs 500000000 ${BIGANN_DIR}/bigann_query.bvecs 0 "" 11 12 6 7 6 7 0 ${N_THREADS} ${N_GPUS} ${USE_SHARD} 0 1 ${MAX_MEM} ${RESULT_DIR}/coarse/coarse_imipq15560_32M_m_replica_full.bin ${RESULT_DIR}/index/index_imipq15560_32M_m_replica_500M.bin ${SEARCH} 0 0 1 0 ${USE_PRECOMP} ${USE_IMI} ${USE_GPU} ${PRINT_GPU_MEM} 0 | tee ${RESULT_DIR}/outs/mpi_2_1gpu_imipq500M_15560_32M_m_27GB_1Mqueries_search1_noprof.txt
### PROFILER ON (Disabled)
# mpirun -np 2 ${PROF_SCRIPT_DIR}/nvprof_wrapper.sh ${RESULT_DIR}/outs/mpi_2_1gpu_imipq500M_15560_32M_m_27GB_1Mqueries_prof ${DEMO_DIR}/demo_imipq_gpu_sift_m_mpi 128 15560 8 8 ${BIGANN_DIR}/bigann_learn.bvecs 32381696 ${BIGANN_DIR}/bigann_base.bvecs 500000000 ${BIGANN_DIR}/bigann_query.bvecs 0 "" 11 12 6 7 6 7 0 ${N_THREADS} ${N_GPUS} ${USE_SHARD} 0 1 ${MAX_MEM} ${RESULT_DIR}/coarse/coarse_imipq15560_32M_m_replica_full.bin ${RESULT_DIR}/index/index_imipq15560_32M_m_replica_500M.bin ${SEARCH} 0 0 1 0 ${USE_PRECOMP} ${USE_IMI} ${USE_GPU} ${PRINT_GPU_MEM} 0 | tee ${RESULT_DIR}/outs/mpi_2_1gpu_imipq500M_15560_32M_m_27GB_1Mqueries_search1.txt
