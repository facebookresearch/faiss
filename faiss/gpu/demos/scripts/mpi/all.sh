#!/bin/bash

# IVFPQ CPU
mpirun -np 1 ${DEMO_DIR}/demo_imipq_gpu_sift_m_mpi 128 126491 8 8 ${BIGANN_DIR}/bigann_learn.bvecs 32381696 ${BIGANN_DIR}/bigann_base.bvecs 1000000000 ${BIGANN_DIR}/bigann_query.bvecs 0 ${BIGANN_DIR}/gnd/idx_1000M.ivecs 0 1 0 12 0 11 0 48 1 1 1 1 28991029248 ${RESULT_DIR}/coarse/coarse_ivfpq126491_32M_full.bin ${RESULT_DIR}/index/index_ivfpq126491_32M.bin 1 0 0 1 1 1 0 0 | tee ${RESULT_DIR}/outs/cpu_ivfpq1000M_126491_32M_m_27GB_numQueries1.txt

# IVFPQ GPU
mpirun -np 1 ${DEMO_DIR}/demo_imipq_gpu_sift_m_mpi 128 126491 8 8 ${BIGANN_DIR}/bigann_learn.bvecs 32381696 ${BIGANN_DIR}/bigann_base.bvecs 1000000000 ${BIGANN_DIR}/bigann_query.bvecs 0 ${BIGANN_DIR}/gnd/idx_1000M.ivecs 0 1 0 12 0 11 0 48 1 1 1 1 28991029248 ${RESULT_DIR}/coarse/coarse_ivfpq126491_32M_full.bin ${RESULT_DIR}/index/index_ivfpq126491_32M.bin 1 0 0 1 1 1 0 1 | tee ${RESULT_DIR}/outs/mp1_1_1gpu_ivfpq1000M_126491_32M_m_27GB_numQueries1.txt

# IMIPQ CPU
mpirun -np 1 ${DEMO_DIR}/demo_imipq_gpu_sift_m_mpi 128 15560 8 8 ${BIGANN_DIR}/bigann_learn.bvecs 32381696 ${BIGANN_DIR}/bigann_base.bvecs 1000000000 ${BIGANN_DIR}/bigann_query.bvecs 0 ${BIGANN_DIR}/gnd/idx_1000M.ivecs 0 1 0 12 0 11 0 48 1 1 1 1 28991029248 ${RESULT_DIR}/coarse/coarse_imipq15560_32M_m_replica_full.bin ${RESULT_DIR}/index/index_imipq15560_32M_m_replica.bin 1 0 0 1 1 1 1 0 | tee ${RESULT_DIR}/outs/cpu_imipq1000M_15560_32M_m_27GB_numQueries1.txt

# IMIPQ GPU
mpirun -np 1 ${DEMO_DIR}/demo_imipq_gpu_sift_m_mpi 128 15560 8 8 ${BIGANN_DIR}/bigann_learn.bvecs 32381696 ${BIGANN_DIR}/bigann_base.bvecs 1000000000 ${BIGANN_DIR}/bigann_query.bvecs 0 "" 3 4 6 7 6 7 0 48 1 1 1 1 28991029248 ${RESULT_DIR}/coarse/coarse_imipq15560_32M_m_replica_full.bin ${RESULT_DIR}/index/index_imipq15560_32M_m_replica.bin 1 0 0 1 0 1 1 1 | tee ${RESULT_DIR}/outs/mpi_1_1gpu_imipq1000M_15560_32M_m_27GB_1Mqueries.txt

# IMIPQ GPU multi-process
## Indexing
mpirun -np 2 ${DEMO_DIR}/demo_imipq_gpu_sift_m_mpi 128 15560 8 8 ${BIGANN_DIR}/bigann_learn.bvecs 32381696 ${BIGANN_DIR}/bigann_base.bvecs 500000000 ${BIGANN_DIR}/bigann_query.bvecs 0 "" 11 12 6 7 6 7 0 48 1 1 0 1 28991029248 ${RESULT_DIR}/coarse/coarse_imipq15560_32M_m_replica_full.bin ${RESULT_DIR}/index/index_imipq15560_32M_m_replica_500M.bin 0 0 0 1 0 1 1 1 | tee ${RESULT_DIR}/outs/mpi_2_1gpu_imipq500M_15560_32M_m_27GB_1Mqueries.txt
## Searching
mpirun -np 2 ${DEMO_DIR}/demo_imipq_gpu_sift_m_mpi 128 15560 8 8 ${BIGANN_DIR}/bigann_learn.bvecs 32381696 ${BIGANN_DIR}/bigann_base.bvecs 500000000 ${BIGANN_DIR}/bigann_query.bvecs 0 "" 11 12 6 7 6 7 0 48 1 1 0 1 28991029248 ${RESULT_DIR}/coarse/coarse_imipq15560_32M_m_replica_full.bin ${RESULT_DIR}/index/index_imipq15560_32M_m_replica_500M.bin 1 0 0 1 0 1 1 1 | tee ${RESULT_DIR}/outs/mpi_2_1gpu_imipq500M_15560_32M_m_27GB_1Mqueries.txt
