#!/bin/bash
FAISS_DIR=.
DEMO_DIR=${FAISS_DIR}/build/faiss/gpu/demos
RESULT_DIR=${FAISS_DIR}/demo_out
GIST_SCRIPT_DIR=${FAISS_DIR}/faiss/gpu/demos/scripts/mpi/gist
PROF_SCRIPT_DIR=${FAISS_DIR}/faiss/gpu/demos/scripts/mpi/sift

N_THREADS=8
USE_GPU=1
USE_SHARD=1
PRINT_GPU_MEM=0
Q_INIT=10
Q_END=11
MAX_MEM=8589934592
MEM_STR=8GB

export DEMO_DIR
export RESULT_DIR
export MAX_MEM
export MEM_STR
export N_THREADS
export USE_GPU
export N_GPUS
export USE_SHARD
export USE_PRECOMP
export USE_IMI
export N_CENTROIDS
export SEARCH
export PRINT_GPU_MEM
export Q_INIT
export Q_END
export N_INIT
export N_END
export K_INIT
export K_END
export PROF_SCRIPT_DIR

# IMIPQ
CENTROID_LIST="500"
USE_IMI=1
USE_PRECOMP=1

N_GPUS_LIST="1 2"
SEARCH_LIST="0 1"

N_INIT=7
N_END=8
K_INIT=7
K_END=8

for N_GPUS in ${N_GPUS_LIST};
do
    for N_CENTROIDS in ${CENTROID_LIST};
    do
        for SEARCH in ${SEARCH_LIST};
        do
            ${GIST_SCRIPT_DIR}/gist_multi_gpu_wrapper.sh
        done
    done
done


N_INIT=6
N_END=7
K_INIT=6
K_END=7

for N_GPUS in ${N_GPUS_LIST};
do
    for N_CENTROIDS in ${CENTROID_LIST};
    do
        for SEARCH in ${SEARCH_LIST};
        do
            ${GIST_SCRIPT_DIR}/gist_multi_gpu_wrapper.sh
        done
    done
done