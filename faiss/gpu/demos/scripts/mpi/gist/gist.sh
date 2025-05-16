#!/bin/bash

FAISS_DIR=.
DEMO_DIR=${FAISS_DIR}/faiss/gpu/demos
RESULT_DIR=${FAISS_DIR}/demo_out
GIST_SCRIPT_DIR=${FAISS_DIR}/faiss/gpu/demos/scripts/mpi/gist

MAX_MEM=3221225472
MEM_STR=3GB
N_THREADS=8
USE_GPU=1
N_GPUS=1
USE_SHARD=1
PRINT_GPU_MEM=0

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

# IMIPQ
CENTROID_LIST="128 226 256 333 419 500 746 1266 2059"
USE_IMI=1
USE_PRECOMP=1
for N_CENTROIDS in ${CENTROID_LIST};
do
    for SEARCH in 0 1;
    do
        ${GIST_SCRIPT_DIR}/gist_wrapper.sh
    done
done

# IVFPQ
USE_IMI=0
CENTROID_LIST="250 500 1000 2000 4000"
for N_CENTROIDS in ${CENTROID_LIST};
do
    USE_PRECOMP=1
    SEARCH=0
    ${GIST_SCRIPT_DIR}/gist_wrapper.sh

    SEARCH=1
    for USE_PRECOMP in 1;
    do
        ${GIST_SCRIPT_DIR}/gist_wrapper.sh
    done
done
