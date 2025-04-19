#!/bin/bash

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
export USE_PROF
export BASE_SIZE
export NUM_DIMENSIONS
export RANDOM_DIR
export CENTROID_LIST
export USE_NVPROF
export COPY_PER_SHARD

N_GPUS_LIST="1"
SEARCH_LIST="0 1"
USE_PROF_LIST="1"

# SHARED MEM
N_INIT=7
N_END=8
K_INIT=7
K_END=8

for N_GPUS in ${N_GPUS_LIST};
do
    for N_CENTROIDS in ${CENTROID_LIST};
    do
        SEARCH=0
        ${RANDOM_DIR}/random_multi_gpu_wrapper.sh

        SEARCH=1
        for USE_PROF in ${USE_PROF_LIST};
        do
            ${RANDOM_DIR}/random_multi_gpu_wrapper.sh
        done
    done
done

# LOCAL MEM
N_INIT=6
N_END=7
K_INIT=7
K_END=8

for N_GPUS in ${N_GPUS_LIST};
do
    for N_CENTROIDS in ${CENTROID_LIST};
    do
        SEARCH=0
        ${RANDOM_DIR}/random_multi_gpu_wrapper.sh

        SEARCH=1
        for USE_PROF in ${USE_PROF_LIST};
        do
            ${RANDOM_DIR}/random_multi_gpu_wrapper.sh
        done
    done
done
