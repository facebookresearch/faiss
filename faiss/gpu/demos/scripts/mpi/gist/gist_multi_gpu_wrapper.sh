#!/bin/bash
BASE_NAME=gist
GIST_DIR=$BASE_DIR/${BASE_NAME}
BASE_SIZE=1000000
VECS_TYPE=fvecs
IS_FLOAT=1
TRAIN_SIZE=$((100 * N_CENTROIDS))

if (( TRAIN_SIZE > 100000 )); then
    TRAIN_SIZE=100000
fi

echo "RUNNING gist_wrapper..."
echo "DEMO_DIR: $DEMO_DIR"
echo "RESULT_DIR: $RESULT_DIR"
echo "MAX_MEM: $MAX_MEM"
echo "MEM_STR: $MEM_STR"
echo "N_THREADS: $N_THREADS"
echo "USE_GPU: $USE_GPU"
echo "N_GPUS: $N_GPUS"
echo "USE_SHARD: $USE_SHARD"
echo "USE_PRECOMP: $USE_PRECOMP"
echo "BASE_NAME: $BASE_NAME"
echo "GIST_DIR: $GIST_DIR"
echo "BASE_SIZE: $BASE_SIZE"
echo "VECS_TYPE: $VECS_TYPE"
echo "IS_FLOAT: $IS_FLOAT"
echo "N_CENTROIDS: $N_CENTROIDS"
echo "TRAIN_SIZE: $TRAIN_SIZE"
echo "USE_IMI: $USE_IMI"
echo "SEARCH: $SEARCH"
echo "PRINT_GPU_MEM: $PRINT_GPU_MEM"
echo "Q_INIT: $Q_INIT"
echo "Q_END: $Q_END"
echo "N_INIT: $N_INIT"
echo "N_END: $N_END"
echo "K_INIT: $K_INIT"
echo "K_END: $K_END"
echo "PROF_SCRIPT_DIR: $PROF_SCRIPT_DIR"
echo "-----------------------------"

INDEX_POSFIX=${BASE_NAME}_imi${USE_IMI}_c${N_CENTROIDS}_t${TRAIN_SIZE}
OUT_NAME=${INDEX_POSFIX}_mem${MEM_STR}_gpu${USEGPU}n${N_GPUS}s${USE_SHARD}_ncpu${N_THREADS}_pre${USE_PRECOMP}_search${SEARCH}_q_${Q_INIT}_${Q_END}_np_${N_INIT}_${N_END}_k_${K_INIT}_${K_END}_prof

if [ $SEARCH = 0 ]; then
    mpirun -np ${N_GPUS} ${DEMO_DIR}/demo_imipq_gpu_sift_m_mpi 960 ${N_CENTROIDS} 8 8 ${GIST_DIR}/${BASE_NAME}_learn.${VECS_TYPE} ${TRAIN_SIZE} ${GIST_DIR}/${BASE_NAME}_base.${VECS_TYPE} ${BASE_SIZE} ${GIST_DIR}/${BASE_NAME}_base.${VECS_TYPE} 0 "" ${Q_INIT} ${Q_END} ${N_INIT} ${N_END} ${K_INIT} ${K_END} ${IS_FLOAT} ${N_THREADS} ${N_GPUS} ${USE_SHARD} 0 1 ${MAX_MEM} ${RESULT_DIR}/coarse/coarse_${INDEX_POSFIX}.bin ${RESULT_DIR}/index/index_${INDEX_POSFIX}.bin ${SEARCH} 0 0 1 0 ${USE_PRECOMP} ${USE_IMI} ${USE_GPU} ${PRINT_GPU_MEM} 0 | tee ${RESULT_DIR}/outs/${OUT_NAME}.txt
else
   mpirun -np ${N_GPUS} ${PROF_SCRIPT_DIR}/nvprof_wrapper.sh ${RESULT_DIR}/outs/${OUT_NAME} ${DEMO_DIR}/demo_imipq_gpu_sift_m_mpi 960 ${N_CENTROIDS} 8 8 ${GIST_DIR}/${BASE_NAME}_learn.${VECS_TYPE} ${TRAIN_SIZE} ${GIST_DIR}/${BASE_NAME}_base.${VECS_TYPE} ${BASE_SIZE} ${GIST_DIR}/${BASE_NAME}_base.${VECS_TYPE} 0 "" ${Q_INIT} ${Q_END} ${N_INIT} ${N_END} ${K_INIT} ${K_END} ${IS_FLOAT} ${N_THREADS} ${N_GPUS} ${USE_SHARD} 0 1 ${MAX_MEM} ${RESULT_DIR}/coarse/coarse_${INDEX_POSFIX}.bin ${RESULT_DIR}/index/index_${INDEX_POSFIX}.bin ${SEARCH} 0 0 1 0 ${USE_PRECOMP} ${USE_IMI} ${USE_GPU} ${PRINT_GPU_MEM} 0 | tee ${RESULT_DIR}/outs/${OUT_NAME}.txt
fi
