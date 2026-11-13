#!/usr/bin/env bash
set -euo pipefail

# Slurm runs a copied script from its spool directory; retain the checkout path.
repo_root=${EMEANS_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
script_path="$repo_root/benchs/e_means/launchers/launch_dino10b_k16m.sh"
result_dir=${EMEANS_RESULT_DIR:-/checkpoint/umami/matthijs/faiss/e_means/dino10b_k16m}

common_args=(
    --n 10_000_000_000 --k $((1 << 24))
    --ivf --nlist 4096 --nprobe 4
    --seed 0 --final-closing-mean
    --batch-size $((1 << 21)) --ssd-ram-gib 120
)

output_complete() {
    [[ -f "$1" ]] && grep -q '^PASS rank=0:' "$1"
}

copy_data() {
    local data=${1:?data directory is required}
    local source_data=/checkpoint/amaia/codegen/swann/datasets/dino_vitl_10B
    mkdir -p "$data/chunked_base_10B"
    echo "Copying DINO10B to $data on $(hostname)"
    cp "$source_data/queries_clean.bvecs" "$data/"
    for chunk in {0000..0049}; do
        cp "$source_data/chunked_base_10B/chunk_$chunk.bvecs" \
            "$data/chunked_base_10B/" &
    done
    wait
    echo "DINO copy complete on $(hostname)"
}

if [[ ${1:-} == "--copy-data" ]]; then
    copy_data "${2:?scratch directory is required}/DINO10B"
    exit 0
fi

if [[ ${1:-} == "--train-node" ]]; then
    scratch_root=${2:?scratch directory is required}
    master_addr=${3:?master address is required}
    master_port=${4:?master port is required}
    shift 4
    export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
    unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
    cd "$repo_root/benchs/e_means"
    torchrun --nnodes="$SLURM_JOB_NUM_NODES" --nproc-per-node=8 \
        --node-rank="$SLURM_PROCID" \
        --master-addr="$master_addr" --master-port="$master_port" \
        demo_training_from_ssd.py \
        "${common_args[@]}" --data "$scratch_root/DINO10B" "$@"
    exit 0
fi

if [[ ${1:-} == "--worker" ]]; then
    name=${2:?job name is required}
    log_file=${3:?log file is required}
    shift 3
    if output_complete "$log_file"; then
        echo "complete: $log_file"
        exit 0
    fi

    if [[ ${SLURM_JOB_NUM_NODES:-1} -gt 1 ]]; then
        # Use the same unique path on each host's local scratch filesystem.
        # The copy step must finish on every host before any torchrun starts.
        scratch_root=$(mktemp -d "/scratch/${RANDOM}.${SLURM_JOB_ID}.XXXXXX")
        trap 'srun --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 --cpu-bind=none --gpu-bind=none rm -rf -- "$scratch_root"' EXIT
        master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '1p')
        master_port=$((20000 + SLURM_JOB_ID % 20000))
        srun --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 \
            --cpu-bind=none --gpu-bind=none --kill-on-bad-exit=1 \
            bash "$script_path" --copy-data "$scratch_root"
        srun --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 \
            --cpu-bind=none --gpu-bind=none --kill-on-bad-exit=1 \
            bash "$script_path" --train-node \
            "$scratch_root" "$master_addr" "$master_port" "$@"
        exit 0
    fi

    # Run the copies on the allocated host, before launching any GPU workers.
    # RANDOM separates jobs; mktemp also makes creation collision-safe.
    scratch_root=$(mktemp -d "/scratch/${RANDOM}.XXXXXX")
    trap 'rm -rf -- "$scratch_root"' EXIT
    data="$scratch_root/DINO10B"
    copy_data "$data"

    export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
    unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
    cd "$repo_root/benchs/e_means"
    srun --ntasks=1 --cpu-bind=none --gpu-bind=none --kill-on-bad-exit=1 \
        python demo_training_from_ssd.py \
        --launch 8 "${common_args[@]}" --data "$data" "$@"
    exit 0
fi

cluster_name=${EMEANS_SLURM_CLUSTER_NAME:-${SLURM_CLUSTER_NAME:-}}
if [[ -z "$cluster_name" ]]; then
    cluster_name=$(scontrol show config | sed -n \
        's/^[[:space:]]*ClusterName[[:space:]]*=[[:space:]]*//p')
fi
case "$cluster_name" in
    fair-cw-use1-1)
        # One launcher task spawns eight GPU workers; each uses 120 GiB RAM
        # for prefetch, with room for initialization and other host buffers.
        slurm_layout=(
            --qos=h100_mdf_explore --account=umami
            --ntasks-per-node=1 --gpus-per-node=8
            --cpus-per-task=128 --mem=1600G
        )
        ;;
    *) echo "unsupported cluster: $cluster_name" >&2; exit 1 ;;
esac

launch_job() {
    local name=${1:?job name is required}
    local nodes=${2:?node count is required}
    shift 2
    local time_limit=12:00:00
    if [[ $nodes -gt 1 ]]; then
        time_limit=24:00:00
    fi
    local log_file="$result_dir/$name.out"
    if [[ -n ${EMEANS_ONLY:-} ]] && ! [[ $name =~ $EMEANS_ONLY ]]; then
        return
    fi
    if output_complete "$log_file"; then
        echo "complete: $name"
        return
    fi
    if [[ -n $(squeue --noheader --name="$name" \
        --states=PENDING,RUNNING --format='%i') ]]; then
        echo "active: $name"
        return
    fi
    mkdir -p "$result_dir"
    local job_id
    job_id=$(sbatch --parsable \
        "${slurm_layout[@]}" --nodes="$nodes" --time="$time_limit" \
        --job-name="$name" --chdir="$repo_root" \
        --output="$log_file" --open-mode=truncate \
        --export="ALL,EMEANS_REPO_ROOT=$repo_root,PYTHONUNBUFFERED=1" \
        "$script_path" --worker "$name" "$log_file" "$@")
    echo "submitted: name=$name job=$job_id log=$log_file"
}

prefix=dino10B_k16M
for tenths in {10..30..2} 15 25 ; do
    epochs=$(printf '%d.%d' "$((tenths / 10))" "$((tenths % 10))")
    launch_job "${prefix}.ivf4096.np4.seed0.ep$epochs.8gpu_1node" 1 \
        --epochs "$epochs"
done

for epochs in 2.0 2.5 3.0; do
    launch_job "${prefix}.exact.seed0.ep$epochs.32gpu_4nodes" 4 \
        --exact --epochs "$epochs"
done
