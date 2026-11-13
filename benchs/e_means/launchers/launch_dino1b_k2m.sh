#!/usr/bin/env bash
set -euo pipefail

# Slurm runs a copied script from its spool directory; retain the checkout path.
repo_root=${EMEANS_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
script_path="$repo_root/benchs/e_means/launchers/$(basename "${BASH_SOURCE[0]}")"
result_dir=${EMEANS_RESULT_DIR:-/checkpoint/umami/matthijs/faiss/e_means/dino1b_k2m}

common_args=(
    --n 1000_000_000 --k $((1 << 21))
    --nlist 1024 --nprobe 4
    --seed 1 --final-closing-mean
    --batch-size $((1 << 21)) --ssd-ram-gib 100
)

output_complete() {
    [[ -f "$1" ]] && grep -q '^PASS rank=0:' "$1"
}

if [[ ${1:-} == "--worker" ]]; then
    name=${2:?job name is required}
    log_file=${3:?log file is required}
    shift 3
    if output_complete "$log_file"; then
        echo "complete: $log_file"
        exit 0
    fi

    # Run the copies on the allocated host, before launching any GPU workers.
    # RANDOM separates jobs; mktemp also makes creation collision-safe.
    scratch_root=$(mktemp -d "/scratch/${RANDOM}.XXXXXX")
    trap 'rm -rf -- "$scratch_root"' EXIT
    data="$scratch_root/DINO10B"
    source_data=/checkpoint/amaia/codegen/swann/datasets/dino_vitl_10B
    mkdir -p "$data/chunked_base_10B"
    echo "Copying DINO1B to $data on $(hostname)"
    cp "$source_data/queries_clean.bvecs" "$data/"
    for chunk in {0000..0004}; do
        cp "$source_data/chunked_base_10B/chunk_$chunk.bvecs" \
            "$data/chunked_base_10B/" &
    done
    wait
    echo "DINO copy complete"

    export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
    unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
    cd "$repo_root/benchs/e_means"
    srun --ntasks=1 --cpu-bind=none --gpu-bind=none --kill-on-bad-exit=1 \
        python demo_training_from_ssd.py \
        "${common_args[@]}" --data "$data" "$@"
    exit 0
fi

cluster_name=${EMEANS_SLURM_CLUSTER_NAME:-${SLURM_CLUSTER_NAME:-}}
if [[ -z "$cluster_name" ]]; then
    cluster_name=$(scontrol show config | sed -n \
        's/^[[:space:]]*ClusterName[[:space:]]*=[[:space:]]*//p')
fi
case "$cluster_name" in
    fair-cw-use1-1)
        # One launcher task spawns GPU workers; each uses 100 GiB RAM
        # for prefetch, with room for initialization and other host buffers.
        slurm_layout=(
            --qos=h100_mdf_explore --account=umami
            --nodes=1 --ntasks-per-node=1
        )
        ;;
    *) echo "unsupported cluster: $cluster_name" >&2; exit 1 ;;
esac

launch_job() {
    local name=${1:?job name is required}
    local gpus=${2:?GPU count is required}
    shift 2
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
        "${slurm_layout[@]}" --gpus-per-node="$gpus" \
        --cpus-per-task="$((16 * gpus))" --mem="$((200 * gpus))G" \
        --time=12:00:00 \
        --job-name="$name" --chdir="$repo_root" \
        --output="$log_file" --open-mode=truncate \
        --export="ALL,EMEANS_REPO_ROOT=$repo_root,PYTHONUNBUFFERED=1" \
        "$script_path" --worker "$name" "$log_file" --launch "$gpus" "$@")
    echo "submitted: name=$name job=$job_id log=$log_file"
}

prefix=dino1B_k2M
for tenths in {10..30..2} 15 25 ; do
    epochs=$(printf '%d.%d' "$((tenths / 10))" "$((tenths % 10))")
    launch_job "${prefix}.ivf1024.np4.seed1.ep$epochs.4gpu_1node" 4 \
        --epochs "$epochs"
done

for epochs in 2.0 2.5 3.0; do
    launch_job "${prefix}.exact.seed1.ep$epochs.8gpu_1node" 8 \
        --exact --epochs "$epochs"
done

for epochs in 2.0 2.5 3.0; do
    launch_job "${prefix}.ivf4096.np16.seed1.ep$epochs.4gpu_1node" 4 \
        --nlist 4096 --nprobe 16 --epochs "$epochs"
done
