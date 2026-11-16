#!/bin/bash
# Submit from the checkout root with the Python environment activated:
# sbatch --partition=... --qos=... --account=... benchs/e_means/analysis/run_synthetic_cpu_comparison.slurm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --job-name=emeans_cpu128_comparison
set -euo pipefail
benchmark_dir=benchs/e_means/analysis
mkdir -p "$benchmark_dir/synthetic_ivf_single128_results" "$benchmark_dir/synthetic_ivf_distributed_results"
srun --exact --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=128 --cpu-bind=threads \
  env OMP_NUM_THREADS=128 MKL_NUM_THREADS=128 OPENBLAS_NUM_THREADS=128 PYTHONUNBUFFERED=1 \
  python "$benchmark_dir/bench_synthetic_ivf_distributed.py" \
  > "$benchmark_dir/synthetic_ivf_single128_results/run.log" 2>&1
srun --exact --nodes=1 --ntasks=4 --ntasks-per-node=4 --cpus-per-task=32 --cpu-bind=threads \
  env OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 PYTHONUNBUFFERED=1 \
  python "$benchmark_dir/bench_synthetic_ivf_distributed.py" \
  > "$benchmark_dir/synthetic_ivf_distributed_results/run.log" 2>&1
