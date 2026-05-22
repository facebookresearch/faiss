# NVIDIA GPU SM Codes Reference

Source: [NVIDIA CUDA GPUs](https://developer.nvidia.com/cuda-gpus) (verified April 2026)

## CUDA Compute Capability → SM Code Mapping

| Product Name | CPU Arch | GPU Code | Compute Capability | SM Code | Notes |
|---|---|---|---|---|---|
| GeForce RTX 2060, RTX 2070, RTX 2080 | x86_64 | TU106/TU102 | 7.5 | `sm_75` | Turing |
| NVIDIA A100 | x86_64 | GA100 | 8.0 | `sm_80` | Ampere data center |
| GeForce RTX 3090, RTX 3080 | x86_64 | GA102 | 8.0 | `sm_80` | Ampere consumer |
| GeForce RTX 3080 Ti, RTX 3070 | x86_64 | GA102/GA104 | 8.6 | `sm_86` | Ampere consumer |
| GeForce RTX 4090, RTX 4080, RTX 4070 | x86_64 | AD102/AD103 | 8.9 | `sm_89` | Ada Lovelace |
| NVIDIA H100, H200 | x86_64 | GH100 | 9.0 | `sm_90` | Hopper |
| GB200, NVIDIA B200 | x86_64 | GB200 | 12.0 | `sm_120` | Blackwell data center |
| RTX PRO 6000/5000/4500/4000/2000 Blackwell | x86_64 | GB202 | 12.0 | `sm_120` | Blackwell workstation |
| GeForce RTX 5090, RTX 5080, RTX 5070 Ti | x86_64 | GB202 | 12.0 | `sm_120` | Blackwell consumer |
| GeForce RTX 5070, RTX 5060 Ti, RTX 5060, RTX 5050 | x86_64 | GB205/GB206 | 12.0 | `sm_120` | Blackwell consumer |
| NVIDIA DGX Spark | aarch64 | GB10 | 12.1 | `sm_121` | Grace Blackwell Superchip |

## FAISS Build Defaults

| Build Target | `CUDA_ARCHS` | SM Codes Compiled |
|---|---|---|
| Standard (x86_64) | `75;80;86;89;90;120` | sm_75 sm_80 sm_86 sm_89 sm_90 sm_120 |
| DGX Spark (aarch64) | `121-real` | sm_121 only |

## Notes

- **sm_75 (Turing)**: Supported in CUDA 13.2 for PTX compilation only; offline library support removed in CUDA 13.0.
- **sm_121 / DGX Spark**: The GB10 chip pairs a Grace CPU (aarch64/sbsa) with a Blackwell GPU die. Build with `build_wheel_spark.sh`; requires `libcuvs-spark.so` from [zbrad/cuvs](https://github.com/zbrad/cuvs).
- **sm_100**: Mentioned in CUDA 13.2 cuBLAS release notes alongside sm_103 but not attributed to any shipping product on the NVIDIA CUDA GPUs product page. Not included in FAISS build defaults.
