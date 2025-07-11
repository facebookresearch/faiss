# Toolchain file for building Faiss with Metal support on macOS

set(CMAKE_SYSTEM_NAME Darwin)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable Metal support
set(FAISS_ENABLE_METAL ON CACHE BOOL "Enable Metal support" FORCE)

# Disable GPU support (CUDA/ROCm)
set(FAISS_ENABLE_GPU OFF CACHE BOOL "Disable CUDA/ROCm support" FORCE)
