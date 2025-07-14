# Toolchain file for building Faiss with Metal support on macOS

set(CMAKE_SYSTEM_NAME Darwin)

# Set the C and CXX compilers to the ones from Homebrew's LLVM
set(CMAKE_C_COMPILER "/opt/homebrew/opt/llvm/bin/clang")
set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++")

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable Metal support
set(FAISS_ENABLE_METAL ON CACHE BOOL "Enable Metal support" FORCE)

# Disable GPU support (CUDA/ROCm)
set(FAISS_ENABLE_GPU OFF CACHE BOOL "Disable CUDA/ROCm support" FORCE)

set(CMAKE_METAL_COMPILER "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/metal")
set(CMAKE_METAL_ARCHIVER "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/metal-ar")
