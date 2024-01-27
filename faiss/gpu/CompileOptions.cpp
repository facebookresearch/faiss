#include <string>

namespace faiss {

// This is the one defined in utils.cpp
// Crossing fingers that the InitGpuCompileOptions_instance will
// be instanciated after this global variable
extern std::string gpu_compile_options;

struct InitGpuCompileOptions {
    InitGpuCompileOptions() {
        gpu_compile_options = "GPU ";
#ifdef USE_NVIDIA_RAFT
        gpu_compile_options += "NVIDIA_RAFT ";
#endif
    }
};

InitGpuCompileOptions InitGpuCompileOptions_instance;

} // namespace faiss
