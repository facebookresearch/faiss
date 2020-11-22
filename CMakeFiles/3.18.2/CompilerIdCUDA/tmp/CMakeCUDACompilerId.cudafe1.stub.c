#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "CMakeCUDACompilerId.fatbin.c"
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
static void __nv_cudaEntityRegisterCallback(void **__T0){__nv_dummy_param_ref(__T0);__nv_save_fatbinhandle_for_managed_rt(__T0);}
static void __sti____cudaRegisterAll(void){__cudaRegisterBinary(__nv_cudaEntityRegisterCallback);}

#pragma GCC diagnostic pop
