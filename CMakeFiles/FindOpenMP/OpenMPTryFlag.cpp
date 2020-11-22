
#include <omp.h>
int main(void) {
#ifdef _OPENMP
  omp_get_max_threads();
  return 0;
#elif defined(__HIP_DEVICE_COMPILE__)
  return 0;
#else
  breaks_on_purpose
#endif
}
