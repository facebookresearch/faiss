#ifndef OMP_H
#define OMP_H

#ifdef __cplusplus
extern "C" {
#endif

typedef void * omp_lock_t;

inline int omp_get_thread_num() { return 0; }
inline int omp_get_max_threads() { return 1; }
inline int omp_get_num_threads() { return 1; }
inline void omp_set_num_threads(int) {}
inline void omp_init_lock(omp_lock_t *) {}
inline void omp_destroy_lock(omp_lock_t *) {}
inline void omp_set_lock(omp_lock_t *) {}
inline void omp_unset_lock(omp_lock_t *) {}
inline int omp_test_lock(omp_lock_t *) { return 1; }
inline int omp_get_nested() { return 0; }
inline void omp_set_nested(int) {}
inline int omp_in_parallel() { return 0; }

#ifdef __cplusplus
}
#endif

#endif
