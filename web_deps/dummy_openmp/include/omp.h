#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// ---- Thread & Execution Info ----
static inline int omp_get_num_threads(void) { return 1; }
static inline int omp_get_max_threads(void) { return 1; }
static inline int omp_get_thread_num(void) { return 0; }
static inline int omp_get_num_procs(void) { return 1; }
static inline int omp_in_parallel(void) { return 0; }
static inline void omp_set_num_threads(int n) { (void)n; }
static inline void omp_set_dynamic(int n) { (void)n; }
static inline int omp_get_dynamic(void) { return 0; }
static inline int omp_get_thread_limit(void) { return 1; }
static inline int omp_get_max_active_levels(void) { return 1; }
static inline int omp_get_level(void) { return 0; }
static inline int omp_get_ancestor_thread_num(int level) {
  (void)level;
  return 0;
}
static inline int omp_get_team_size(int level) {
  (void)level;
  return 1;
}
static inline int omp_get_active_level(void) { return 0; }

// ---- Nested parallelism ----
static inline int omp_get_nested(void) { return 0; }
static inline void omp_set_nested(int nested) { (void)nested; }

// ---- Timing ----
static inline double omp_get_wtime(void) { return 0.0; }
static inline double omp_get_wtick(void) { return 1e-6; }

// ---- Locking (simple) ----
typedef int omp_lock_t;
static inline void omp_init_lock(omp_lock_t *lock) { (void)lock; }
static inline void omp_destroy_lock(omp_lock_t *lock) { (void)lock; }
static inline void omp_set_lock(omp_lock_t *lock) { (void)lock; }
static inline void omp_unset_lock(omp_lock_t *lock) { (void)lock; }
static inline int omp_test_lock(omp_lock_t *lock) {
  (void)lock;
  return 1;
}

// ---- Locking (nested) ----
typedef int omp_nest_lock_t;
static inline void omp_init_nest_lock(omp_nest_lock_t *lock) { (void)lock; }
static inline void omp_destroy_nest_lock(omp_nest_lock_t *lock) { (void)lock; }
static inline void omp_set_nest_lock(omp_nest_lock_t *lock) { (void)lock; }
static inline void omp_unset_nest_lock(omp_nest_lock_t *lock) { (void)lock; }
static inline int omp_test_nest_lock(omp_nest_lock_t *lock) {
  (void)lock;
  return 1;
}

// ---- Affinity / Proc bind (optional, rarely used in Faiss) ----
static inline int omp_get_place_num_procs(int place_num) {
  (void)place_num;
  return 1;
}
static inline int omp_get_place_proc_ids(int place_num, int *ids) {
  (void)place_num;
  (void)ids;
  return 0;
}

#ifdef __cplusplus
}
#endif
