// faiss/impl/PanoramaStats.h
#ifndef FAISS_PANORAMA_STATS_H
#define FAISS_PANORAMA_STATS_H

#include <faiss/impl/platform_macros.h>

namespace faiss {

/// Statistics are not robust to internal threading nor when
/// multiple Panorama searches are performed in parallel.
/// Use these values in a single-threaded context to accurately
/// gauge Panorama's pruning effectiveness.
struct PanoramaStats {
    uint64_t total_dims_scanned; // total dimensions scanned
    uint64_t total_dims;  	 // total dimensions
    float pct_dims_scanned;      // average percentage of dimensions scanned
    
    PanoramaStats() { reset(); }
    void reset();
    void add(const PanoramaStats& other);
};

// Single global var for all Panorama indexes
FAISS_API extern PanoramaStats indexPanorama_stats;

} // namespace faiss

#endif