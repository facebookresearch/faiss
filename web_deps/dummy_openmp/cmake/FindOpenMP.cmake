# Mark the package as found
set(OpenMP_FOUND TRUE)
set(OpenMP_CXX_FOUND TRUE)
set(OpenMP_VERSION "Dummy")

if (NOT TARGET OpenMP::OpenMP_CXX)
    # Create dummy IMPORTED target
    add_library(OpenMP::OpenMP_CXX INTERFACE IMPORTED GLOBAL)
    
    # Suppressing unknown pragmas makes ignoring openmp pragmas
    set_target_properties(OpenMP::OpenMP_CXX PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/../include"
        INTERFACE_COMPILE_OPTIONS "-Wno-unknown-pragmas"
    )
endif()
