list(APPEND srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/demo_ivfpq_indexing_gpu.cpp)

foreach(source ${srcs})
    get_filename_component(name ${source} NAME_WE)
    add_executable(${name} ${source})
    target_link_libraries(${name} ${faiss_lib_gpu} ${faiss_lib} ${CUDA_LINKER_LIBS})
endforeach(source)
