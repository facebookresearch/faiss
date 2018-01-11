find_package(SWIG REQUIRED)
find_package(PythonLibs REQUIRED)
include(${SWIG_USE_FILE})

include_directories(${PYTHON_INCLUDE_PATH})
include_directories(${PROJECT_SOURCE_DIR})

set(SWIG_MODULE swig${PROJECT_NAME})

set_source_files_properties(${SWIG_MODULE}.i PROPERTIES CPLUSPLUS ON)
# Add swig module
swig_add_module(${SWIG_MODULE} python
    swigfaiss.i)
swig_link_libraries(${SWIG_MODULE} ${PROJECT_NAME} ${PYTHON_LIBRARIES})

# Files to install with Python
set(PYTHON_INSTALL_FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${SWIG_MODULE}.py
    ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/_${SWIG_MODULE}.so)

# Configure setup.py and copy to output directory
set(SETUP_PY_IN ${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in)
set(SETUP_PY_OUT ${CMAKE_CURRENT_BINARY_DIR}/setup.py)
configure_file(${SETUP_PY_IN} ${SETUP_PY_OUT})

add_custom_target(install-py
    DEPENDS _${SWIG_MODULE}
    COMMAND python ${SETUP_PY_OUT} install)
