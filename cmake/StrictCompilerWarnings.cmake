# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Strict compiler warnings module for code quality enforcement
# This module provides functions to enable compiler warnings at various levels

# Warning levels:
#   0 = Disabled (default for normal builds)
#   1 = Basic warnings (Wall, Wextra) - Recommended minimum
#   2 = Standard warnings (adds Wpedantic, Wshadow, etc.)
#   3 = Strict warnings (all recommended warnings)
set(FAISS_WARNING_LEVEL "0" CACHE STRING "Warning level: 0=off, 1=basic, 2=standard, 3=strict")
set_property(CACHE FAISS_WARNING_LEVEL PROPERTY STRINGS "0" "1" "2" "3")

option(FAISS_WARNINGS_AS_ERRORS "Treat warnings as errors (-Werror)" OFF)

# Function to apply warnings to a target based on the warning level
function(faiss_add_warnings target)
    if(FAISS_WARNING_LEVEL EQUAL 0)
        return()
    endif()

    # Level 1: Basic warnings
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:
                -Wall
                -Wextra
            >
        )
    elseif(MSVC)
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/W3>
        )
    endif()

    # Level 2: Standard warnings
    if(FAISS_WARNING_LEVEL GREATER_EQUAL 2)
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(${target} PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:
                    -Wpedantic
                    -Wshadow
                    -Wnon-virtual-dtor
                    -Wunused
                    -Woverloaded-virtual
                    -Wformat=2
                    -Wimplicit-fallthrough
                >
            )
        elseif(MSVC)
            target_compile_options(${target} PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive->
            )
        endif()
    endif()

    # Level 3: Strict warnings
    if(FAISS_WARNING_LEVEL GREATER_EQUAL 3)
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
            target_compile_options(${target} PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:
                    -Wold-style-cast
                    -Wcast-align
                    -Wconversion
                    -Wsign-conversion
                    -Wnull-dereference
                    -Wdouble-promotion
                    -Wmisleading-indentation
                    -Wduplicated-cond
                    -Wduplicated-branches
                    -Wlogical-op
                    -Wuseless-cast
                >
            )
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            target_compile_options(${target} PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:
                    -Wold-style-cast
                    -Wcast-align
                    -Wconversion
                    -Wsign-conversion
                    -Wnull-dereference
                    -Wdouble-promotion
                >
            )
        endif()
    endif()

    # Warnings as errors
    if(FAISS_WARNINGS_AS_ERRORS)
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(${target} PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-Werror>
            )
        elseif(MSVC)
            target_compile_options(${target} PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:/WX>
            )
        endif()
    endif()
endfunction()

# Print warning configuration
function(faiss_print_warning_config)
    if(FAISS_WARNING_LEVEL GREATER 0)
        message(STATUS "FAISS Warning Level: ${FAISS_WARNING_LEVEL}")
        if(FAISS_WARNINGS_AS_ERRORS)
            message(STATUS "FAISS Warnings as Errors: ON")
        endif()
    endif()
endfunction()
