# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Finds libaio and defines the imported target LibAIO::LibAIO. Installed next
# to faiss-config.cmake, which finds libaio with it for a static Faiss.

find_path(LIBAIO_INCLUDE_DIR NAMES libaio.h)
find_library(LIBAIO_LIBRARY NAMES aio)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LibAIO
  REQUIRED_VARS LIBAIO_LIBRARY LIBAIO_INCLUDE_DIR)

if(LibAIO_FOUND AND NOT TARGET LibAIO::LibAIO)
  add_library(LibAIO::LibAIO UNKNOWN IMPORTED)
  set_target_properties(LibAIO::LibAIO PROPERTIES
    IMPORTED_LOCATION "${LIBAIO_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${LIBAIO_INCLUDE_DIR}")
endif()

mark_as_advanced(LIBAIO_INCLUDE_DIR LIBAIO_LIBRARY)
