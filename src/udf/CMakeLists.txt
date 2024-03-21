project(libUDF LANGUAGES CXX)
cmake_minimum_required(VERSION 3.11)

set_property(GLOBAL PROPERTY RULE_MESSAGES OFF)

set (CMAKE_CXX_STANDARD 17)

set(NEKRS_USE_DFLOAT_FLOAT OFF CACHE BOOL "use dfloat = float")

set(MPI_CXX_SKIP_MPICXX TRUE)
find_package(MPI REQUIRED)

set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")

#set(CMAKE_VERBOSE_MAKEFILE on)

set(GSDIR ${NEKRS_INSTALL_DIR}/gslib)
set(FINDPTSDIR ${NEKRS_INSTALL_DIR}/pointInterpolation/findpts)

set(INCLUDE_DIRS
  ${CASE_DIR}
  ${FINDPTSDIR}
  ${GSDIR}
  ${NEKRS_INSTALL_DIR}
  ${NEKRS_INSTALL_DIR}/include
  ${NEKRS_INSTALL_DIR}/include/mesh
  ${NEKRS_INSTALL_DIR}/include/core
  ${NEKRS_INSTALL_DIR}/include/core/io
  ${NEKRS_INSTALL_DIR}/include/core/ogs
  ${NEKRS_INSTALL_DIR}/include/core/linAlg
  ${NEKRS_INSTALL_DIR}/include/utils
  ${NEKRS_INSTALL_DIR}/include/udf
  ${NEKRS_INSTALL_DIR}/include/plugins
  ${NEKRS_INSTALL_DIR}/include/nrs
  ${NEKRS_INSTALL_DIR}/include/nrs/io
  ${NEKRS_INSTALL_DIR}/include/nrs/plugins
  ${NEKRS_INSTALL_DIR}/include/nrs/postProcessing
  ${NEKRS_INSTALL_DIR}/include/nrs/nekInterface
  ${NEKRS_INSTALL_DIR}/include/nrs/neknek
  ${NEKRS_INSTALL_DIR}/include/nrs/bdry
  ${NEKRS_INSTALL_DIR}/include/setup
  ${NEKRS_INSTALL_DIR}/include/elliptic
  ${NEKRS_INSTALL_DIR}/include/nrs/cds
  ${NEKRS_INSTALL_DIR}/include/nrs/cds/cvode
  ${NEKRS_INSTALL_DIR}/include/nrs/cds/regularization
  ${NEKRS_INSTALL_DIR}/include/pointInterpolation
  ${NEKRS_INSTALL_DIR}/include/pointInterpolation/findpts
  ${NEKRS_INSTALL_DIR}/occa/include
)

# dummy
add_library(OKL EXCLUDE_FROM_ALL okl.cpp)
target_include_directories(OKL PRIVATE $ENV{NEKRS_UDF_INCLUDES} ${INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(OKL PRIVATE -D__okl__)

add_library(udf SHARED udf.cpp)
target_link_libraries(udf PRIVATE MPI::MPI_CXX)
target_include_directories(udf PRIVATE $ENV{NEKRS_UDF_INCLUDES} ${INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})

if(cpptrace_FOUND)
  find_package(cpptrace QUIET PATHS ${NEKRS_INSTALL_DIR})
  target_link_libraries(udf PRIVATE cpptrace::cpptrace)
endif()

if(NEKRS_USE_DFLOAT_FLOAT)
  target_compile_definitions(udf PUBLIC NEKRS_USE_DFLOAT_FLOAT)
endif()

if(APPLE)
target_link_options(udf PRIVATE "LINKER:-undefined,dynamic_lookup")
endif()
target_link_options(udf PRIVATE "$ENV{NEKRS_UDF_LDFLAGS}")

if(EXISTS "udf.cmake") 
    include(udf.cmake)
endif()
