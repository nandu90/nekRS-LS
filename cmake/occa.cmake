function(add_occa)

set(OCCA_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/occa")
add_subdirectory(${OCCA_SOURCE_DIR} ${occa_content_BINARY_DIR})

message("-- OCCA_CUDA_ENABLED: ${OCCA_CUDA_ENABLED}")
message("-- OCCA_HIP_ENABLED: ${OCCA_HIP_ENABLED}")
message("-- OCCA_DPCPP_ENABLED: ${OCCA_DPCPP_ENABLED}")
message("-- OCCA_OPENCL_ENABLED: ${OCCA_OPENCL_ENABLED}")
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

endfunction()
