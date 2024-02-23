set(HYPRE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/hypre)
set(HYPRE_FLAGS_EXTRA "-fPIC ${CMAKE_C_COMPILE_OPTIONS_VISIBILITY}hidden")

set(HYPRE_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/HYPRE_BUILD-prefix)
set(HYPRE_BUILD_DIR ${HYPRE_INSTALL_DIR}/src/HYPRE_BUILD)

ExternalProject_Add(
   HYPRE_BUILD
   URL "${HYPRE_SOURCE_DIR}" 
   CONFIGURE_COMMAND cd ${HYPRE_BUILD_DIR}/src && ./configure 
   --with-extra-CFLAGS=${HYPRE_FLAGS_EXTRA}
   --with-extra-CXXFLAGS=${HYPRE_FLAGS_EXTRA} 
   --disable-shared --enable-single --enable-mixedint --disable-fortran
   ${HYPRE_CONFIGURE_FLAGS}
   --prefix=${HYPRE_INSTALL_DIR}
   BUILD_COMMAND "" 
   INSTALL_COMMAND cd ${HYPRE_BUILD_DIR}/src && $(MAKE) install
)

add_library(nekrs-hypre SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/elliptic/amgSolver/hypre/hypreWrapper.cpp)
add_dependencies(nekrs-hypre HYPRE_BUILD)
target_include_directories(nekrs-hypre PRIVATE ${HYPRE_INSTALL_DIR}/include)
target_link_libraries(nekrs-hypre PUBLIC MPI::MPI_C 
                                  PRIVATE ${HYPRE_INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}HYPRE.a)
set_target_properties(nekrs-hypre PROPERTIES CXX_VISIBILITY_PRESET hidden)

if(ENABLE_HYPRE_GPU AND (OCCA_CUDA_ENABLED OR OCCA_HIP_ENABLED))

if(OCCA_CUDA_ENABLED)
  enable_language(CUDA)
  find_package(CUDAToolkit 11.0 REQUIRED)

  set(HYPRE_DEP CUDA::cudart)
  list(APPEND HYPRE_DEP CUDA::curand)
  list(APPEND HYPRE_DEP CUDA::cublas)
  list(APPEND HYPRE_DEP CUDA::cusparse)
  list(APPEND HYPRE_DEP CUDA::cusolver)

  set(HYPRE_CONFIGURE_FLAGS --with-cuda)
  if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.1.0")
    list(APPEND HYPRE_CONFIGURE_FLAGS "--with-gpu-arch='70 80'")
  endif()

  if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.0.0")
    list(APPEND HYPRE_CONFIGURE_FLAGS "--with-gpu-arch='70 80 90'")
    list(APPEND HYPRE_CONFIGURE_FLAGS --enable-device-malloc-async)
  endif()
else(OCCA_HIP_ENABLED)
  enable_language(HIP)
  find_package(rocrand REQUIRED)
  find_package(rocblas REQUIRED)
  find_package(rocsparse REQUIRED)
  find_package(rocsolver REQUIRED)

  set(HYPRE_DEP roc::rocrand)
  list(APPEND HYPRE_DEP roc::rocblas)
  list(APPEND HYPRE_DEP roc::rocsparse)
  list(APPEND HYPRE_DEP roc::rocsolver)

  set(HYPRE_CONFIGURE_FLAGS --with-hip)
endif()

if(NEKRS_GPU_MPI)
  list(APPEND HYPRE_CONFIGURE_FLAGS --enable-gpu-aware-mpi)
endif()

  set(HYPRE_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/HYPRE_BUILD_DEVICE-prefix)
  set(HYPRE_BUILD_DIR ${HYPRE_INSTALL_DIR}/src/HYPRE_BUILD_DEVICE)

  ExternalProject_Add(
   HYPRE_BUILD_DEVICE
   URL "${HYPRE_SOURCE_DIR}" 
   CONFIGURE_COMMAND cd ${HYPRE_BUILD_DIR}/src && ./configure
     CUCC=${CMAKE_CXX_COMPILER}
     --prefix=${HYPRE_INSTALL_DIR}
     --with-extra-CFLAGS=${HYPRE_FLAGS_EXTRA} 
     --with-extra-CXXFLAGS=${HYPRE_FLAGS_EXTRA} 
     --with-extra-CUFLAGS=${HYPRE_FLAGS_EXTRA}
     --disable-shared --enable-single --enable-mixedint --disable-fortran
     ${HYPRE_CONFIGURE_FLAGS}
   BUILD_COMMAND "" 
   INSTALL_COMMAND cd ${HYPRE_BUILD_DIR}/src && $(MAKE) install
  )

  add_library(nekrs-hypre-device SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/elliptic/amgSolver/hypre/hypreWrapperDevice.cpp)
  add_dependencies(nekrs-hypre-device HYPRE_BUILD_DEVICE)
  target_compile_definitions(nekrs-hypre-device PRIVATE -DENABLE_HYPRE_GPU)
  target_include_directories(nekrs-hypre-device PRIVATE ${HYPRE_INSTALL_DIR}/include)
  target_link_libraries(nekrs-hypre-device 
                        PUBLIC libocca MPI::MPI_C 
                        PRIVATE ${HYPRE_INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}HYPRE.a 
			${HYPRE_DEP}) 
  set_target_properties(nekrs-hypre-device PROPERTIES CXX_VISIBILITY_PRESET hidden)

else()
  #dummy
  message(WARNING "HYPRE device support disabled!")
  add_library(nekrs-hypre-device SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/elliptic/amgSolver/hypre/hypreWrapperDevice.cpp)
  target_link_libraries(nekrs-hypre-device PUBLIC libocca MPI::MPI_C) 
endif()

unset(HYPRE_BUILD_DIR)
unset(HYPRE_SOURCE_DIR)
unset(HYPRE_CONFIGURE_FLAGS) 
unset(HYPRE_INSTALL_DIR)
unset(HYPRE_FLAGS_EXTRA)
unset(HYPRE_DEP)
