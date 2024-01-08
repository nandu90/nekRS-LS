include(config/bench.cmake)
include(config/mesh.cmake)
include(config/elliptic.cmake)
include(config/gslib.cmake)

set(NRS_SRC 
    src/lib/nekrs.cpp
    src/core/io/writeFld.cpp
    src/utils/fileUtils.cpp
    src/utils/sha1.cpp
    src/utils/inipp.cpp
    src/utils/unifdef.c
    src/utils/mysort.cpp
    src/utils/parallelSort.cpp
    src/utils/tinyexpr.c
    src/utils/setupAide.cpp
    src/core/printHeader.cpp
    src/nrs/regularization/lowPassFilter.cpp
    src/nrs/regularization/avm.cpp
    src/nrs/bdry/bcMap.cpp
    src/core/compileKernels.cpp
    src/nrs/bdry/alignment.cpp
    src/neknek/registerNekNekKernels.cpp
    src/nrs/postProcessing/registerPostProcessingKernels.cpp
    src/solvers/elliptic/registerEllipticKernels.cpp
    src/solvers/elliptic/registerEllipticPreconditionerKernels.cpp
    src/nrs/cds/registerCdsKernels.cpp
    src/linAlg/registerLinAlgKernels.cpp
    src/mesh/registerMeshKernels.cpp
    src/core/LVector.cpp
    src/nrs/bdry/createEToBV.cpp
    src/nrs/registerNrsKernels.cpp
    src/nrs/numberActiveFields.cpp
    src/nrs/cfl.cpp
    src/nrs/nrs.cpp
    src/nrs/applyDirichlet.cpp
    src/nrs/timeStepper.cpp
    src/nrs/evaluateProperties.cpp
    src/nrs/subCycling.cpp
    src/nrs/tombo.cpp
    src/nrs/constantFlowRate.cpp
    src/nrs/Urst.cpp
    src/nrs/cds/cds.cpp
    src/nrs/cds/cdsSolve.cpp
    src/core/io/parsePar.cpp
    src/core/io/re2Reader.cpp
    src/core/io/configReader.cpp
    src/core/timer.cpp
    src/core/platform.cpp
    src/core/comm.cpp
    src/core/flopCounter.cpp
    src/core/kernelRequestManager.cpp
    src/core/device.cpp
    src/linAlg/linAlg.cpp
    src/linAlg/matrixConditionNumber.cpp
    src/linAlg/matrixInverse.cpp
    src/linAlg/matrixEig.cpp
    src/linAlg/matrixTranspose.cpp
    src/linAlg/matrixRightSolve.cpp
    src/plugins/tavg.cpp
    src/nrs/plugins/velRecycling.cpp
    src//nrs/plugins/RANSktau.cpp
    src/nrs/plugins/lowMach.cpp
    src/nrs/plugins/lpm.cpp
    src/pointInterpolation/findpts/findpts.cpp
    src/pointInterpolation/pointInterpolation.cpp
    src/neknek/neknek.cpp
    src/neknek/fixCoupledSurfaceFlux.cpp
    src/udf/udf.cpp
    src/udf/compileUDFKernels.cpp
    src/nekInterface/nekInterfaceAdapter.cpp
    src/nrs/postProcessing/strainRotationRate.cpp
    src/nrs/postProcessing/viscousDrag.cpp
    src/nrs/postProcessing/Qcriterion.cpp
    src/solvers/cvode/registerCvodeKernels.cpp
    src/solvers/cvode/cvode.cpp
    src/solvers/cvode/cbGMRES.cpp
    ${BENCH_SOURCES}
    ${MESH_SOURCES}
    ${ELLIPTIC_SOURCES}
    ${OGS_SOURCES}
    ${FINDPTS_SOURCES}
)

set(NRS_INCLUDE
    src
    src/nrs/bdry
    src/core
    src/utils
    src/lib
    src/core/io
    src/udf
    src/nrs/regularization
    src/linAlg
    src/nrs
    src/nrs/plugins
    src/plugins
    src/neknek
    src/nrs/cds
    src/pointInterpolation/findpts
    src/pointInterpolation
    src/solvers/cvode
    ${BENCH_SOURCE_DIR}
    ${BENCH_SOURCE_DIR}/core
    ${BENCH_SOURCE_DIR}/fdm
    ${BENCH_SOURCE_DIR}/axHelm
    ${BENCH_SOURCE_DIR}/advsub
    ${MESH_SOURCE_DIR}
    ${NEKINTERFACEDIR}
    ${OGS_SOURCE_DIR}/include
    ${OGS_SOURCE_DIR}
    ${FINDPTS_SOURCE_DIR}
    ${ELLIPTIC_SOURCE_DIR}
    PRIVATE
    ${ELLIPTIC_SOURCE_DIR}/amgSolver/hypre
    ${ELLIPTIC_SOURCE_DIR}/amgSolver/amgx
    ${ELLIPTIC_SOURCE_DIR}/MG
)

set_property(
   SOURCE src/core/printHeader.cpp 
   APPEND PROPERTY COMPILE_DEFINITIONS
   GITCOMMITHASH="${GIT_COMMIT_HASH}"
   NEKRS_VERSION=${PROJECT_VERSION_MAJOR}
   NEKRS_SUBVERSION=${PROJECT_VERSION_MINOR}
   NEKRS_PATCHVERSION=${PROJECT_VERSION_PATCH}
)

add_library(nekrs-lib SHARED ${NRS_SRC})
if (NEKRS_BUILD_FLOAT)
  add_library(nekrs-lib-fp32 SHARED ${NRS_SRC})
endif()

set_target_properties(nekrs-lib PROPERTIES LINKER_LANGUAGE CXX OUTPUT_NAME nekrs)
if (NEKRS_BUILD_FLOAT)
  set_target_properties(nekrs-lib-fp32 PROPERTIES LINKER_LANGUAGE CXX OUTPUT_NAME nekrs-fp32)
endif()

target_include_directories(nekrs-lib PUBLIC ${CMAKE_CURRENT_BINARY_DIR} ${NRS_INCLUDE}) 
if (NEKRS_BUILD_FLOAT)
  target_include_directories(nekrs-lib-fp32 PUBLIC ${CMAKE_CURRENT_BINARY_DIR} ${NRS_INCLUDE}) 
endif()

if (NEKRS_BUILD_FLOAT)
  target_compile_definitions(nekrs-lib-fp32 PUBLIC -DNEKRS_USE_DFLOAT_FLOAT)
  target_compile_definitions(nekrs-lib-fp32 PUBLIC -DOGS_USE_DFLOAT_FLOAT)
endif()

add_executable(nekrs-bin src/main.cpp)
if (NEKRS_BUILD_FLOAT)
  add_executable(nekrs-bin-fp32 src/main.cpp)
endif()

target_include_directories(nekrs-bin PRIVATE src/lib src/utils)
set_target_properties(nekrs-bin PROPERTIES LINKER_LANGUAGE CXX OUTPUT_NAME nekrs)
if (NEKRS_BUILD_FLOAT)
  target_include_directories(nekrs-bin-fp32 PRIVATE src/lib src/utils)
  set_target_properties(nekrs-bin-fp32 PROPERTIES LINKER_LANGUAGE CXX OUTPUT_NAME nekrs-fp32)
endif()
