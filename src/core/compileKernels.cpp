#include "LVector.hpp"
#include "nekrsSys.hpp"
#include "compileKernels.hpp"
#include "bcMap.hpp"
#include "mesh.h"
#include "ogs.hpp"
#include "ogsKernels.hpp"
#include "udf.hpp"
#include <vector>
#include <tuple>
#include "findpts.hpp"
#include "fileUtils.hpp"
#include "LVector.hpp"

std::string createOptionsPrefix(std::string section)
{
  std::string prefix = section + std::string(" ");
  if (section.find("temperature") != std::string::npos) {
    prefix = std::string("scalar00 ");
  }
  std::transform(prefix.begin(), prefix.end(), prefix.begin(), [](unsigned char c) {
    return std::toupper(c);
  });
  return prefix;
}

void platform_t::compileKernels()
{

  MPI_Barrier(platform->comm.mpiComm);

  bcMap::addKernelConstants(platform->kernelInfo);

  occa::properties kernelInfoBC = compileUDFKernels(); // includes plugins

  const double tStart = MPI_Wtime();
  if (platform->comm.mpiRank == 0 && !platform->options.compareArgs("BUILD ONLY", "TRUE")) {
    std::cout << "benchmarking hot kernels ..." << std::endl;
  }

  // register what to compile
  registerCoreKernels();
  registerMeshKernels(kernelInfoBC);
  registerPointInterpolationKernels();
  if (platform->solver->id() == "nrs")
    registerNrsKernels(kernelInfoBC);

  if (platform->comm.mpiRank == 0) {
    printf("JIT compiling kernels (this may take awhile if they are not in cache) ...\n");
    fflush(stdout);
  }

  // perform actual compilation (if needed) and load
  platform->kernels.compile();
  {
    const bool buildNodeLocal = platform->cacheLocal;
    const bool buildOnly = platform->options.compareArgs("BUILD ONLY", "TRUE");
    auto communicator = buildNodeLocal ? platform->comm.mpiCommLocal : platform->comm.mpiComm;
    auto &plat = platform;
    ogsBuildKernel_t buildKernel =
        [plat](const std::string &fileName, const std::string &kernelName, const occa::properties &props) {
          return plat->device.buildKernel(fileName, kernelName, props);
        };
    oogs::compile(platform->device.occaDevice(),
                  buildKernel,
                  platform->device.mode(),
                  communicator,
                  buildOnly);
  }

  MPI_Barrier(platform->comm.mpiComm);
  const double loadTime = MPI_Wtime() - tStart;

  fflush(stdout);
  if (platform->comm.mpiRank == 0) {
    std::ofstream ofs;
    ofs.open(occa::env::OCCA_CACHE_DIR + "cache/compile.timestamp",
             std::ofstream::out | std::ofstream::trunc);
    ofs.close();
  }

  platform->timer.set("loadKernels", loadTime);
  if (platform->comm.mpiRank == 0) {
    printf("done (%gs)\n\n", loadTime);
  }
  fflush(stdout);

  // assign platform related kernels
  std::string kernelName;
  kernelName = "core-copyDfloatToPfloat";
  platform->copyDfloatToPfloatKernel = platform->kernels.get(kernelName);

  kernelName = "core-copyPfloatToDfloat";
  platform->copyPfloatToDfloatKernel = platform->kernels.get(kernelName);

  kernelName = "core-copyDfloatToDouble";
  platform->copyDfloatToDoubleKernel = platform->kernels.get(kernelName);

}
