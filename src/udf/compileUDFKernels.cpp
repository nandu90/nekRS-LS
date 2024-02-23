#include "nekrsSys.hpp"
#include "compileKernels.hpp"
#include "udf.hpp"

occa::properties compileUDFKernels()
{
  const bool buildNodeLocal = platform->cacheLocal;

  std::string installDir;
  installDir.assign(getenv("NEKRS_HOME"));
  int N;
  platform->options.getArgs("POLYNOMIAL DEGREE", N);
  deviceKernelProperties kernelInfo(platform->kernelInfo + meshKernelProperties(N));

  MPI_Barrier(platform->comm.mpiComm);
  const double tStart = MPI_Wtime();
  if (platform->comm.mpiRank == 0) {
    std::cout << "loading udf kernels ... " << std::endl;
  }

  const std::string bcDataFile = installDir + "/include/nrs/bdry/bcData.h";
  kernelInfo.include() += bcDataFile.c_str();
  kernelInfo.okl_include_paths() += std::string(fs::current_path()).c_str();

  if (udf.loadKernels) {
    udf.loadKernels(kernelInfo);
    // kernelInfoBC might now may include user-defined props
  }

  if (udf.autoloadKernels) {
    udf.autoloadKernels(kernelInfo);
  }

  // internal BC kernels call device functions defined in this file
  std::string oklFileCache;
  platform->options.getArgs("OKL FILE CACHE", oklFileCache);
  kernelInfo.include() += realpath(oklFileCache.c_str(), NULL);

  udf.autoloadPlugins(kernelInfo);

  // just to bail out early in case included source doesn't compile
  {
    const std::string dummyKernelName = "compileUDFKernelsTest";
    const std::string dummyKernelStr =
        std::string("@kernel void compileUDFKernelsTest(int N) {"
                    "  for (int i = 0; i < N; ++i; @tile(64, @outer, @inner)) {}"
                    "}");

    if (platform->comm.mpiRank == 0) {
      platform->device.occaDevice().buildKernelFromString(dummyKernelStr, dummyKernelName, kernelInfo);
    }
  }

  MPI_Barrier(platform->comm.mpiComm);
  const double loadTime = MPI_Wtime() - tStart;
  if (platform->comm.mpiRank == 0) {
    printf("done (%gs)\n", loadTime);
  }
  fflush(stdout);

  return static_cast<occa::properties>(kernelInfo);
}
