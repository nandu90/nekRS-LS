#include "bcMap.hpp"
#include "mesh.h"
#include <nrs.hpp>
#include <compileKernels.hpp>
#include <limits>

void registerNekNekKernels()
{
  dlong N;
  platform->options.getArgs("POLYNOMIAL DEGREE", N);

  const std::string oklpath = getenv("NEKRS_KERNEL_DIR");

  std::string kernelName = "copyNekNekPoints";
  std::string fileName = oklpath + "/nrs/neknek/" + kernelName + ".okl";
  platform->kernelRequests.add(kernelName, fileName, platform->kernelInfo);

  auto surfaceFluxKernelInfo = platform->kernelInfo;
  surfaceFluxKernelInfo += meshKernelProperties(N);
  bcMap::addKernelConstants(surfaceFluxKernelInfo);
  kernelName = "computeFlux";
  fileName = oklpath + "/nrs/neknek/" + kernelName + ".okl";
  platform->kernelRequests.add(kernelName, fileName, surfaceFluxKernelInfo);

  kernelName = "fixSurfaceFlux";
  fileName = oklpath + "/nrs/neknek/" + kernelName + ".okl";
  platform->kernelRequests.add(kernelName, fileName, surfaceFluxKernelInfo);
}
