#include "platform.hpp"
#include "compileKernels.hpp"
#include "mesh.h"

void registerMeshKernels(occa::properties kernelInfoBC)
{
  int p, pCub = 0;
  platform->options.getArgs("POLYNOMIAL DEGREE", p);
  platform->options.getArgs("CUBATURE POLYNOMIAL DEGREE", pCub);

  std::vector<int> Nlist = {p};
  if (p != 2) {
    Nlist.push_back(2);
  }

  for (auto &N : Nlist) {
    const int Nq = N + 1;
    const int cubNq = (N == p) ? pCub + 1 : 1;
    const int Np = Nq * Nq * Nq;
    const int cubNp = cubNq * cubNq * cubNq;

    const std::string meshPrefix = "mesh-";
    const std::string orderSuffix = "_" + std::to_string(N);

    auto kernelInfo = platform->kernelInfo + meshKernelProperties(N);
    std::string oklpath = getenv("NEKRS_KERNEL_DIR");

    std::string fileName;
    std::string kernelName;

    {
      auto prop = kernelInfo;
      prop["defines/p_ndot"] = 0;
      kernelName = "surfaceAreaNormalMultiplyIntegrateHex3D";
      fileName = oklpath + "/mesh/" + kernelName + ".okl";
      platform->kernels.add(meshPrefix + kernelName, fileName, prop);

      prop["defines/p_ndot"] = 1;
      kernelName = "surfaceAreaNormalMultiplyIntegrateHex3D-ndot";
      platform->kernels.add(meshPrefix + kernelName, fileName, prop);
    }

    kernelName = "setBIDHex3D";
    fileName = oklpath + "/mesh/" + kernelName + ".okl";
    platform->kernels.add(meshPrefix + kernelName + orderSuffix, fileName, kernelInfo);

    kernelName = "distanceHex3D";
    fileName = oklpath + "/mesh/" + kernelName + ".okl";
    platform->kernels.add(meshPrefix + kernelName + orderSuffix, fileName, kernelInfo);

    auto hlongSumKernelInfo = kernelInfo;
    hlongSumKernelInfo["defines/dfloat"] = hlongString;
    kernelName = "sum";
    fileName = oklpath + "/core/linAlg/" + kernelName + ".okl";
    platform->kernels.add("hlong-" + meshPrefix + kernelName + orderSuffix, fileName, hlongSumKernelInfo);

    occa::properties meshKernelInfo = kernelInfo;
    meshKernelInfo["defines/p_cubNq"] = cubNq;
    meshKernelInfo["defines/p_cubNp"] = cubNp;

    kernelName = "geometricFactorsHex3D";
    fileName = oklpath + "/mesh/" + kernelName + ".okl";
    platform->kernels.add(meshPrefix + kernelName + orderSuffix, fileName, meshKernelInfo);

    kernelName = "cubatureGeometricFactorsHex3D";
    fileName = oklpath + "/mesh/" + kernelName + ".okl";
    platform->kernels.add(meshPrefix + kernelName + orderSuffix, fileName, meshKernelInfo);

    kernelName = "surfaceGeometricFactorsHex3D";
    fileName = oklpath + "/mesh/" + kernelName + ".okl";
    platform->kernels.add(meshPrefix + kernelName + orderSuffix, fileName, meshKernelInfo);

    if (N == p) {
      kernelName = "velocityDirichletBCHex3D";
      fileName = oklpath + "/mesh/" + kernelName + ".okl";
      platform->kernels.add(meshPrefix + kernelName, fileName, kernelInfoBC);

      meshKernelInfo = kernelInfo;
      int nAB = 3;
      platform->options.getArgs("MESH INTEGRATION ORDER", nAB);
      meshKernelInfo["defines/p_nAB"] = nAB;
      kernelName = "nStagesSumVector";
      fileName = oklpath + "/core/" + kernelName + ".okl";
      platform->kernels.add(meshPrefix + kernelName, fileName, meshKernelInfo);
    }

    for (const std::string dir : {"XY", "XZ", "YZ"}) {
      auto props = kernelInfo;
      props["includes"].asArray();
      props["includes"] += oklpath + "/mesh/planarAveraging.h";

      kernelName = "gatherPlanarValues" + dir;
      fileName = oklpath + "/mesh/" + kernelName + ".okl";
      platform->kernels.add(kernelName, fileName, props);

      kernelName = "scatterPlanarValues" + dir;
      fileName = oklpath + "/mesh/" + kernelName + ".okl";
      platform->kernels.add(kernelName, fileName, props);
    }
  }
}
