#include <registerKernels.hpp>
#include <tuple>

void registerLinAlgKernels()
{
  occa::properties kernelInfo = platform->kernelInfo;

  const std::string oklDir = getenv("NEKRS_KERNEL_DIR") + std::string("/platform/linAlg/");
  const bool serial = platform->serial;

  const std::string extension = serial ? ".c" : ".okl";
  const std::vector<std::pair<std::string, bool>> allKernels{
      {"fill", false},
      {"pfill", false},
      {"vabs", false},
      {"add", false},
      {"scale", false},
      {"scaleMany", false},
      {"axpby", true},
      {"paxpby", true},
      {"axpbyMany", true},
      {"paxpbyMany", true},
      {"axpbyz", false},
      {"axpbyzMany", false},
      {"axmy", true},
      {"paxmy", false},
      {"axmyMany", true},
      {"axmyVector", true},
      {"axmyz", false},
      {"paxmyz", true},
      {"axmyzMany", false},
      {"paxmyzMany", false},
      {"ady", false},
      {"adyz", false},
      {"adyMany", false},
      {"padyMany", false},
      {"axdy", false},
      {"aydx", false},
      {"aydxMany", false},
      {"axdyz", false},
      {"sum", false},
      {"sumMany", false},
      {"min", false},
      {"max", false},
      {"amax", false},
      {"amaxMany", false},
      {"norm1", true},
      {"norm1Many", true},
      {"norm2", true},
      {"norm2Many", true},
      {"weightedNorm1", true},
      {"weightedNorm1Many", true},
      {"weightedSqrSum", true},
      {"innerProd", true},
      {"weightedInnerProd", true},
      {"weightedInnerProdMany", true},
      {"pweightedInnerProdMany", true},
      {"weightedInnerProdMulti", false},
      {"pweightedInnerProdMulti", false},
      {"weightedInnerProdMultiDevice", false},
      {"crossProduct", false},
      {"unitVector", false},
      {"entrywiseMag", false},
      {"linearCombination", false},
      {"relativeError", false},
      {"absoluteError", false},
      {"magSqrVector", false},
      {"magSqrSymTensor", false},
      {"magSqrSymTensorDiag", false},
      {"magSqrTensor", false},
      {"mask", false},
      {"pmask", false},
  };

  std::string kernelName;
  std::string fileName;
  bool nativeSerialImplementation;
  const std::string prefix = "linAlg::";

  for (auto &&nameAndSerialImpl : allKernels) {
    std::tie(kernelName, nativeSerialImplementation) = nameAndSerialImpl;
    const std::string extension = (serial && nativeSerialImplementation) ? ".c" : ".okl";
    const bool pfloatKernel = (kernelName.front() == 'p') ? true : false;

    if (pfloatKernel && (sizeof(dfloat) == sizeof(pfloat))) continue; 

    fileName = kernelName;
    occa::properties props = kernelInfo;
    if (pfloatKernel) {
      props["defines/dfloat"] = pfloatString;
      fileName.erase(0, 1);
    }

    platform->kernelRequests.add(prefix + kernelName, oklDir + fileName + extension, props);
  }

  {
    auto props = kernelInfo;
    props["defines/dfloat"] = hlongString;
    kernelName = "sum";
    fileName = oklDir + kernelName + ".okl";
    platform->kernelRequests.add(prefix + "hlong-" + kernelName, fileName, props);
  }
}
