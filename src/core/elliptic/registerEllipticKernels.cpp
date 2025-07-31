#include <registerKernels.hpp>
#include "elliptic.h"
#include "benchmarkAx.hpp"

void registerEllipticKernels(std::string section, bool stressForm)
{
  int N;
  platform->options.getArgs("POLYNOMIAL DEGREE", N);
  const std::string optionsPrefix = createOptionsPrefix(section);

  occa::properties kernelInfo = platform->kernelInfo;
  kernelInfo["defines"].asObject();
  kernelInfo["includes"].asArray();
  kernelInfo["header"].asArray();
  kernelInfo["flags"].asObject();
  kernelInfo["include_paths"].asArray();
  kernelInfo += meshKernelProperties(N);

  const int poisson = platform->options.compareArgs(optionsPrefix + "HELMHOLTZ TYPE", "POISSON");

  const bool blockSolver = [&]() {
    if (platform->options.compareArgs(optionsPrefix + "BLOCK SOLVER", "TRUE")) {
      return true;
    }
    if (stressForm) {
      return true;
    }

    return false;
  }();
  const int Nfields = (blockSolver) ? 3 : 1;

  const bool serial = platform->serial;
  const std::string fileNameExtension = (serial) ? ".c" : ".okl";
  const std::string sectionIdentifier = std::to_string(Nfields) + "-";
  const std::string suffix = "Hex3D";

  {
    const std::string oklpath = getenv("NEKRS_KERNEL_DIR") + std::string("/core/elliptic/");
    std::string fileName, kernelName;

    {
      const std::string extension = ".okl";
      occa::properties properties = platform->kernelInfo;

      kernelName = "fusedCopyDfloatToPfloat";
      fileName = oklpath + kernelName + fileNameExtension;
      platform->kernelRequests.add(kernelName, fileName, properties);

      properties["defines/p_Nfields"] = Nfields;

      kernelName = "multiScaledAddwOffset";
      fileName = oklpath + kernelName + extension;
      platform->kernelRequests.add(sectionIdentifier + kernelName, fileName, properties);

      kernelName = "accumulate";
      fileName = oklpath + kernelName + extension;
      platform->kernelRequests.add(sectionIdentifier + kernelName, fileName, properties);
    }
  }

  int nelgt, nelgv;
  const std::string meshFile = platform->options.getArgs("MESH FILE");
  re2::nelg(meshFile, nelgt, nelgv, platform->comm.mpiComm);
  const int NelemBenchmark = nelgv / platform->comm.mpiCommSize;
  bool verbose = platform->verbose();
  const int verbosity = verbose ? 2 : 1;

  for (auto &&coeffField : {true, false}) {
    if (platform->options.compareArgs(optionsPrefix + "ELLIPTIC COEFF FIELD", "TRUE") != coeffField) {
      continue;
    }

    auto axKernel = benchmarkAx(NelemBenchmark,
                                N + 1,
                                N,
                                !coeffField,
                                poisson,
                                false,
                                sizeof(dfloat),
                                Nfields,
                                stressForm,
                                verbosity,
                                targetTimeBenchmark,
                                platform->options.compareArgs("KERNEL AUTOTUNING", "FALSE") ? false : true);

    if (platform->options.compareArgs("BUILD ONLY", "FALSE")) {
      std::string kernelNamePrefix = (poisson) ? "poisson-" : "";
      kernelNamePrefix += "elliptic";
      if (blockSolver) {
        kernelNamePrefix += (stressForm) ? "Stress" : "Block";
      }

      std::string kernelName = "AxCoeff";
      if (platform->options.compareArgs("ELEMENT MAP", "TRILINEAR")) {
        kernelName += "Trilinear";
      }
      kernelName += suffix;

      platform->kernelRequests.add(kernelNamePrefix + "Partial" + kernelName, axKernel);
    }
  }
}
