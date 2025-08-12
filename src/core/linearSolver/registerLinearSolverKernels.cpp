#include "registerKernels.hpp"
#include "linearSolver.hpp"

namespace
{

void registerGMRESKernels(int Nfields)
{
  const std::string oklpath = getenv("NEKRS_KERNEL_DIR") + std::string("/core/linearSolver/");
  std::string fileName;
  const bool serial = platform->serial;

  const std::string fileNameExtension = (serial) ? ".c" : ".okl";
  const std::string sectionIdentifier = std::string("gmres::") + std::to_string(Nfields) + "-";

  occa::properties gmresKernelInfo = platform->kernelInfo;
  gmresKernelInfo["defines/p_Nfields"] = Nfields;

  std::string kernelName = "gramSchmidtOrthogonalization";
  fileName = oklpath + kernelName + fileNameExtension;
  platform->kernelRequests.add(sectionIdentifier + kernelName, fileName, gmresKernelInfo);

  kernelName = "updatePGMRESSolution";
  fileName = oklpath + kernelName + ".okl";
  platform->kernelRequests.add(sectionIdentifier + kernelName, fileName, gmresKernelInfo);

  kernelName = "PGMRESSolution";
  fileName = oklpath + kernelName + fileNameExtension;
  platform->kernelRequests.add(sectionIdentifier + kernelName, fileName, gmresKernelInfo);


  kernelName = "fusedResidualAndNorm";
  fileName = oklpath + kernelName + fileNameExtension;
  platform->kernelRequests.add(sectionIdentifier + kernelName, fileName, gmresKernelInfo);
}

void registerCGKernels(int Nfields, bool usePfloat = false)
{
  const std::string oklpath = getenv("NEKRS_KERNEL_DIR") + std::string("/core/linearSolver/");

  const bool serial = platform->serial;

  const std::string fileNameExtension = (serial) ? ".c" : ".okl";

  const auto sectionIdentifier = std::string("cg::") + (usePfloat ? std::string("pfloat") : std::string("")) +
                                 std::to_string(Nfields) + "-";

  std::string kernelName;
  std::string fileName;

  occa::properties properties = platform->kernelInfo;
  properties["defines/p_Nfields"] = Nfields;

  if (usePfloat) {
    properties["defines/dfloat"] = pfloatString;
  }

  kernelName = "blockUpdatePCG";
  fileName = oklpath + kernelName + fileNameExtension;
  platform->kernelRequests.add(sectionIdentifier + kernelName, fileName, properties);

  occa::properties combinedPCGInfo = properties;

  kernelName = "combinedPCGPreMatVec";
  fileName = oklpath + kernelName + fileNameExtension;
  platform->kernelRequests.add(sectionIdentifier + kernelName, fileName, combinedPCGInfo);

  kernelName = "combinedPCGUpdateConvergedSolution";
  fileName = oklpath + kernelName + fileNameExtension;
  platform->kernelRequests.add(sectionIdentifier + kernelName, fileName, combinedPCGInfo);

  combinedPCGInfo["defines/p_nReduction"] = CombinedPCGId::nReduction;
  combinedPCGInfo["defines/p_gamma"] = CombinedPCGId::gamma;
  combinedPCGInfo["defines/p_a"] = CombinedPCGId::a;
  combinedPCGInfo["defines/p_b"] = CombinedPCGId::b;
  combinedPCGInfo["defines/p_c"] = CombinedPCGId::c;
  combinedPCGInfo["defines/p_d"] = CombinedPCGId::d;
  combinedPCGInfo["defines/p_e"] = CombinedPCGId::e;
  combinedPCGInfo["defines/p_f"] = CombinedPCGId::f;

  kernelName = "combinedPCGPostMatVec";
  fileName = oklpath + kernelName + fileNameExtension;
  platform->kernelRequests.add(sectionIdentifier + kernelName, fileName, combinedPCGInfo);
}

} // namespace

void registerLinearSolverKernels()
{
  if (platform->comm.mpiRank() == 0 && platform->verbose()) {
    std::cout << "registerLinearSolverKernels" << std::endl;
  }

  registerGMRESKernels(1);
  registerGMRESKernels(3);
  registerCGKernels(1);
  registerCGKernels(3);

  if (!std::is_same<pfloat, dfloat>::value) {
    registerCGKernels(1, true); // Krylov based coarse solve
  }
}
