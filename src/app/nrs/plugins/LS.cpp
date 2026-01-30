#include "nrs.hpp"
#include "platform.hpp"
#include "nekInterfaceAdapter.hpp"
#include "LS.hpp"
#include "linAlg.hpp"

// private members
namespace
{
nrs_t *nrs;
occa::kernel signlsKernel;
occa::kernel normalVectorKernel;
bool buildKernelCalled = false;
};

void LS::buildKernel(occa::properties _kernelInfo)
{
  std::cout << "ENTERING LS::buildKernel()...\n";

  occa::properties kernelInfo;
  kernelInfo += _kernelInfo;

  auto buildKernel = [&kernelInfo](const std::string &kernelName) {
    const auto path = getenv("NEKRS_KERNEL_DIR") + std::string("/app/nrs/plugins/");
    const auto fileName = path + "LS.okl";
    const auto reqName = "LS::";
    if (platform->options.compareArgs("REGISTER ONLY", "TRUE")) {
      platform->kernelRequests.add(reqName, fileName, kernelInfo);
      return occa::kernel();
    } else {
      buildKernelCalled = 1;
      return platform->kernelRequests.load(reqName, kernelName);
    }
  };

  signlsKernel = buildKernel("signls");
  normalVectorKernel = buildKernel("normalVector");

  std::cout << "EXITING LS::buildKernel()...\n";
}

void LS::updateSourceTerms()
{
  std::cout << "ENTERING LS::updateSourceTerms()...\n";

  auto mesh = nrs->meshV;

  // compute interface normals - currently we store this in nrs->fluid->o_U
  launchKernel("core-gradientVolumeHex3D",
               mesh->Nelements,
               mesh->o_vgeo,
               mesh->o_D,
               nrs->fluid->fieldOffset,
               nrs->scalar->o_solution("phi"),
               nrs->fluid->o_U);
  oogs::startFinish(nrs->fluid->o_U, mesh->dim, nrs->fluid->fieldOffset, ogsDfloat, ogsAdd, mesh->oogs);
  platform->linAlg->axmyVector(mesh->Nlocal, nrs->fluid->fieldOffset, 0, 1.0, mesh->o_invLMM, nrs->fluid->o_U);

  // compute sign function
  auto o_signls = platform->deviceMemoryPool.reserve<dfloat>(nrs->fluid->fieldOffset);
  signlsKernel(mesh->Nlocal, nrs->scalar->o_solution("phi"), o_signls);

  normalVectorKernel(mesh->Nlocal, nrs->fluid->fieldOffset, o_signls, nrs->fluid->o_U);

  // update source term
  nrs->scalar->o_explicitTerms("phi").copyFrom(o_signls);

  std::cout << "EXITING LS::updateSourceTerms()...\n";
}

void LS::setup()
{
  std::cout << "ENTERING LS::setup()...\n";

  static bool isInitialized = false;
  if (isInitialized) {
    return;
  }
  isInitialized = true;

  nrs = dynamic_cast<nrs_t *>(platform->app);

  std::cout << "EXITING LS::setup()...\n";
}
