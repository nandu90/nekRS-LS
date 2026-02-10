#include "app.hpp"
#include "nrs.hpp"
#include "platform.hpp"
#include "LS.hpp"
#include "linAlg.hpp"
#include "solver.hpp"
#include <stdexcept>

// private members
namespace
{
nrs_t *nrs;
occa::kernel signlsKernel;
occa::kernel normalVectorKernel;
bool buildKernelCalled = false;
std::unique_ptr<ls_t> tlsr = nullptr;
occa::memory o_coeffEXT, o_coeffBDF;
int advectionSubcycingSteps = 0;
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
  if (!nrs || !nrs->meshV) {
    throw std::runtime_error("LS::setup: nrs or nrs->meshV is null (mesh not initialized)");
  }

  int nBDF;
  int nEXT;
  platform->options.getArgs("BDF ORDER", nBDF);
  platform->options.getArgs("EXT ORDER", nEXT);
  platform->options.getArgs("SUBCYCLING STEPS", advectionSubcycingSteps);
  if (advectionSubcycingSteps) {
    nEXT = nBDF;
    platform->options.setArgs("EXT ORDER", std::to_string(nEXT));
  }
  nekrsCheck(nEXT < nBDF,
             platform->comm.mpiComm(),
             EXIT_FAILURE,
             "%s\n",
             "EXT order needs to be >= BDF order!");
  o_coeffEXT = platform->device.malloc<dfloat>(nEXT);
  o_coeffBDF = platform->device.malloc<dfloat>(nBDF);

  dlong fieldOffset = nrs->meshV->fieldOffset;
  dlong cubatureOffset = -1;
  auto cubOffset = fieldOffset;
  if (platform->options.compareArgs("ADVECTION TYPE", "CUBATURE")) {
    cubOffset = std::max(cubOffset, nrs->meshV->Nelements * nrs->meshV->cubNp);
  }
  cubatureOffset = alignStride<dfloat>(cubOffset);

  tlsr = [&]() {
    lsConfig_t cfg;
    cfg.g0 = &nrs->g0;
    cfg.dt = nrs->dt;
    cfg.dpdt = false;
    cfg.o_coeffBDF = o_coeffBDF;
    cfg.o_coeffEXT = o_coeffEXT;
    cfg.fieldOffset = fieldOffset;
    cfg.vCubatureOffset = cubatureOffset;
    cfg.mesh = nrs->meshV;
    cfg.meshV = nrs->meshV;
    return std::make_unique<ls_t>(cfg);
  }();

  std::cout << "EXITING LS::setup()...\n";
}


ls_t::ls_t(lsConfig_t &cfg)
{
  std::cout << "ENTERING LS::init()...\n";
  if (platform->comm.mpiRank() == 0) {
    std::cout << "================ " << "SETUP LEVEL-SET" << " ===============\n";
  }
  std::cout << "EXITING LS::init()...\n";
}

void ls_t::solve(double time, int stage) {}

void ls_t::saveSolutionState() {}
void ls_t::restoreSolutionState() {}
void ls_t::lagSolution() {}

void ls_t::setTimeIntegrationCoeffs(int tstep) {}

void ls_t::extrapolateSolution() {}

void ls_t::applyDirichlet(double time) {}
void ls_t::setupEllipticSolver() {}

void ls_t::finalize() {}
