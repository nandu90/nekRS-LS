#include "nrs.hpp"
#include "platform.hpp"
#include "nekInterfaceAdapter.hpp"
#include "RANSktau.hpp"
#include "linAlg.hpp"

// private members
namespace
{
nrs_t *nrs;

int kFieldIndex;

dfloat rho;
dfloat mueLam;

occa::memory o_mut;

occa::memory o_k;
occa::memory o_tau;

occa::memory o_implicitKtau;

occa::kernel computeKernel;
occa::kernel mueKernel;
occa::kernel limitKernel;

occa::kernel SijMag2OiOjSkKernel;

bool buildKernelCalled = false;
bool setupCalled = false;

dfloat coeff[] = {
    0.6,       // sigma_k
    0.5,       // sigma_tau
    1.0,       // alpinf_str
    0.0708,    // beta0
    0.41,      // kappa
    0.09,      // betainf_str
    0.0,       // sigd_min
    1.0 / 8.0, // sigd_max
    400.0,     // fb_c1st
    400.0,     // fb_c2st
    85.0,      // fb_c1
    100.0,     // fb_c2
    0.52,      // alp_inf
    1e-8,      // TINY
    0          // Pope correction
};

occa::memory implicitK(double time, int scalarIdx)
{
  auto &scalar = nrs->scalar;

  if (scalarIdx == kFieldIndex) {
    return o_implicitKtau.slice(0 * scalar->fieldOffset(), scalar->fieldOffset());
  }
  if (scalarIdx == kFieldIndex + 1) {
    return o_implicitKtau.slice(1 * scalar->fieldOffset(), scalar->fieldOffset());
  }
  return o_NULL;
}

} // namespace

void RANSktau::buildKernel(occa::properties _kernelInfo)
{
  occa::properties kernelInfo;
  if (!kernelInfo.get<std::string>("defines/p_sigma_k").size()) {
    kernelInfo["defines/p_sigma_k"] = coeff[0];
  }
  if (!kernelInfo.get<std::string>("defines/p_sigma_tau").size()) {
    kernelInfo["defines/p_sigma_tau"] = coeff[1];
  }
  if (!kernelInfo.get<std::string>("defines/p_alpinf_str").size()) {
    kernelInfo["defines/p_alpinf_str"] = coeff[2];
  }
  if (!kernelInfo.get<std::string>("defines/p_beta0").size()) {
    kernelInfo["defines/p_beta0"] = coeff[3];
  }
  if (!kernelInfo.get<std::string>("defines/p_kappa").size()) {
    kernelInfo["defines/p_kappa"] = coeff[4];
  }
  if (!kernelInfo.get<std::string>("defines/p_betainf_str").size()) {
    kernelInfo["defines/p_betainf_str"] = coeff[5];
  }
  if (!kernelInfo.get<std::string>("defines/p_ibetainf_str3").size()) {
    kernelInfo["defines/p_ibetainf_str3"] = 1 / pow(coeff[5], 3);
  }
  if (!kernelInfo.get<std::string>("defines/p_sigd_min").size()) {
    kernelInfo["defines/p_sigd_min"] = coeff[6];
  }
  if (!kernelInfo.get<std::string>("defines/p_sigd_max").size()) {
    kernelInfo["defines/p_sigd_max"] = coeff[7];
  }
  if (!kernelInfo.get<std::string>("defines/p_fb_c1st").size()) {
    kernelInfo["defines/p_fb_c1st"] = coeff[8];
  }
  if (!kernelInfo.get<std::string>("defines/p_fb_c2st").size()) {
    kernelInfo["defines/p_fb_c2st"] = coeff[9];
  }
  if (!kernelInfo.get<std::string>("defines/p_fb_c1").size()) {
    kernelInfo["defines/p_fb_c1"] = coeff[10];
  }
  if (!kernelInfo.get<std::string>("defines/p_fb_c2").size()) {
    kernelInfo["defines/p_fb_c2"] = coeff[11];
  }
  if (!kernelInfo.get<std::string>("defines/p_alp_inf").size()) {
    kernelInfo["defines/p_alp_inf"] = coeff[12];
  }
  if (!kernelInfo.get<std::string>("defines/p_tiny").size()) {
    kernelInfo["defines/p_tiny"] = coeff[13];
  }
  if (!kernelInfo.get<std::string>("defines/p_pope").size()) {
    kernelInfo["defines/p_pope"] = coeff[14];
  }

  if (platform->comm.mpiRank == 0 && platform->verbose()) {
    std::cout << "\nRANSktau settings\n";
    std::cout << kernelInfo << std::endl;
  }

  kernelInfo += _kernelInfo;

  auto buildKernel = [&kernelInfo](const std::string &kernelName) {
    const auto path = getenv("NEKRS_KERNEL_DIR") + std::string("/app/nrs/plugins/");
    const auto fileName = path + "RANSktau.okl";
    const auto reqName = "RANSktau::";
    if (platform->options.compareArgs("REGISTER ONLY", "TRUE")) {
      platform->kernelRequests.add(reqName, fileName, kernelInfo);
      return occa::kernel();
    } else {
      buildKernelCalled = 1;
      return platform->kernelRequests.load(reqName, kernelName);
    }
  };

  computeKernel = buildKernel("RANSktauComputeHex3D");
  mueKernel = buildKernel("mue");
  limitKernel = buildKernel("limit");
  SijMag2OiOjSkKernel = buildKernel("SijMag2OiOjSk");

  int Nscalar;
  platform->options.getArgs("NUMBER OF SCALARS", Nscalar);

  nekrsCheck(Nscalar < 2, platform->comm.mpiComm, EXIT_FAILURE, "%s\n", "Nscalar needs to be >= 2!");
  platform->options.setArgs("FLUID STRESSFORMULATION", "TRUE");
}

void RANSktau::updateProperties()
{
  nekrsCheck(!setupCalled || !buildKernelCalled,
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "%s\n",
             "called prior to tavg::setup()!");

  platform->options.getArgs("FLUID VISCOSITY", mueLam);
  platform->options.getArgs("FLUID DENSITY", rho);

  limitKernel(nrs->fluid->mesh->Nlocal, o_k, o_tau);
  mueKernel(nrs->fluid->mesh->Nlocal,
            nrs->fluid->fieldOffset,
            rho,
            mueLam,
            o_k,
            o_tau,
            o_mut,
            nrs->fluid->o_mue,
            nrs->scalar->o_diff + nrs->scalar->fieldOffsetScan[kFieldIndex]);
}

const deviceMemory<dfloat> RANSktau::o_mue_t()
{
  deviceMemory<dfloat> out(o_mut);
  return out;
}

void RANSktau::updateSourceTerms()
{
  nekrsCheck(!setupCalled || !buildKernelCalled,
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "%s\n",
             "called prior to tavg::setup()!");

  auto mesh = nrs->fluid->mesh;
  auto &scalar = nrs->scalar;

  occa::memory o_OiOjSk = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
  occa::memory o_SijMag2 = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);

  auto o_SijOij = nrs->strainRotationRate();

  SijMag2OiOjSkKernel(mesh->Nlocal, nrs->fluid->fieldOffset, 1, o_SijOij, o_OiOjSk, o_SijMag2);

  platform->options.getArgs("FLUID VISCOSITY", mueLam);
  platform->options.getArgs("FLUID DENSITY", rho);

  computeKernel(mesh->Nelements,
                nrs->fluid->fieldOffset, // assumes offset is always the same
                rho,
                mueLam,
                mesh->o_vgeo,
                mesh->o_D,
                o_k,
                o_tau,
                o_SijMag2,
                o_OiOjSk,
                o_implicitKtau,
                scalar->o_EXT + scalar->fieldOffsetScan[kFieldIndex]);
}

void RANSktau::setup(int ifld)
{
  static bool isInitialized = false;
  if (isInitialized) {
    return;
  }
  isInitialized = true;

  nrs = dynamic_cast<nrs_t *>(platform->app);
  kFieldIndex = ifld; // tauFieldIndex is assumed to be kFieldIndex+1

  auto &scalar = nrs->scalar;
  nekrsCheck(scalar->NSfields < kFieldIndex + 1,
             platform->comm.mpiComm,
             EXIT_FAILURE,
             "%s\n",
             "number of scalar fields too low!");

  for (int i = 0; i < 2; i++) {
    auto &scalar = nrs->scalar;

    platform->options.getArgs("FLUID DENSITY", rho);
    auto o_rho =
        scalar->o_rho.slice(scalar->fieldOffsetScan[kFieldIndex + i], scalar->mesh(kFieldIndex + i)->Nlocal);
    platform->linAlg->fill(o_rho.size(), rho, o_rho);

    const std::string sid = scalarDigitStr(kFieldIndex + i);
    nekrsCheck(!platform->options.getArgs("SCALAR" + sid + " DIFFUSIVITY").empty() ||
                   !platform->options.getArgs("SCALAR" + sid + " DENSITY").empty(),
               platform->comm.mpiComm,
               EXIT_FAILURE,
               "%s\n",
               "illegal property specification for k/tau in par!");
  }

  o_k = scalar->o_S + scalar->fieldOffsetScan[kFieldIndex];
  o_tau = scalar->o_S + scalar->fieldOffsetScan[kFieldIndex + 1];
  o_mut = platform->device.malloc<dfloat>(scalar->mesh(kFieldIndex)->Nlocal);
  o_implicitKtau = platform->device.malloc<dfloat>(2 * scalar->fieldOffset());

  scalar->userImplicitLinearTerm = implicitK;

  setupCalled = true;
}
