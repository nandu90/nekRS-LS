#include "cds.hpp"
#include "lowPassFilter.hpp"
#include "avm.hpp"
#include "bcMap.hpp"

cds_t::cds_t(cdsConfig_t& cfg)
{
  auto& options = platform->options;
  const std::string section = "cds-";
  platform_t *platform = platform_t::getInstance();

  this->NSfields = cfg.Nscalar;

  this->mesh.resize(this->NSfields);
  this->fieldOffset.resize(this->NSfields);
  this->fieldOffsetScan.resize(this->NSfields);
  this->solver.resize(this->NSfields);
  this->compute.resize(this->NSfields);
  this->cvodeSolve.resize(this->NSfields);
  this->filterS.resize(this->NSfields);

  this->NVfields = cfg.dim;
  this->g0 = cfg.g0;
  this->idt = cfg.idt;
  this->dt = cfg.dt;
  this->nBDF = cfg.nBDF;
  this->o_coeffBDF = cfg.o_coeffBDF;
  this->nEXT = cfg.nEXT;
  this->o_coeffEXT = cfg.o_coeffEXT;
  this->o_usrwrk = cfg.o_usrwrk;
  this->vFieldOffset = cfg.fieldOffset;
  this->Nsubsteps = cfg.Nsubsteps;
  this->vCubatureOffset = cfg.vCubatureOffset;
  this->fieldOffset[0] = cfg.fieldOffset; // same for all scalars - at least for now
  this->o_ellipticCoeff = cfg.o_ellipticCoeff;
  this->o_U = cfg.o_U;
  this->o_Ue = cfg.o_Ue;
  this->o_Urst = cfg.o_Urst;
  this->o_relUrst = cfg.o_relUrst;
  this->mesh[0] = cfg.mesh;
  this->meshV = cfg.meshV;

  auto mesh = this->mesh[0];

  this->fieldOffsetScan[0] = 0;
  dlong sum = this->fieldOffset[0];
  for (int s = 1; s < this->NSfields; ++s) {
    this->fieldOffset[s] = this->fieldOffset[0];
    this->fieldOffsetScan[s] = sum;
    sum += this->fieldOffset[s];
    this->mesh[s] = this->meshV;
  }
  this->fieldOffsetSum = sum;

  this->o_fieldOffsetScan = platform->device.malloc<dlong>(this->NSfields, this->fieldOffsetScan.data());

  this->gsh =  oogs::setup(mesh->ogs, 1, this->fieldOffset[0], ogsDfloat, NULL, OOGS_AUTO) ;
  this->gshT = (mesh != this->meshV) ? oogs::setup(mesh->ogs, 1, this->fieldOffset[0], ogsDfloat, NULL, OOGS_AUTO) : this->gsh;

  this->S = (dfloat *)calloc(std::max(this->nBDF, this->nEXT) * this->fieldOffsetSum, sizeof(dfloat));

  this->o_prop = platform->device.malloc<dfloat>(2 * this->fieldOffsetSum);
  this->o_diff = this->o_prop.slice(0 * this->fieldOffsetSum);
  this->o_rho = this->o_prop.slice(1 * this->fieldOffsetSum);

  for (int is = 0; is < this->NSfields; is++) {
    const std::string sid = scalarDigitStr(is);

    if (options.compareArgs("SCALAR" + sid + " SOLVER", "NONE")) {
      continue;
    }

    dfloat diff = 1;
    dfloat rho = 1;
    options.getArgs("SCALAR" + sid + " DIFFUSIVITY", diff);
    options.getArgs("SCALAR" + sid + " DENSITY", rho);

    auto o_diff = this->o_diff + this->fieldOffsetScan[is];
    auto o_rho = this->o_rho + this->fieldOffsetScan[is];
    platform->linAlg->fill(mesh->Nlocal, diff, o_diff);
    platform->linAlg->fill(mesh->Nlocal, rho, o_rho);
  }

  this->anyCvodeSolver = false;
  this->anyEllipticSolver = false;

  this->EToBOffset = this->mesh[0]->Nelements * this->mesh[0]->Nfaces;

  this->EToB = (int *)calloc(this->EToBOffset * this->NSfields, sizeof(int));

  for (int is = 0; is < this->NSfields; is++) {
    std::string sid = scalarDigitStr(is);

    this->compute[is] = 1;
    if (options.compareArgs("SCALAR" + sid + " SOLVER", "NONE")) {
      this->compute[is] = 0;
      this->cvodeSolve[is] = 0;
      continue;
    }

    this->cvodeSolve[is] = options.compareArgs("SCALAR" + sid + " SOLVER", "CVODE");
    this->anyCvodeSolver |= this->cvodeSolve[is];
    this->anyEllipticSolver |= (!this->cvodeSolve[is] && this->compute[is]);

    mesh_t *mesh;
    (is) ? mesh = this->meshV : mesh = this->mesh[0]; // only first scalar can be a CHT mesh

    int cnt = 0;
    for (int e = 0; e < mesh->Nelements; e++) {
      for (int f = 0; f < mesh->Nfaces; f++) {
        this->EToB[cnt + this->EToBOffset * is] = bcMap::id(mesh->EToB[f + e * mesh->Nfaces], "scalar" + sid);
        cnt++;
      }
    }
  }
  this->o_EToB = platform->device.malloc<int>(this->EToBOffset * this->NSfields, this->EToB);

  this->o_compute = platform->device.malloc<dlong>(this->NSfields, this->compute.data());
  this->o_cvodeSolve = platform->device.malloc<dlong>(this->NSfields, this->cvodeSolve.data());

  int nFieldsAlloc = this->anyEllipticSolver ? std::max(this->nBDF, this->nEXT) : 1;
  this->o_S = platform->device.malloc<dfloat>(nFieldsAlloc * this->fieldOffsetSum, this->S);

  nFieldsAlloc = this->anyEllipticSolver ? this->nEXT : 1;
  this->o_FS = platform->device.malloc<dfloat>(nFieldsAlloc * this->fieldOffsetSum);

  if (this->anyEllipticSolver) {
    this->o_Se = platform->device.malloc<dfloat>(this->fieldOffsetSum);
    this->o_BF = platform->device.malloc<dfloat>(this->fieldOffsetSum);
  }

  bool scalarFilteringEnabled = false;
  bool avmEnabled = false;
  {
    for (int is = 0; is < this->NSfields; is++) {
      std::string sid = scalarDigitStr(is);

      if (!options.compareArgs("SCALAR" + sid + " REGULARIZATION METHOD", "NONE")) {
        scalarFilteringEnabled = true;
      }

      if (options.compareArgs("SCALAR" + sid + " REGULARIZATION METHOD", "AVM_RESIDUAL")) {
        avmEnabled = true;
      }
      if (options.compareArgs("SCALAR" + sid + " REGULARIZATION METHOD", "AVM_HIGHEST_MODAL_DECAY")) {
        avmEnabled = true;
      }
    }
  }

  this->applyFilter = 0;

  if (scalarFilteringEnabled) {

    std::vector<dlong> applyFilterRT(this->NSfields, 0);
    const dlong Nmodes = this->mesh[0]->N + 1;
    this->o_filterRT = platform->device.malloc<dfloat>(this->NSfields * Nmodes * Nmodes);
    this->o_filterS = platform->device.malloc<dfloat>(this->NSfields);
    this->o_applyFilterRT = platform->device.malloc<dlong>(this->NSfields);
    for (int is = 0; is < this->NSfields; is++) {
      std::string sid = scalarDigitStr(is);

      if (options.compareArgs("SCALAR" + sid + " REGULARIZATION METHOD", "NONE")) {
        continue;
      }
      if (!this->compute[is]) {
        continue;
      }

      if (options.compareArgs("SCALAR" + sid + " REGULARIZATION METHOD", "HPFRT")) {
        int filterNc = -1;
        options.getArgs("SCALAR" + sid + " HPFRT MODES", filterNc);
        dfloat filterS;
        options.getArgs("SCALAR" + sid + " HPFRT STRENGTH", filterS);
        filterS = -1.0 * fabs(filterS);
        this->filterS[is] = filterS;

        this->o_filterRT.copyFrom(lowPassFilter(this->mesh[is], filterNc), Nmodes * Nmodes, is * Nmodes * Nmodes);

        applyFilterRT[is] = 1;
        this->applyFilter = 1;
      }
    }

    this->o_filterS.copyFrom(this->filterS.data(), this->NSfields);
    this->o_applyFilterRT.copyFrom(applyFilterRT.data(), this->NSfields);

    if (avmEnabled) {
      avm::setup(this);
    }
  }

  std::string kernelName;
  const std::string suffix = "Hex3D";
  {
    kernelName = "strongAdvectionVolume" + suffix;
    this->strongAdvectionVolumeKernel = platform->kernels.get(section + kernelName);

    if (platform->options.compareArgs("ADVECTION TYPE", "CUBATURE")) {
      kernelName = "strongAdvectionCubatureVolume" + suffix;
      this->strongAdvectionCubatureVolumeKernel = platform->kernels.get(section + kernelName);
    }

    kernelName = "advectMeshVelocity" + suffix;
    this->advectMeshVelocityKernel = platform->kernels.get(section + kernelName);

    kernelName = "maskCopy";
    this->maskCopyKernel = platform->kernels.get(section + kernelName);

    kernelName = "maskCopy2";
    this->maskCopy2Kernel = platform->kernels.get(section + kernelName);

    kernelName = "sumMakef";
    this->sumMakefKernel = platform->kernels.get(section + kernelName);

    kernelName = "neumannBC" + suffix;
    this->neumannBCKernel = platform->kernels.get(section + kernelName);
    kernelName = "dirichletBC";
    this->dirichletBCKernel = platform->kernels.get(section + kernelName);

    kernelName = "setEllipticCoeff";
    this->setEllipticCoeffKernel = platform->kernels.get(section + kernelName);

    kernelName = "filterRT" + suffix;
    this->filterRTKernel = platform->kernels.get(section + kernelName);

    if (this->Nsubsteps) {
      if (platform->options.compareArgs("ADVECTION TYPE", "CUBATURE")) {
        kernelName = "subCycleStrongCubatureVolume" + suffix;
        this->subCycleStrongCubatureVolumeKernel = platform->kernels.get(section + kernelName);
      }
      kernelName = "subCycleStrongVolume" + suffix;
      this->subCycleStrongVolumeKernel = platform->kernels.get(section + kernelName);
    }
  }

  this->cvode = nullptr;
}
