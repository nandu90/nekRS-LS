#include "nrs.hpp"
#include "bdry.hpp"
#include "bcMap.hpp"
#include "nekInterfaceAdapter.hpp"
#include "udf.hpp"
#include "lowPassFilter.hpp"
#include "avm.hpp"
#include "re2Reader.hpp"

#include "cds.hpp"

static void printICMinMax(nrs_t *nrs)
{
  if (platform->comm.mpiRank == 0) {
    printf("================= INITIAL CONDITION ====================\n");
  }

  {
    auto mesh = nrs->_mesh;
    auto o_x = mesh->o_x;
    auto o_y = mesh->o_y;
    auto o_z = mesh->o_z;

    const auto xMin = platform->linAlg->min(mesh->Nlocal, o_x, platform->comm.mpiComm);
    const auto yMin = platform->linAlg->min(mesh->Nlocal, o_y, platform->comm.mpiComm);
    const auto zMin = platform->linAlg->min(mesh->Nlocal, o_z, platform->comm.mpiComm);
    const auto xMax = platform->linAlg->max(mesh->Nlocal, o_x, platform->comm.mpiComm);
    const auto yMax = platform->linAlg->max(mesh->Nlocal, o_y, platform->comm.mpiComm);
    const auto zMax = platform->linAlg->max(mesh->Nlocal, o_z, platform->comm.mpiComm);
    if (platform->comm.mpiRank == 0) {
      printf("XYZ   min/max: %g %g  %g %g  %g %g\n", xMin, xMax, yMin, yMax, zMin, zMax);
    }
  }

  if (platform->options.compareArgs("MOVING MESH", "TRUE")) {
    auto mesh = nrs->_mesh;
    auto o_ux = mesh->o_U + 0 * nrs->fieldOffset;
    auto o_uy = mesh->o_U + 1 * nrs->fieldOffset;
    auto o_uz = mesh->o_U + 2 * nrs->fieldOffset;
    const auto uxMin = platform->linAlg->min(mesh->Nlocal, o_ux, platform->comm.mpiComm);
    const auto uyMin = platform->linAlg->min(mesh->Nlocal, o_uy, platform->comm.mpiComm);
    const auto uzMin = platform->linAlg->min(mesh->Nlocal, o_uz, platform->comm.mpiComm);
    const auto uxMax = platform->linAlg->max(mesh->Nlocal, o_ux, platform->comm.mpiComm);
    const auto uyMax = platform->linAlg->max(mesh->Nlocal, o_uy, platform->comm.mpiComm);
    const auto uzMax = platform->linAlg->max(mesh->Nlocal, o_uz, platform->comm.mpiComm);
    if (platform->comm.mpiRank == 0) {
      printf("UMSH  min/max: %g %g  %g %g  %g %g\n", uxMin, uxMax, uyMin, uyMax, uzMin, uzMax);
    }
  }

  {
    auto mesh = nrs->meshV;
    auto o_ux = nrs->o_U + 0 * nrs->fieldOffset;
    auto o_uy = nrs->o_U + 1 * nrs->fieldOffset;
    auto o_uz = nrs->o_U + 2 * nrs->fieldOffset;
    const auto uxMin = platform->linAlg->min(mesh->Nlocal, o_ux, platform->comm.mpiComm);
    const auto uyMin = platform->linAlg->min(mesh->Nlocal, o_uy, platform->comm.mpiComm);
    const auto uzMin = platform->linAlg->min(mesh->Nlocal, o_uz, platform->comm.mpiComm);
    const auto uxMax = platform->linAlg->max(mesh->Nlocal, o_ux, platform->comm.mpiComm);
    const auto uyMax = platform->linAlg->max(mesh->Nlocal, o_uy, platform->comm.mpiComm);
    const auto uzMax = platform->linAlg->max(mesh->Nlocal, o_uz, platform->comm.mpiComm);
    if (platform->comm.mpiRank == 0) {
      printf("U     min/max: %g %g  %g %g  %g %g\n", uxMin, uxMax, uyMin, uyMax, uzMin, uzMax);
    }
  }

  {
    auto mesh = nrs->meshV;
    const auto prMin = platform->linAlg->min(mesh->Nlocal, nrs->o_P, platform->comm.mpiComm);
    const auto prMax = platform->linAlg->max(mesh->Nlocal, nrs->o_P, platform->comm.mpiComm);
    if (platform->comm.mpiRank == 0) {
      printf("P     min/max: %g %g\n", prMin, prMax);
    }
  }

  if (nrs->Nscalar) {
    auto cds = nrs->cds;
    if (platform->comm.mpiRank == 0) {
      printf("S     min/max:");
    }

    int cnt = 0;
    for (int is = 0; is < cds->NSfields; is++) {
      cnt++;

      mesh_t *mesh;
      (is) ? mesh = cds->meshV : mesh = cds->mesh[0]; // only first scalar can be a CHT mesh

      auto o_si = nrs->cds->o_S + nrs->cds->fieldOffsetScan[is];
      const auto siMin = platform->linAlg->min(mesh->Nlocal, o_si, platform->comm.mpiComm);
      const auto siMax = platform->linAlg->max(mesh->Nlocal, o_si, platform->comm.mpiComm);
      if (platform->comm.mpiRank == 0) {
        if (cnt > 1) {
          printf("  ");
        } else {
          printf(" ");
        }
        printf("%g %g", siMin, siMax);
      }
    }
    if (platform->comm.mpiRank == 0) {
      printf("\n");
    }
  }
}

nrs_t::nrs_t(MPI_Comm comm, bool ms, setupAide &options)
{
  platform_t *platform = platform_t::getInstance();
  device_t &device = platform->device;

  this->multiSession = ms;
  this->kernelInfo = new occa::properties();
  *(this->kernelInfo) = platform->kernelInfo;
  occa::properties &kernelInfo = *this->kernelInfo;
  kernelInfo["defines"].asObject();
  kernelInfo["includes"].asArray();
  kernelInfo["header"].asArray();
  kernelInfo["flags"].asObject();
  kernelInfo["include_paths"].asArray();

  int N, cubN;
  platform->options.getArgs("POLYNOMIAL DEGREE", N);
  platform->options.getArgs("CUBATURE POLYNOMIAL DEGREE", cubN);
  platform->options.getArgs("NUMBER OF SCALARS", this->Nscalar);
  platform->options.getArgs("MESH DIMENSION", this->dim);
  platform->options.getArgs("ELEMENT TYPE", this->elementType);

  {
#if 1
    if (platform->device.mode() == "Serial") {
      platform->options.setArgs("ENABLE GS COMM OVERLAP", "FALSE");
    }
#endif

    if (platform->comm.mpiCommSize == 1) {
      platform->options.setArgs("ENABLE GS COMM OVERLAP", "FALSE");
    }

    if (platform->comm.mpiRank == 0 && platform->options.compareArgs("ENABLE GS COMM OVERLAP", "FALSE")) {
      std::cout << "ENABLE GS COMM OVERLAP disabled\n\n";
    }
  }

  this->flow = 1;
  if (platform->options.compareArgs("VELOCITY SOLVER", "NONE")) {
    this->flow = 0;
  }

  if (this->flow) {
    if (platform->options.compareArgs("VELOCITY STRESSFORMULATION", "TRUE")) {
      platform->options.setArgs("VELOCITY BLOCK SOLVER", "TRUE");
    }
  }

  {
    int nelgt, nelgv;
    const std::string meshFile = options.getArgs("MESH FILE");
    re2::nelg(meshFile, nelgt, nelgv, platform->comm.mpiComm);

    nekrsCheck(nelgt != nelgv && platform->options.compareArgs("MOVING MESH", "TRUE"),
               platform->comm.mpiComm,
               EXIT_FAILURE,
               "%s\n",
               "Conjugate heat transfer not supported in a moving mesh!");

    nekrsCheck(nelgt != nelgv && !platform->options.compareArgs("SCALAR00 IS TEMPERATURE", "TRUE"),
               platform->comm.mpiComm,
               EXIT_FAILURE,
               "%s\n",
               "Conjugate heat transfer requires a temperature field!");

    bool coupled = neknekCoupled();
    nekrsCheck(nelgt != nelgv && coupled,
               platform->comm.mpiComm,
               EXIT_FAILURE,
               "%s\n",
               "Conjugate heat transfer + neknek not supported!");
  }

  // init nek
  {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    std::string casename;
    platform->options.getArgs("CASENAME", casename);

    nek::setup(this);
  }

  this->cht = 0;
  {
    hlong NelementsV = nekData.nelv;
    hlong NelementsT = nekData.nelt;
    MPI_Allreduce(MPI_IN_PLACE, &NelementsV, 1, MPI_HLONG, MPI_SUM, platform->comm.mpiComm);
    MPI_Allreduce(MPI_IN_PLACE, &NelementsT, 1, MPI_HLONG, MPI_SUM, platform->comm.mpiComm);
    if ((NelementsT > NelementsV) && this->Nscalar) {
      this->cht = 1;
    }

    nekrsCheck(this->cht && (NelementsT <= NelementsV),
               MPI_COMM_SELF,
               EXIT_FAILURE,
               "%s\n",
               "Invalid solid element partitioning");
  }

  this->_mesh = createMesh(comm, N, cubN, this->cht, kernelInfo);
  this->meshV = (mesh_t *)this->_mesh->fluid;
  mesh_t *mesh = this->meshV;

  // verify boundary conditions
  {
    auto fields = nrsFieldsToSolve(options);

    for (const auto &field : fields) {
      auto msh = (this->cht && (field == "scalar00" || field == "mesh")) ? this->_mesh : mesh;
      nekrsCheck(msh->Nbid != bcMap::size(field),
                 platform->comm.mpiComm,
                 EXIT_FAILURE,
                 "Size of %s boundaryTypeMap (%d) does not match number of boundary IDs in mesh (%d)!\n",
                 field.c_str(),
                 bcMap::size(field),
                 msh->Nbid);
    }

    bcMap::checkBoundaryAlignment(this->meshV);
  }

  this->NVfields = this->dim;

  platform->options.getArgs("SUBCYCLING STEPS", this->Nsubsteps);
  platform->options.getArgs("DT", this->dt[0]);

  this->idt = 1 / this->dt[0];
  this->g0 = 1;

  platform->options.getArgs("BDF ORDER", this->nBDF);
  platform->options.getArgs("EXT ORDER", this->nEXT);
  if (this->Nsubsteps) {
    this->nEXT = this->nBDF;
  }

  nekrsCheck(this->nEXT < this->nBDF,
             platform->comm.mpiComm,
             EXIT_FAILURE,
             "%s\n",
             "EXT order needs to be >= BDF order!");

  this->coeffEXT = (dfloat *)calloc(this->nEXT, sizeof(dfloat));
  this->coeffBDF = (dfloat *)calloc(this->nBDF, sizeof(dfloat));

  this->nRK = 4;

  dfloat mue = 1;
  dfloat rho = 1;
  platform->options.getArgs("VISCOSITY", mue);
  platform->options.getArgs("DENSITY", rho);

  { // setup fieldOffset
    this->fieldOffset = mesh->Np * (mesh->Nelements + mesh->totalHaloPairs);
    mesh_t *meshT = this->_mesh;
    this->fieldOffset = std::max(this->fieldOffset, meshT->Np * (meshT->Nelements + meshT->totalHaloPairs));
    this->fieldOffset = alignStride<dfloat>(this->fieldOffset);
  }
  this->_mesh->fieldOffset = this->fieldOffset;

  { // setup cubatureOffset
    if (platform->options.compareArgs("ADVECTION TYPE", "CUBATURE")) {
      this->cubatureOffset = std::max(this->fieldOffset, mesh->Nelements * mesh->cubNp);
    } else {
      this->cubatureOffset = this->fieldOffset;
    }
    this->cubatureOffset = alignStride<dfloat>(this->cubatureOffset);
  }

  if (this->Nsubsteps) {
    int Sorder;
    platform->options.getArgs("SUBCYCLING TIME ORDER", Sorder);
    if (Sorder == 4 && this->nRK == 4) { // ERK(4,4)
      dfloat rka[4] = {0.0, 1.0 / 2.0, 1.0 / 2.0, 1.0};
      dfloat rkb[4] = {1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0};
      dfloat rkc[4] = {0.0, 1.0 / 2.0, 1.0 / 2.0, 1.0};
      this->coeffsfRK = (dfloat *)calloc(this->nRK, sizeof(dfloat));
      this->weightsRK = (dfloat *)calloc(this->nRK, sizeof(dfloat));
      this->nodesRK = (dfloat *)calloc(this->nRK, sizeof(dfloat));
      memcpy(this->coeffsfRK, rka, this->nRK * sizeof(dfloat));
      memcpy(this->weightsRK, rkb, this->nRK * sizeof(dfloat));
      memcpy(this->nodesRK, rkc, this->nRK * sizeof(dfloat));
    } else {
      nekrsCheck(true, platform->comm.mpiComm, EXIT_FAILURE, "%s\n", "Unsupported subcycling scheme!");
    }
    this->o_coeffsfRK = device.malloc<dfloat>(this->nRK, this->coeffsfRK);
    this->o_weightsRK = device.malloc<dfloat>(this->nRK, this->weightsRK);
  }

  if (options.compareArgs("MOVING MESH", "TRUE")) {
    const int nBDF = std::max(this->nBDF, this->nEXT);
    auto o_tmp = platform->device.malloc<dfloat>(mesh->Nlocal);
    o_tmp.copyFrom(mesh->o_LMM, mesh->Nlocal);
    mesh->o_LMM.free();
    mesh->o_LMM = platform->device.malloc<dfloat>(this->fieldOffset * nBDF);
    mesh->o_LMM.copyFrom(o_tmp, mesh->Nlocal);

    o_tmp.copyFrom(mesh->o_invLMM, mesh->Nlocal);
    mesh->o_invLMM.free();
    mesh->o_invLMM = platform->device.malloc<dfloat>(this->fieldOffset * nBDF);
    mesh->o_invLMM.copyFrom(o_tmp, mesh->Nlocal);

    const int nAB = std::max(this->nEXT, mesh->nAB);
    mesh->U = (dfloat *)calloc(this->NVfields * this->fieldOffset * nAB, sizeof(dfloat));
    mesh->o_U = platform->device.malloc<dfloat>(this->NVfields * nAB * this->fieldOffset, mesh->U);
    mesh->o_Ue = platform->device.malloc<dfloat>(this->NVfields * nAB * this->fieldOffset);
    if (this->Nsubsteps) {
      mesh->o_divU = platform->device.malloc<dfloat>(this->fieldOffset * nAB);
    }
  }

  {
    const dlong Nstates = this->Nsubsteps ? std::max(this->nBDF, this->nEXT) : 1;
    bool useCVODE = platform->options.compareArgs("CVODE", "TRUE");
    if ((useCVODE || this->Nsubsteps) && platform->options.compareArgs("MOVING MESH", "TRUE")) {
      this->o_relUrst = platform->device.malloc<dfloat>(Nstates * this->NVfields * this->cubatureOffset);
    }
    if (!this->Nsubsteps || platform->options.compareArgs("MOVING MESH", "FALSE")) {
      this->o_Urst = platform->device.malloc<dfloat>(Nstates * this->NVfields * this->cubatureOffset);
    }
  }

  this->U =
      (dfloat *)calloc(this->NVfields * std::max(this->nBDF, this->nEXT) * this->fieldOffset, sizeof(dfloat));
  this->o_U =
      platform->device.malloc<dfloat>(this->NVfields * std::max(this->nBDF, this->nEXT) * this->fieldOffset,
                                      this->U);

  this->o_Ue = platform->device.malloc<dfloat>(this->NVfields * this->fieldOffset);

  this->P = (dfloat *)calloc(this->fieldOffset, sizeof(dfloat));
  this->o_P = platform->device.malloc<dfloat>(this->fieldOffset, this->P);

  this->o_BF = platform->device.malloc<dfloat>(this->NVfields * this->fieldOffset);
  this->o_FU = platform->device.malloc<dfloat>(this->NVfields * this->nEXT * this->fieldOffset);

  this->o_ellipticCoeff = device.malloc<dfloat>(2 * this->fieldOffset);

  int nProperties = 2;
  if (!options.compareArgs("MESH SOLVER", "NONE")) {
    nProperties = 4;
  }

  this->o_prop = device.malloc<dfloat>(nProperties * this->fieldOffset);
  this->o_mue = this->o_prop.slice(0 * this->fieldOffset);
  this->o_rho = this->o_prop.slice(1 * this->fieldOffset);
  if (!options.compareArgs("MESH SOLVER", "NONE")) {
    this->o_meshMue = this->o_prop.slice(2 * this->fieldOffset);
    this->o_meshRho = this->o_prop.slice(3 * this->fieldOffset);
  }

  platform->linAlg->fill(mesh->Nlocal, mue, this->o_mue);
  platform->linAlg->fill(mesh->Nlocal, rho, this->o_rho);
  if (!options.compareArgs("MESH SOLVER", "NONE")) {
    auto o_mue = this->o_prop + 2 * this->fieldOffset;
    auto o_rho = this->o_prop + 3 * this->fieldOffset;
    platform->linAlg->fill(mesh->Nlocal, 1.0, o_mue);
    platform->linAlg->fill(mesh->Nlocal, 0.0, o_rho);
  }

  if (platform->options.compareArgs("CONSTANT FLOW RATE", "TRUE")) {
    this->o_Uc = platform->device.malloc<dfloat>(this->NVfields * this->fieldOffset);
    this->o_Pc = platform->device.malloc<dfloat>(this->fieldOffset);
    this->o_prevProp = device.malloc<dfloat>(2 * this->fieldOffset);
    this->o_prevProp.copyFrom(this->o_prop, this->o_prevProp.length());
  }

  this->o_div = device.malloc<dfloat>(this->fieldOffset);

  this->o_coeffEXT = platform->device.malloc<dfloat>(this->nEXT, this->coeffEXT);
  this->o_coeffBDF = platform->device.malloc<dfloat>(this->nBDF, this->coeffBDF);

  this->qqt = oogs::setup(mesh->ogs, this->NVfields, this->fieldOffset, ogsDfloat, NULL, OOGS_AUTO);
  this->gsh = this->qqt;

  if (!options.compareArgs("MESH SOLVER", "NONE")) {
    mesh_t *meshT = this->_mesh;
    this->gshMesh = oogs::setup(meshT->ogs, this->NVfields, this->fieldOffset, ogsDfloat, NULL, OOGS_AUTO);
  }

  if (this->flow) {
    this->EToB = (int *)calloc(mesh->Nelements * mesh->Nfaces, sizeof(int));
    int cnt = 0;
    for (int e = 0; e < mesh->Nelements; e++) {
      for (int f = 0; f < mesh->Nfaces; f++) {
        this->EToB[cnt] = bcMap::id(mesh->EToB[f + e * mesh->Nfaces], "velocity");
        cnt++;
      }
    }
    this->o_EToB = device.malloc<int>(mesh->Nelements * mesh->Nfaces, this->EToB);
  }

  if (!platform->options.compareArgs("MESH SOLVER", "NONE")) {
    this->EToBMeshVelocity = (int *)calloc(mesh->Nelements * mesh->Nfaces, sizeof(int));
    int cnt = 0;
    for (int e = 0; e < mesh->Nelements; e++) {
      for (int f = 0; f < mesh->Nfaces; f++) {
        int bc = bcMap::id(mesh->EToB[f + e * mesh->Nfaces], "mesh");
        this->EToBMeshVelocity[cnt] = bcMap::id(mesh->EToB[f + e * mesh->Nfaces], "mesh");
        cnt++;
      }
    }
    this->o_EToBMeshVelocity = device.malloc<int>(mesh->Nelements * mesh->Nfaces, this->EToBMeshVelocity);
  }

  if (platform->options.compareArgs("VELOCITY REGULARIZATION METHOD", "HPFRT")) {

    this->filterNc = -1;
    dfloat filterS;
    platform->options.getArgs("VELOCITY HPFRT STRENGTH", filterS);
    platform->options.getArgs("VELOCITY HPFRT MODES", this->filterNc);
    filterS = -1.0 * fabs(filterS);
    this->filterS = filterS;

    this->o_filterRT = lowPassFilter(this->meshV, this->filterNc);
  }

  // build kernels
  std::string kernelName;
  const std::string suffix = "Hex3D";
  {
    const std::string section = "nrs-";
    kernelName = "nStagesSum3";
    this->nStagesSum3Kernel = platform->kernels.get(section + kernelName);

    kernelName = "computeFieldDotNormal";
    this->computeFieldDotNormalKernel = platform->kernels.get(section + kernelName);

    kernelName = "computeFaceCentroid";
    this->computeFaceCentroidKernel = platform->kernels.get(section + kernelName);

    {
      kernelName = "strongAdvectionVolume" + suffix;
      this->strongAdvectionVolumeKernel = platform->kernels.get(section + kernelName);

      if (platform->options.compareArgs("ADVECTION TYPE", "CUBATURE")) {
        kernelName = "strongAdvectionCubatureVolume" + suffix;
        this->strongAdvectionCubatureVolumeKernel = platform->kernels.get(section + kernelName);
      }
    }

    kernelName = "curl" + suffix;
    this->curlKernel = platform->kernels.get(section + kernelName);

    kernelName = "SijOij" + suffix;
    this->SijOijKernel = platform->kernels.get(section + kernelName);

    kernelName = "gradientVolume" + suffix;
    this->gradientVolumeKernel = platform->kernels.get(section + kernelName);

    kernelName = "wGradientVolume" + suffix;
    this->wgradientVolumeKernel = platform->kernels.get(section + kernelName);

    {
      kernelName = "sumMakef";
      this->sumMakefKernel = platform->kernels.get(section + kernelName);
    }

    kernelName = "wDivergenceVolume" + suffix;
    this->wDivergenceVolumeKernel = platform->kernels.get(section + kernelName);
    kernelName = "divergenceVolume" + suffix;
    this->divergenceVolumeKernel = platform->kernels.get(section + kernelName);

    kernelName = "divergenceSurface" + suffix;
    this->divergenceSurfaceKernel = platform->kernels.get(section + kernelName);

    kernelName = "advectMeshVelocity" + suffix;
    this->advectMeshVelocityKernel = platform->kernels.get(section + kernelName);

    kernelName = "pressureRhs" + suffix;
    this->pressureRhsKernel = platform->kernels.get(section + kernelName);

    kernelName = "pressureStress" + suffix;
    this->pressureStressKernel = platform->kernels.get(section + kernelName);

    kernelName = "pressureDirichletBC" + suffix;
    this->pressureDirichletBCKernel = platform->kernels.get(section + kernelName);

    kernelName = "velocityRhs" + suffix;
    this->velocityRhsKernel = platform->kernels.get(section + kernelName);

    kernelName = "averageNormalBcType";
    this->averageNormalBcTypeKernel = platform->kernels.get(section + kernelName);

    kernelName = "fixZeroNormalMask";
    this->fixZeroNormalMaskKernel = platform->kernels.get(section + kernelName);

    kernelName = "applyZeroNormalMask";
    this->applyZeroNormalMaskKernel = platform->kernels.get(section + kernelName);

    kernelName = "initializeZeroNormalMask";
    this->initializeZeroNormalMaskKernel = platform->kernels.get(section + kernelName);

    kernelName = "velocityDirichletBC" + suffix;
    this->velocityDirichletBCKernel = platform->kernels.get(section + kernelName);

    kernelName = "velocityNeumannBC" + suffix;
    this->velocityNeumannBCKernel = platform->kernels.get(section + kernelName);

    kernelName = "UrstCubature" + suffix;
    this->UrstCubatureKernel = platform->kernels.get(section + kernelName);

    kernelName = "Urst" + suffix;
    this->UrstKernel = platform->kernels.get(section + kernelName);

    if (this->Nsubsteps) {
      if (platform->options.compareArgs("ADVECTION TYPE", "CUBATURE")) {
        kernelName = "subCycleStrongCubatureVolume" + suffix;
        this->subCycleStrongCubatureVolumeKernel = platform->kernels.get(section + kernelName);
      }
      kernelName = "subCycleStrongVolume" + suffix;
      this->subCycleStrongVolumeKernel = platform->kernels.get(section + kernelName);

      kernelName = "subCycleRK";
      this->subCycleRKKernel = platform->kernels.get(section + kernelName);

      kernelName = "subCycleInitU0";
      this->subCycleInitU0Kernel = platform->kernels.get(section + kernelName);
    }

    kernelName = "extrapolate";
    this->extrapolateKernel = platform->kernels.get(section + kernelName);

    kernelName = "maskCopy";
    this->maskCopyKernel = platform->kernels.get(section + kernelName);

    kernelName = "maskCopy2";
    this->maskCopy2Kernel = platform->kernels.get(section + kernelName);

    kernelName = "mask";
    this->maskKernel = platform->kernels.get(section + kernelName);

    kernelName = "filterRT" + suffix;
    this->filterRTKernel = platform->kernels.get(section + kernelName);

    kernelName = "cfl" + suffix;
    this->cflKernel = platform->kernels.get(section + kernelName);

    kernelName = "pressureAddQtl";
    this->pressureAddQtlKernel = platform->kernels.get(section + kernelName);

    kernelName = "setEllipticCoeff";
    this->setEllipticCoeffKernel = platform->kernels.get(section + kernelName);
    kernelName = "setEllipticCoeffPressure";
    this->setEllipticCoeffPressureKernel = platform->kernels.get(section + kernelName);
  }

  if (this->Nscalar) {
    cdsConfig_t cfg;

    cfg.Nscalar = this->Nscalar;
    cfg.dim = this->NVfields;
    cfg.g0 = &this->g0;
    cfg.idt = &this->idt;
    cfg.dt = this->dt;
    cfg.nBDF = this->nBDF;
    cfg.o_coeffBDF = this->o_coeffBDF;
    cfg.nEXT = this->nEXT;
    cfg.o_coeffEXT = this->o_coeffEXT;
    cfg.o_usrwrk = &this->o_usrwrk;
    cfg.vFieldOffset = this->fieldOffset;
    cfg.Nsubsteps = this->Nsubsteps;
    cfg.vCubatureOffset = this->cubatureOffset;
    cfg.fieldOffset = this->fieldOffset;
    cfg.o_ellipticCoeff = this->o_ellipticCoeff;
    cfg.o_U = this->o_U;
    cfg.o_Ue = this->o_Ue;
    cfg.o_Urst = this->o_Urst;
    cfg.o_relUrst = this->o_relUrst;
    cfg.mesh = this->_mesh;
    cfg.meshV = this->meshV;

    this->cds = new cds_t(cfg);

    if (this->cds->anyCvodeSolver) {
      this->cvode = new cvode_t(this);
      this->cds->cvode = this->cvode;
    }
  }

  if (!platform->options.getArgs("RESTART FILE NAME").empty()) {
    std::string fileName;
    platform->options.getArgs("RESTART FILE NAME", fileName);
    nek::restartFromFile(fileName);

    double startTime;
    nek::copyFromNek(startTime);
    platform->options.setArgs("START TIME", to_string_f(startTime));
  } else {
    nek::getIC();
  }

  // udf setup
  if (platform->comm.mpiRank == 0) {
    printf("calling UDF_Setup ... \n");
  }
  fflush(stdout);

  udf.setup(this);

  if (platform->comm.mpiRank == 0) {
    printf("done\n");
  }
  fflush(stdout);

  this->p0the = this->p0th[0];

  // in case the user modifies mesh in udf.setup
  this->_mesh->o_x.copyFrom(this->_mesh->x);
  this->_mesh->o_y.copyFrom(this->_mesh->y);
  this->_mesh->o_z.copyFrom(this->_mesh->z);
  if (this->meshV != this->_mesh) {
    this->meshV->update();
  }
  this->_mesh->update();

  // in case the user sets IC in udf.setup
  this->o_U.copyFrom(this->U);
  this->o_P.copyFrom(this->P);
  if (this->Nscalar) {
    this->cds->o_S.copyFrom(this->cds->S);
  }
  if (options.compareArgs("MOVING MESH", "TRUE")) {
    mesh->o_U.copyFrom(mesh->U);
  }

  double startTime;
  platform->options.getArgs("START TIME", startTime);

  // ensure both codes see the same mesh + IC
  nek::ocopyToNek(startTime, 0);

  // update props based on IC
  this->evaluateProperties(startTime);

  // CVODE can only be initialized once the initial condition
  // is known, however, a user may need to set function ptrs
  // on to cvode_t object.
  // Hence, the actual CVODE initialization part of cvode_t
  // is done below.
  if (this->cvode) {
    this->cvode->initialize();
  }

  if (platform->comm.mpiRank == 0) {
    std::cout << std::endl;
  }
  printMeshMetrics(this->_mesh);

  printICMinMax(this);

  // setup elliptic solver

  if (this->Nscalar) {
    cds_t *cds = this->cds;

    for (int is = 0; is < cds->NSfields; is++) {
      std::string sid = scalarDigitStr(is);

      if (!cds->compute[is]) {
        continue;
      }

      mesh_t *mesh;
      (is) ? mesh = cds->meshV : mesh = cds->mesh[0]; // only first scalar can be a CHT mesh

      const auto solverName = cds->cvodeSolve[is] ? "CVODE" : "ELLIPTIC";
      if (platform->comm.mpiRank == 0) {
        std::cout << "================= " << solverName << " SETUP SCALAR" << sid << " ===============\n";
      }

      const int nbrBIDs = bcMap::size("scalar" + sid);
      for (int bID = 1; bID <= nbrBIDs; bID++) {
        std::string bcTypeText(bcMap::text(bID, "scalar" + sid));
        if (platform->comm.mpiRank == 0 && bcTypeText.size()) {
          printf("bID %d -> bcType %s\n", bID, bcTypeText.c_str());
        }
      }

      if (cds->cvodeSolve[is]) {
        continue;
      }

      cds->solver[is] = new elliptic_t();
      cds->solver[is]->name = "scalar" + sid;
      cds->solver[is]->Nfields = 1;
      cds->solver[is]->fieldOffset = this->fieldOffset;
      cds->solver[is]->mesh = mesh;

      cds->solver[is]->poisson = 0;

      cds->setEllipticCoeffKernel(mesh->Nlocal,
                                  *(cds->g0) * *(cds->idt),
                                  cds->fieldOffsetScan[is],
                                  this->fieldOffset,
                                  0,
                                  cds->o_diff,
                                  cds->o_rho,
                                  o_NULL,
                                  cds->o_ellipticCoeff);

      cds->solver[is]->o_lambda0 = cds->o_ellipticCoeff.slice(0 * this->fieldOffset);
      cds->solver[is]->o_lambda1 = cds->o_ellipticCoeff.slice(1 * this->fieldOffset);

      cds->solver[is]->EToB = (int *)calloc(mesh->Nelements * mesh->Nfaces, sizeof(int));
      for (dlong e = 0; e < mesh->Nelements; e++) {
        for (int f = 0; f < mesh->Nfaces; f++) {
          const int bID = mesh->EToB[f + e * mesh->Nfaces];
          cds->solver[is]->EToB[f + e * mesh->Nfaces] = bcMap::ellipticType(bID, "scalar" + sid);
        }
      }

      ellipticSolveSetup(cds->solver[is]);
    }
  }

  if (this->flow) {
    if (platform->comm.mpiRank == 0) {
      printf("================ ELLIPTIC SETUP VELOCITY ================\n");
    }

    this->uvwSolver = NULL;

    bool unalignedBoundary = bcMap::unalignedMixedBoundary("velocity");

    nekrsCheck(unalignedBoundary && !options.compareArgs("VELOCITY BLOCK SOLVER", "TRUE"),
               platform->comm.mpiComm,
               EXIT_FAILURE,
               "%s\n",
               "SHL or unaligned SYM boundaries require solver = pcg+block");

    if (platform->options.compareArgs("VELOCITY BLOCK SOLVER", "TRUE")) {
      this->uvwSolver = new elliptic_t();
    }

    for (int bID = 1; bID <= bcMap::size("velocity"); bID++) {
      std::string bcTypeText(bcMap::text(bID, "velocity"));
      if (platform->comm.mpiRank == 0 && bcTypeText.size()) {
        printf("bID %d -> bcType %s\n", bID, bcTypeText.c_str());
      }
    }

    this->setEllipticCoeffKernel(mesh->Nlocal,
                                 this->g0 * this->idt,
                                 0 * this->fieldOffset,
                                 this->fieldOffset,
                                 0,
                                 this->o_mue,
                                 this->o_rho,
                                 o_NULL,
                                 this->o_ellipticCoeff);

    if (this->uvwSolver) {
      this->uvwSolver->name = "velocity";
      this->uvwSolver->stressForm = 0;
      if (options.compareArgs("VELOCITY STRESSFORMULATION", "TRUE")) {
        this->uvwSolver->stressForm = 1;
      }
      this->uvwSolver->Nfields = this->NVfields;
      this->uvwSolver->fieldOffset = this->fieldOffset;
      this->uvwSolver->mesh = mesh;
      this->uvwSolver->o_lambda0 = this->o_ellipticCoeff.slice(0 * this->fieldOffset);
      this->uvwSolver->o_lambda1 = this->o_ellipticCoeff.slice(1 * this->fieldOffset);
      this->uvwSolver->poisson = 0;
      this->uvwSolver->EToB =
          (int *)calloc(mesh->Nelements * mesh->Nfaces * this->uvwSolver->Nfields, sizeof(int));
      for (int fld = 0; fld < this->uvwSolver->Nfields; fld++) {
        std::string key;
        if (fld == 0) {
          key = "x-velocity";
        }
        if (fld == 1) {
          key = "y-velocity";
        }
        if (fld == 2) {
          key = "z-velocity";
        }
        for (dlong e = 0; e < mesh->Nelements; e++) {
          for (int f = 0; f < mesh->Nfaces; f++) {
            const int offset = fld * mesh->Nelements * mesh->Nfaces;
            const int bID = mesh->EToB[f + e * mesh->Nfaces];
            this->uvwSolver->EToB[f + e * mesh->Nfaces + offset] = bcMap::ellipticType(bID, key);
          }
        }
      }

      if (unalignedBoundary) {
        this->o_zeroNormalMaskVelocity =
            platform->device.malloc<dfloat>(this->uvwSolver->Nfields * this->uvwSolver->fieldOffset);
        this->o_EToBVVelocity = platform->device.malloc<int>(this->meshV->Nlocal);
        createEToBV(this->meshV, this->uvwSolver->EToB, this->o_EToBVVelocity);
        auto o_EToB = platform->device.malloc<int>(mesh->Nelements * mesh->Nfaces * this->uvwSolver->Nfields,
                                                   this->uvwSolver->EToB);
        createZeroNormalMask(this, mesh, o_EToB, this->o_EToBVVelocity, this->o_zeroNormalMaskVelocity);

        this->uvwSolver->applyZeroNormalMask =
            [this, mesh](dlong Nelements, const occa::memory &o_elementList, occa::memory &o_x) {
              applyZeroNormalMask(this,
                                  mesh,
                                  Nelements,
                                  o_elementList,
                                  this->uvwSolver->o_EToB,
                                  this->o_zeroNormalMaskVelocity,
                                  o_x);
            };
      }
      ellipticSolveSetup(this->uvwSolver);
    } else {
      this->uSolver = new elliptic_t();
      this->uSolver->name = "velocity";
      this->uSolver->Nfields = 1;
      this->uSolver->fieldOffset = this->fieldOffset;
      this->uSolver->mesh = mesh;
      this->uSolver->o_lambda0 = this->o_ellipticCoeff.slice(0 * this->fieldOffset);
      this->uSolver->o_lambda1 = this->o_ellipticCoeff.slice(1 * this->fieldOffset);
      this->uSolver->poisson = 0;
      this->uSolver->EToB = (int *)calloc(mesh->Nelements * mesh->Nfaces, sizeof(int));
      for (dlong e = 0; e < mesh->Nelements; e++) {
        for (int f = 0; f < mesh->Nfaces; f++) {
          const int bID = mesh->EToB[f + e * mesh->Nfaces];
          this->uSolver->EToB[f + e * mesh->Nfaces] = bcMap::ellipticType(bID, "x-velocity");
        }
      }

      ellipticSolveSetup(this->uSolver);

      this->vSolver = new elliptic_t();
      this->vSolver->name = "velocity";
      this->vSolver->Nfields = 1;
      this->vSolver->fieldOffset = this->fieldOffset;
      this->vSolver->mesh = mesh;
      this->vSolver->o_lambda0 = this->o_ellipticCoeff.slice(0 * this->fieldOffset);
      this->vSolver->o_lambda1 = this->o_ellipticCoeff.slice(1 * this->fieldOffset);
      this->vSolver->poisson = 0;
      this->vSolver->EToB = (int *)calloc(mesh->Nelements * mesh->Nfaces, sizeof(int));
      for (dlong e = 0; e < mesh->Nelements; e++) {
        for (int f = 0; f < mesh->Nfaces; f++) {
          const int bID = mesh->EToB[f + e * mesh->Nfaces];
          this->vSolver->EToB[f + e * mesh->Nfaces] = bcMap::ellipticType(bID, "y-velocity");
        }
      }

      ellipticSolveSetup(this->vSolver);

      this->wSolver = new elliptic_t();
      this->wSolver->name = "velocity";
      this->wSolver->Nfields = 1;
      this->wSolver->fieldOffset = this->fieldOffset;
      this->wSolver->mesh = mesh;
      this->wSolver->o_lambda0 = this->o_ellipticCoeff.slice(0 * this->fieldOffset);
      this->wSolver->o_lambda1 = this->o_ellipticCoeff.slice(1 * this->fieldOffset);
      this->wSolver->poisson = 0;
      this->wSolver->EToB = (int *)calloc(mesh->Nelements * mesh->Nfaces, sizeof(int));
      for (dlong e = 0; e < mesh->Nelements; e++) {
        for (int f = 0; f < mesh->Nfaces; f++) {
          const int bID = mesh->EToB[f + e * mesh->Nfaces];
          this->wSolver->EToB[f + e * mesh->Nfaces] = bcMap::ellipticType(bID, "z-velocity");
        }
      }

      ellipticSolveSetup(this->wSolver);
    }
  } // flow

  if (this->flow) {
    if (platform->comm.mpiRank == 0) {
      printf("================ ELLIPTIC SETUP PRESSURE ================\n");
    }

    this->pSolver = new elliptic_t();
    this->pSolver->name = "pressure";
    this->pSolver->Nfields = 1;
    this->pSolver->fieldOffset = this->fieldOffset;
    this->pSolver->mesh = mesh;

    this->pSolver->poisson = 1;

    // lambda0 = 1/rho  lambda1 = 0
    this->setEllipticCoeffPressureKernel(mesh->Nlocal, this->fieldOffset, this->o_rho, this->o_ellipticCoeff);

    this->pSolver->o_lambda0 = this->o_ellipticCoeff.slice(0 * this->fieldOffset);
    this->pSolver->lambda0Avg = platform->linAlg->innerProd(mesh->Nlocal,
                                                            mesh->o_LMM,
                                                            this->pSolver->o_lambda0,
                                                            platform->comm.mpiComm) /
                                mesh->volume;
    this->pSolver->o_lambda1 = this->o_ellipticCoeff.slice(1 * this->fieldOffset);

    this->pSolver->EToB = (int *)calloc(mesh->Nelements * mesh->Nfaces, sizeof(int));
    for (dlong e = 0; e < mesh->Nelements; e++) {
      for (int f = 0; f < mesh->Nfaces; f++) {
        const int bID = mesh->EToB[f + e * mesh->Nfaces];
        this->pSolver->EToB[f + e * mesh->Nfaces] = bcMap::ellipticType(bID, "pressure");
      }
    }

    ellipticSolveSetup(this->pSolver);

  } // flow

  if (!options.compareArgs("MESH SOLVER", "NONE")) {
    mesh_t *mesh = this->_mesh;

    if (platform->comm.mpiRank == 0) {
      printf("================ ELLIPTIC SETUP MESH ================\n");
    }

    const int nbrBIDs = bcMap::size("mesh");
    for (int bID = 1; bID <= nbrBIDs; bID++) {
      std::string bcTypeText(bcMap::text(bID, "mesh"));
      if (platform->comm.mpiRank == 0 && bcTypeText.size()) {
        printf("bID %d -> bcType %s\n", bID, bcTypeText.c_str());
      }
    }

    this->setEllipticCoeffKernel(mesh->Nlocal,
                                 1.0,
                                 0 * this->fieldOffset,
                                 this->fieldOffset,
                                 0,
                                 this->o_meshMue,
                                 this->o_meshRho,
                                 o_NULL,
                                 this->o_ellipticCoeff);

    this->meshSolver = new elliptic_t();
    this->meshSolver->name = "mesh";
    this->meshSolver->stressForm = 0;
    if (options.compareArgs("MESH STRESSFORMULATION", "TRUE")) {
      this->meshSolver->stressForm = 1;
    }
    this->meshSolver->Nfields = this->NVfields;
    this->meshSolver->fieldOffset = this->fieldOffset;
    this->meshSolver->mesh = mesh;
    this->meshSolver->o_lambda0 = this->o_ellipticCoeff.slice(0 * this->fieldOffset);
    this->meshSolver->o_lambda1 = this->o_ellipticCoeff.slice(1 * this->fieldOffset);
    this->meshSolver->poisson = 1;

    this->meshSolver->EToB =
        (int *)calloc(mesh->Nelements * mesh->Nfaces * this->meshSolver->Nfields, sizeof(int));
    for (int fld = 0; fld < this->meshSolver->Nfields; fld++) {
      std::string key;
      if (fld == 0) {
        key = "x-mesh";
      }
      if (fld == 1) {
        key = "y-mesh";
      }
      if (fld == 2) {
        key = "z-mesh";
      }
      for (dlong e = 0; e < mesh->Nelements; e++) {
        for (int f = 0; f < mesh->Nfaces; f++) {
          const int offset = fld * mesh->Nelements * mesh->Nfaces;
          const int bID = mesh->EToB[f + e * mesh->Nfaces];
          this->meshSolver->EToB[f + e * mesh->Nfaces + offset] = bcMap::ellipticType(bID, key);
        }
      }
    }

    bool unalignedBoundary = bcMap::unalignedMixedBoundary("mesh");
    if (unalignedBoundary) {
      this->o_zeroNormalMaskMeshVelocity =
          platform->device.malloc<dfloat>(this->meshSolver->Nfields * this->meshSolver->fieldOffset);
      this->o_EToBVMeshVelocity = platform->device.malloc<int>(mesh->Nlocal);
      auto o_EToB = platform->device.malloc<int>(mesh->Nelements * mesh->Nfaces * this->meshSolver->Nfields,
                                                 this->meshSolver->EToB);
      createEToBV(mesh, this->meshSolver->EToB, this->o_EToBVMeshVelocity);
      createZeroNormalMask(this, mesh, o_EToB, this->o_EToBVMeshVelocity, this->o_zeroNormalMaskMeshVelocity);
      this->meshSolver->applyZeroNormalMask =
          [this, mesh](dlong Nelements, const occa::memory &o_elementList, occa::memory &o_x) {
            applyZeroNormalMask(this,
                                mesh,
                                Nelements,
                                o_elementList,
                                this->meshSolver->o_EToB,
                                this->o_zeroNormalMaskMeshVelocity,
                                o_x);
          };
    }
    ellipticSolveSetup(this->meshSolver);
  }
}

void nrs_t::finalize()
{
  if (this->uSolver) {
    delete this->uSolver;
  }
  if (this->vSolver) {
    delete this->vSolver;
  }
  if (this->wSolver) {
    delete this->wSolver;
  }
  if (this->uvwSolver) {
    delete this->uvwSolver;
  }
  if (this->pSolver) {
    delete this->pSolver;
  }
  for (int is; is < this->Nscalar; is++) {
    if (this->cds->solver[is]) {
      delete this->cds->solver[is];
    }
  }
  if (this->cvode) {
    delete this->cvode;
  }
  if (this->meshSolver) {
    delete this->meshSolver;
  }
}
