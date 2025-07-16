#include "platform.hpp"
#include "elliptic.h"
#include "elliptic.hpp"
#include "ellipticPrecon.h"
#include "maskedFaceIds.hpp"

elliptic::elliptic(const std::string &name,
                   mesh_t *mesh,
                   dlong fieldOffset,
                   const std::vector<int> &EToBIn,
                   const occa::memory &o_lambda0,
                   const occa::memory &o_lambda1)
{
  solver = new elliptic_t();
  solver->name = name;
  solver->mesh = mesh;

  _EToB = EToBIn;
  solver->EToB = _EToB.data();

  solver->fieldOffset = (fieldOffset <= 0) ? alignStride<dfloat>(mesh->Nlocal) : fieldOffset;
  _setup(o_lambda0, o_lambda1);
}

elliptic::~elliptic()
{
  if (solver->precon) {
    delete solver->precon;
  }
  solver->o_EToB.free();
}

void elliptic::updatePreconditioner()
{
  MPI_Barrier(platform->comm.mpiComm);
  auto tStart = MPI_Wtime();
  if (platform->comm.mpiRank == 0) {
    printf("updating preconditioner for %s ... \n", solver->name.c_str());
  }
  auto precon = solver->precon;

  if (solver->options.compareArgs("PRECONDITIONER", "JACOBI")) {
    ellipticUpdateJacobi(solver);
  } else if (solver->options.compareArgs("PRECONDITIONER", "MULTIGRID")) {
    MGSolver_t::multigridLevel** levels = precon->MGSolver->levels;
    auto elliptic = dynamic_cast<pMGLevel*>(levels[0])->elliptic;                            

    if (solver->options.compareArgs("MULTIGRID SMOOTHER", "DAMPEDJACOBI")) {
      ellipticMultiGridUpdateLambda(solver);
    } else if (solver->options.compareArgs("MULTIGRID SMOOTHER", "ASM") || solver->options.compareArgs("MULTIGRID SMOOTHER", "RAS")) {
      for (int n = 0; n < elliptic->nLevels - 1; n++) {
        auto level = dynamic_cast<pMGLevel*>(levels[n]);
        level->updateSmootherSchwarz(solver);
      }
    }

    if (solver->options.compareArgs("MULTIGRID SMOOTHER", "CHEBYSHEV")) {
      for (int n = 0; n < elliptic->nLevels - 1; n++) {
        auto level = dynamic_cast<pMGLevel*>(levels[n]);
        level->updateSetupSmootherChebyshev();
      }
    }

    if (solver->options.compareArgs("MULTIGRID COARSE SOLVE", "TRUE")) {
      ellipticCoarseGridSetup(elliptic, 1);
    }
  }

  MPI_Barrier(platform->comm.mpiComm);
  if (platform->comm.mpiRank == 0) {
    printf("done (%gs)\n", MPI_Wtime() - tStart);
  }
  fflush(stdout);
}

void elliptic::solve(const occa::memory &o_lambda0,
                     const occa::memory &o_lambda1,
                     const occa::memory &RHS,
                     occa::memory x)
{
  _solve(o_lambda0, o_lambda1, RHS, x);
};

std::string &elliptic::name() const
{
  return solver->name;
}

setupAide &elliptic::options()
{
  return solver->options;
}

int elliptic::Niter() const
{
  return solver->Niter;
};

void elliptic::Niter(int val)
{
  solver->Niter = val;
};

dlong elliptic::fieldOffset() const
{
  return solver->fieldOffset;
};

bool elliptic::nullSpace() const
{
  return solver->nullspace;
};

dfloat elliptic::initialResidual() const
{
  return solver->res00Norm;
};

void elliptic::initialResidual(dfloat val)
{
  solver->res00Norm = val;
};

dfloat elliptic::initialGuessResidual() const
{
  return solver->res0Norm;
};

void elliptic::initialGuessResidual(dfloat val)
{
  solver->res0Norm = val;
};

dfloat elliptic::finalResidual() const
{
  return solver->resNorm;
};

void elliptic::finalResidual(dfloat val)
{
  solver->resNorm = val;
};

dlong elliptic::Nmasked()
{
  return solver->Nmasked;
};

occa::memory elliptic::o_maskIds() const
{
  return solver->o_maskIds;
};

occa::memory elliptic::o_EToB() const
{
  return solver->o_EToB;
};

std::vector<int> elliptic::EToB() const
{
  return _EToB;
};

int elliptic::Nfields() const
{
  return solver->Nfields;
};

void elliptic::applyZeroNormalMask(
    const std::function<void(dlong Nelements, const occa::memory &o_elementList, occa::memory &o_x)> &f)
{
  solver->applyZeroNormalMask = f;
};

void elliptic::userPreconditioner(const std::function<void(const occa::memory &o_r, occa::memory &o_z)> &f)
{
  solver->userPreconditioner = f;
};

std::tuple<int, int> elliptic::projectionCounters() const
{
  if (solver->solutionProjection) {
    return {solver->solutionProjection->getPrevNumVecsProjection(),
            solver->solutionProjection->getMaxNumVecsProjection()};
  }
  return {0, 0};
};

void elliptic::_solve(const occa::memory &o_lambda0,
                      const occa::memory &o_lambda1,
                      const occa::memory &o_rhs,
                      occa::memory o_x)
{
  auto &elliptic = this->solver;

  elliptic->o_lambda0 = o_lambda0;
  elliptic->o_lambda1 = o_lambda1;

  auto &options = elliptic->options;
  auto &precon = elliptic->precon;
  auto &mesh = elliptic->mesh;

  int maxIter = 999;
  options.getArgs("MAXIMUM ITERATIONS", maxIter);

  const int verbose = platform->verbose();
  const auto movingMesh = platform->options.compareArgs("MOVING MESH", "TRUE");

  auto printNorm = [&](const occa::memory &o_u, const std::string &txt) {
    const dfloat norm = platform->linAlg->weightedNorm2Many(mesh->Nlocal,
                                                            elliptic->Nfields,
                                                            elliptic->fieldOffset,
                                                            elliptic->o_invDegree,
                                                            o_u,
                                                            platform->comm.mpiComm);
    if (platform->comm.mpiRank == 0) {
      printf("%s %s norm: %.15e\n", elliptic->name.c_str(), txt.c_str(), norm);
    }
    nekrsCheck(std::isnan(norm),
               MPI_COMM_SELF,
               EXIT_FAILURE,
               "%s unreasonable %s!\n",
               elliptic->name.c_str(),
               txt.c_str());
  };

  auto o_x0 = platform->deviceMemoryPool.reserve<dfloat>(
      (elliptic->Nfields > 1) ? elliptic->Nfields * elliptic->fieldOffset : mesh->Nlocal);
  nekrsCheck(o_x.size() < o_x0.size(), MPI_COMM_SELF, EXIT_FAILURE, "%s!\n", "unreasonable size of o_x");
  nekrsCheck(o_rhs.size() < o_x.size(), MPI_COMM_SELF, EXIT_FAILURE, "%s!\n", "unreasonable size of o_rhs");

  auto updateResidualWeight = [&]() {
    if (platform->options.compareArgs("LINEAR SOLVER STOPPING CRITERION TYPE", "LEGACY")) {
      if (!elliptic->o_residualWeight.isInitialized()) {
        elliptic->o_residualWeight = platform->device.malloc<dfloat>(mesh->Nlocal);
      }
      elliptic->o_residualWeight.copyFrom(elliptic->o_invDegree);
//      platform->linAlg->scale(mesh->Nlocal, 1 / mesh->volume, elliptic->o_residualWeight);
      platform->linAlg->scale(mesh->Nlocal, 1.0, elliptic->o_residualWeight);

    } else if (platform->options.compareArgs("LINEAR SOLVER STOPPING CRITERION TYPE", "l2_RESIDUAL")) {
      if (!elliptic->o_residualWeight.isInitialized()) {
        elliptic->o_residualWeight = platform->device.malloc<dfloat>(mesh->Nlocal);
      }
      auto Nglobal = mesh->NelementsGlobal * mesh->Np;
      platform->linAlg->axmyz(mesh->Nlocal,
                              1. / Nglobal,
                              mesh->o_invAJw,
                              mesh->o_invAJw,
                              elliptic->o_residualWeight);
      platform->linAlg->axmy(mesh->Nlocal, 1.0, elliptic->o_invDegree, elliptic->o_residualWeight);

#if 0
      nekrsCheck(elliptic->nullspace,
                 MPI_COMM_SELF,
                 EXIT_FAILURE,
                 "%s\n",
                 "STOPPING CRITERION l2_RESIDUAL is currently unsupported if there is a non-trival nullspace");
#endif
    } else if (platform->options.compareArgs("LINEAR SOLVER STOPPING CRITERION TYPE", "L2_RESIDUAL")) {
      elliptic->o_residualWeight = mesh->o_invAJwTimesInvDegree;
    } else {
      const auto txt = platform->options.getArgs("LINEAR SOLVER STOPPING CRITERION TYPE");
      nekrsAbort(MPI_COMM_SELF,
                 EXIT_FAILURE,
                 "%s <%s>\n",
                 "Invalid LINEAR SOLVER STOPPING CRITERION TYPE",
                 txt.c_str());
    }
  };

  if (!elliptic->o_residualWeight.isInitialized()) {
    updateResidualWeight();
  } else if (movingMesh) {
    updateResidualWeight();
  }

  std::string timerName = elliptic->name;
  if (timerName.find("scalar") != std::string::npos) {
    timerName = "scalar";
  }

  ellipticAllocateWorkspace(elliptic);

  if (options.compareArgs("ELLIPTIC PRECO COEFF FIELD", "TRUE")) {
    if (options.compareArgs("PRECONDITIONER", "MULTIGRID")) {
      ellipticMultiGridUpdateLambda(elliptic);
    }

    if (options.compareArgs("PRECONDITIONER", "JACOBI") ||
        options.compareArgs("MULTIGRID SMOOTHER", "DAMPEDJACOBI")) {
      ellipticUpdateJacobi(elliptic);
    }
  }

  o_x0.copyFrom(o_x);
  if (platform->verbose()) {
    printNorm(o_x0, "o_x0");
    printNorm(o_rhs, "o_rhs");
  }

  // compute initial residual r = rhs - Ax0
  auto o_r = [&]() {
    auto &o_Ap = o_x;
    ellipticAx(elliptic, mesh->Nelements, mesh->o_elementList, o_x0, o_Ap, dfloatString);

    auto o_r = platform->deviceMemoryPool.reserve<dfloat>(o_x0.size());
    platform->linAlg
        ->axpbyzMany(mesh->Nlocal, elliptic->Nfields, elliptic->fieldOffset, -1.0, o_Ap, 1.0, o_rhs, o_r);

    if (elliptic->nullspace) {
      ellipticZeroMean(elliptic, o_r);
    }
    ellipticApplyMask(elliptic, o_r, dfloatString);
    oogs::startFinish(o_r, elliptic->Nfields, elliptic->fieldOffset, ogsDfloat, ogsAdd, elliptic->oogs);

    return o_r;
  }();

  if (platform->verbose()) {
    printNorm(o_r, "o_r");
  }

  const auto rdotr = [&]() {
    return platform->linAlg->weightedNorm2Many(mesh->Nlocal,
                                               elliptic->Nfields,
                                               elliptic->fieldOffset,
                                               elliptic->o_residualWeight,
                                               o_r,
                                               platform->comm.mpiComm);
  };

  if (options.compareArgs("INITIAL GUESS", "PROJECTION") ||
      options.compareArgs("INITIAL GUESS", "PROJECTION-ACONJ")) {

    platform->timer.tic(timerName + " proj pre", 1);
    elliptic->res00Norm = rdotr();
    nekrsCheck(std::isnan(elliptic->res00Norm),
               MPI_COMM_SELF,
               EXIT_FAILURE,
               "%s unreasonable res00Norm!\n",
               elliptic->name.c_str());

    elliptic->solutionProjection->pre(o_r);

    platform->timer.toc(timerName + " proj pre");
  }

  elliptic->res0Norm = rdotr();
  nekrsCheck(std::isnan(elliptic->res0Norm),
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "%s unreasonable res00Norm!\n",
             elliptic->name.c_str());

  // linear solve
  {
    // absolute tol
    dfloat tol = 1e-6;
    options.getArgs("SOLVER TOLERANCE", tol);

    // absolute tol + relative
    if (!options.getArgs("SOLVER RELATIVE TOLERANCE").empty()) {
      dfloat relTol;
      options.getArgs("SOLVER RELATIVE TOLERANCE", relTol);
      tol = std::max(relTol * elliptic->res0Norm, tol);
    } else { // relative and absolute tolerance are the same
      if (options.compareArgs("LINEAR SOLVER STOPPING CRITERION", "RELATIVE")) {
        tol *= elliptic->res0Norm;
      }
    }

    elliptic->resNorm = elliptic->res0Norm;
    platform->linAlg->fill(o_x.size(), 0.0, o_x);

    if (options.compareArgs("SOLVER", "PCG")) {
      elliptic->Niter = pcg(elliptic, tol, maxIter, elliptic->resNorm, o_r, o_x);
    } else if (options.compareArgs("SOLVER", "PGMRES")) {
      elliptic->Niter = pgmres(elliptic, tol, maxIter, elliptic->resNorm, o_r, o_x);
    } else {
      nekrsAbort(platform->comm.mpiComm,
                 EXIT_FAILURE,
                 "Unknown linear solver %s!\n",
                 options.getArgs("SOLVER").c_str());
    }

    if (elliptic->Niter == maxIter && platform->comm.mpiRank == 0) {
      printf("iteration limit of %s linear solver reached!\n", elliptic->name.c_str());
    }
  }

  if (options.compareArgs("INITIAL GUESS", "PROJECTION") ||
      options.compareArgs("INITIAL GUESS", "PROJECTION-ACONJ")) {
    platform->timer.tic(timerName + " proj post", 1);
    elliptic->solutionProjection->post(o_x);
    platform->timer.toc(timerName + " proj post");
  } else {
    elliptic->res00Norm = elliptic->res0Norm;
  }

  platform->linAlg->axpbyMany(mesh->Nlocal, elliptic->Nfields, elliptic->fieldOffset, 1.0, o_x0, 1.0, o_x);

  if (elliptic->nullspace) {
    ellipticZeroMean(elliptic, o_x);
  }

  elliptic->o_lambda0 = nullptr;
  elliptic->o_lambda1 = nullptr;
  ellipticFreeWorkspace(elliptic);
}

void checkConfig(elliptic_t *elliptic)
{
  mesh_t *mesh = elliptic->mesh;
  setupAide &options = elliptic->options;

  int err = 0;

  if (!options.compareArgs("DISCRETIZATION", "CONTINUOUS")) {
    if (platform->comm.mpiRank == 0) {
      printf("solver only supports CG\n");
    }
    err++;
  }

  if (elliptic->elementType != HEXAHEDRA) {
    if (platform->comm.mpiRank == 0) {
      printf("solver only supports HEX elements\n");
    }
    err++;
  }

  if (elliptic->blockSolver && options.compareArgs("PRECONDITIONER", "MULTIGRID")) {
    if (platform->comm.mpiRank == 0) {
      printf("Block solver does not support multigrid preconditioner\n");
    }
    err++;
  }

  if (!elliptic->poisson && options.compareArgs("PRECONDITIONER", "MULTIGRID") &&
      !options.compareArgs("MULTIGRID SMOOTHER", "DAMPEDJACOBI")) {
    if (platform->comm.mpiRank == 0) {
      printf("Non-Poisson type equations require Jacobi multigrid smoother\n");
    }
    err++;
  }

  if (options.compareArgs("SOLVER", "PCG+COMBINED") && !options.compareArgs("PRECONDITIONER", "JACO")) {
    if (platform->comm.mpiRank == 0) {
      printf("combinedPCG requires Jacobi preconditioner!\n");
    }
    err++;
  }

  if (elliptic->mesh->ogs == NULL) {
    if (platform->comm.mpiRank == 0) {
      printf("mesh->ogs == NULL!");
    }
    err++;
  }

  if (elliptic->Nfields < 1 || elliptic->Nfields > 3) {
    if (platform->comm.mpiRank == 0) {
      printf("Invalid Nfields = %d!", elliptic->Nfields);
    }
    err++;
  }

  nekrsCheck(elliptic->EToB == nullptr,
             platform->comm.mpiComm,
             EXIT_FAILURE,
             "%s",
             "elliptic->EToB not allocated!\n");

  {
    int found = 0;
    for (int fld = 0; fld < elliptic->Nfields; fld++) {
      for (dlong e = 0; e < mesh->Nelements; e++) {
        for (int f = 0; f < mesh->Nfaces; f++) {
          const int offset = fld * mesh->Nelements * mesh->Nfaces;
          const int bc = elliptic->EToB[f + e * mesh->Nfaces + offset];
          if (bc == ellipticBcType::ZERO_NORMAL || bc == ellipticBcType::ZERO_TANGENTIAL) {
            found = 1;
          }
        }
      }
    }
    MPI_Allreduce(MPI_IN_PLACE, &found, 1, MPI_INT, MPI_MAX, platform->comm.mpiComm);
    if (found && !elliptic->blockSolver) {
      if (platform->comm.mpiRank == 0) {
        printf("Unaligned BCs require block solver!\n");
      }
      err++;
    }
  }

  nekrsCheck(err, platform->comm.mpiComm, EXIT_FAILURE, "%s", "\n");
}

void elliptic::_setup(const occa::memory &o_lambda0, const occa::memory &o_lambda1)
{
  auto &elliptic = solver;

  MPI_Barrier(platform->comm.mpiComm);
  const double tStart = MPI_Wtime();

  nekrsCheck(elliptic->name.size() == 0,
             platform->comm.mpiComm,
             EXIT_FAILURE,
             "%s\n",
             "Empty elliptic solver name!");

  elliptic->o_lambda0 = o_lambda0;
  elliptic->o_lambda1 = o_lambda1;

  elliptic->lambda0Avg = platform->linAlg->innerProd(elliptic->mesh->Nlocal,
                                                     elliptic->mesh->o_LMM,
                                                     elliptic->o_lambda0,
                                                     platform->comm.mpiComm) /
                         elliptic->mesh->volume;

  if (elliptic->o_lambda1.isInitialized()) {
    elliptic->lambda1Avg = platform->linAlg->innerProd(elliptic->mesh->Nlocal,
                                                       elliptic->mesh->o_LMM,
                                                       elliptic->o_lambda1,
                                                       platform->comm.mpiComm) /
                           elliptic->mesh->volume;
  }

  nekrsCheck(!std::isnormal(elliptic->lambda0Avg) || elliptic->lambda0Avg == 0,
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "unreasonable lambda0Avg=%g!\n",
             elliptic->lambda0Avg);

  elliptic->poisson = (elliptic->o_lambda1.isInitialized()) ? 0 : 1;

  platform->options.getArgs("ELEMENT TYPE", elliptic->elementType);
  elliptic->options.setArgs("DISCRETIZATION", "CONTINUOUS");

  // create private options based on platform
  for (auto &entry : platform->options.keyWordToDataMap) {
    std::string prefix = upperCase(elliptic->name);
    if (entry.first.find(prefix) == 0) {
      std::string key = entry.first;
      key.erase(0, prefix.size() + 1);
      elliptic->options.setArgs(key, entry.second);
    }
  }

  if (platform->device.mode() == "Serial" && solver->options.compareArgs("MULTIGRID COARSE SOLVE", "TRUE")) {
    elliptic->options.setArgs("COARSE SOLVER LOCATION", "CPU");
  }

  if (platform->comm.mpiRank == 0 && platform->verbose()) {
    std::cout << elliptic->options << std::endl;
  }

  elliptic->stressForm = 0;
  if (elliptic->options.compareArgs("STRESSFORMULATION", "TRUE")) {
    elliptic->stressForm = 1;
  }

  elliptic->Nfields = 1;
  elliptic->options.getArgs("NFIELDS", elliptic->Nfields);
  elliptic->blockSolver = elliptic->Nfields > 1;

  setupAide &options = elliptic->options;
  const int verbose = platform->verbose() ? 1 : 0;

  mesh_t *mesh = elliptic->mesh;
  const dlong Nlocal = mesh->Np * mesh->Nelements;

  const dlong Nblocks = (Nlocal + BLOCKSIZE - 1) / BLOCKSIZE;

  elliptic->o_EToB =
      platform->device.malloc<int>(mesh->Nelements * mesh->Nfaces * elliptic->Nfields, elliptic->EToB);

  checkConfig(elliptic);

  if (options.compareArgs("SOLVER", "PGMRES")) {
    initializeGmresData(elliptic);
    const std::string sectionIdentifier = std::to_string(elliptic->Nfields) + "-";
    elliptic->gramSchmidtOrthogonalizationKernel =
        platform->kernelRequests.load(sectionIdentifier + "gramSchmidtOrthogonalization");
    elliptic->updatePGMRESSolutionKernel =
        platform->kernelRequests.load(sectionIdentifier + "updatePGMRESSolution");
    elliptic->fusedResidualAndNormKernel =
        platform->kernelRequests.load(sectionIdentifier + "fusedResidualAndNorm");
  }

  if (options.compareArgs("SOLVER", "PCG+COMBINED")) {
    const std::string sectionIdentifier = std::to_string(elliptic->Nfields) + "-";
    elliptic->combinedPCGPreMatVecKernel =
        platform->kernelRequests.load(sectionIdentifier + "combinedPCGPreMatVec");
    elliptic->combinedPCGPostMatVecKernel =
        platform->kernelRequests.load(sectionIdentifier + "combinedPCGPostMatVec");
    elliptic->combinedPCGUpdateConvergedSolutionKernel =
        platform->kernelRequests.load(sectionIdentifier + "combinedPCGUpdateConvergedSolution");
  }

  int Nreductions = 1;
  if (options.compareArgs("SOLVER", "PCG+COMBINED")) {
    Nreductions = CombinedPCGId::nReduction;
  }

  elliptic->nullspace = 0;
  if (elliptic->poisson) {
    int nullspace = 1;

    // check based on BC
    for (int fld = 0; fld < elliptic->Nfields; fld++) {
      for (dlong e = 0; e < mesh->Nelements; e++) {
        for (int f = 0; f < mesh->Nfaces; f++) {
          const int offset = fld * mesh->Nelements * mesh->Nfaces;
          const int bc = elliptic->EToB[f + e * mesh->Nfaces + offset];
          if (bc > 0 && bc != ellipticBcType::NEUMANN) {
            nullspace = 0;
          }
        }
      }
    }
    MPI_Allreduce(MPI_IN_PLACE, &nullspace, 1, MPI_INT, MPI_MIN, platform->comm.mpiComm);
    elliptic->nullspace = nullspace;
    if (platform->comm.mpiRank == 0 && elliptic->nullspace) {
      printf("non-trivial nullSpace detected\n");
    }
  }

  { // setup masked gs handle
    ogs_t *ogs = (elliptic->blockSolver) ? mesh->ogs : nullptr;
    const auto [Nmasked, o_maskIds, NmaskedLocal, o_maskIdsLocal, NmaskedGlobal, o_maskIdsGlobal] =
        maskedFaceIds(mesh,
                      elliptic->fieldOffset,
                      elliptic->Nfields,
                      elliptic->fieldOffset,
                      elliptic->EToB,
                      ellipticBcType::DIRICHLET);

    elliptic->Nmasked = Nmasked;
    elliptic->o_maskIds = o_maskIds;
    elliptic->NmaskedLocal = NmaskedLocal;
    elliptic->o_maskIdsLocal = o_maskIdsLocal;
    elliptic->NmaskedGlobal = NmaskedGlobal;
    elliptic->o_maskIdsGlobal = o_maskIdsGlobal;

    if (!ogs) {
      nekrsCheck(elliptic->Nfields > 1,
                 platform->comm.mpiComm,
                 EXIT_FAILURE,
                 "%s\n",
                 "Creating a masked gs handle for nFields > 1 is currently not supported!");

      std::vector<hlong> maskedGlobalIds(mesh->Nlocal);
      memcpy(maskedGlobalIds.data(), mesh->globalIds, mesh->Nlocal * sizeof(hlong));
      std::vector<dlong> maskIds(Nmasked);
      o_maskIds.copyTo(maskIds.data());
      for (dlong n = 0; n < Nmasked; n++) {
        maskedGlobalIds[maskIds[n]] = 0;
      }
      ogs = ogsSetup(mesh->Nlocal,
                     maskedGlobalIds.data(),
                     platform->comm.mpiComm,
                     1,
                     platform->device.occaDevice());
    }
    elliptic->ogs = ogs;
    elliptic->o_invDegree = elliptic->ogs->o_invDegree;
  }

  {
    std::string kernelName;
    const std::string suffix = "Hex3D";
    const std::string sectionIdentifier = std::to_string(elliptic->Nfields) + "-";
    const std::string poissonPrefix = elliptic->poisson ? "poisson-" : "";

    if (options.compareArgs("PRECONDITIONER", "JACOBI")) {
      kernelName = "ellipticBlockBuildDiagonal" + suffix;
      elliptic->ellipticBlockBuildDiagonalKernel = platform->kernelRequests.load(poissonPrefix + kernelName);
    }

    kernelName = "fusedCopyDfloatToPfloat";
    elliptic->fusedCopyDfloatToPfloatKernel = platform->kernelRequests.load(kernelName);

    std::string kernelNamePrefix = poissonPrefix;
    kernelNamePrefix += "elliptic";
    if (elliptic->blockSolver) {
      kernelNamePrefix += (elliptic->stressForm) ? "Stress" : "Block";
    }

    kernelName = "AxCoeff";
    if (platform->options.compareArgs("ELEMENT MAP", "TRILINEAR")) {
      kernelName += "Trilinear";
    }
    kernelName += suffix;

    elliptic->AxKernel = platform->kernelRequests.load(kernelNamePrefix + "Partial" + kernelName);

    elliptic->updatePCGKernel = platform->kernelRequests.load(sectionIdentifier + "ellipticBlockUpdatePCG");
  }

  oogs_mode oogsMode = OOGS_AUTO;
  elliptic->oogs =
      oogs::setup(elliptic->ogs, elliptic->Nfields, elliptic->fieldOffset, ogsDfloat, NULL, oogsMode);
  elliptic->oogsAx = elliptic->oogs;

  if (platform->options.compareArgs("ENABLE GS COMM OVERLAP", "TRUE")) {
    const auto Nlocal = elliptic->Nfields * static_cast<size_t>(elliptic->fieldOffset);
    auto o_p = platform->deviceMemoryPool.reserve<dfloat>(Nlocal);
    auto o_Ap = platform->deviceMemoryPool.reserve<dfloat>(Nlocal);

    auto timeEllipticOperator = [&]() {
      const int Nsamples = 10;
      ellipticOperator(elliptic, o_p, o_Ap, dfloatString);

      platform->device.finish();
      MPI_Barrier(platform->comm.mpiComm);
      const double start = MPI_Wtime();

      for (int test = 0; test < Nsamples; ++test) {
        ellipticOperator(elliptic, o_p, o_Ap, dfloatString);
      }

      platform->device.finish();
      double elapsed = (MPI_Wtime() - start) / Nsamples;
      MPI_Allreduce(MPI_IN_PLACE, &elapsed, 1, MPI_DOUBLE, MPI_MAX, platform->comm.mpiComm);

      return elapsed;
    };

    auto nonOverlappedTime = timeEllipticOperator();
    auto callback = [&]() {
      ellipticAx(elliptic,
                 mesh->NlocalGatherElements,
                 mesh->o_localGatherElementList,
                 o_p,
                 o_Ap,
                 dfloatString);
    };
    elliptic->oogsAx =
        oogs::setup(elliptic->ogs, elliptic->Nfields, elliptic->fieldOffset, ogsDfloat, callback, oogsMode);

    auto overlappedTime = timeEllipticOperator();
    if (overlappedTime > nonOverlappedTime) {
      elliptic->oogsAx = elliptic->oogs;
    }

    if (platform->comm.mpiRank == 0) {
      printf("testing Ax overlap %.2es %.2es ", nonOverlappedTime, overlappedTime);
      if (elliptic->oogsAx != elliptic->oogs) {
        printf("(overlap enabled)");
      }

      printf("\n");
    }
  }

  ellipticPreconditionerSetup(elliptic, elliptic->ogs);

  if (options.compareArgs("INITIAL GUESS", "PROJECTION") ||
      options.compareArgs("INITIAL GUESS", "PROJECTION-ACONJ")) {
    dlong nVecsProject = 8;
    options.getArgs("RESIDUAL PROJECTION VECTORS", nVecsProject);

    dlong nStepsStart = 5;
    options.getArgs("RESIDUAL PROJECTION START", nStepsStart);

    SolutionProjection::ProjectionType type = SolutionProjection::ProjectionType::CLASSIC;
    if (options.compareArgs("INITIAL GUESS", "PROJECTION-ACONJ")) {
      type = SolutionProjection::ProjectionType::ACONJ;
    } else if (options.compareArgs("INITIAL GUESS", "PROJECTION")) {
      type = SolutionProjection::ProjectionType::CLASSIC;
    }

    elliptic->solutionProjection = new SolutionProjection(*elliptic, type, nVecsProject, nStepsStart);
  }

  elliptic->o_lambda0 = nullptr;
  elliptic->o_lambda1 = nullptr;

  MPI_Barrier(platform->comm.mpiComm);
  if (platform->comm.mpiRank == 0) {
    printf("done (%gs)\n", MPI_Wtime() - tStart);
  }
  fflush(stdout);
}
