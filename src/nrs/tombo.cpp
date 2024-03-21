#include "nrs.hpp"
#include "udf.hpp"
#include "linAlg.hpp"
#include "neknek.hpp"
#include "bdry.hpp"
#include "bcMap.hpp"
#include <limits>

namespace tombo
{
occa::memory pressureSolve(nrs_t *nrs, double time, int stage)
{
  auto mesh = nrs->meshV;

  double flopCount = 0.0;
  platform->timer.tic("pressure rhs", 1);

  occa::memory o_stressTerm = platform->o_memPool.reserve<dfloat>(nrs->NVfields * nrs->fieldOffset);
  {
    occa::memory o_curl = platform->o_memPool.reserve<dfloat>(nrs->NVfields * nrs->fieldOffset);

    nrs->curlKernel(mesh->Nelements, 1, mesh->o_vgeo, mesh->o_D, nrs->fieldOffset, nrs->o_Ue, o_curl);
    flopCount += static_cast<double>(mesh->Nelements) * (18 * mesh->Np * mesh->Nq + 36 * mesh->Np);

    oogs::startFinish(o_curl, nrs->NVfields, nrs->fieldOffset, ogsDfloat, ogsAdd, nrs->gsh);

    platform->linAlg->axmyVector(mesh->Nlocal, nrs->fieldOffset, 0, 1.0, nrs->meshV->o_invLMM, o_curl);
    flopCount += mesh->Nlocal;

    nrs->curlKernel(mesh->Nelements, 1, mesh->o_vgeo, mesh->o_D, nrs->fieldOffset, o_curl, o_stressTerm);
    flopCount += static_cast<double>(mesh->Nelements) * (18 * mesh->Np * mesh->Nq + 36 * mesh->Np);
  }

  if (platform->options.compareArgs("VELOCITY STRESSFORMULATION", "TRUE")) {
    nrs->pressureStressKernel(mesh->Nelements,
                              mesh->o_vgeo,
                              mesh->o_D,
                              nrs->fieldOffset,
                              nrs->o_mue,
                              nrs->o_Ue,
                              nrs->o_div,
                              o_stressTerm);
    flopCount += static_cast<double>(mesh->Nelements) * (18 * mesh->Nq * mesh->Np + 100 * mesh->Np);
  }

  occa::memory o_gradDiv = platform->o_memPool.reserve<dfloat>(nrs->NVfields * nrs->fieldOffset);
  nrs->gradientVolumeKernel(mesh->Nelements,
                            mesh->o_vgeo,
                            mesh->o_D,
                            nrs->fieldOffset,
                            nrs->o_div,
                            o_gradDiv);
  flopCount += static_cast<double>(mesh->Nelements) * (6 * mesh->Np * mesh->Nq + 18 * mesh->Np);

  auto o_rhs = platform->o_memPool.reserve<dfloat>(nrs->NVfields * nrs->fieldOffset);

  auto o_lambda0 = platform->o_memPool.reserve<dfloat>(mesh->Nlocal);
  platform->linAlg->adyz(mesh->Nlocal, 1.0, nrs->o_rho, o_lambda0);

  if (platform->options.compareArgs("PRESSURE VISCOUS TERMS", "TRUE")) {
    nrs->pressureRhsKernel(mesh->Nlocal,
                           nrs->fieldOffset,
                           nrs->o_mue,
                           o_lambda0, /* 1/rho */
                           nrs->o_BF,
                           o_stressTerm,
                           o_gradDiv,
                           o_rhs);
    flopCount += 12 * static_cast<double>(mesh->Nlocal);
  } else {
    o_rhs.copyFrom(nrs->o_BF);

    auto o_BFx = nrs->o_BF.slice(0 * nrs->fieldOffset, mesh->Nlocal);
    auto o_BFy = nrs->o_BF.slice(1 * nrs->fieldOffset, mesh->Nlocal);
    auto o_BFz = nrs->o_BF.slice(2 * nrs->fieldOffset, mesh->Nlocal);

    platform->linAlg->axmy(mesh->Nlocal, 1.0, o_lambda0, o_BFx);
    platform->linAlg->axmy(mesh->Nlocal, 1.0, o_lambda0, o_BFy);
    platform->linAlg->axmy(mesh->Nlocal, 1.0, o_lambda0, o_BFz);
  }

  oogs::startFinish(o_rhs, nrs->NVfields, nrs->fieldOffset, ogsDfloat, ogsAdd, nrs->gsh);

  platform->linAlg->axmyVector(mesh->Nlocal, nrs->fieldOffset, 0, 1.0, nrs->meshV->o_invLMM, o_rhs);

  occa::memory o_pRhs = platform->o_memPool.reserve<dfloat>(nrs->fieldOffset);

  nrs->wDivergenceVolumeKernel(mesh->Nelements, mesh->o_vgeo, mesh->o_D, nrs->fieldOffset, o_rhs, o_pRhs);
  flopCount += static_cast<double>(mesh->Nelements) * (6 * mesh->Np * mesh->Nq + 18 * mesh->Np);

  // now adding to o_rhs

  nrs->pressureAddQtlKernel(mesh->Nlocal, mesh->o_LMM, nrs->g0 * 1 / nrs->dt[0], nrs->o_div, o_pRhs);
  flopCount += 3 * mesh->Nlocal;

  nrs->divergenceSurfaceKernel(mesh->Nelements,
                               mesh->o_sgeo,
                               mesh->o_vmapM,
                               nrs->o_EToB,
                               nrs->g0 * 1 / nrs->dt[0],
                               nrs->fieldOffset,
                               o_rhs,
                               nrs->o_U,
                               o_pRhs);
  flopCount += 25 * static_cast<double>(mesh->Nelements) * mesh->Nq * mesh->Nq;

  platform->timer.toc("pressure rhs");
  platform->flopCounter->add("pressure RHS", flopCount);

  occa::memory o_p = platform->o_memPool.reserve<dfloat>(nrs->fieldOffset);
  o_p.copyFrom(nrs->o_P);

  nrs->pSolver->solve(o_lambda0, o_NULL, static_cast<const occa::memory>(o_pRhs), o_p);

  if (platform->verbose) {
    const dfloat debugNorm = platform->linAlg->weightedNorm2Many(mesh->Nlocal,
                                                                 1,
                                                                 nrs->fieldOffset,
                                                                 mesh->ogs->o_invDegree,
                                                                 o_p,
                                                                 platform->comm.mpiComm);
    if (platform->comm.mpiRank == 0) {
      printf("p norm: %.15e\n", debugNorm);
    }
  }

  return o_p;
}

occa::memory velocitySolve(nrs_t *nrs, double time, int stage)
{
  auto mesh = nrs->meshV;

  occa::memory o_gradMueDiv = platform->o_memPool.reserve<dfloat>(nrs->NVfields * nrs->fieldOffset);
  occa::memory o_gradP = platform->o_memPool.reserve<dfloat>(nrs->NVfields * nrs->fieldOffset);

  double flopCount = 0.0;
  platform->timer.tic("velocity rhs", 1);

  {
    occa::memory o_mueDiv = platform->o_memPool.reserve<dfloat>(nrs->fieldOffset);

    platform->linAlg->axmyz(mesh->Nlocal,
                            (platform->options.compareArgs("VELOCITY STRESSFORMULATION", "TRUE")) ? -2. / 3
                                                                                                  : 1. / 3,
                            nrs->o_mue,
                            nrs->o_div,
                            o_mueDiv);

    nrs->gradientVolumeKernel(mesh->Nelements,
                              mesh->o_vgeo,
                              mesh->o_D,
                              nrs->fieldOffset,
                              o_mueDiv,
                              o_gradMueDiv);
    flopCount += static_cast<double>(mesh->Nelements) * (6 * mesh->Np * mesh->Nq + 18 * mesh->Np);

    nrs->wgradientVolumeKernel(mesh->Nelements, mesh->o_vgeo, mesh->o_D, nrs->fieldOffset, nrs->o_P, o_gradP);
    flopCount += static_cast<double>(mesh->Nelements) * 18 * (mesh->Np * mesh->Nq + mesh->Np);
  }

  occa::memory o_rhs = platform->o_memPool.reserve<dfloat>(nrs->NVfields * nrs->fieldOffset);

  nrs->velocityRhsKernel(mesh->Nlocal, nrs->fieldOffset, nrs->o_BF, o_gradMueDiv, o_gradP, o_rhs);

  flopCount += 9 * mesh->Nlocal;

  o_gradMueDiv.free();
  o_gradP.free();

  nrs->velocityNeumannBCKernel(mesh->Nelements,
                               nrs->fieldOffset,
                               mesh->o_sgeo,
                               mesh->o_vmapM,
                               mesh->o_EToB,
                               nrs->o_EToB,
                               time,
                               mesh->o_x,
                               mesh->o_y,
                               mesh->o_z,
                               nrs->o_rho,
                               nrs->o_mue,
                               nrs->o_usrwrk,
                               nrs->o_Ue,
                               o_rhs);
  flopCount += static_cast<double>(mesh->Nelements) * (3 * mesh->Np + 36 * mesh->Nq * mesh->Nq);

  platform->timer.toc("velocity rhs");
  platform->flopCounter->add("velocity RHS", flopCount);

  auto o_U = platform->o_memPool.reserve<dfloat>(nrs->NVfields * nrs->fieldOffset);
  o_U.copyFrom(platform->options.compareArgs("VELOCITY INITIAL GUESS", "EXTRAPOLATION") && stage == 1
                   ? nrs->o_Ue
                   : nrs->o_U);

  const auto o_lambda0 = nrs->o_mue;

  const auto o_lambda1 = [&]()
  {
    auto o_l = platform->o_memPool.reserve<dfloat>(mesh->Nlocal);
    if (nrs->userVelocityImplicitLinearTerm) {
      auto o_implicitLT = nrs->userVelocityImplicitLinearTerm(time);
      platform->linAlg
          ->axpbyz(mesh->Nlocal, nrs->g0 / nrs->dt[0], nrs->o_rho, 1.0, o_implicitLT, o_l);
    } else {
      platform->linAlg->axpby(mesh->Nlocal, nrs->g0 / nrs->dt[0], nrs->o_rho, 0.0, o_l);
    }
    return o_l;
  }();

  if (nrs->uvwSolver) {
    nrs->uvwSolver->solve(o_lambda0, o_lambda1, static_cast<const occa::memory>(o_rhs), o_U);
  } else {
    const auto o_rhsX = o_rhs.slice(0 * nrs->fieldOffset);
    const auto o_rhsY = o_rhs.slice(1 * nrs->fieldOffset);
    const auto o_rhsZ = o_rhs.slice(2 * nrs->fieldOffset);
    nrs->uSolver->solve(o_lambda0, o_lambda1, o_rhsX, o_U.slice(0 * nrs->fieldOffset));
    nrs->vSolver->solve(o_lambda0, o_lambda1, o_rhsY, o_U.slice(1 * nrs->fieldOffset));
    nrs->wSolver->solve(o_lambda0, o_lambda1, o_rhsZ, o_U.slice(2 * nrs->fieldOffset));
  }

  if (platform->verbose) {
    const dfloat debugNorm = platform->linAlg->weightedNorm2Many(mesh->Nlocal,
                                                                 mesh->dim,
                                                                 nrs->fieldOffset,
                                                                 mesh->ogs->o_invDegree,
                                                                 o_U,
                                                                 platform->comm.mpiComm);
    if (platform->comm.mpiRank == 0) {
      printf("U norm: %.15e\n", debugNorm);
    }
  }

  return o_U;
}

} // namespace tombo
