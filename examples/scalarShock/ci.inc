#include <math.h>

static int ciMode = 0;

#define EPS_SVV 4.85e-3
#define EPS_AVM 8.54e-3

dfloat exactSolution(dfloat x);

void ciSetup(MPI_Comm comm, setupAide &options)
{
  options.setArgs("CHECKPOINT INTERVAL", "0");
  options.setArgs("END TIME", std::string("1.0"));

  options.setArgs("DT", std::string("1e-3"));

  options.setArgs("SCALAR00 SOLVER TOLERANCE", std::string("1e-8"));

  options.setArgs("SCALAR01 SOLVER TOLERANCE", std::string("1e-8"));
}

void ciTestErrors(nrs_t *nrs, double time, int tstep)
{
  if (!nrs->lastStep) {
    return;
  }

  auto mesh = nrs->meshV;
  auto &scalar = nrs->scalar;

  std::vector<dfloat> sexact(mesh->Nlocal, 0.0);
  auto [x, y, z] = mesh->xyzHost();
  for(int n = 0; n < mesh->Nlocal; n++) {
    sexact[n] = exactSolution(x[n]);
  }

  auto o_exact = platform->device.malloc<dfloat>(sexact.size(), sexact.data());
  auto o_svv = scalar->o_solution("s00");
  auto o_avm = scalar->o_solution("s01");

  auto o_err = platform->device.malloc<dfloat>(mesh->Nlocal);

  platform->linAlg->axpbyz(mesh->Nlocal, 1.0, o_svv, -1.0, o_exact, o_err);
  const auto svvnorm = platform->linAlg->weightedNorm1(mesh->Nlocal, mesh->o_LMM, o_err, platform->comm.mpiComm());

  platform->linAlg->axpbyz(mesh->Nlocal, 1.0, o_avm, -1.0, o_exact, o_err);
  const auto avmnorm = platform->linAlg->weightedNorm1(mesh->Nlocal, mesh->o_LMM, o_err, platform->comm.mpiComm());

  if (platform->comm.mpiRank() == 0) {
    CiEvalTest("L1-Err norm SVV: " + to_string_f(svvnorm) + " L1-Err norm AVM: " + to_string_f(avmnorm), svvnorm < EPS_SVV && avmnorm < EPS_AVM);
  }
}
