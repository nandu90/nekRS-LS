#include <array>

#include "platform.hpp"
#include "svv.hpp"

/**
 * Spectrally Vanishing Viscosity method (https://doi.org/10.1016/j.jcp.2026.114961) 
 **/

namespace //private
{
  occa::memory o_B;
  occa::memory o_Binv;

  bool setupCalled = false; 

  dfloat meshScale = -1e10;
  bool meshScalerCalled = false;

  dfloat PNLEG(const dfloat Z, const int N)
  {
    dfloat p1 = 1.0;
    if(N==0) return p1;

    dfloat p2 = Z;
    dfloat p3 = p2;

    for (int k=1; k < N; k++){
      dfloat fk = k;
      p3 = ((2.0 * fk + 1.0) * Z * p2 - fk * p1) / (fk + 1.0);
      p1 = p2;
      p2 = p3;
    }

    return p3;
  }

  std::vector<dfloat> legendreBasis(mesh_t* mesh)
  {
    std::vector<dfloat> B(mesh->Nq * mesh->Nq, 0.0);

    if(mesh->Nq == 1) B[0] = 1.0;

    for (int i = 0; i < mesh->Nq; i++){
      for (int j = 0; j < mesh->Nq; j++) {
        B[j * mesh->Nq + i] = PNLEG(mesh->r[j], i);
      }
    }
    return B;
  }

} //namespace

dfloat svv::svvMeshScale(mesh_t *mesh, MPI_Comm _comm)
{
  if(meshScalerCalled)
    return meshScale;

  meshScalerCalled = true;

  auto o_svvf = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nelements);

  launchKernel("core-svv::svvMeshScale", mesh->Nelements, mesh->o_vgeo, o_svvf);
  std::vector<dfloat> svvf(mesh->Nelements);
  o_svvf.copyTo(svvf.data(), mesh->Nelements);

  for (dlong n = 0 ; n < mesh->Nelements; n++) {
    meshScale = meshScale > svvf[n] ? meshScale : svvf[n];
  }

  MPI_Allreduce(MPI_IN_PLACE, &meshScale, 1, MPI_DFLOAT, MPI_MAX, _comm);

  return meshScale;
}

void svv::convoluteDerivative(mesh_t* mesh, occa::memory& o_filterPower, occa::memory& o_svvD)
{
  if(!setupCalled) {
    auto B = legendreBasis(mesh);
    auto Binv = platform->linAlg->matrixInverse(mesh->Nq, B);

    o_B = platform->device.malloc<dfloat>(mesh->Nq * mesh->Nq, B.data());
    o_Binv = platform->device.malloc<dfloat>(mesh->Nq * mesh->Nq, Binv.data());
  }

  launchKernel("core-svv::convoluteDerivative",
               mesh->Nelements,
               o_B,
               o_Binv,
               mesh->o_D,
               o_filterPower,
               o_svvD);

  setupCalled = true;
}
