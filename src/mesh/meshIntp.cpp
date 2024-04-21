#include "mesh.h"
#include "platform.hpp"

void mesh_t::interpolate(mesh_t *meshC, const occa::memory& o_z, occa::memory& o_zC)
{
  const auto Nc = meshC->N;

  auto o_J = [&]() 
  {
    static std::array<occa::memory, this->maxNqIntp> o_J;
    if (o_J[Nc].isInitialized()) return o_J[Nc];

    std::vector<dfloat> J(this->Nq * meshC->Nq);
    InterpolationMatrix1D(this->N, this->Nq, this->r, meshC->Nq, meshC->r, J.data());
    o_J[Nc] = platform->device.malloc<dfloat>(J.size());
    o_J[Nc].copyFrom(J.data());
    return o_J[Nc];
  };

  this->intpKernel[Nc](this->Nelements, o_J(), o_z, o_zC);
}

void mesh_t::map2Uniform(int Nu, const occa::memory& o_z, occa::memory& o_zU)
{
  nekrsCheck(Nu > this->maxNqIntp - 1, 
             MPI_COMM_SELF, 
             EXIT_FAILURE, 
             "%s\n", 
             "N of uniform grid has to be smaller or equal to %d", this->maxNqIntp - 1);

  auto o_J = [&]() 
  {
    static std::array<occa::memory, this->maxNqIntp> o_J;
    if (o_J[Nu].isInitialized()) return o_J[Nu];

    std::vector<dfloat> r(Nu + 1);
    r[0] = -1.;
    r[Nu] = 1.;

    const auto dr = (r[Nu] - r[0]) / Nu;
    for(int i = 1; i < Nu; i++) r[i] = r[i-1] + dr;
  
    std::vector<dfloat> J(this->Nq * r.size());
    InterpolationMatrix1D(this->N, this->Nq, this->r, r.size(), r.data(), J.data());
    o_J[Nu] = platform->device.malloc<dfloat>(J.size());
    o_J[Nu].copyFrom(J.data());
    return o_J[Nu];
  };

  auto kernel = this->intpKernel[Nu];
  kernel(this->Nelements, o_J(), o_z, o_zU);
}

void mesh_t::map2Uniform(const occa::memory& o_z, occa::memory& o_zU)
{
  map2Uniform(this->N, o_z, o_zU); 
}
