#include "mesh.h"
#include "platform.hpp"

// interpolate to M-points
occa::memory mesh_t::intpMatrix(std::vector<dfloat> M)
{
  nekrsCheck(M.size() > mesh_t::maxNqIntp,
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "target N has to be smaller or equal to %d", mesh_t::maxNqIntp - 1);

  std::vector<dfloat> J(this->Nq * M.size());
  InterpolationMatrix1D(this->N, this->Nq, this->r, M.size(), M.data(), J.data());

  auto transposeJ = [&]()
  {
    std::vector<dfloat> Jt(J.size());
    for (int i = 0; i < this->Nq; i++) {
      for (int j = 0; j < M.size(); j++) {
        Jt[i * M.size() + j] = J[j * this->Nq + i];
      }
    }
    return Jt;
  };

  auto o_J = platform->device.malloc<dfloat>(J.size());
  o_J.copyFrom((M.size() < this->Nq) ? J.data() : transposeJ().data());

  return o_J;
}

void mesh_t::interpolate(const occa::memory& o_z, mesh_t *mesh, occa::memory& o_zM, bool uniform, dlong nel_)
{
  auto nel = (nel_ > 0) ? nel_ : this->Nelements;

  nekrsCheck(mesh->Nq > mesh_t::maxNqIntp,
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "target N has to be smaller or equal to %d", mesh_t::maxNqIntp - 1);

  if (uniform) {

    static std::array<occa::memory, mesh_t::maxNqIntp> o_Juni;
    if (!o_Juni[mesh->N].isInitialized()) {

      auto M = [&]()
      {
        std::vector<dfloat> r(mesh->N + 1);
        r[0] = -1.0;
        r[mesh->N] = 1.0;

        const auto dr = (r[mesh->N] - r[0]) / mesh->N;
        for(int i = 1; i < mesh->N; i++) r[i] = r[i-1] + dr;
        return r;
      }();

      o_Juni[mesh->N] = intpMatrix(M);
    }

    this->intpKernel[mesh->N](nel, o_Juni[mesh->N], o_z, o_zM);

  } else {

    static std::array<occa::memory, mesh_t::maxNqIntp> o_Jgll;
    if (!o_Jgll[mesh->N].isInitialized()) {
      std::vector<dfloat> M(mesh->Nq);
      for(int i = 0; i < M.size(); i++) M[i] = mesh->r[i];
      o_Jgll[mesh->N] = intpMatrix(M);
    }

    this->intpKernel[mesh->N](nel, o_Jgll[mesh->N], o_z, o_zM);
  }

}
