#include <limits>
#include "nrs.hpp"
#include "linAlg.hpp"

occa::memory cds_t::solve(const int is, double time, int stage)
{
  std::string sid = scalarDigitStr(is);

  mesh_t* mesh = this->mesh[0];
  if(is) {
    mesh = this->meshV;
  }

  platform->timer.tic("scalar rhs", 1);

  auto o_rhs = platform->o_memPool.reserve<dfloat>(this->fieldOffset[is]);
  o_rhs.copyFrom(this->o_BF, this->fieldOffset[is], 0, this->fieldOffsetScan[is]);

  this->neumannBCKernel(mesh->Nelements,
                       1,
                       mesh->o_sgeo,
                       mesh->o_vmapM,
                       mesh->o_EToB,
                       is,
                       time,
                       this->fieldOffset[is],
                       0,
                       this->EToBOffset,
                       mesh->o_x,
                       mesh->o_y,
                       mesh->o_z,
                       this->o_Ue,
                       this->o_S,
                       this->o_EToB,
                       this->o_diff,
                       this->o_rho,
                       *(this->o_usrwrk),
                       o_rhs);

  platform->timer.toc("scalar rhs");

  auto o_S = [&]()
  {
     auto o_S0 = platform->o_memPool.reserve<dfloat>(this->fieldOffset[is]);
     if (platform->options.compareArgs("SCALAR" + sid + " INITIAL GUESS", "EXTRAPOLATION") && stage == 1)
       o_S0.copyFrom(this->o_Se, this->fieldOffset[is], 0, this->fieldOffsetScan[is]);
     else
       o_S0.copyFrom(this->o_S, this->fieldOffset[is], 0, this->fieldOffsetScan[is]);
     
     return o_S0;
  }();

  ellipticSolve(this->solver[is], o_rhs, o_S);

  return o_S;
}


