#include "nrs.hpp"
#include "platform.hpp"
#include "linAlg.hpp"

void nrs_t::Qcriterion(const occa::memory& o_U, occa::memory& o_Q)
{
  occa::memory o_SijOij = 
      platform->o_memPool.reserve<dfloat>(3 * this->NVfields * static_cast<size_t>(this->fieldOffset));
  this->strainRotationRate(true, true, o_U, o_SijOij); 

  auto kernel = platform->kernels.get("Qcriterion");
  kernel(this->meshV->Nlocal, this->fieldOffset, this->o_div, o_SijOij, o_Q);
}
