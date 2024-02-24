#include "nrs.hpp"
#include "platform.hpp"
#include "linAlg.hpp"

void nrs_t::Qcriterion(const occa::memory& o_U, occa::memory& o_Q)
{
  auto o_SijOij = this->strainRotationRate(o_U); 

  auto kernel = platform->kernels.get("nrs->Qcriterion");
  kernel(this->meshV->Nlocal, this->fieldOffset, this->o_div, o_SijOij, o_Q);
}
