#include "nrs.hpp"

void nrs_t::strainRotationRate(bool smooth, bool rotationRate, const occa::memory &o_U, occa::memory &o_SO)
{
  mesh_t *mesh = this->meshV;

  const int nFields = (rotationRate) ? 2 * this->NVfields + this->NVfields : 2 * this->NVfields;

  nekrsCheck(o_SO.length() < (nFields * this->fieldOffset),
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "o_SO too small to store %d fields!\n",
             nFields);

  this->SijOijKernel(mesh->Nelements,
                     this->fieldOffset,
                     (int)rotationRate,
                     (int)smooth,
                     mesh->o_vgeo,
                     mesh->o_D,
                     o_U,
                     o_SO);

  if (smooth) {
    oogs::startFinish(o_SO, nFields, this->fieldOffset, ogsDfloat, ogsAdd, this->gsh);

    platform->linAlg->axmyMany(mesh->Nlocal, nFields, this->fieldOffset, 0, 1.0, mesh->o_invLMM, o_SO);
  }
}

void nrs_t::strainRate(bool smooth, const occa::memory &o_U, occa::memory &o_S)
{
  strainRotationRate(smooth, false, o_U, o_S);
}
