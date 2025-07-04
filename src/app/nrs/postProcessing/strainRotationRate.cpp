#include "nrs.hpp"

static occa::memory
_strainRotationRate(mesh_t *mesh, bool rotationRate, dlong offset, const occa::memory &o_U, bool smooth)
{
  const int nFields = (rotationRate) ? 2 * mesh->dim + mesh->dim : 2 * mesh->dim;

  auto o_SO = platform->deviceMemoryPool.reserve<dfloat>(nFields * offset);

  launchKernel("nrs-SijOijHex3D",
               mesh->Nelements,
               offset,
               static_cast<int>(rotationRate),
               static_cast<int>(smooth),
               mesh->o_vgeo,
               mesh->o_D,
               o_U,
               o_SO);

  if (smooth) {
    oogs::startFinish(o_SO, nFields, offset, ogsDfloat, ogsAdd, mesh->oogs3);

    platform->linAlg->axmyMany(mesh->Nlocal, nFields, offset, 0, 1.0, mesh->o_invLMM, o_SO);
  }

  return o_SO;
}

occa::memory nrs_t::strainRotationRate(bool smooth)
{
  return _strainRotationRate(meshV, true, fluid->fieldOffset, fluid->o_U, smooth);
}

occa::memory nrs_t::strainRotationRate(dlong offset, const occa::memory &o_U, bool smooth)
{
  return _strainRotationRate(meshV, true, offset, o_U, smooth);
}

occa::memory nrs_t::strainRate(bool smooth)
{
  return _strainRotationRate(meshV, false, fluid->fieldOffset, fluid->o_U, smooth);
}

occa::memory nrs_t::strainRate(dlong offset, const occa::memory &o_U, bool smooth)
{
  return _strainRotationRate(meshV, false, offset, o_U, smooth);
}
