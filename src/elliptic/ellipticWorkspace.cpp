#include "elliptic.h"
#include "ellipticPrecon.h"

void ellipticAllocateWorkspace(elliptic_t *elliptic)
{
  const auto Nlocal = elliptic->Nfields * static_cast<size_t>(elliptic->fieldOffset);

  elliptic->o_p = platform->o_memPool.reserve<dfloat>(Nlocal);
  elliptic->o_z = platform->o_memPool.reserve<dfloat>(Nlocal);
  elliptic->o_Ap = platform->o_memPool.reserve<dfloat>(Nlocal);
  elliptic->o_rPfloat = platform->o_memPool.reserve<pfloat>(Nlocal);
  elliptic->o_zPfloat = platform->o_memPool.reserve<pfloat>(Nlocal);

  if (elliptic->options.compareArgs("SOLVER", "PCG+COMBINED")) {
      elliptic->o_v = platform->o_memPool.reserve<dfloat>(Nlocal);
  }

  if (elliptic->gmresData) {
    const auto flexible = elliptic->gmresData->flexible;
    const auto nRestartVectors = elliptic->gmresData->nRestartVectors;
    elliptic->gmresData->o_V = 
      platform->o_memPool.reserve<dfloat>(Nlocal * nRestartVectors);
    elliptic->gmresData->o_Z = 
      platform->o_memPool.reserve<dfloat>(Nlocal * ((flexible) ? nRestartVectors : 1));
  }

  if (elliptic->precon) {
    if (elliptic->precon->MGSolver) {
      elliptic->precon->MGSolver->allocateWorkStorage();
    }
  }
}

void ellipticFreeWorkspace(elliptic_t *elliptic)
{
  if (elliptic->precon) {
    if (elliptic->precon->MGSolver) {
      elliptic->precon->MGSolver->freeWorkStorage();
    }
  }

  if(elliptic->o_v.isInitialized()) 
    elliptic->o_v.free();

  if (elliptic->gmresData) {
    if(elliptic->gmresData->o_V.isInitialized()) elliptic->gmresData->o_V.free();
    if(elliptic->gmresData->o_Z.isInitialized()) elliptic->gmresData->o_Z.free();
  }

  elliptic->o_p.free();
  elliptic->o_z.free();
  elliptic->o_Ap.free();
  elliptic->o_rPfloat.free();
  elliptic->o_zPfloat.free();
}
