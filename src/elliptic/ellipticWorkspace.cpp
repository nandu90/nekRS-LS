#include "elliptic.h"
#include "ellipticPrecon.h"

void ellipticAllocateWorkspace(elliptic_t *elliptic)
{
  elliptic->o_p = platform->o_memPool.reserve<dfloat>(elliptic->Nfields * elliptic->fieldOffset);
  elliptic->o_z = platform->o_memPool.reserve<dfloat>(elliptic->Nfields * elliptic->fieldOffset);
  elliptic->o_Ap = platform->o_memPool.reserve<dfloat>(elliptic->Nfields * elliptic->fieldOffset);
  elliptic->o_x0 = platform->o_memPool.reserve<dfloat>(elliptic->Nfields * elliptic->fieldOffset);
  elliptic->o_rPfloat = platform->o_memPool.reserve<pfloat>(elliptic->Nfields * elliptic->fieldOffset);
  elliptic->o_zPfloat = platform->o_memPool.reserve<pfloat>(elliptic->Nfields * elliptic->fieldOffset);

  if (elliptic->options.compareArgs("SOLVER", "PCG+COMBINED")) {
    elliptic->o_v = platform->o_memPool.reserve<dfloat>(elliptic->Nfields * elliptic->fieldOffset);
  }

#if 1
  if (elliptic->gmresData) {
    const auto flexible = elliptic->gmresData->flexible;
    const auto nRestartVectors = elliptic->gmresData->nRestartVectors;
    elliptic->gmresData->o_V = 
      platform->o_memPool.reserve<dfloat>(static_cast<size_t>(elliptic->fieldOffset) * elliptic->Nfields * nRestartVectors);
    elliptic->gmresData->o_Z = 
      platform->o_memPool.reserve<dfloat>(static_cast<size_t>(elliptic->fieldOffset) * elliptic->Nfields * ((flexible) ? nRestartVectors : 1));
  }
#endif

#if 1
  if (elliptic->precon) {
    if (elliptic->precon->MGSolver) {
      elliptic->precon->MGSolver->allocateWorkStorage();
    }
  }
#endif 
}

void ellipticFreeWorkspace(elliptic_t *elliptic)
{
  elliptic->o_p.free();
  elliptic->o_z.free();
  elliptic->o_Ap.free();
  elliptic->o_x0.free();
  elliptic->o_rPfloat.free();
  elliptic->o_zPfloat.free();

  if (elliptic->options.compareArgs("SOLVER", "PCG+COMBINED")) {
    elliptic->o_v.free();
  }
#if 1
  if (elliptic->gmresData) {
    if(elliptic->gmresData->o_V.isInitialized()) elliptic->gmresData->o_V.free();
    if(elliptic->gmresData->o_Z.isInitialized()) elliptic->gmresData->o_Z.free();
  }
#endif

#if 1
  if (elliptic->precon) {
    if (elliptic->precon->MGSolver) {
      elliptic->precon->MGSolver->freeWorkStorage();
    }
  }
#endif 
}

