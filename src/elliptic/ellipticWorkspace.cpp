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
}

