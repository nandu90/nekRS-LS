#include "platform.hpp"
#include "mesh.h"
#include "ellipticBcTypes.h"

static std::vector<int> createEToBV(const mesh_t* mesh, const std::vector<int>& EToB)
{
  const int largeNumber = 1 << 20;

  std::vector<int> EToBV(mesh->Nlocal, largeNumber);
  for (dlong e = 0; e < mesh->Nelements; e++) {
    for (int f = 0; f < mesh->Nfaces; f++) {
      int bc = EToB[f + e * mesh->Nfaces];
      if (bc > 0) {
        for (int n = 0; n < mesh->Nfp; n++) {
          int fid = mesh->faceNodes[n + f * mesh->Nfp];
          EToBV[fid + e * mesh->Np] = std::min(bc, EToBV[fid + e * mesh->Np]);
        }
      }
    }
  }

  ogsGatherScatter(EToBV.data(), ogsInt, ogsMin, mesh->ogs);

  for (dlong n = 0; n < mesh->Nlocal; n++) {
    if (EToBV[n] == largeNumber) {
      EToBV[n] = 0;
    }
  }

  return EToBV;
}

occa::memory mesh_t::createZeroNormalMask(dlong fieldOffset,
                                          const occa::memory &o_EToB)
{
  auto o_mask = platform->device.malloc<dfloat>(fieldOffset * dim);

  auto o_EToBV = [&]()
  {
    std::vector<int> EToB(o_EToB.size());
    o_EToB.copyTo(EToB.data());

    auto EToBV = createEToBV(this, EToB);
    auto o_EToBV = platform->deviceMemoryPool.reserve<int>(EToBV.size());
    o_EToBV.copyFrom(EToBV.data());
    return o_EToBV;
  }();

  static occa::kernel initializeZeroNormalMaskKernel;
  if(!initializeZeroNormalMaskKernel.isInitialized()) 
    initializeZeroNormalMaskKernel = platform->kernelRequests.load("mesh-initializeZeroNormalMask");   

  initializeZeroNormalMaskKernel(Nlocal, fieldOffset, o_EToBV, o_mask);

  // normal xyz + count
  occa::memory o_avgNormal =
      platform->deviceMemoryPool.reserve<dfloat>((dim + 1) * fieldOffset);

  const int bcType = ellipticBcType::ZERO_NORMAL;

  static occa::kernel averageNormalBcTypeKernel;
  if(!averageNormalBcTypeKernel.isInitialized()) 
    averageNormalBcTypeKernel = platform->kernelRequests.load("mesh-averageNormalBcType");   

  averageNormalBcTypeKernel(Nelements,
                            fieldOffset,
                            bcType,
                            o_sgeo,
                            o_vmapM,
                            o_EToB,
                            o_avgNormal);

  oogs::startFinish(o_avgNormal, dim + 1, fieldOffset, ogsDfloat, ogsAdd, oogs);

  static occa::kernel fixZeroNormalMaskKernel;
  if(!fixZeroNormalMaskKernel.isInitialized()) 
    fixZeroNormalMaskKernel = platform->kernelRequests.load("mesh-fixZeroNormalMask");   

  fixZeroNormalMaskKernel(Nelements,
                          fieldOffset,
                          o_sgeo,
                          o_vmapM,
                          o_EToB,
                          o_avgNormal,
                          o_mask);

  oogs::startFinish(o_mask, dim, fieldOffset, ogsDfloat, ogsMin, oogs);

  return o_mask;
}

