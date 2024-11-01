#if !defined(nekrs_cjp_hpp_)
#define nekrs_cjp_hpp_

#include "platform.hpp"
#include "mesh.h"
#include "opSEM.hpp"

static void addCJP(mesh_t *mesh, dfloat coef, dlong fieldOffset, const occa::memory& o_U, const occa::memory& o_S, occa::memory& o_out)
{
   // (n * o_grad)
   auto o_grad = opSEM::strongGrad(mesh, fieldOffset, o_S);
   auto o_nDotGradFace = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nelements * mesh->Nfaces * mesh->Nfp);
   auto o_h2 = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nelements * mesh->Nfaces * mesh->Nfp);

   static occa::kernel cjpHelperKernel;
   if (!cjpHelperKernel.isInitialized()) cjpHelperKernel = platform->kernelRequests.load("cjpHelperHex3D"); 
   cjpHelperKernel(mesh->Nelements, 
                   fieldOffset, 
                   mesh->o_x, 
                   mesh->o_y, 
                   mesh->o_z, 
                   mesh->o_vmapM,
                   mesh->o_Jw,
                   mesh->o_sgeo, 
                   o_grad,
                   o_h2,
                   o_nDotGradFace);

   static oogs_t* gshFace = nullptr;
   if (!gshFace) {
     gshFace = oogs::setup(o_nDotGradFace.size(),
                           mesh->globalFaceIds, 
                           1,
                           0,
                           ogsDfloat,
                           platform->comm.mpiComm,
                           1,
                           platform->device.occaDevice(),
                           NULL,
                           OOGS_AUTO);
   }

   oogs::startFinish(o_nDotGradFace, 1, 0, ogsDfloat, ogsAdd, gshFace);

#if 0
   static auto o_tmp =  platform->device.malloc<dfloat>(mesh->Nlocal);
   cjpHelper2Kernel(mesh->Nelements, 
                    o_nDotGradFace, 
                    o_tmp);

   deviceMemory<dfloat> d_tmp(o_tmp); 
   addUserCheckpointField("scalar01", std::vector<deviceMemory<dfloat>>{d_tmp}); 

#endif

   const dfloat penaltyFactor = -coef * std::pow(static_cast<dfloat>(mesh->Nq), static_cast<dfloat>(-4));
   std::cout << "addCJP " << coef << std::endl;

   static occa::kernel cjpKernel;
   if (!cjpKernel.isInitialized()) cjpKernel = platform->kernelRequests.load("cjpHex3D"); 

   cjpKernel(mesh->Nelements,
             fieldOffset, 
             penaltyFactor,
             mesh->o_D,
             mesh->o_vmapM,
             mesh->o_vgeo,
             mesh->o_sgeo,
             o_h2,
             o_U,
             o_nDotGradFace,
             o_out);
}

#endif
