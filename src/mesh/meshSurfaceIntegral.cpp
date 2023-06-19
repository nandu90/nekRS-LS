#include <mesh.h>
#include "platform.hpp"

namespace {
  dfloat *sum;
  dfloat *sumFace;
  occa::memory o_sumFace;
  occa::memory h_sumFace;
}

static std::vector<dfloat> integral(mesh_t *mesh, int Nfields, dlong fieldOffset, bool vector, int nbID,
                                    const occa::memory& o_bID, const occa::memory& o_fld)
{
  const auto Nbytes = (Nfields * sizeof(dfloat)) * mesh->Nelements;

  if (o_sumFace.size() < Nbytes) {
    if (o_sumFace.size()) o_sumFace.free();
    o_sumFace = platform->device.malloc(Nbytes);

    if (h_sumFace.size()) h_sumFace.free();
    h_sumFace = platform->device.mallocHost(Nbytes);
    sumFace = (dfloat *) h_sumFace.ptr();

    if (sum) free(sum);
    sum = (dfloat *) std::malloc(Nbytes);
  }

  if (vector)
    mesh->surfaceIntegralVectorKernel(mesh->Nelements, 
                                Nfields,
                                fieldOffset, 
                                nbID,
                                o_bID,
                                mesh->o_sgeo, 
                                mesh->o_vmapM, 
                                mesh->o_EToB,  
                                o_fld,
                                o_sumFace);
  else 
    mesh->surfaceIntegralKernel(mesh->Nelements, 
                          Nfields,
                          fieldOffset, 
                          nbID,
                          o_bID,
                          mesh->o_sgeo, 
                          mesh->o_vmapM, 
                          mesh->o_EToB,  
                          o_fld,
                          o_sumFace);

  o_sumFace.copyTo(sumFace, Nbytes);

  for (int j = 0; j < Nfields + 1; ++j) {
    sum[j] = 0;
    for (int i = 0; i < mesh->Nelements; ++i) {
      sum[j] += sumFace[i + j * mesh->Nelements];
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, sum, Nfields, MPI_DFLOAT, MPI_SUM, platform->comm.mpiComm);

  std::vector<dfloat> out; 
  for (int i = 0; i < Nfields; ++i)
    out.push_back(sum[i]);
  
  return out;
}

std::vector<dfloat> mesh_t::surfaceIntegral(int nbID, const occa::memory& o_bID, const occa::memory& o_fld)
{
  return integral(this, 1, static_cast<dlong>(0), false, nbID, o_bID, o_fld);
}

std::vector<dfloat> mesh_t::surfaceIntegralVector(dlong fieldOffset, int nbID, const occa::memory& o_bID, const occa::memory& o_fld)
{
  return integral(this, 1, fieldOffset, true, nbID, o_bID, o_fld);
}

std::vector<dfloat> mesh_t::surfaceIntegralMany(int Nfields, dlong fieldOffset, int nbID,
                                                const occa::memory& o_bID, const occa::memory& o_fld)
{
  return integral(this, Nfields, fieldOffset, false, nbID, o_bID, o_fld);
}
