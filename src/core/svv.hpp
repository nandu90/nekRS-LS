#if !defined(nekrs_svv_hpp_)
#define nekrs_svv_hpp_

#include "mesh.h"

namespace svv{

void setup(mesh_t *mesh_, const dlong fieldOffset, const dlong NSfields);
void convoluteDerivative(const dfloat NSVV, occa::memory& o_svvD, occa::memory& o_svvDT);
void computeViscosityScale(const dlong vFieldOffset, const occa::memory& o_U, occa::memory& o_svvmu);

}

#endif
