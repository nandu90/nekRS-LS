#if !defined(nekrs_avm_hpp_)
#define nekrs_avm_hpp_

#include "mesh.h"
#include "QQt.hpp"

namespace avm{

void setup(mesh_t *mesh_, oogs_t *gsh_);
occa::memory viscosity(dlong UFieldOffset, const occa::memory& o_U, const occa::memory& o_S,
                       dfloat scalingCoeff = 1.0, dfloat logS0 = 2.0, dfloat kappa = 1.0, bool makeCont = false);
}

#endif
