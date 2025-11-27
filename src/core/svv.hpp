#if !defined(nekrs_svv_hpp_)
#define nekrs_svv_hpp_

#include "mesh.h"

namespace svv{

void convoluteDerivative(mesh_t* mesh, const dfloat NSVV, occa::memory& o_svvD, occa::memory& o_svvDT);

}

#endif
