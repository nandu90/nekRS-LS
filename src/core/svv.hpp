#if !defined(nekrs_svv_hpp_)
#define nekrs_svv_hpp_

#include "mesh.h"

namespace svv{

void convoluteDerivative(mesh_t* mesh, occa::memory& o_filterPower, occa::memory& o_svvD);

}

#endif
