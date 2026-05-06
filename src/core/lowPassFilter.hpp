#if !defined(nekrs_lowPassFilter_hpp_)
#define nekrs_lowPassFilter_hpp_

#include "platform.hpp"
#include "mesh.h"
occa::memory lowPassFilterSetup(mesh_t *mesh, const dlong filterNc, bool cutOff = false, bool C0 = false);

#endif
