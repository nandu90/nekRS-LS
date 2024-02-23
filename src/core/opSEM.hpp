#if !defined(nekrs_opSEM_hpp_)
#define nekrs_opSEM_hpp_

#include "platform.hpp"
#include "mesh.h"

namespace opSEM 
{

occa::memory grad(mesh_t *mesh, const occa::memory& o_in, dlong offset); 
occa::memory strongGrad(mesh_t *mesh, const occa::memory& o_in, dlong offset); 
occa::memory divergence(mesh_t *mesh, const occa::memory& o_in, dlong offset); 
occa::memory strongDivergence(mesh_t *mesh, const occa::memory& o_in, dlong offset); 
occa::memory laplacian(mesh_t *mesh, const occa::memory& o_lambda, const occa::memory& o_in, dlong offset);
occa::memory strongLaplacian(mesh_t *mesh, const occa::memory& o_lambda, const occa::memory& o_in, dlong offset);

}

#endif
