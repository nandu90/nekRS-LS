#if !defined(nekrs_velRecycling_hpp_)
#define nekrs_velRecycling_hpp_


#include "nrs.hpp"
#include "nekInterfaceAdapter.hpp"

namespace velRecycling
{
void buildKernel(occa::properties kernelInfo);
void copy();

// gs-based
// sssumes elements are arranged such that the z-dimension varies the least
void setup(occa::memory o_wrk_, const hlong eOffset, const int bID_,
           const dfloat wbar_);

// interpolation-based
void setup(occa::memory o_wrk_,
           const dfloat xOffset, const dfloat yOffset, const dfloat zOffset, 
           const int bID_,  const dfloat wbar_);
}

#endif
