#if !defined(nekrs_LS_hpp_)
#define nekrs_LS_hpp_

#include "nrs.hpp"
#include "nekInterfaceAdapter.hpp"

namespace LS
{
void buildKernel(occa::properties kernelInfo);
void updateSourceTerms();
void setup();
}

#endif
