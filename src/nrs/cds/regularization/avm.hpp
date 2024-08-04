#if !defined(nekrs_avm_hpp_)
#define nekrs_avm_hpp_

#include "cds.hpp"
namespace avm{
void setup(cds_t* cds);
occa::memory viscosity(const dlong scalarIndex, occa::memory o_S);
void apply(const dlong scalarIndex, occa::memory o_S);
}

#endif
