#include "platform.hpp"
#include "nekInterfaceAdapter.hpp"
#include "fld.hpp"

namespace fld
{
ElementFilter elementFilter;
}

void fld::write(std::string suffix, double t, int step, int outXYZ, int FP64,
                const occa::memory& o_u, const occa::memory& o_p, const occa::memory& o_s,
                int NSfields, int Nout, bool uniform)
{
  nek::outfld(suffix.c_str(), t, step, outXYZ, FP64, o_u, o_p, o_s, NSfields, Nout, uniform); 
}

void fld::write(std::string suffix, double t, int step, int outXYZ, int FP64,
                const occa::memory& o_s, int NSfields, int Nout, bool uniform)
{
  write(suffix, t, step, outXYZ, FP64, o_NULL, o_NULL, o_s, NSfields, Nout, uniform); 
}


