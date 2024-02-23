#if !defined(nekrs_io_hpp_)
#define nekrs_io_hpp_

#include "nrs.hpp"

void writeFld(nrs_t *nrs, dfloat t, int step, int Nro = 0);
void writeFld(nrs_t *nrs, dfloat t, int step, std::string suffix, int Nro = 0);
void writeFld(nrs_t *nrs, dfloat t, int step, int outXYZ, int FP64, int Nro = 0);
void writeFld(nrs_t *nrs, dfloat t, int step, int outXYZ, int FP64, std::string suffix, int Nro = 0);

void writeFld(std::string suffix, dfloat t, int step, int outXYZ, int FP64,
              const occa::memory& o_s, int NSfields, int Nro = 0);

void writeFld(std::string suffix, dfloat t, int step, int outXYZ, int FP64,
              const occa::memory& o_u, const occa::memory& o_p,  const occa::memory& o_s,
              int NSfields, int Nro = 0);

#endif
