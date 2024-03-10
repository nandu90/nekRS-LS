#if !defined(nekrs_io_hpp_)
#define nekrs_io_hpp_

#include "platform.hpp"

class ElementFilter
{
  public:
    // mask: local element ids
    void set(const std::vector<long>& mask) 
    { 
      _mask = mask;
    };
    void clear() { _mask.clear(); };

    std::vector<long> mask() const { return _mask; };

  private:
    std::vector<long> _mask;
};

namespace fld {

extern ElementFilter elementFilter;

void write(std::string suffix, double t, int step, int outXYZ, int FP64,
           const occa::memory& o_s, int NSfields, int Nout = 0, bool uniform = false);

void write(std::string suffix, double t, int step, int outXYZ, int FP64,
           const occa::memory& o_u, const occa::memory& o_p,  const occa::memory& o_s,
           int NSfields, int Nout = 0, bool uniform = false);

}

#endif
