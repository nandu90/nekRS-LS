/*

The MIT License (MIT)

Copyright (c) 2017 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#ifndef LINALG_HPP
#define LINALG_HPP

#include "platform.hpp"

#define USE_WEIGHTED_INNER_PROD_MULTI_DEVICE 0

class linAlg_t
{
private:
  occa::properties kernelInfo;
  MPI_Comm comm;
  int blocksize;
  bool serial;

  int timer = 0;

  void runTimers();

  ~linAlg_t();
  linAlg_t();
  static linAlg_t *singleton;

  template <typename T = dfloat> static std::string getKnlPrefix()
  {
    std::string val;
    if (std::is_same<T, pfloat>::value && sizeof(dfloat) != sizeof(pfloat)) {
      val = std::string("p");
    } else if (std::is_same<T, dfloat>::value) {
      //
    } else {
      nekrsAbort(MPI_COMM_SELF, EXIT_FAILURE, "%s", "unsupported data type on input!\n");
    }

    return "linAlg::" + val;
  }

  template <typename T = dfloat> static occa::memory getScratch(size_t n, bool host = false)
  {
    if (std::is_same<T, pfloat>::value) {
      if (host) {
        return platform->memoryPool.reserve<pfloat>(n);
      }
      return platform->deviceMemoryPool.reserve<pfloat>(n);
    } else {
      if (host) {
        return platform->memoryPool.reserve<dfloat>(n);
      }
      return platform->deviceMemoryPool.reserve<dfloat>(n);
    }
  }

public:
  static linAlg_t *getInstance();

  void enableTimer();
  void disableTimer();

#include "linAlg.tpp"

  // z = x \cross y
  void crossProduct(const dlong N,
                    const dlong fieldOffset,
                    const occa::memory &o_x,
                    const occa::memory &o_y,
                    occa::memory &o_z);

  void unitVector(const dlong N, const dlong fieldOffset, occa::memory &o_v);

  // o_b[n] = \sqrt{\sum_{i=0}^{Nfields-1} o_a[n+i*fieldOffset]^2}
  void entrywiseMag(const dlong N,
                    const dlong Nfields,
                    const dlong fieldOffset,
                    const occa::memory &o_a,
                    occa::memory &o_b);

  void magSqrVector(const dlong N, const dlong fieldOffset, const occa::memory &o_u, occa::memory &o_mag);

  void
  magSqrSymTensor(const dlong N, const dlong fieldOffset, const occa::memory &o_tensor, occa::memory &o_mag);

  void magSqrSymTensorDiag(const dlong N,
                           const dlong fieldOffset,
                           const occa::memory &o_tensor,
                           occa::memory &o_mag);

  void
  magSqrTensor(const dlong N, const dlong fieldOffset, const occa::memory &o_tensor, occa::memory &o_mag);

  // o_y[n] = x_{Nfields} * coeff_{Nfields} + \sum_{i=0}^{Nfields-1} coeff_i * x_i
  void linearCombination(const dlong N,
                         const dlong Nfields,
                         const dlong fieldOffset,
                         const occa::memory &o_coeff,
                         const occa::memory &o_x,
                         occa::memory &o_y);

  dfloat maxRelativeError(const dlong N,
                          const int Nfields,
                          const dlong fieldOffset,
                          const dfloat absTol,
                          const occa::memory &o_u,
                          const occa::memory &o_uRef,
                          MPI_Comm comm);

  dfloat maxAbsoluteError(const dlong N,
                          const int Nfields,
                          const dlong fieldOffset,
                          const dfloat absTol,
                          const occa::memory &o_u,
                          const occa::memory &o_uRef,
                          MPI_Comm comm);

  // matrix is in row major ordering
  std::vector<dfloat> matrixInverse(const int N, const std::vector<dfloat> &A);
  std::vector<dfloat> matrixPseudoInverse(const int N, const std::vector<dfloat> &A);
  std::vector<dfloat> matrixTranspose(const int N, const std::vector<dfloat> &A);
};

#endif
