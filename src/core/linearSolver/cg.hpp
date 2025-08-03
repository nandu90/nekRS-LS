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

#include <limits>
#include <array>
#include "platform.hpp"
#include "linAlg.hpp"

//#define DEBUG

template <typename T> class cg : public linearSolver
{

public:
  cg(dlong _Nlocal,
     int _Nfields,
     dlong _fieldOffset,
     const occa::memory &_o_weight,
     bool _flexible,
     bool _combined,
     std::function<void(const occa::memory &o_q, occa::memory &o_Aq)> _Ax,
     std::function<void(const occa::memory &o_r, occa::memory &o_z)> _preco)
  {
    this->Nlocal = _Nlocal;
    this->Nfields = _Nfields;
    this->fieldOffset = _fieldOffset;
    o_weight = _o_weight;
    flexible = _flexible;
    combined = _combined;
    Ax = _Ax;
    preco = _preco;

    this->tiny = 10 * std::numeric_limits<T>::min();
    this->FPfactor = (std::is_same<T, pfloat>::value) ? 0.5 : 1.0;
    auto type = ((std::is_same<T, pfloat>::value && !std::is_same<dfloat, pfloat>::value) ? std::string("pfloat") : std::string(""));
    this->knlPrefix = std::string("cg::") + type + std::to_string(this->Nfields) + "-";
  };

  int solve(const dfloat tol, const int MAXIT, dfloat &rdotr, occa::memory &o_r, occa::memory &o_x) override
  {
    {
      const auto n =
          (this->Nfields > 1) ? this->Nfields * static_cast<size_t>(this->fieldOffset) : this->Nlocal;

      o_p = platform->deviceMemoryPool.reserve<T>(n);
      platform->linAlg->fill<T>(o_p.size(), 0.0, o_p);

      o_z = (preco) ? platform->deviceMemoryPool.reserve<T>(n) : o_r;
      o_Ap = platform->deviceMemoryPool.reserve<T>(n);
      if (combined) {
        o_v = platform->deviceMemoryPool.reserve<T>(n);
      }
    }

    const auto Nblock = [&]() { return (this->Nlocal + BLOCKSIZE - 1) / BLOCKSIZE; }();

    auto Nreductions = [&]() {
      int n = 1;
      if (combined) {
        n = CombinedPCGId::nReduction;
      }
      return n;
    }();

    h_tmpReductions = platform->memoryPool.reserve<T>(Nreductions * Nblock);
    o_tmpReductions = platform->deviceMemoryPool.reserve<T>(h_tmpReductions.size());

    if (platform->comm.mpiRank == 0 && platform->verbose()) {
      auto txt = (combined && this->o_invDiagA.isInitialized() || preco) ? std::string("P") : std::string(""); 
      txt += (combined) ? std::string("CCG") : std::string("CG") + ((flexible) ? "-flex" : "");
      printf("%s %s: initial res norm %.15e target %e \n", txt.c_str(), this->_name.c_str(), rdotr, tol);
    }

    const auto Niter = [&]() {
      if (combined) {
        return runCombined(tol, MAXIT, rdotr, o_r, o_x);
      } else {
        return runStandard(tol, MAXIT, rdotr, o_r, o_x);
      }
    }();

    o_p.free();
    if (o_z != o_r) {
      o_z.free();
    }
    o_Ap.free();
    if (combined) {
      o_v.free();
    }

    o_tmpReductions.free();
    h_tmpReductions.free();

    return Niter;
  };

private:
  occa::memory o_p;
  occa::memory o_z;
  occa::memory o_Ap;
  occa::memory o_v;
  occa::memory o_tmpReductions;
  occa::memory h_tmpReductions;

  occa::memory o_weight;

  bool flexible;
  bool combined;

  std::function<void(const occa::memory &o_q, occa::memory &o_Aq)> Ax;
  std::function<void(const occa::memory &o_r, occa::memory &o_z)> preco;

  dfloat update(const occa::memory &o_p,
                const occa::memory &o_Ap,
                const T alpha,
                occa::memory &o_x,
                occa::memory &o_r)
  {
    const bool serial = platform->serial;

    // r <= r - alpha*A*p
    // dot(r,r)
    launchKernel(this->knlPrefix + "blockUpdatePCG",
                 this->Nlocal,
                 this->fieldOffset,
                 o_weight,
                 o_Ap,
                 alpha,
                 o_r,
                 o_tmpReductions);

    dfloat rdotr1 = 0;
#ifdef ELLIPTIC_ENABLE_TIMER
    platform->timer.tic("dotp");
#endif
    if (serial) {
      rdotr1 = *(o_tmpReductions.ptr<T>());
    } else {
      auto tmp = h_tmpReductions.ptr<T>();
      o_tmpReductions.copyTo(tmp);
      for (int n = 0; n < o_tmpReductions.size(); ++n) {
        rdotr1 += tmp[n];
      }
    }

    // x <= x + alpha*p
    platform->linAlg->axpbyMany<T>(this->Nlocal, this->Nfields, this->fieldOffset, static_cast<dfloat>(alpha), o_p, 1.0, o_x);

    MPI_Allreduce(MPI_IN_PLACE, &rdotr1, 1, MPI_DFLOAT, MPI_SUM, platform->comm.mpiComm);
#ifdef ELLIPTIC_ENABLE_TIMER
    platform->timer.toc("dotp");
#endif

    platform->flopCounter->add("UpdatePCG",
                               this->FPfactor * this->Nfields * static_cast<double>(this->Nlocal) * 6 +
                                   this->Nlocal);

    return rdotr1;
  };

  void combinedReductions(const occa::memory &o_Minv,
                          const occa::memory &o_v,
                          const occa::memory &o_p,
                          const occa::memory &o_r,
                          std::array<T, CombinedPCGId::nReduction> &reductions)
  {
    const bool serial = platform->serial;
    launchKernel(this->knlPrefix + "combinedPCGPostMatVec",
                 this->Nlocal,
                 this->fieldOffset,
                 (this->o_invDiagA.isInitialized()) ? 1 : 0,
                 o_weight,
                 o_weight,
                 o_Minv,
                 o_v,
                 o_p,
                 o_r,
                 o_tmpReductions);
    if (serial) {
      auto ptr = o_tmpReductions.ptr<T>();
      std::copy(ptr, ptr + CombinedPCGId::nReduction, reductions.begin());
    } else {
      auto tmp = h_tmpReductions.ptr<T>();
      o_tmpReductions.copyTo(tmp);
      std::fill(reductions.begin(), reductions.end(), 0.0);

      const dlong Nblock = (this->Nlocal + BLOCKSIZE - 1) / BLOCKSIZE;

      for (int red = 0; red < CombinedPCGId::nReduction; ++red) {
        for (int n = 0; n < Nblock; ++n) {
          reductions[red] += tmp[n + Nblock * red];
        }
      }
    }

    const auto mpiType = (std::is_same<T, float>::value) ? MPI_FLOAT : MPI_DFLOAT;
    MPI_Allreduce(MPI_IN_PLACE, reductions.data(), CombinedPCGId::nReduction, MPI_DFLOAT, MPI_SUM, platform->comm.mpiComm);

    platform->flopCounter->add("CombinedPCGReductions",
                               this->FPfactor * this->Nfields * static_cast<double>(this->Nlocal) * 3 * 7);
  };

  int runStandard(const dfloat tol, const int MAXIT, dfloat &rdotr, occa::memory &o_r, occa::memory &o_x)
  {
    dfloat rdotz1;
    dfloat alpha;

    int iter = 0;
    do {
      iter++;
      const dfloat rdotz2 = rdotz1;
      if (preco) {
        preco(o_r, o_z);

        rdotz1 = platform->linAlg->weightedInnerProdMany<T>(this->Nlocal,
                                                         this->Nfields,
                                                         this->fieldOffset,
                                                         o_weight,
                                                         o_r,
                                                         o_z,
                                                         platform->comm.mpiComm);
      } else {
        rdotz1 = rdotr;
      }

      if (platform->comm.mpiRank == 0) {
        nekrsCheck(std::isnan(rdotz1),
                   MPI_COMM_SELF,
                   EXIT_FAILURE,
                   "%s\n",
                   "Detected invalid rdotz norm while running linear solver!");
      }

#ifdef DEBUG
      printf("norm rdotz1: %.15e\n", rdotz1);
#endif

      dfloat beta = 0;
      if (iter > 1) {
        beta = rdotz1 / rdotz2;
        if (flexible) {
          const auto zdotAp = platform->linAlg->weightedInnerProdMany<T>(this->Nlocal,
                                                                         this->Nfields,
                                                                         this->fieldOffset,
                                                                         o_weight,
                                                                         o_z,
                                                                         o_Ap,
                                                                         platform->comm.mpiComm);
          beta = -alpha * zdotAp / rdotz2;
#ifdef DEBUG
          printf("norm zdotAp: %.15e\n", zdotAp);
#endif
        }
      }

#ifdef DEBUG
      printf("beta: %.15e\n", beta);
#endif

      platform->linAlg->axpbyMany<T>(this->Nlocal, this->Nfields, this->fieldOffset, 1.0, o_z, static_cast<dfloat>(beta), o_p);

      Ax(o_p, o_Ap);

      const dfloat pAp = platform->linAlg->weightedInnerProdMany<T>(this->Nlocal,
                                                                 this->Nfields,
                                                                 this->fieldOffset,
                                                                 o_weight,
                                                                 o_p,
                                                                 o_Ap,
                                                                 platform->comm.mpiComm);
      alpha = rdotz1 / (pAp + this->tiny);

#ifdef DEBUG
      printf("norm pAp: %.15e\n", pAp);
      printf("alpha: %.15e\n", alpha);
#endif

      //  x <= x + alpha*p
      //  r <= r - alpha*A*p
      //  dot(r,r)
      rdotr = std::sqrt(update(o_p, o_Ap, alpha, o_x, o_r));
#ifdef DEBUG
      printf("rdotr: %.15e\n", rdotr);
#endif
      if (platform->comm.mpiRank == 0) {
        nekrsCheck(std::isnan(rdotr),
                   MPI_COMM_SELF,
                   EXIT_FAILURE,
                   "%s\n",
                   "Detected invalid resiual norm while running linear solver!");
      }

      if (platform->verbose() && (platform->comm.mpiRank == 0)) {
        printf("it %d r norm %.15e\n", iter, rdotr);
      }
    } while (rdotr > tol && iter < MAXIT);

    return iter;
  };

  // Algo 5 from https://arxiv.org/pdf/2205.08909.pdf
  int runCombined(const dfloat tol, const int MAXIT, dfloat &rdotr, occa::memory &o_r, occa::memory &o_x)
  {
    T betakm1 = 0;
    T betakm2 = 0;
    T alphak = 0;
    T alphakm1 = 0;
    T alphakm2 = 0;

    std::array<T, CombinedPCGId::nReduction> reductions;

    auto &o_Minv = (this->o_invDiagA.isInitialized()) ? this->o_invDiagA : o_NULL;
    platform->linAlg->fill<T>(o_v.size(), 0.0, o_v);

    int iter = 0;
    do {
      iter++;
      const dlong updateX = iter > 1 && iter % 2 == 1;

      launchKernel(this->knlPrefix + "combinedPCGPreMatVec",
                   this->Nlocal,
                   updateX,
                   (o_Minv.isInitialized()) ? 1 : 0,
                   this->fieldOffset,
                   alphakm1,
                   alphakm2,
                   betakm1,
                   betakm2,
                   (updateX) ? alphakm2 / betakm2 : static_cast<T>(0),
                   o_Minv,
                   o_v,
                   o_p,
                   o_x,
                   o_r);

      platform->flopCounter->add("CombinedPCGPreMatVecKernel",
                                 this->FPfactor * this->Nfields * static_cast<double>(this->Nlocal) * 0.5 *
                                     (11 + 5));

      Ax(o_p, o_v);

      combinedReductions(o_Minv, o_v, o_p, o_r, reductions);

      const auto gammak = reductions[CombinedPCGId::gamma];
      const auto ak = reductions[CombinedPCGId::a];
      const auto bk = reductions[CombinedPCGId::b];
      const auto ck = reductions[CombinedPCGId::c];
      const auto dk = reductions[CombinedPCGId::d];
      const auto ek = reductions[CombinedPCGId::e];
      const auto fk = reductions[CombinedPCGId::f];

      alphak = dk / (ak + this->tiny);

      if (platform->comm.mpiRank == 0) {
        nekrsCheck(std::isnan(dk),
                   MPI_COMM_SELF,
                   EXIT_FAILURE,
                   "%s\n",
                   "Detected invalid rdotz norm while running linear solver!");
      }

#ifdef DEBUG
      printf("alpha: %.15e\n", alphak);
      printf("norm pAp: %.15e\n", ak); // ak = p^T A p
      printf("norm rdotz1: %.15e\n", dk);
#endif

      // r_{k+1}^T r_{k+1} = (r - alpha v)^T (r - alpha v)
      rdotr = gammak + alphak * (-2. * bk + alphak * ck);
      rdotr = std::sqrt(std::abs(rdotr));
#ifdef DEBUG
      printf("rdotr: %.15e\n", rdotr);
#endif
      if (platform->comm.mpiRank == 0) {
        nekrsCheck(std::isnan(rdotr),
                   MPI_COMM_SELF,
                   EXIT_FAILURE,
                   "%s\n",
                   "Detected invalid resiual norm while running linear solver!");
      }
      if (platform->verbose() && (platform->comm.mpiRank == 0)) {
        printf("it %d r norm %.15e\n", iter, rdotr);
      }

      // converged, update solution prior to exit
      if (rdotr <= tol) {
        const dlong singleVectorUpdate = iter % 2 == 1;
        if (platform->comm.mpiRank == 0) {
          nekrsCheck(!singleVectorUpdate && betakm1 == 0,
                     MPI_COMM_SELF,
                     EXIT_FAILURE,
                     "%s\n",
                     "Cannot update solution as beta == 0!");
        }
        const T alphaInvBeta = (!singleVectorUpdate) ? alphakm1 / betakm1 : 0;

        launchKernel(this->knlPrefix + "combinedPCGUpdateConvergedSolution",
                     this->Nlocal,
                     singleVectorUpdate,
                     (o_Minv.isInitialized()) ? 1 : 0,
                     this->fieldOffset,
                     alphak,
                     alphakm1,
                     betakm1,
                     alphaInvBeta,
                     o_Minv,
                     o_p,
                     o_r,
                     o_x);
      }

      betakm2 = betakm1;
      betakm1 = std::abs(1 + alphak * (-2. * ek + alphak * fk) / dk);

#ifdef DEBUG
      printf("beta: %.15e\n", betakm1);
#endif

      alphakm2 = alphakm1;
      alphakm1 = alphak;

    } while (rdotr > tol && iter < MAXIT);

    return iter;
  };
};
