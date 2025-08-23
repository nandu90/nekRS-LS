#include "platform.hpp"
#include "linAlg.hpp"

template <typename T> class gmres : public linearSolver
{

public:
  gmres(dlong _Nlocal,
        int _Nfields,
        dlong _fieldOffset,
        const occa::memory &_o_weight,
        int _nRestartVectors,
        bool _flexible,
        bool _iR,
        std::function<void(const occa::memory &o_q, occa::memory &o_Aq)> _Ax,
        std::function<void(const occa::memory &o_r, occa::memory &o_z)> _preco)
  {
    this->Nlocal = _Nlocal;
    this->Nfields = _Nfields;
    this->fieldOffset = _fieldOffset;
    o_weight = _o_weight;
    nRestartVectors = _nRestartVectors;
    flexible = _flexible;
    Ax = _Ax;
    preco = _preco;

#if 0 // disabled for now as it's not yet supported
    iR = _iR;
#else
    iR = false;
#endif

    Nblock = (this->Nlocal + BLOCKSIZE - 1) / BLOCKSIZE;

    this->tiny = 10 * std::numeric_limits<T>::min();
    this->FPfactor = (std::is_same<T, pfloat>::value) ? 0.5 : 1.0;
    this->knlPrefix = std::string("gmres::") + ((std::is_same<T, double>::value) ? "double::" : "float::")
                      + std::to_string(this->Nfields) + "::";

    residualKernel = platform->kernelRequests.load(this->knlPrefix + "fusedResidualAndNorm");
    correctionKernel = platform->kernelRequests.load(this->knlPrefix + "PGMRESSolution");
    updateSolutionKernel = platform->kernelRequests.load(this->knlPrefix + "updatePGMRESSolution");
    gsOrthoKernel = platform->kernelRequests.load(this->knlPrefix + "gramSchmidtOrthogonalization");
  };

  void solve(const dfloat tol,
             const int _maxIter,
             const occa::memory &o_rIn,
             occa::memory &o_xIn) override
  {
    this->r0Norm = platform->linAlg->weightedNorm2Many<T>(this->Nlocal,
                                                          this->Nfields, 
                                                          this->fieldOffset,
                                                          this->o_weight,
                                                          o_rIn,
                                                          platform->comm.mpiComm());
    
    nekrsCheck(std::isnan(this->r0Norm),
               MPI_COMM_SELF,
               EXIT_FAILURE,
               "%s unreasonable initial residual norm!\n",
               this->_name.c_str());
    
    this->rNorm = this->r0Norm;

    maxIter = _maxIter;
    nRestartVectors = std::min(nRestartVectors, maxIter);

    H.resize((nRestartVectors + 1) * (nRestartVectors + 1));
    sn.resize(nRestartVectors);
    cs.resize(nRestartVectors);
    s.resize(nRestartVectors + 1);

    o_y = platform->deviceMemoryPool.reserve<T>(nRestartVectors);
    h_y = platform->memoryPool.reserve<T>(o_y.size());

    o_scratch = platform->deviceMemoryPool.reserve<T>(nRestartVectors * Nblock);
    h_scratch = platform->memoryPool.reserve<T>(o_scratch.size());

    {
      const auto n = (this->Nfields > 1 ) ? this->Nfields * static_cast<size_t>(this->fieldOffset) : this->Nlocal;
 
      o_tmp = platform->deviceMemoryPool.reserve<T>(n);
      o_w = platform->deviceMemoryPool.reserve<T>(n);

      o_xCorr = platform->deviceMemoryPool.reserve<T>(n);

      o_r0 = o_rIn;
      o_r = platform->deviceMemoryPool.reserve<double>(n);
      
      if (std::is_same<T, float>::value) {
        platform->copyFloatToDoubleKernel(o_r.size(), o_r0, o_r);
      } else {
        o_r.copyFrom(o_r0);
      }

      platform->linAlg->fill<T>(n, 0.0, o_xIn);
      if (std::is_same<T, float>::value) {
        o_x = platform->deviceMemoryPool.reserve<double>(n);
        platform->copyFloatToDoubleKernel(o_x.size(), o_xIn, o_x);
      } else {
        o_x = o_xIn.slice(0, n); 
      }

      o_V = platform->deviceMemoryPool.reserve<T>(alignStride<T>(n) * nRestartVectors);
      o_Z = platform->deviceMemoryPool.reserve<T>(alignStride<T>(n) * ((flexible) ? nRestartVectors : 1));
    }

    if (platform->comm.mpiRank() == 0 && platform->verbose()) {
      auto txt = (preco) ? std::string("P") : std::string("");
      txt += std::string("GMRES") + ((flexible) ? "-flex" : "");
      printf("%s %s: initial res norm %.15e target %e \n", txt.c_str(), this->_name.c_str(), this->rNorm, tol);
    }

    int Niter = 0;
    do {
      auto NiterInner = runInner(tol, Niter, this->rNorm);
      Niter += NiterInner;

      if (iR) {
        updateSolution(NiterInner);
        this->rNorm = updateResidual();
        if (platform->comm.mpiRank() == 0 && platform->verbose()) {
          std::cout << "r-norm: " << this->rNorm << std::endl;
        }
      }

      // test for exit conditions
      if (this->rNorm < tol || Niter == maxIter) {
        if (!iR) {
          updateSolution(NiterInner, false);
        }
        break;
      }

      if (!iR) {
        updateSolution(nRestartVectors);
        this->rNorm = updateResidual();
        if (platform->comm.mpiRank() == 0 && platform->verbose()) {
         std::cout << "restarting r-norm: " << this->rNorm << std::endl;
        }
      }

    } while (this->rNorm > tol && Niter < maxIter);

    _nIter = Niter;

    if (std::is_same<T, float>::value) {
      platform->copyDoubleToFloatKernel(o_x.size(), o_x, o_xIn);
    }

    o_r.free();
    o_x.free();
    o_xCorr.free();

    o_y.free();
    h_y.free();

    o_scratch.free();
    h_scratch.free();

    o_tmp.free();
    o_r0.free();
    o_w.free();
    o_V.free();
    o_Z.free();
  };

private:
  occa::memory o_weight;
  bool flexible;
  bool iR;

  int maxIter;
  int nRestartVectors;
  int Nblock; 

  occa::memory o_V;
  occa::memory o_Z;
  occa::memory o_r0;
  occa::memory o_tmp;
  occa::memory o_w;

  occa::memory o_r;
  occa::memory o_x;
  occa::memory o_xCorr;

  occa::memory o_scratch;
  occa::memory h_scratch;

  occa::memory o_y;
  occa::memory h_y;

  std::vector<dfloat> H;
  std::vector<dfloat> sn;
  std::vector<dfloat> cs;
  std::vector<dfloat> s;

  std::function<void(const occa::memory &o_q, occa::memory &o_Aq)> Ax;
  std::function<void(const occa::memory &o_r, occa::memory &o_z)> preco;

  occa::kernel residualKernel;
  occa::kernel correctionKernel;
  occa::kernel updateSolutionKernel;
  occa::kernel gsOrthoKernel;

  double updateResidual()
  {
    auto o_wrk = platform->deviceMemoryPool.reserve<double>(Nblock);

    occa::memory o_xT = o_w;
    if (std::is_same<T, float>::value) {
      platform->copyDoubleToFloatKernel(o_x.size(), o_x, o_xT);
    } else {
      o_xT = o_x;
    }
  
    // r = r0 - Ax
    residualKernel(Nblock,
                   this->Nlocal,
                   this->fieldOffset,
                   o_weight,
                   o_r0,
                   [&](){ Ax(o_xT, o_tmp); return o_tmp; }(),
                   o_r,
                   o_wrk);

    auto flopCount = this->FPfactor * 4 * this->Nfields * static_cast<double>(this->Nlocal);
    platform->flopCounter->add("gmres evaluate residual and norm", flopCount);

    double nr = 0.0;
    if (platform->serial) {
      nr = *(o_wrk.ptr<double>());
    } else {
      auto h_wrk = platform->memoryPool.reserve<double>(o_wrk.size());
      auto wrk = h_wrk.template ptr<double>();
      o_wrk.copyTo(wrk);
      for (dlong n = 0; n < h_wrk.size(); ++n) {
        nr += wrk[n];
      }
    }
    MPI_Allreduce(MPI_IN_PLACE, &nr, 1, MPI_DOUBLE, MPI_SUM, platform->comm.mpiComm());

    return std::sqrt(nr);
  };

  void updateSolution(const int gmresUpdateSize, bool runPreco = true) 
  {
    for (int k = gmresUpdateSize - 1; k >= 0; --k) {
      auto y = h_y.template ptr<T>();

      y[k] = s[k];

      for (int m = k + 1; m < gmresUpdateSize; ++m) {
        y[k] -= H[k + m * (nRestartVectors + 1)] * y[m];
      }

      y[k] /= (H[k + k * (nRestartVectors + 1)] + this->tiny);
    }

    o_y.copyFrom(h_y.template ptr<T>(), gmresUpdateSize);

    correctionKernel(this->Nlocal,
                     this->fieldOffset,
                     gmresUpdateSize,
                     o_y,
                     (flexible) ? o_Z : o_V,
                     (flexible) ? o_xCorr : o_w);

    if (!flexible) {
      if (runPreco) {
       preco(o_w, o_xCorr);
      } else {
       o_xCorr.copyFrom(o_w);
      }
    }

    updateSolutionKernel(this->Nlocal,
                         this->fieldOffset,
                         o_xCorr,
                         o_x);

    auto flopCount = this->FPfactor * (gmresUpdateSize+1) * this->Nfields * static_cast<double>(this->Nlocal);
    platform->flopCounter->add("gmresUpdate", flopCount);
  };

  int runInner(const dfloat tol, const int iter0, dfloat &rdotr)
  {
    const auto offset = o_V.size() / nRestartVectors;

    dfloat nr = s[0] = rdotr;

    // init o_V0
    auto o_rT = [&]() {
      auto o_rT = o_w;
      if (std::is_same<T, float>::value) {
        platform->copyDoubleToFloatKernel(o_r.size(), o_r, o_w);
        return o_w;
      } else {
        return o_r;
      }
    }();
    platform->linAlg
       ->axpbyMany<T>(this->Nlocal, this->Nfields, this->fieldOffset, 1. / (nr + this->tiny), o_rT, 0.0, o_V);

    int i;
    for (i = 0; i < nRestartVectors; ++i) {

      auto o_Mv = flexible ? o_Z + i * offset : o_Z;
      preco(o_V + i * offset, o_Mv);

      Ax(static_cast<const occa::memory>(o_Mv), o_w);

      // 1 pass classical Gram-Schmidt (project o_w onto o_V)
      {
#if USE_WEIGHTED_INNER_PROD_MULTI_DEVICE
        platform->linAlg->weightedInnerProdMulti<T>(this->Nlocal,
                                                    (i + 1),
                                                    this->Nfields,
                                                    this->fieldOffset,
                                                    o_weight,
                                                    o_V,
                                                    o_w,
                                                    platform->comm.mpiComm(),
                                                    o_y);
        o_y.copyTo(h_y.template ptr<T>(), (i + 1));
#else
        platform->linAlg->weightedInnerProdMulti<T>(this->Nlocal,
                                                    (i + 1),
                                                    this->Nfields,
                                                    this->fieldOffset,
                                                    o_weight,
                                                    o_V,
                                                    o_w,
                                                    platform->comm.mpiComm(),
                                                    h_y.template ptr<T>());
        o_y.copyFrom(h_y.template ptr<T>(), (i + 1));
#endif

        gsOrthoKernel(Nblock,
                      this->Nlocal,
                      this->fieldOffset,
                      (i + 1),
                      o_weight,
                      o_y,
                      o_V,
                      o_w,
                      o_scratch);

        double flopCount = FPfactor * 5 * (i + 1) * this->Nfields * static_cast<double>(this->Nlocal);
        platform->flopCounter->add("gramSchmidt", flopCount);
      }

      // normalize
      auto nw = [&]() {
        dfloat norm = 0.0;
        if (platform->serial) {
          norm = *(o_scratch.ptr<T>());
        } else {
          auto scratch = h_scratch.template ptr<T>();
          o_scratch.copyTo(scratch);
          for (int k = 0; k < o_scratch.size(); ++k) {
            norm += scratch[k];
          }
        }
        MPI_Allreduce(MPI_IN_PLACE, &norm, 1, MPI_DFLOAT, MPI_SUM, platform->comm.mpiComm());
        return std::sqrt(norm);
      }();
      if (i < nRestartVectors - 1) {
        auto o_Vi = o_V + (i + 1) * offset;
        platform->linAlg->axpbyMany<T>(this->Nlocal,
                                       this->Nfields,
                                       this->fieldOffset,
                                       1. / (nw + this->tiny),
                                       o_w,
                                       0,
                                       o_Vi);
      }

      // apply Givens rotations to new column
      //  H(i+1,i) = ||w||_2
      H[i + 1 + i * (nRestartVectors + 1)] = nw;

      // apply Givens rotation
      for (int k = 0; k <= i; ++k) {
        H[k + i * (nRestartVectors + 1)] = h_y.template ptr<T>()[k];
      }

      for (int k = 0; k < i; ++k) {
        const dfloat h1 = H[k + i * (nRestartVectors + 1)];
        const dfloat h2 = H[k + 1 + i * (nRestartVectors + 1)];

        H[k + i * (nRestartVectors + 1)] = cs[k] * h1 + sn[k] * h2;
        H[k + 1 + i * (nRestartVectors + 1)] = -sn[k] * h1 + cs[k] * h2;
      }

      // form i-th rotation matrix
      const auto h1 = H[i + i * (nRestartVectors + 1)];
      const auto h2 = H[i + 1 + i * (nRestartVectors + 1)];
      const auto hr = 1 / (std::sqrt(h1 * h1 + h2 * h2) + this->tiny);
      cs[i] = h1 * hr;
      sn[i] = h2 * hr;

      H[i + i * (nRestartVectors + 1)] = cs[i] * h1 + sn[i] * h2;
      H[i + 1 + i * (nRestartVectors + 1)] = 0;

      // approximate residual norm
      s[i + 1] = -sn[i] * s[i];
      s[i] = cs[i] * s[i];

      rdotr = std::abs(s[i + 1]);

      if (platform->comm.mpiRank() == 0) {
        nekrsCheck(std::isnan(rdotr),
                   MPI_COMM_SELF,
                   EXIT_FAILURE,
                   "%s\n",
                   "Detected invalid resiual norm while running linear solver!");
      }

      if (!iR) {
        const auto iter = iter0 + i + 1;
        if (platform->verbose() && platform->comm.mpiRank() == 0) {
          printf("it %d r norm %.15e\n", iter, rdotr);
        }

        if (rdotr < tol || iter == maxIter) {
          return i+1;
        }
      }
    }

    return nRestartVectors;
  };
};
