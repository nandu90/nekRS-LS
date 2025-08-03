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

    this->tiny = 10 * std::numeric_limits<T>::min();
    this->FPfactor = (std::is_same<T, pfloat>::value) ? 0.5 : 1.0;
    auto type = ((std::is_same<T, pfloat>::value && !std::is_same<dfloat, pfloat>::value) ? std::string("pfloat") : std::string(""));
    this->knlPrefix = std::string("gmres::") + type + std::to_string(this->Nfields) + "-";
  };

  int solve(const dfloat tol, const int MAXIT, dfloat &rdotr, occa::memory &o_r, occa::memory &o_x) override
  {
    nRestartVectors = std::min(nRestartVectors, MAXIT);

    H.resize((nRestartVectors + 1) * (nRestartVectors + 1));
    sn.resize(nRestartVectors);
    cs.resize(nRestartVectors);
    s.resize(nRestartVectors + 1);

    o_y = platform->deviceMemoryPool.reserve<T>(nRestartVectors);
    auto h_y = platform->memoryPool.reserve<T>(o_y.size());
    y = h_y.template ptr<T>();

    o_scratch =
        platform->deviceMemoryPool.reserve<T>(nRestartVectors * ((this->Nlocal + BLOCKSIZE - 1) / BLOCKSIZE));
    auto h_scratch = platform->memoryPool.reserve<T>(o_scratch.size());
    scratch = h_scratch.template ptr<T>();

    const auto n = this->Nfields * static_cast<size_t>(this->fieldOffset);

    o_tmp = platform->deviceMemoryPool.reserve<T>(n);
    o_r0 = platform->deviceMemoryPool.reserve<T>(n);
    o_w = platform->deviceMemoryPool.reserve<T>(n);

    o_V = platform->deviceMemoryPool.reserve<T>(n * nRestartVectors);
    o_Z = platform->deviceMemoryPool.reserve<T>(n * ((flexible) ? nRestartVectors : 1));

    if (platform->comm.mpiRank == 0 && platform->verbose()) {
      auto txt = (preco) ? std::string("P") : std::string(""); 
      txt += std::string("GMRES") + ((flexible) ? "-flex" : "");
      printf("%s %s: initial res norm %.15e target %e \n", txt.c_str(), this->_name.c_str(), rdotr, tol);
    }

    const auto Niter = run(tol, MAXIT, rdotr, o_r, o_x);

    o_y.free();
    y = nullptr;

    o_scratch.free();
    scratch = nullptr;

    o_tmp.free();
    o_r0.free();
    o_w.free();
    o_V.free();
    o_Z.free();

    return Niter;
  };

private:
  occa::memory o_weight;
  bool flexible;

  int nRestartVectors;

  occa::memory o_V;
  occa::memory o_Z;
  occa::memory o_r0;
  occa::memory o_tmp;
  occa::memory o_w;

  occa::memory o_scratch;
  T *scratch;

  occa::memory o_y;
  T *y;

  std::vector<dfloat> H;
  std::vector<dfloat> sn;
  std::vector<dfloat> cs;
  std::vector<dfloat> s;

  std::function<void(const occa::memory &o_q, occa::memory &o_Aq)> Ax;
  std::function<void(const occa::memory &o_r, occa::memory &o_z)> preco;

  int run(const dfloat tol, const int MAXIT, dfloat &rdotr, occa::memory &o_r, occa::memory &o_x)
  {
    const auto offset = this->fieldOffset * this->Nfields;
    const int Nblock = (this->Nlocal + BLOCKSIZE - 1) / BLOCKSIZE;

    o_r0.copyFrom(o_r, o_r.size());

    dfloat nr = rdotr;

    auto update = [&](int gmresUpdateSize, bool runPreco = true) {
      for (int k = gmresUpdateSize - 1; k >= 0; --k) {
        y[k] = s[k];

        for (int m = k + 1; m < gmresUpdateSize; ++m) {
          y[k] -= H[k + m * (nRestartVectors + 1)] * y[m];
        }

        y[k] /= (H[k + k * (nRestartVectors + 1)] + this->tiny);
      }

      o_y.copyFrom(y, gmresUpdateSize);

      if (flexible) {
        launchKernel(this->knlPrefix + "updatePGMRESSolution",
                     this->Nlocal,
                     this->fieldOffset,
                     gmresUpdateSize,
                     o_y,
                     o_Z,
                     o_x);
      } else {
        platform->linAlg->fill<T>(this->Nfields * this->fieldOffset, 0.0, o_tmp);
        launchKernel(this->knlPrefix + "updatePGMRESSolution",
                     this->Nlocal,
                     this->fieldOffset,
                     gmresUpdateSize,
                     o_y,
                     o_V,
                     o_tmp);

        auto &o_tmp2 = (runPreco) ? o_w : o_tmp;
        if (runPreco) {
          preco(o_tmp, o_tmp2);
        }
        platform->linAlg->axpbyMany<T>(this->Nlocal, this->Nfields, this->fieldOffset, 1.0, o_tmp2, 1.0, o_x);

        double flopCount =
            this->FPfactor * 2 * gmresUpdateSize * this->Nfields * static_cast<double>(this->Nlocal);
        platform->flopCounter->add("gmresUpdate", flopCount);
      }
    };

    int iter;
    for (iter = 0; iter < MAXIT;) {
      s[0] = nr;

      // normalize
      platform->linAlg
          ->axpbyMany<T>(this->Nlocal, this->Nfields, this->fieldOffset, 1. / (nr + this->tiny), o_r, 0.0, o_V);

      for (int i = 0; i < nRestartVectors; ++i) {

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
                                                   platform->comm.mpiComm,
                                                   o_y);
          o_y.copyTo(y, (i + 1));
#else
          platform->linAlg->weightedInnerProdMulti<T>(this->Nlocal,
                                                   (i + 1),
                                                   this->Nfields,
                                                   this->fieldOffset,
                                                   o_weight,
                                                   o_V,
                                                   o_w,
                                                   platform->comm.mpiComm,
                                                   y);
          o_y.copyFrom(y, (i + 1));
#endif

          launchKernel(this->knlPrefix + "gramSchmidtOrthogonalization",
                       Nblock,
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
            o_scratch.copyTo(scratch, Nblock);
            for (int k = 0; k < Nblock; ++k) {
              norm += scratch[k];
            }
          }
          MPI_Allreduce(MPI_IN_PLACE, &norm, 1, MPI_DFLOAT, MPI_SUM, platform->comm.mpiComm);
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
          H[k + i * (nRestartVectors + 1)] = y[k];
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

        iter++;
        rdotr = std::abs(s[i + 1]);

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

        if (rdotr < tol || iter == MAXIT) {
          // update approximation
          update(i + 1, false);
          break;
        }
      }

      // exit if tolerance is reached
      if (rdotr < tol || iter == MAXIT) {
        break;
      }

      // update approximation
      update(nRestartVectors);

      Ax(o_x, o_tmp);

      {
        launchKernel(this->knlPrefix + "fusedResidualAndNorm",
                     Nblock,
                     this->Nlocal,
                     this->fieldOffset,
                     o_weight,
                     o_r0,
                     o_tmp,
                     o_r,
                     o_scratch);

        auto flopCount = this->FPfactor * 4 * this->Nfields * static_cast<double>(this->Nlocal);
        platform->flopCounter->add("gmres evaluate residual and norm", flopCount);
      }

      if (platform->serial) {
        nr = *(o_scratch.ptr<T>());
      } else {
        o_scratch.copyTo(scratch, Nblock);
        nr = 0.0;
        for (dlong n = 0; n < Nblock; ++n) {
          nr += scratch[n];
        }
      }
      MPI_Allreduce(MPI_IN_PLACE, &nr, 1, MPI_DFLOAT, MPI_SUM, platform->comm.mpiComm);
      nr = std::sqrt(nr);
      rdotr = nr;

      if (rdotr <= tol) {
        return iter;
      }
    }

    return iter;
  };
};
