#include <array>

#include "platform.hpp"
#include "svv.hpp"
#include "udf.hpp"


/**
 * Spectrally Vanishing Viscosity method (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5397463) 
 **/

namespace svv
{
  oogs_t *gsh = nullptr;
  mesh_t *mesh = nullptr;

  occa::memory o_svvf;

  occa::memory o_scale;
  occa::memory o_applySVV;

  bool enableSVV = false;
  bool isScaleInitialized = false;
  bool movingMesh = false;

  dlong fieldOffset;
  dlong NSfields;

  std::vector<dfloat> squareMatrixMultiply(int M, const std::vector<dfloat>& A_, const std::vector<dfloat>& B_)
  {
    double alpha = 1.0, beta = 0.0;
    char trans = 'T';
    std::vector<double> A(A_.begin(), A_.end());
    std::vector<double> B(B_.begin(), B_.end());
    std::vector<double> C(M * M, 0.0);
    dgemm_(&trans, &trans, &M, &M, &M, &alpha, A.data(), &M, B.data(), &M, &beta, C.data(), &M);

    std::vector<dfloat> C_(C.begin(), C.end());
    auto out = platform->linAlg->matrixTranspose(M, C_);
    return out;
  }

  dfloat PNLEG(const dfloat Z, const int N)
  {
    dfloat p1 = 1.0;
    if(N==0) return p1;

    dfloat p2 = Z;
    dfloat p3 = p2;

    for (int k=1; k < N; k++){
      dfloat fk = k;
      p3 = ((2.0 * fk + 1.0) * Z * p2 - fk * p1) / (fk + 1.0);
      p1 = p2;
      p2 = p3;
    }

    return p3;
  }

  std::vector<dfloat> legendreBasis()
  {
    std::vector<dfloat> B(mesh->Nq * mesh->Nq, 0.0);

    if(mesh->Nq == 1) B[0] = 1.0;

    for (int i = 0; i < mesh->Nq; i++){
      for (int j = 0; j < mesh->Nq; j++) {
        B[j * mesh->Nq + i] = PNLEG(mesh->r[j], i);
      }
    }
    return B;
  }

  void convoluteDerivative(const dfloat NSVV, occa::memory& o_svvD, occa::memory& o_svvDT)
  {
    auto B = legendreBasis();
    auto Binv = platform->linAlg->matrixInverse(mesh->Nq, B);

    std::vector<dfloat> Q(mesh->Nq * mesh->Nq, 0.0);
    for (int i=0; i < mesh->Nq; i++){
      Q[i * mesh->Nq + i] = pow(i/ dfloat(mesh->N), dfloat(mesh->N)/NSVV);
    }

    auto tmp = squareMatrixMultiply(mesh->Nq, B, Q);

    auto cmat = squareMatrixMultiply(mesh->Nq, tmp, Binv);

    std::vector<dfloat> D(mesh->D, mesh->D + mesh->Nq * mesh->Nq);
    auto svvD = squareMatrixMultiply(mesh->Nq, cmat, D);
    auto svvDT = platform->linAlg->matrixTranspose(mesh->Nq, svvD);

    o_svvD.copyFrom(svvD.data());
    o_svvDT.copyFrom(svvDT.data());
  }

  void setup(mesh_t *mesh_, const dlong _fieldOffset, const dlong _NSfields)
  {
    mesh = mesh_;
    gsh = mesh->oogs;

    fieldOffset = _fieldOffset;
    NSfields = _NSfields;

    o_svvf = platform->device.malloc<dfloat>(fieldOffset);

    enableSVV = true;

    std::vector<dfloat> scale(NSfields, 0.1);

    std::vector<dlong> applySVV(NSfields, 0);

    for (int is = 0; is < NSfields; is++) {
      const auto sid = scalarDigitStr(is);

      if(platform->options.compareArgs("SCALAR" + sid + " REGULARIZATION METHOD","SVV")) {
        applySVV[is] = 1;

        platform->options.getArgs("SCALAR" + sid + " SVV SCALING COEFF", scale[is]);
      }
    }

    o_applySVV = platform->device.malloc<dlong>(NSfields, applySVV.data());
    o_scale = platform->device.malloc<dfloat>(NSfields, scale.data());

    movingMesh = platform->options.compareArgs("MOVING MESH","TRUE");
  }

  void computeViscosityScale(const dlong vfieldOffset, const occa::memory& o_U, occa::memory& o_svvmu)
  {
    if(!enableSVV) return;

    if(!isScaleInitialized || movingMesh) 
      launchKernel("core-svv::svvMeshScale", mesh->Nelements, mesh->o_vgeo, o_svvf);

    launchKernel("core-svv::svvViscosityScale",
                 mesh->Nelements * mesh->Np,
                 vfieldOffset,
                 fieldOffset,
                 NSfields,
                 o_applySVV,
                 o_scale,
                 o_U,
                 o_svvf,
                 o_svvmu);

    isScaleInitialized = true;
  }
} //namespace svv
