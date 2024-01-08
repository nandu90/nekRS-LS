#if !defined(nekrs_nekrs_hpp_)
#define nekrs_nekrs_hpp_

#include "nekrsSys.hpp"
#include "mesh3D.h"
#include "elliptic.h"
#include "cds.hpp"
#include "linAlg.hpp"
#include "timer.hpp"
#include "platform.hpp"
#include "neknek.hpp"
#include "cvode.hpp"
#include "fldFile.hpp"
#include "randomVector.hpp"

struct nrs_t {

  bool multiSession;

  int dim, elementType;

  mesh_t *_mesh = nullptr;
  mesh_t *meshV = nullptr;

  elliptic_t *uSolver = nullptr;
  elliptic_t *vSolver = nullptr;
  elliptic_t *wSolver = nullptr;
  elliptic_t *uvwSolver = nullptr;
  elliptic_t *pSolver = nullptr;
  elliptic_t *meshSolver = nullptr;

  cds_t *cds = nullptr;

  neknek_t *neknek = nullptr;
  cvode_t *cvode = nullptr;

  oogs_t *gsh = nullptr;
  oogs_t *qqt = nullptr;

  oogs_t *gshMesh = nullptr;

  dlong ellipticWrkOffset;

  int flow;
  int cht;
  int Nscalar;
  int NVfields;

  dlong fieldOffset;
  dlong cubatureOffset;

  int timeStepConverged;

  dfloat dt[3], idt;
  dfloat g0, ig0;
  dfloat CFL, unitTimeCFL;

  double timePrevious;

  dfloat p0th[3] = {0.0, 0.0, 0.0};
  dfloat p0the = 0.0;
  dfloat dp0thdt;

  dfloat alpha0Ref;

  int nEXT;
  int nBDF;

  int tstep;
  int lastStep;
  int isOutputStep;
  int outputForceStep;

  int nRK, Nsubsteps;
  dfloat *coeffsfRK, *weightsRK, *nodesRK;
  occa::memory o_coeffsfRK, o_weightsRK;

  dfloat *U, *P;
  occa::memory o_U, o_P;

  occa::memory o_Ue;

  occa::memory o_div;

  occa::memory o_rho, o_mue;
  occa::memory o_meshRho, o_meshMue;

  dfloat *usrwrk;
  occa::memory o_usrwrk;

  occa::memory o_idH;

  occa::memory o_BF;
  occa::memory o_BFDiag;

  occa::memory o_FU;

  occa::memory o_prop, o_ellipticCoeff;

  dfloat *coeffEXT, *coeffBDF;
  occa::memory o_coeffEXT, o_coeffBDF;

  int *EToB;
  int *EToBMeshVelocity;
  occa::memory o_EToB;
  occa::memory o_EToBMeshVelocity;

  occa::memory o_EToBVVelocity;
  occa::memory o_EToBVMeshVelocity;

  occa::memory o_Uc, o_Pc;
  occa::memory o_prevProp;

  occa::memory o_relUrst;
  occa::memory o_Urst;

  occa::properties *kernelInfo;

  int filterNc;
  dfloat filterS;
  occa::memory o_filterRT;

  occa::kernel filterRTKernel;
  occa::kernel advectMeshVelocityKernel;
  occa::kernel pressureAddQtlKernel;
  occa::kernel pressureStressKernel;
  occa::kernel extrapolateKernel;

  occa::kernel subCycleRKKernel;
  occa::kernel subCycleInitU0Kernel;
  occa::kernel nStagesSum3Kernel;
  occa::kernel wgradientVolumeKernel;

  occa::kernel subCycleStrongCubatureVolumeKernel;
  occa::kernel subCycleStrongVolumeKernel;

  occa::kernel computeFaceCentroidKernel;
  occa::kernel computeFieldDotNormalKernel;

  occa::kernel UrstCubatureKernel;
  occa::kernel UrstKernel;

  occa::kernel advectionVolumeKernel;
  occa::kernel advectionCubatureVolumeKernel;

  occa::kernel strongAdvectionVolumeKernel;
  occa::kernel strongAdvectionCubatureVolumeKernel;

  occa::kernel gradientVolumeKernel;

  occa::kernel wDivergenceVolumeKernel;
  occa::kernel divergenceVolumeKernel;
  occa::kernel divergenceSurfaceKernel;

  occa::kernel divergenceStrongVolumeKernel;
  occa::kernel sumMakefKernel;
  occa::kernel pressureRhsKernel;
  occa::kernel pressureDirichletBCKernel;

  occa::kernel velocityRhsKernel;
  occa::kernel velocityNeumannBCKernel;
  occa::kernel velocityDirichletBCKernel;

  occa::kernel cflKernel;

  occa::kernel setEllipticCoeffKernel;
  occa::kernel setEllipticCoeffPressureKernel;

  occa::kernel curlKernel;

  occa::kernel SijOijKernel;

  occa::kernel maskCopyKernel;
  occa::kernel maskCopy2Kernel;
  occa::kernel maskKernel;

  occa::memory o_zeroNormalMaskVelocity;
  occa::memory o_zeroNormalMaskMeshVelocity;
  occa::kernel averageNormalBcTypeKernel;
  occa::kernel fixZeroNormalMaskKernel;
  occa::kernel initializeZeroNormalMaskKernel;

  occa::kernel applyZeroNormalMaskKernel;

  nrs_t(MPI_Comm comm, bool ms, setupAide &options);

  void finalize();

  void evaluateProperties(const double timeNew);

  int numberActiveFields();

  dfloat viscousDrag(int nbID, const occa::memory &o_bID, occa::memory &o_Sij);

  //       ( SO0          )         (     SO8  SO7)
  // Sij = ( SO3  SO1     )  Oij =  (          SO6)
  //       ( SO5  SO4  SO2)         (             )
  void strainRotationRate(bool smooth, bool rotationRate, const occa::memory &o_U, occa::memory &o_SO);

  void strainRate(bool smooth, const occa::memory &o_U, occa::memory &o_S);

  void Qcriterion(const occa::memory &o_U, occa::memory &o_Q);
};

static std::vector<std::string> nrsFieldsToSolve(setupAide &options)
{
  int Nscalar = 0;
  options.getArgs("NUMBER OF SCALARS", Nscalar);

  std::vector<std::string> fields;

  if (!options.compareArgs("MESH SOLVER", "NONE")) {
    fields.push_back("mesh");
  }

  if (!options.compareArgs("VELOCITY SOLVER", "NONE")) {
    fields.push_back("velocity");
  }

  for (int i = 0; i < Nscalar; i++) {
    const auto sid = scalarDigitStr(i);
    if (!options.compareArgs("SCALAR" + sid + " SOLVER", "NONE")) {
      fields.push_back("scalar" + sid);
    }
  }
  return fields;
}

#endif
