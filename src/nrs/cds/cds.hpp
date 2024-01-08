#ifndef CDS_H
#define CDS_H

#include "nekrsSys.hpp"
#include "mesh3D.h"
#include "elliptic.h"
#include "neknek.hpp"
#include "cvode.hpp"

struct cdsConfig_t {
  int Nscalar;
  mesh_t *mesh;
  mesh_t *meshV;
  int dim;
  dfloat *g0;
  dfloat *idt;
  dfloat *dt;
  int nBDF;
  occa::memory o_coeffBDF;
  int nEXT;
  occa::memory o_coeffEXT;
  occa::memory *o_usrwrk;
  dlong vFieldOffset;
  dlong vCubatureOffset;
  dlong fieldOffset;
  int Nsubsteps;
  occa::memory o_ellipticCoeff;
  occa::memory o_U;
  occa::memory o_Ue;
  occa::memory o_Urst;
  occa::memory o_relUrst;
};

struct cds_t {
  static constexpr double targetTimeBenchmark{0.2};

  cds_t(cdsConfig_t &cfg);
  occa::memory solve(int i, double time, int stage);

  std::vector<mesh_t *> mesh;
  std::vector<dlong> fieldOffset;
  std::vector<dlong> fieldOffsetScan;
  occa::memory o_fieldOffsetScan;
  dlong fieldOffsetSum;
  mesh_t *meshV;
  std::vector<elliptic_t *> solver;
  neknek_t *neknek = nullptr;
  cvode_t *cvode = nullptr;

  bool anyCvodeSolver = false;
  bool anyEllipticSolver = false;

  int NVfields;
  int NSfields;

  oogs_t *gsh, *gshT;

  dlong vFieldOffset;
  dlong vCubatureOffset;
  dfloat *idt;
  dfloat *dt;
  dfloat *g0;

  int nEXT;
  int nBDF;

  std::vector<int> compute;
  std::vector<int> cvodeSolve;
  occa::memory o_compute;
  occa::memory o_cvodeSolve;

  dfloat *S;

  int filterNc;
  std::vector<dfloat> filterS;
  dfloat *filterM;
  occa::memory o_applyFilterRT;
  occa::memory o_filterS;
  occa::memory o_filterRT;
  int applyFilter;

  int Nsubsteps;

  int *EToB;
  occa::memory o_EToB;
  dlong EToBOffset;

  occa::memory *o_usrwrk;

  occa::memory o_U;
  occa::memory o_Ue;
  occa::memory o_relUrst;
  occa::memory o_Urst;

  occa::memory o_S, o_Se;
  occa::memory o_prop, o_ellipticCoeff;
  occa::memory o_rho, o_diff;
  occa::memory o_FS, o_BF, o_BFDiag;

  occa::memory o_coeffEXT, o_coeffBDF;

  occa::kernel sumMakefKernel;
  occa::kernel subCycleStrongCubatureVolumeKernel;
  occa::kernel subCycleStrongVolumeKernel;
  occa::kernel filterRTKernel;
  occa::kernel advectionVolumeKernel;
  occa::kernel advectionSurfaceKernel;
  occa::kernel advectionCubatureVolumeKernel;
  occa::kernel advectionCubatureSurfaceKernel;
  occa::kernel strongAdvectionVolumeKernel;
  occa::kernel strongAdvectionCubatureVolumeKernel;
  occa::kernel advectMeshVelocityKernel;
  occa::kernel neumannBCKernel;
  occa::kernel dirichletBCKernel;
  occa::kernel setEllipticCoeffKernel;
  occa::kernel maskCopyKernel;
  occa::kernel maskCopy2Kernel;

  occa::properties *kernelInfo;
};

#endif
