#include "cvode.hpp"
#include "elliptic.h"
#include "inipp.hpp"
#include "nrs.hpp"
#include "nekInterfaceAdapter.hpp"
#include "Urst.hpp"
#include <cstdlib>
#include <limits>
#include <array>
#include <numeric>
#include "nrssys.hpp"
#include "ogs.hpp"
#include "udf.hpp"

#include "timeStepper.hpp"
#include "plugins/lowMach.hpp"
#include "bdry.hpp"
#include "tabularPrinter.hpp"

#ifdef ENABLE_CVODE

// cvode includes
#include "sunlinsol/sunlinsol_spgmr.h"
#include "sundials/sundials_types.h"
#include "sundials/sundials_math.h"
#include "cvode/cvode.h"

#include "nvector/nvector_serial.h"
#include "nvector/nvector_mpiplusx.h"

#ifdef ENABLE_CUDA
#include "nvector/nvector_cuda.h"
#endif
#ifdef ENABLE_HIP
#include "nvector/nvector_hip.h"
#endif

#include "getN_VectorMemory.hpp"
#include "cbGMRES.hpp"

#endif

namespace {

void check_retval(void *returnvalue, const char *funcname, int opt)
{
  int *retval;

  /* Check if SUNDIALS function returned NULL pointer - no memory allocated */

  if (opt == 0 && returnvalue == NULL) {
    nrsAbort(MPI_COMM_SELF,
             EXIT_FAILURE,
             "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
             funcname);
  }
  /* Check if retval < 0 */
  else if (opt == 1) {
    retval = (int *)returnvalue;
    if (*retval < 0) {
      nrsAbort(MPI_COMM_SELF,
               EXIT_FAILURE,
               "\nSUNDIALS_ERROR: %s() failed with retval = %d\n\n",
               funcname,
               *retval);
    }
  }
  /* Check if function returned NULL pointer - no memory allocated */
  else if (opt == 2 && returnvalue == NULL) {
    nrsAbort(MPI_COMM_SELF,
             EXIT_FAILURE,
             "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
             funcname);
  }
}

} // namespace

cvode_t::cvode_t(nrs_t *_nrs)
{
  if (platform->comm.mpiRank == 0) {
    std::cout << "initializing CVODE ...\n";
  }

  this->nrs = _nrs;
  auto cds = nrs->cds;

  o_coeffExt = platform->device.malloc<dfloat>(maxTimestepperOrder);

  o_U = platform->device.malloc<dfloat>(nrs->nEXT * nrs->NVfields * nrs->fieldOffset);

  if (platform->options.compareArgs("MOVING MESH", "TRUE")) {
    o_meshU = platform->device.malloc<dfloat>(nrs->nEXT * nrs->NVfields * nrs->fieldOffset);
    o_xyz0 = platform->device.malloc<dfloat>(nrs->NVfields * nrs->fieldOffset);
  }

  if (platform->options.getArgs("CVODE RECYCLE PROPERTIES").empty()) {
    platform->options.setArgs("CVODE RECYCLE PROPERTIES", "TRUE");
  }
  recycleProperties = platform->options.compareArgs("CVODE RECYCLE PROPERTIES", "TRUE");

  this->scalarIds = std::vector<dlong>();
  this->cvodeScalarIds = std::vector<dlong>(cds->NSfields, -1);

  Nscalar = 0;

  this->minCvodeScalarId = std::numeric_limits<dlong>::max();
  this->maxCvodeScalarId = -std::numeric_limits<dlong>::max();

  for (int is = 0; is < cds->NSfields; is++) {
    if (!cds->compute[is]) {
      continue;
    }
    if (!cds->cvodeSolve[is]) {
      continue;
    }
    this->minCvodeScalarId = std::min(minCvodeScalarId, is);
    this->maxCvodeScalarId = std::max(maxCvodeScalarId, is);
  }

  // assumption: CVODE scalars are contiguous
  bool valid = true;
  for (int is = this->minCvodeScalarId; is <= this->maxCvodeScalarId; ++is) {
    valid &= cds->compute[is];
    valid &= cds->cvodeSolve[is];
  }

  nrsCheck(!valid, platform->comm.mpiComm, EXIT_FAILURE, "%s\n", "CVODE scalars must be contiguous");

  std::vector<mesh_t*> meshes;

  for (int is = 0; is < cds->NSfields; is++) {
    if (!cds->compute[is]) {
      continue;
    }
    if (!cds->cvodeSolve[is]) {
      continue;
    }

    cvodeScalarIds[is] = Nscalar;
    scalarIds.push_back(is);

    Nscalar++;

    meshes.push_back(cds->mesh[is]);
  }

  YLVec = std::make_shared<LVector_t<dfloat>>(meshes, false);
  YdotLVec = std::make_shared<LVector_t<dfloat>>(meshes, false);

  std::vector<dlong> lengths_(this->Nscalar, 0);
  this->nEq = 0;
  for(int is = 0; is < Nscalar; ++is){
    const auto Nlocal = YLVec->Nlocal(is);
    this->nEq += Nlocal;
    lengths_[is] = Nlocal;
  }

  // use compact L-vector layout for CVODE
  YLVec->fieldOffsets(lengths_);
  YdotLVec->fieldOffsets(lengths_);

  o_scalarIds = platform->device.malloc<dlong>(scalarIds.size(), scalarIds.data());
  o_cvodeScalarIds = platform->device.malloc<dlong>(cvodeScalarIds.size(), cvodeScalarIds.data());

  setupDirichletMask();

  this->weakLaplacianKernel = platform->kernels.get("cvode_t::weakLaplacianHex3D");
  this->extrapolateDirichletKernel = platform->kernels.get("cvode_t::extrapolateDirichlet");
  this->mapToMaskedPointKernel = platform->kernels.get("cvode_t::mapToMaskedPoint");
  this->errorWeightKernel = platform->kernels.get("cvode_t::errorWeight");
  this->fusedAddRhoDivKernel = platform->kernels.get("cvode_t::rhoDiv");

  isInitialized = false;

  verboseCVODE = platform->options.compareArgs("CVODE VERBOSE", "TRUE");
  sharedRho = platform->options.compareArgs("CVODE SHARED RHO", "TRUE");
  if (scalarIds.size() < 2) {
    sharedRho = false;
  }

  this->gsh = oogs::setup(nrs->meshV->ogs, this->Nscalar, nrs->fieldOffset, ogsDfloat, NULL, OOGS_AUTO);

  mixedPrecisionJtvEnabled = platform->options.compareArgs("CVODE MIXED PRECISION JTV", "TRUE");
  nrsCheck(mixedPrecisionJtvEnabled,
           platform->comm.mpiComm,
           EXIT_FAILURE,
           "%s\n",
           "CVODE MIXED PRECISION JTV = TRUE not supported yet");

  {
    auto mesh = cds->mesh[0];
    o_vgeoPfloat = platform->device.malloc<pfloat>(mesh->Nelements * mesh->Np * mesh->Nvgeo);
    platform->copyDfloatToPfloatKernel(mesh->Nelements * mesh->Np * mesh->Nvgeo, mesh->o_vgeo, o_vgeoPfloat);
  }

  o_invRhoCpAvg = platform->o_memPool.reserve<dfloat>(cds->fieldOffset[0]);

  if (platform->comm.mpiRank == 0) {
    std::cout << "done\n";
  }

}

#ifdef ENABLE_CVODE
int cvode_t::cvodeRHS(realtype time, N_Vector Y, N_Vector Ydot) {
  auto o_y = getN_VectorMemory(sunrealtype, Y);
  auto o_ydot = getN_VectorMemory(sunrealtype, Ydot);
  
  this->YLVec->optr(o_y);
  this->YdotLVec->optr(o_ydot);

  this->rhs(time, *(this->YLVec), *(this->YdotLVec));

  return 0;
}

// callback used by cvode to compute Jv
// linear solver requires matVec: Mv = (I - gamma*J)v
int cvode_t::cvodeJtv(N_Vector v, N_Vector Jv, realtype t, N_Vector y, N_Vector fy, N_Vector ytemp) 
{
#if 0
  int iter, retval;

  retval = CVodeGetErrWeights(cvodeMem, ytemp);
  if (retval != CV_SUCCESS)
    return (retval);

  auto o_ytemp = getN_VectorMemory(sunrealtype, ytemp);
  auto o_Jv = getN_VectorMemory(sunrealtype, Jv);

  /* Initialize perturbation to 1/||v|| */
  const realtype sig = sigScale / N_VWrmsNorm_MPIManyVector(v, ytemp);
  const realtype siginv = 1.0 / sig;

  std::cout << "sig = " << sig << std::endl;

  N_VLinearSum(sig, v, 1.0, y, ytemp);
  this->jtvRhs(t, o_ytemp, o_Jv);
  N_VLinearSum(siginv, Jv, -siginv, fy, Jv);

  return (0);
#else
  const auto o_v = getN_VectorMemory(sunrealtype, v);
  auto o_Jv = getN_VectorMemory(sunrealtype, Jv);
  const auto o_y = getN_VectorMemory(sunrealtype, y);
  const auto o_fy = getN_VectorMemory(sunrealtype, fy);
  auto o_ytemp = getN_VectorMemory(sunrealtype, ytemp);

  return this->jtv(t, o_v, o_y, o_fy, o_ytemp, o_Jv);
#endif
}

// Jv = [f(y + v*sig) - f(y)]/sig, where sig = sigScale / ||v||_WRMS, 
int cvode_t::jtv(double t, const occa::memory& o_v, const occa::memory& o_y, const occa::memory& o_fy, 
                 occa::memory& o_work, occa::memory& o_Jv) 
{
  if (detailedTimersEnabled)
    platform->timer.tic(timerName + "solve::cvode::linearSolve::jtv", 0);

  const auto norm = sqrt(platform->linAlg->weightedSqrSum(this->nEq, o_ewt, o_v, platform->comm.mpiComm) / this->nEqTotal);
  const auto sig = sigScale / norm;
  const auto siginv = 1.0 / sig;

  platform->linAlg->axpbyz(this->nEq, sig, o_v, 1.0, o_y, o_work);
  this->jtvRhs(t, o_work, o_Jv);
  platform->linAlg->axpbyz(this->nEq, siginv, o_Jv, -siginv,  o_fy, o_Jv);

  if (detailedTimersEnabled)
    platform->timer.toc(timerName + "solve::cvode::linearSolve::jtv");

  return 0;
}

int cvode_t::cvodeErrorWt(N_Vector y, N_Vector ewt)
{
  if (detailedTimersEnabled)
    platform->timer.tic(timerName + "solve::cvode::errorWt", 0);

  auto o_y = getN_VectorMemory(sunrealtype, y);
  o_ewt = getN_VectorMemory(sunrealtype, ewt);
 
  auto cds = nrs->cds;
  const auto cvodeCHT = nrs->cht && cds->cvodeSolve[0];
  if(cvodeCHT){
    this->errorWeightKernel(YLVec->Nlocal(0),
                            1,
                            this->relTol,
                            YLVec->invDegree(0),
                            this->o_absTol,
                            o_y,
                            o_ewt);
    if(this->Nscalar > 1) {
      auto NlocalL = YLVec->Nlocal(1);
      auto o_yV = o_y + NlocalL;
      auto o_ewtV = o_ewt + NlocalL;
      auto o_absV = this->o_absTol + 1;
      this->errorWeightKernel(NlocalL,
                              this->Nscalar-1,
                              this->relTol,
                              YLVec->invDegree(1),
                              o_absV,
                              o_yV,
                              o_ewtV);
    }
  } else {
    this->errorWeightKernel(YLVec->Nlocal(0),
                            this->Nscalar,
                            this->relTol,
                            YLVec->invDegree(0),
                            this->o_absTol,
                            o_y,
                            o_ewt);
  }

  if (detailedTimersEnabled)
    platform->timer.toc(timerName + "solve::cvode::errorWt");

  return 0;
}
#endif

void cvode_t::initialize()
{

  if (isInitialized) {
    return;
  }
  isInitialized = true;

  auto *cds = nrs->cds;

  int retval;

#ifdef ENABLE_CVODE
  auto fwdCvodeRHS = [](double time, N_Vector Y, N_Vector Ydot, void *user_data) {
    return static_cast<cvode_t*>(user_data)->cvodeRHS(time, Y, Ydot);
  };
 
  auto fwdLinearSolve = [](SUNLinearSolver S, SUNMatrix A, N_Vector x, N_Vector b, realtype tol) {
    const std::string timerName = "cvode_t::";
    std::string solverType;
    platform->options.getArgs("CVODE SOLVER", solverType);
 
    platform->timer.tic(timerName + "solve::cvode::linearSolve", 0);
    int retVal;
    if(solverType == "CBGMRES") {
      retVal = cbGMRES(S, x, b, tol);
    } else if (solverType == "GMRES"){
      retVal = SUNLinSolSolve_SPGMR(S, NULL, x, b, tol);
    }
    platform->timer.toc(timerName + "solve::cvode::linearSolve");
    return retVal;
  };

  // same as cvLsDQJtimes, but with scaling for sig
  auto fwdCvodeJtv = [](N_Vector v, N_Vector Jv, realtype t, N_Vector y, N_Vector fy, void *user_data, N_Vector work) {
    return static_cast<cvode_t*>(user_data)->cvodeJtv(v, Jv, t, y, fy, work);
  };

  SUNContext sunctx = nullptr;
  retval = SUNContext_Create((void *)&platform->comm.mpiComm, &sunctx);
  check_retval(&retval, "SUNContext_Create", 1);

  const int blockSize = BLOCKSIZE;

  {
    this->y = nullptr;
    if (platform->device.mode() == "CUDA") {
#ifdef ENABLE_CUDA
      auto occaStream = platform->device.occaDevice().getStream();
      auto backendStreamNative = occa::unwrap(occaStream);
      SUNCudaThreadDirectExecPolicy stream_exec_policy(blockSize, *static_cast<cudaStream_t*>(backendStreamNative));
      SUNCudaBlockReduceExecPolicy reduce_exec_policy(blockSize, 0, *static_cast<cudaStream_t*>(backendStreamNative));
      this->y = N_VNew_Cuda(this->nEq, sunctx);
      check_retval((void *)this->y, "N_VNew_Cuda", 0);

      retval = N_VSetKernelExecPolicy_Cuda(this->y, &stream_exec_policy, &reduce_exec_policy);
      check_retval(&retval, "N_VSetKernelExecPolicy_Cuda", 0);

      retval = N_VEnableFusedOps_Cuda(this->y , SUNTRUE);
      check_retval(&retval, "N_VEnableFusedOps_Cuda", 1);
#else
      nrsCheck(true,
               platform->comm.mpiComm,
               EXIT_FAILURE,
               "%s",
               "CVODE ENABLE_CUDA not enabled, despite mode being CUDA!\n");
#endif
    } else if (platform->device.mode() == "HIP") {
#ifdef ENABLE_HIP
      auto occaStream = platform->device.occaDevice().getStream();
      auto backendStreamNative = occa::unwrap(occaStream);
      SUNHipThreadDirectExecPolicy stream_exec_policy(blockSize, *static_cast<hipStream_t*>(backendStreamNative));
      SUNHupBlockReduceExecPolicy reduce_exec_policy(blockSize, 0, *static_cast<hipStream_t*>(backendStreamNative));

      this->y = N_VNew_Hip(data->nEq, sunctx);
      check_retval((void *)this->y, "N_VNew_Hip", 0);

      retVal = N_VEnableFusedOps_Hip(this->y, SUNTRUE);
      check_retval(&retval, "N_VEnableFusedOps_Hip", 1);
#else
      nrsCheck(true,
               platform->comm.mpiComm,
               EXIT_FAILURE,
               "%s",
               "CVODE ENABLE_HIP not enabled, despite mode being HIP!\n");
#endif
    } else if (platform->device.mode() == "Serial") {
      this->y = N_VNew_Serial(this->nEq, sunctx);
      check_retval((void *)this->y, "N_VNew_Serial", 0);

      retval = N_VEnableFusedOps_Serial(this->y, SUNTRUE);
      check_retval(&retval, "N_VEnableFusedOps_Serial", 1);
    }
    this->cvodeY = N_VMake_MPIPlusX(platform->comm.mpiComm, this->y, sunctx);
    this->nEqTotal = N_VGetLength(this->cvodeY);

    retval = N_VEnableFusedOps_MPIPlusX(this->cvodeY, SUNTRUE);
    check_retval(&retval, "N_VEnableFusedOps_MPIPlusX", 1);
  }

  o_cvodeY = getN_VectorMemory(sunrealtype, cvodeY);

  YLVec->optr(o_cvodeY);
  YdotLVec->optr(o_cvodeY);

  // set initial condition
  nrsToCv(nrs->cds->o_S, *YLVec, false);

  if (platform->options.getArgs("CVODE ADVECTION TYPE").empty()) {
    const auto dealiasing = platform->options.compareArgs("ADVECTION TYPE", "CUBATURE") ? true : false;
    if (dealiasing)
      platform->options.setArgs("CVODE ADVECTION TYPE", "CUBATURE+CONVECTIVE");
    else
      platform->options.setArgs("CVODE ADVECTION TYPE", "CONVECTIVE");
  }

  if (platform->options.getArgs("CVODE INTEGRATOR").empty())
    platform->options.setArgs("CVODE INTEGRATOR", "BDF");

  int integrator; 
  if (platform->options.compareArgs("CVODE INTEGRATOR", "BDF")) {
    integrator = CV_BDF;
  } else if (platform->options.compareArgs("CVODE INTEGRATOR", "ADAMS")) {
    integrator = CV_ADAMS;
  } else {
    nrsAbort(platform->comm.mpiComm, EXIT_FAILURE, "%s",
             "Invalid CVODE INTEGRATOR!\n");
  }

  this->cvodeMem = CVodeCreate(integrator, sunctx);

  auto T0 = 0.0;
  platform->options.getArgs("START TIME", T0);

  this->relTol = 1e-4;
  platform->options.getArgs("CVODE RELATIVE TOLERANCE", this->relTol);

  double absTol = 1e-6;

  // populate absolute tolerance vector
  std::vector<dfloat> absTols(this->Nscalar, absTol);
  for (int is = 0; is < cds->NSfields; is++) {
    if (!cds->compute[is]) {
      continue;
    }
    if (!cds->cvodeSolve[is]) {
      continue;
    }

    const auto cvodeScalarId = cvodeScalarIds[is];
    dfloat absTolScalar = absTol;
    platform->options.getArgs("SCALAR" + scalarDigitStr(is) + " CVODE ABSOLUTE TOLERANCE", absTolScalar);
    absTols[is] = absTolScalar;
  }

  this->o_absTol = platform->device.malloc<dfloat>(this->Nscalar, absTols.data());

  if (platform->options.getArgs("CVODE SIGMA SCALE").empty())
    platform->options.setArgs("CVODE SIGMA SCALE", "1.0");

  platform->options.getArgs("CVODE SIGMA SCALE", this->sigScale);

  check_retval((void *)this->cvodeMem, "CVodeCreate", 0);
  retval = CVodeInit(this->cvodeMem, fwdCvodeRHS, T0, this->cvodeY);
  check_retval(&retval, "CVodeInit", 1);

  // provide function to compute weights to account for multiplicity
  // same as cvEwtSetSV, but with scaling for multiplicity
  CVEwtFn fwdCvodeErrorWt = [](N_Vector y, N_Vector ewt, void *user_data) {
    return static_cast<cvode_t*>(user_data)->cvodeErrorWt(y, ewt);
  };

  retval = CVodeWFtolerances(this->cvodeMem, fwdCvodeErrorWt);
  check_retval(&retval, "CVodeWFtolerances", 1);

  if (platform->options.getArgs("CVODE STOP TIME").empty())
    platform->options.setArgs("CVODE STOP TIME", "TRUE");

  if (platform->options.getArgs("CVODE SOLVER").empty())
    platform->options.setArgs("CVODE SOLVER", "GMRES");

  platform->options.getArgs("CVODE SOLVER", this->linearSolverType);
 
  SUNLinearSolver LS;

  if (this->linearSolverType.find("GMRES") != std::string::npos) {
    if (platform->options.getArgs("CVODE GMRES BASIS VECTORS").empty())
      platform->options.setArgs("CVODE GMRES BASIS VECTORS", "10");

    int nVectors;
    platform->options.getArgs("CVODE GMRES BASIS VECTORS", nVectors);

    LS = SUNLinSol_SPGMR(cvodeY, PREC_NONE, nVectors, sunctx);
    check_retval(&retval, "SUNLinSol_SPFGMR", 1);
    LS->ops->solve = fwdLinearSolve;
  } else {
    nrsCheck(true,
             platform->comm.mpiComm,
             EXIT_FAILURE,
             "CVODE SOLVER %s not supported!\n",
             linearSolverType.c_str());
  }

  retval = CVodeSetLinearSolver(this->cvodeMem, LS, NULL);
  check_retval(&retval, "CVodeSetLinearSolver", 1);

  // custom settings
#if 1
  if (platform->options.getArgs("CVODE GS TYPE").empty())
    platform->options.setArgs("CVODE GS TYPE", "CLASSICAL");

  if (linearSolverType.find("GMRES") !=std::string::npos) {
    if(platform->options.compareArgs("CVODE GS TYPE", "CLASSICAL")) {
      SUNLinSol_SPGMRSetGSType(LS, SUN_CLASSICAL_GS);
      check_retval(&retval, "SUNLinSol_SPGMRSetGSType", 1);
    } else if (platform->options.compareArgs("CVODE GS TYPE", "MODIFIED")) {
      nrsCheck(this->linearSolverType == "CBGMRES", 
               platform->comm.mpiComm, EXIT_FAILURE, "%s",
               "CVODE GS TYPE MODIFIED not available for CBGMRES!\n");
    } else {
      nrsAbort(platform->comm.mpiComm, EXIT_FAILURE, "%s",
               "Invalid CVODE GS TYPE!\n");
    }
  }

  retval = CVodeSetJacTimes(this->cvodeMem, NULL, fwdCvodeJtv);
  check_retval(&retval, "CVodeSetJacTimes", 1);

  if (platform->options.getArgs("CVODE MAX STEPS").empty())
    platform->options.setArgs("CVODE MAX STEPS", "100");

  int mxsteps;
  platform->options.getArgs("CVODE MAX STEPS", mxsteps);
  retval = CVodeSetMaxNumSteps(this->cvodeMem, mxsteps);

  if (platform->options.getArgs("CVODE MAX TIMESTEPPER ORDER").empty())
    platform->options.setArgs("CVODE MAX TIMESTEPPER ORDER", "3");

  int maxOrder;
  platform->options.getArgs("CVODE MAX TIMESTEPPER ORDER", maxOrder);
  retval = CVodeSetMaxOrd(this->cvodeMem, maxOrder);
  check_retval(&retval, "CVodeSetMaxOrd", 1);

  // non-linear convergence factor
  double epsNl = 0.1;
  retval = CVodeSetNonlinConvCoef(this->cvodeMem, epsNl);
  check_retval(&retval, "CVodeSetNonlinConvCoef", 1);

  // linear convergence safety factor
  // linear residual satisfies |r|_wrms < 0.1 * epsLin * epsNl
  if (platform->options.getArgs("CVODE EPS LIN").empty())
    platform->options.setArgs("CVODE EPS LIN", "0.5");

  double epsLin;
  platform->options.getArgs("CVODE EPS LIN", epsLin);
  retval = CVodeSetEpsLin(this->cvodeMem, epsLin);
  check_retval(&retval, "CVodeSetEpsLin", 1);
#endif

  // set user data as
  retval = CVodeSetUserData(this->cvodeMem, this);
  check_retval(&retval, "CVodeSetUserData", 1);

  if (platform->comm.mpiRank != 0)
    CVodeSetErrFile(this->cvodeMem, NULL); 

  resetCounters();

#else
  nrsCheck(true, platform->comm.mpiComm, EXIT_FAILURE, "%s\n", "cvode was not enabled");
#endif
}

cvode_t::~cvode_t()
{
  if (o_xyz0.size()) {
    o_xyz0.free();
  }
  if (o_coeffExt.size()) {
    o_coeffExt.free();
  }
  if (o_cvodeScalarIds.size()) {
    o_cvodeScalarIds.free();
  }
  if (o_scalarIds.size()) {
    o_scalarIds.free();
  }
  if (o_absTol.size()) {
    o_absTol.free();
  }

#ifdef ENABLE_CVODE
  // despite documentation, this function does not exist?
  // N_VDestroy_MPIPlusX(cvodeY);

  if (platform->device.mode() == "CUDA") {
#ifdef ENABLE_CUDA
    N_VDestroy_Cuda(y);
#endif
  } else if (platform->device.mode() == "HIP") {
#ifdef ENABLE_HIP
    N_VDestroy_HIP(y);
#endif
  } else if (platform->device.mode() == "Serial") {
    N_VDestroy_Serial(y);
  }

  CVodeFree(&cvodeMem);
#endif
}

void cvode_t::setupDirichletMask()
{
  auto cds = nrs->cds;
  auto mesh = cds->mesh[0];
  int NelemT = mesh->Nelements;
  int NelemV = nrs->meshV->Nelements;
  std::vector<dlong> EToB(Nscalar * NelemT * mesh->Nfaces, 0);
  for (int cvodeScalar = 0; cvodeScalar < Nscalar; ++cvodeScalar) {
    const int fOffset = cvodeScalar * mesh->Nelements * mesh->Nfaces;
    const auto is = scalarIds.at(cvodeScalar);
    const auto sid = scalarDigitStr(is);
    for (dlong e = 0; e < NelemT; e++) {
      for (int f = 0; f < mesh->Nfaces; f++) {
        const int bID = mesh->EToB[f + e * mesh->Nfaces];
        EToB[f + e * mesh->Nfaces + fOffset] = bcMap::ellipticType(bID, "scalar" + sid);

        // Since EToB must include all of the faces on the T-mesh, we need to explicitly
        // mark the faces on the V-mesh as not having a boundary
        // This ensures that there is no effect in the resulting mask
        if (e >= NelemV && is != 0) {
          EToB[f + e * mesh->Nfaces + fOffset] = NO_OP;
        }
      }
    }
  }

  // generate mask using ellipticOgs call
  dlong NmaskedLocal = 0;
  dlong NmaskedGlobal = 0;

  occa::memory o_maskIdsLocal;
  occa::memory o_maskIdsGlobal;
  ellipticOgs(mesh,
              mesh->Nlocal,
              this->Nscalar,
              nrs->fieldOffset,
              EToB.data(),
              this->Nmasked,
              this->o_maskIds,
              NmaskedLocal,
              o_maskIdsLocal,
              NmaskedGlobal,
              o_maskIdsGlobal,
              &(mesh->ogs));

  if (o_maskIdsLocal.size()) {
    o_maskIdsLocal.free();
  }
  if (o_maskIdsGlobal.size()) {
    o_maskIdsGlobal.free();
  }

  maskOffset = alignStride<dfloat>(Nmasked);
  o_maskValues = platform->device.malloc<dfloat>(nrs->nEXT * maskOffset);
}

void cvode_t::applyDirichlet(double time)
{
  // extrapolate masked Dirichlet values to current time state
  // NOTE: this can only be applied after the extrapolation order is reached
  // to avoid introducing CVODE convergence issues in the first few time steps
  if (this->isRhsEvaluation() && (this->externalTStep > nrs->nEXT)) {

    if (this->Nmasked == 0) {
      return;
    }

    auto cds = nrs->cds;
    const int extOrder = std::min(this->externalTStep, nrs->nEXT);
    this->extrapolateDirichletKernel(this->Nmasked,
                                     this->maskOffset,
                                     extOrder,
                                     this->o_maskIds,
                                     o_coeffExt,
                                     this->o_maskValues,
                                     cds->o_S + this->minCvodeScalarId * nrs->fieldOffset);
    return;
  }

  // lower than any other possible Dirichlet value
  static constexpr dfloat TINY = -1e30;
  cds_t *cds = nrs->cds;

  auto o_S_start = platform->o_memPool.reserve<dfloat>(cds->fieldOffsetSum);
  const auto neknekFieldOffset = cds->neknek ? cds->neknek->fieldOffset() : 0;

  for (int is = 0; is < cds->NSfields; is++) {
    if (!cds->compute[is]) {
      continue;
    }
    if (!cds->cvodeSolve[is]) {
      continue;
    }

    mesh_t *mesh = cds->mesh[0];
    oogs_t *gsh = cds->gshT;
    if (is) {
      mesh = cds->meshV;
      gsh = cds->gsh;
    }

    const auto cvodeScalarId = cvodeScalarIds[is];

    auto o_diff_i = cds->o_diff + cds->fieldOffsetScan[is];
    auto o_rho_i = cds->o_rho + cds->fieldOffsetScan[is];
    auto o_Si = o_S_start + cds->fieldOffsetScan[cvodeScalarId];

    platform->linAlg->fill(cds->fieldOffset[is], TINY, o_Si);

    for (int sweep = 0; sweep < 2; sweep++) {
      cds->dirichletBCKernel(mesh->Nelements,
                             cds->fieldOffset[is],
                             is,
                             time,
                             mesh->o_sgeo,
                             mesh->o_x,
                             mesh->o_y,
                             mesh->o_z,
                             mesh->o_vmapM,
                             mesh->o_EToB,
                             cds->o_EToB + is * cds->EToBOffset,
                             cds->o_U,
                             o_diff_i,
                             o_rho_i,
                             neknekFieldOffset,
                             cds->neknek ? cds->neknek->o_pointMap() : o_NULL,
                             cds->neknek ? cds->neknek->o_U() : o_NULL,
                             cds->neknek ? cds->neknek->o_S() : o_NULL,
                             *(cds->o_usrwrk),
                             o_Si);
      if (sweep == 0) {
        oogs::startFinish(o_Si, 1, cds->fieldOffset[is], ogsDfloat, ogsMax, gsh);
      }
      if (sweep == 1) {
        oogs::startFinish(o_Si, 1, cds->fieldOffset[is], ogsDfloat, ogsMin, gsh);
      }
    }
  }

  if (this->Nmasked == 0) {
    return;
  }

  cds->maskCopyKernel(this->Nmasked,
                      0,
                      cds->fieldOffsetScan[minCvodeScalarId],
                      o_maskIds,
                      o_S_start,
                      cds->o_S);

  // o_maskValues must be at state t0 to be lagged by the subsequent CVODE solve call
  if (this->isRhsEvaluation()) {
    return;
  }

  this->mapToMaskedPointKernel(this->Nmasked, o_maskIds, o_S_start, o_maskValues);
}

void cvode_t::rhs(double time, const  LVector_t<dfloat> & o_y,  LVector_t<dfloat> & o_ydot)
{
  const auto tag = this->rhsTagName();
  const auto saveTimerScope = timerScope;
  timerScope = tag;
  if (detailedTimersEnabled)
    platform->timer.tic(tag, 0);
  this->setIsRhsEvaluation(true);

  if (userRHS) {
    userRHS(nrs, time, tExternal, o_y, o_ydot);
  } else {
    defaultRHS(time, tExternal, o_y, o_ydot);
  }

  this->setIsRhsEvaluation(false);
  if (detailedTimersEnabled)
    platform->timer.toc(tag);
  timerScope = saveTimerScope;
}

int cvode_t::jtvRhs(double time, const occa::memory& o_y, occa::memory& o_ydot) 
{
  auto o_yy = o_y; /* kludge because optr takes occa::memory&*/
  this->YLVec->optr(o_yy);
  this->YdotLVec->optr(o_ydot);

  this->jtvRhs(time, *(this->YLVec), *(this->YdotLVec));

  return 0;
}

void cvode_t::jtvRhs(double time, const  LVector_t<dfloat> & o_y,  LVector_t<dfloat> & o_ydot)
{
  this->setIsJacobianEvaluation(true);

  if (userJacobian) {
    userJacobian(nrs, time, tExternal, o_y, o_ydot);
  } else {
    this->rhs(time, o_y, o_ydot);
  }

  this->setIsJacobianEvaluation(false);
}

void cvode_t::defaultRHS(double time, double t0, const  LVector_t<dfloat> & o_y,  LVector_t<dfloat> & o_ydot)
{
  const bool movingMesh = platform->options.compareArgs("MOVING MESH", "TRUE");
  mesh_t *mesh = nrs->meshV;
  if (nrs->cht) {
    mesh = nrs->cds->mesh[0];
  }

  auto *cds = nrs->cds;
  
  if (time != tprev) {

    if (detailedTimersEnabled) {
      platform->timer.tic(timerScope + "::extrapolate", 0);
    }

    if (platform->comm.mpiRank == 0 && time - tprev > 0 && verboseCVODE) {
      std::cout << "tCv= " << time << ", dt= " << time - tprev << std::endl;
    }

    const dfloat cvodeDt = time - t0;
    tprev = time;

    dtCvode[0] = cvodeDt;
    dtCvode[1] = nrs->dt[1];
    dtCvode[2] = nrs->dt[2];

    // evaluate external variables at integrator time using extrapolation
    const int bdfOrder = std::min(this->externalTStep, nrs->nBDF);
    const int extOrder = std::min(this->externalTStep, nrs->nEXT);
    nek::extCoeff(_coeffEXT.data(), dtCvode.data(), extOrder, bdfOrder);
    nek::bdfCoeff(&this->_g0, _coeffBDF.data(), dtCvode.data(), bdfOrder);
    for (int i = nrs->nEXT; i > extOrder; i--) {
      _coeffEXT[i - 1] = 0.0;
    }
    for (int i = nrs->nBDF; i > bdfOrder; i--) {
      _coeffBDF[i - 1] = 0.0;
    }

    if (nrs->pSolver) {
      if (nrs->pSolver->allNeumann && platform->options.compareArgs("LOWMACH", "TRUE")) {
        nrs->p0the = 0.0;
        for (int ext = 0; ext < extOrder; ++ext) {
          nrs->p0the += _coeffEXT[ext] * nrs->p0th[ext];
        }
      }
    }

    o_coeffExt.copyFrom(_coeffEXT.data(), maxTimestepperOrder);

    nrs->extrapolateKernel(nrs->meshV->Nlocal,
                           nrs->NVfields,
                           extOrder,
                           nrs->fieldOffset,
                           o_coeffExt,
                           this->o_U,
                           nrs->o_U);

    if (movingMesh) {

      const int meshOrder = std::min(this->externalTStep, mesh->nAB);
      nek::coeffAB(mesh->coeffAB, dtCvode.data(), meshOrder);
      for (int i = 0; i < meshOrder; ++i) {
        mesh->coeffAB[i] *= dtCvode[0];
      }
      for (int i = mesh->nAB; i > meshOrder; i--) {
        mesh->coeffAB[i - 1] = 0.0;
      }
      mesh->o_coeffAB.copyFrom(mesh->coeffAB, mesh->nAB);

      // restore mesh coordinates prior to integration
      {
        mesh->o_x.copyFrom(this->o_xyz0, mesh->Nlocal, 0, 0 * nrs->fieldOffset);
        mesh->o_y.copyFrom(this->o_xyz0, mesh->Nlocal, 0, 1 * nrs->fieldOffset);
        mesh->o_z.copyFrom(this->o_xyz0, mesh->Nlocal, 0, 2 * nrs->fieldOffset);
      }

      mesh->move();

      nrs->extrapolateKernel(mesh->Nlocal,
                             nrs->NVfields,
                             extOrder,
                             nrs->fieldOffset,
                             o_coeffExt,
                             this->o_meshU,
                             mesh->o_U);
    }

    computeUrst(nrs, true);

    if (detailedTimersEnabled) {
      platform->timer.toc(timerScope + "::extrapolate");
    }
  }

  cvToNrs(o_y, cds->o_S, false);

  if (detailedTimersEnabled) {
    platform->timer.tic(timerScope + "::applyDirichlet", 0);
  }

  this->applyDirichlet(time);

  if (detailedTimersEnabled) {
    platform->timer.toc(timerScope + "::applyDirichlet");
  }

  const bool chtCVODE = nrs->cht && cds->cvodeSolve[0];

  if (!(isJacobianEvaluation() && recycleProperties)) {
    platform->timer.tic(timerScope + "::evaluateProperties", 0);

    evaluateProperties(nrs, time);

    if (chtCVODE) {
      auto mesh = cds->mesh[0];
      o_invRhoCpAvg.copyFrom(cds->o_rho);
      platform->linAlg->axmy(mesh->Nlocal, 1.0, mesh->o_LMM, o_invRhoCpAvg);
      oogs::startFinish(o_invRhoCpAvg, 1, nrs->fieldOffset, ogsDfloat, ogsAdd, cds->gshT);
      platform->linAlg->axmy(mesh->Nlocal, 1.0, mesh->o_invLMM, o_invRhoCpAvg);
      platform->linAlg->ady(mesh->Nlocal, 1.0, o_invRhoCpAvg);
    } else {
      platform->linAlg->adyz(mesh->Nlocal, 1.0, cds->o_rho, o_invRhoCpAvg);
    }

    platform->timer.toc(timerScope + "::evaluateProperties");
  }

  platform->linAlg->fill(cds->fieldOffsetSum, 0.0, cds->o_FS);
  makeq(time);

  auto applyOgsOperation = [&](auto ogsFunc) {
    if (chtCVODE) {
      ogsFunc(cds->o_FS, 1, cds->fieldOffset[0], ogsDfloat, ogsAdd, cds->gshT);
    }

    if (!chtCVODE || (chtCVODE && this->Nscalar > 1)) {
      dlong startScalar = minCvodeScalarId;
      dlong numScalars = this->Nscalar;
      if (chtCVODE) {
        startScalar++;
        numScalars--;
      }
      auto o_fld = cds->o_FS + nrs->cds->fieldOffsetScan[startScalar];
      ogsFunc(o_fld, numScalars, cds->fieldOffset[startScalar], ogsDfloat, ogsAdd, this->gsh);
    }
  };

  if (detailedTimersEnabled) {
    platform->timer.tic(timerScope + "::gatherScatterAndLocalPoint", 0);
  }

  applyOgsOperation(oogs::start);

  if (userLocalPointSource) {
    platform->timer.tic(timerScope + "::gatherScatterAndLocalPoint::localPointSource", 0);
    userLocalPointSource(nrs, o_y, o_ydot);
    platform->timer.toc(timerScope + "::gatherScatterAndLocalPoint::localPointSource");
    cvToNrs(o_ydot, this->o_pointSource, true);
  }

  applyOgsOperation(oogs::finish);

  if (detailedTimersEnabled) {
    platform->timer.toc(timerScope + "::gatherScatterAndLocalPoint");
    platform->timer.tic(timerScope + "::fusedAddRhoDiv", 0);
  }

  // (o_FS + userLocalPointSource) / rho
  if (chtCVODE) {
    if (userLocalPointSource) {
      platform->linAlg->axpby(cds->mesh[0]->Nlocal, 1.0, this->o_pointSource, 1.0, cds->o_FS);
    }
    platform->linAlg->axmy(cds->mesh[0]->Nlocal, 1.0, o_invRhoCpAvg, cds->o_FS);
  }
  if (!chtCVODE || (chtCVODE && this->Nscalar > 1)) {
    dlong startScalar = minCvodeScalarId;
    dlong numScalars = this->Nscalar;
    if (chtCVODE) {
      startScalar++;
      numScalars--;
    }
    auto o_FS_start = cds->o_FS + cds->fieldOffsetScan[startScalar];
    auto o_rho_start = cds->o_rho + cds->fieldOffsetScan[startScalar];
    occa::memory o_ptSource_start;

    if (userLocalPointSource) {
      o_ptSource_start = this->o_pointSource + cds->fieldOffsetScan[startScalar];
    }

    const int useFieldRho = !sharedRho;
    this->fusedAddRhoDivKernel(cds->meshV->Nlocal,
                               numScalars,
                               nrs->fieldOffset,
                               useFieldRho,
                               o_rho_start,
                               o_ptSource_start,
                               o_FS_start);
  }

  if (detailedTimersEnabled) {
    platform->timer.toc(timerScope + "::fusedAddRhoDiv");
  }

  // add dpdt term to temperature eqn
  if (nrs->pSolver) {
    if (platform->options.compareArgs("LOWMACH", "TRUE") && nrs->pSolver->allNeumann) {
      if (detailedTimersEnabled) {
        platform->timer.tic(timerScope + "::dp0thdt", 0);
      }
 
      // evaluate dp0thdt (evaluate together with divergence)
      if (!(isJacobianEvaluation() && recycleProperties)) {
        if (detailedTimersEnabled) {
          platform->timer.tic(timerScope + "::dp0thdt::udfDiv", 0);
        }
        cds->cvode->setTimerScope(timerScope + "::dp0thdt::udfDiv");
 
        platform->linAlg->fill(mesh->Nlocal, 0.0, nrs->o_div);
        udf.div(nrs, time, nrs->o_div);
 
        cds->cvode->setTimerScope(timerScope);
        if (detailedTimersEnabled)
          platform->timer.toc(timerScope + "::dp0thdt::udfDiv");
      }
 
      platform->linAlg->axpby(mesh->Nlocal, nrs->dp0thdt * nrs->alpha0Ref, o_invRhoCpAvg, 1.0, cds->o_FS); 
 
      if (detailedTimersEnabled) {
        platform->timer.toc(timerScope + "::dp0thdt");
      }
    }
  }

  if (detailedTimersEnabled) {
    platform->timer.tic(timerScope + "::maskDirichlet", 0);
  }

  auto o_FS_start = cds->o_FS + cds->fieldOffsetScan[minCvodeScalarId];
  if (this->Nmasked > 0) {
    nrs->maskKernel(this->Nmasked, this->o_maskIds, o_FS_start);
  }

  if (detailedTimersEnabled) {
    platform->timer.toc(timerScope + "::maskDirichlet");
  }

  nrsToCv(cds->o_FS, o_ydot, true);
}

void cvode_t::makeq(double time)
{

  const auto timerScopeSave = timerScope;
  timerScope = timerScopeSave + "::makeq";
  platform->timer.tic(timerScope, 0);

  auto *cds = nrs->cds;
  auto o_FS = nrs->cds->o_FS;

  bool useRelativeVelocity = platform->options.compareArgs("MOVING MESH", "TRUE");
  auto &o_Urst = useRelativeVelocity ? cds->o_relUrst : cds->o_Urst;

  if (udf.sEqnSource) {
    const auto makeQScope = timerScope;
    // ensure that user/application forward the correct base timer
    timerScope = makeQScope + "::udfSEqnSource";
    platform->timer.tic(timerScope, 0);
    udf.sEqnSource(nrs, time, cds->o_S, o_FS);
    platform->timer.toc(timerScope);
    timerScope = makeQScope;
  }

  auto applyTerms = [&](mesh_t *mesh, dlong scalarStart, dlong Nscalar, bool chtPass) {
    if (cds->applyFilter) {
      cds->filterRTKernel(cds->meshV->Nelements,
                          scalarStart,
                          Nscalar,
                          nrs->fieldOffset,
                          cds->o_applyFilterRT,
                          cds->o_filterRT,
                          cds->o_filterS,
                          cds->o_rho,
                          cds->o_S,
                          o_FS);
    }

    double flops = 6 * mesh->Np * mesh->Nq + 4 * mesh->Np;
    flops *= static_cast<double>(mesh->Nelements);
    flops *= Nscalar;
    platform->flopCounter->add("scalarFilterRT", flops);


    if (platform->options.compareArgs("ADVECTION", "TRUE")) {
      if (detailedTimersEnabled) {
        platform->timer.tic(timerScope + "::advection", 0);
      }

      int applyLMM = 1;
      if (chtPass) {
        applyLMM = 0;
      }

      if (platform->options.compareArgs("CVODE ADVECTION TYPE", "CUBATURE")) {
        cds->strongAdvectionCubatureVolumeKernel(cds->meshV->Nelements,
                                                 Nscalar,
                                                 applyLMM,
                                                 mesh->o_LMM,
                                                 mesh->o_vgeo,
                                                 mesh->o_cubDiffInterpT,
                                                 mesh->o_cubInterpT,
                                                 mesh->o_cubProjectT,
                                                 cds->o_compute + scalarStart,
                                                 cds->o_fieldOffsetScan + scalarStart,
                                                 cds->vFieldOffset,
                                                 cds->vCubatureOffset,
                                                 cds->o_S,
                                                 o_Urst,
                                                 cds->o_rho,
                                                 o_FS);
      } else {
        cds->strongAdvectionVolumeKernel(cds->meshV->Nelements,
                                         Nscalar,
                                         applyLMM,
                                         mesh->o_LMM,
                                         mesh->o_vgeo,
                                         mesh->o_D,
                                         cds->o_compute + scalarStart,
                                         cds->o_fieldOffsetScan + scalarStart,
                                         cds->vFieldOffset,
                                         cds->o_S,
                                         o_Urst,
                                         cds->o_rho,
                                         o_FS);
      }

      // During a CHT pass, the LMM term is not applied in the advection kernels as
      // the advection term is only defined in the V-mesh portion.
      // Therefore, apply the LMM term here over the entire T-mesh.
      if (chtPass) {
        auto o_FS_start = o_FS + cds->fieldOffsetScan[scalarStart];
        platform->linAlg->axmyMany(mesh->Nlocal, Nscalar, nrs->fieldOffset, 0, 1.0, mesh->o_LMM, o_FS_start);
      }

      if (detailedTimersEnabled) {
        platform->timer.toc(timerScope + "::advection");
      }

      timeStepper::advectionFlops(cds->mesh[scalarStart], Nscalar);
    } else {
      auto o_FS_start = o_FS + cds->fieldOffsetScan[scalarStart];
      platform->linAlg->axmyMany(mesh->Nlocal, Nscalar, nrs->fieldOffset, 0, 1.0, mesh->o_LMM, o_FS_start);
    }

    if (detailedTimersEnabled) {
      platform->timer.tic(timerScope + "::neumannBC", 0);
    }

    // weak Laplacian + boundary terms
    cds->neumannBCKernel(mesh->Nelements,
                         Nscalar,
                         mesh->o_sgeo,
                         mesh->o_vmapM,
                         mesh->o_EToB,
                         scalarStart,
                         time,
                         nrs->fieldOffset,
                         nrs->fieldOffset,
                         cds->EToBOffset,
                         mesh->o_x,
                         mesh->o_y,
                         mesh->o_z,
                         nrs->o_U,
                         cds->o_S,
                         cds->o_EToB,
                         cds->o_diff,
                         cds->o_rho,
                         *(cds->o_usrwrk),
                         cds->o_FS);

    if (detailedTimersEnabled) {
      platform->timer.toc(timerScope + "::neumannBC");
      platform->timer.tic(timerScope + "::weakLaplacian", 0);
    }

    weakLaplacianKernel(mesh->Nelements,
                        Nscalar,
                        cds->o_fieldOffsetScan + scalarStart,
                        mesh->o_ggeo,
                        mesh->o_invLMM,
                        mesh->o_D,
                        cds->o_diff,
                        cds->o_S,
                        o_FS);

    if (detailedTimersEnabled) {
      platform->timer.toc(timerScope + "::weakLaplacian");
    }
  };

  const bool chtCVODE = nrs->cht && cds->cvodeSolve[0];
  if (chtCVODE) {
    applyTerms(cds->mesh[0], 0, 1, true);
  }

  if (!chtCVODE || (chtCVODE && this->Nscalar > 1)) {

    // Starting scalar id and number of scalars to process in this block
    // note, in non-CHT cases, this is just the number of CVODE scalars.
    // In CHT cases where the temperature is solved via CVODE,
    // this is one less than the number of CVODE scalars.
    dlong startScalar = minCvodeScalarId;
    dlong numScalars = this->Nscalar;
    if (chtCVODE) {
      startScalar++;
      numScalars--;
    }
    applyTerms(cds->meshV, startScalar, numScalars, false);
  }

  platform->timer.toc(timerScope);
  timerScope = timerScopeSave;
}

void cvode_t::nrsToCv(occa::memory o_EField,  LVector_t<dfloat> & o_LField, bool isYdot)
{
  if (detailedTimersEnabled) {
    platform->timer.tic(timerScope + "::nrsToCv", 0);
  }
  
  auto cds = nrs->cds;
  auto o_E = o_EField + cds->fieldOffsetScan[this->minCvodeScalarId];
  o_LField.copyFromE(nrs->fieldOffset, o_E);

  if (userPostNrsToCv) {
    userPostNrsToCv(nrs, o_LField, isYdot);
  }
  if (detailedTimersEnabled) {
    platform->timer.toc(timerScope + "::nrsToCv");
  }
}

void cvode_t::cvToNrs(const  LVector_t<dfloat> & o_LField, occa::memory o_EField, bool isYdot)
{
  if (detailedTimersEnabled) {
    platform->timer.tic(timerScope + "::cvToNrs", 0);
  }
  
  auto cds = nrs->cds;
  
  auto o_E = o_EField + cds->fieldOffsetScan[this->minCvodeScalarId];
  o_LField.copyToE(nrs->fieldOffset, o_E);

  if(userPostCvToNrs) {
    userPostCvToNrs(nrs, o_EField, isYdot);
  }

  if (detailedTimersEnabled) {
    platform->timer.toc(timerScope + "::cvToNrs");
  }
}

void cvode_t::solve(double t0, double t1, int tstep)
{
#ifdef ENABLE_CVODE
  platform->timer.tic(timerName + "solve", 0);
  timerScope = timerName + "solve";

  // update solver statistics from previous state
  auto setPreviousCounters = [&](bool reset = false) {
    if (reset) {
      prevNsteps = 0;
      prevNrhs = 0;
      prevNni = 0; 
      prevNli = 0;
    } else {
      int retval = 0;
      retval = CVodeGetNumSteps(cvodeMem, &prevNsteps);
      check_retval(&retval, "CVodeGetNumSteps", 1);
      retval = CVodeGetNumRhsEvals(cvodeMem, &prevNrhs);
      check_retval(&retval, "CVodeGetNumRhsEvals", 1);
      retval = CVodeGetNumNonlinSolvIters(cvodeMem, &prevNni);
      check_retval(&retval, "CVodeGetNumNonlinSolvIters", 1);
      retval = CVodeGetNumLinIters(cvodeMem, &prevNli);
      check_retval(&retval, "CVodeGetNumLinIters", 1);
    }
  };

  mesh_t *mesh = nrs->meshV;
  if (nrs->cht) {
    mesh = nrs->cds->mesh[0];
  }

  bool movingMesh = platform->options.compareArgs("MOVING MESH", "TRUE");

  // lag solution state and update current state
  for (int s = nrs->nEXT; s > 1; s--) {
    const auto N = nrs->NVfields * nrs->fieldOffset;
    o_U.copyFrom(o_U, N, (s - 1) * N, (s - 2) * N);
  }
  o_U.copyFrom(nrs->o_U, nrs->NVfields * nrs->fieldOffset);

  const auto p0theSave = nrs->p0the;

  if (movingMesh) {

    for (int s = nrs->nEXT; s > 1; s--) {
      const auto N = nrs->NVfields * nrs->fieldOffset;
      o_meshU.copyFrom(o_meshU, N, (s - 1) * N, (s - 2) * N);
    }
    o_meshU.copyFrom(mesh->o_U, nrs->NVfields * nrs->fieldOffset);

    o_xyz0.copyFrom(mesh->o_x, mesh->Nlocal, 0 * nrs->fieldOffset, 0);
    o_xyz0.copyFrom(mesh->o_y, mesh->Nlocal, 1 * nrs->fieldOffset, 0);
    o_xyz0.copyFrom(mesh->o_z, mesh->Nlocal, 2 * nrs->fieldOffset, 0);
  }

  YLVec->optr(o_cvodeY);
  nrsToCv(nrs->cds->o_S, *YLVec, false);

  this->tExternal = t0;
  this->externalTStep = tstep;

  double t;
  int retval = 0;

  // integrate only to time t1
  if (platform->options.compareArgs("CVODE STOP TIME", "TRUE")) {
    retval = CVodeSetStopTime(cvodeMem, t1);
    nrsCheck(retval < 0, MPI_COMM_SELF, EXIT_FAILURE, "%s", "Error calling CVodeSetStopTime\n");
  }

  if (!platform->options.getArgs("CVODE HMAX RATIO").empty()) {
    double hmax;
    platform->options.getArgs("CVODE HMAX RATIO", hmax);
    hmax *= nrs->dt[0];
    retval = CVodeSetMaxStep(cvodeMem, hmax);
    nrsCheck(retval < 0, MPI_COMM_SELF, EXIT_FAILURE, "%s", "Error calling CVodeSetMaxStep\n");
  }

  const auto oldScope = timerScope;
  timerScope = oldScope + "::cvode";
  if (detailedTimersEnabled) {
    platform->timer.tic(timerScope, 0);
  }

  // call cvode solver
  if(platform->verbose && platform->comm.mpiRank == 0)
    std::cout << "calling cvode ...\n";
  setPreviousCounters();
  retval = CVode(cvodeMem, t1, cvodeY, &t, CV_NORMAL);
  updateCounters();

  // restart if needed
  if (retval != CV_SUCCESS) {
    const auto maxRestarts = 5;
    int cnt = 0;
    while (retval != CV_SUCCESS) {
      cnt++;
      if (platform->comm.mpiRank == 0) {
        std::cout << "restarting CVODE ...\n";
      }
      retval = CVodeReInit(cvodeMem, t, cvodeY);
      check_retval(&retval, "CVodeReInit", 1);

      this->tprev = std::numeric_limits<double>::max();
      setPreviousCounters(true);
      retval = CVode(cvodeMem, t1, cvodeY, &t, CV_NORMAL);
      updateCounters();

      if(cnt > maxRestarts) {
        if (platform->comm.mpiRank == 0) {
          std::cout << "giving up ...\n";
        }
      }
    }
  }
  if(platform->verbose && platform->comm.mpiRank == 0)
    std::cout << "done\n"; 

  nrsCheck(retval < 0, MPI_COMM_SELF, EXIT_FAILURE, "%s", "CVODE failed after restart\n");

  if (detailedTimersEnabled) {
    platform->timer.toc(timerScope);
  }
  timerScope = oldScope;

  YLVec->optr(o_cvodeY);
  cvToNrs(*YLVec, nrs->cds->o_S, false);

  if (detailedTimersEnabled) {
    platform->timer.tic(timerScope + "::restore", 0);
  }

  // restore previous state
  nrs->o_U.copyFrom(o_U, nrs->NVfields * nrs->fieldOffset);

  if (movingMesh) {
    mesh->o_x.copyFrom(o_xyz0, mesh->Nlocal, 0, 0 * nrs->fieldOffset);
    mesh->o_y.copyFrom(o_xyz0, mesh->Nlocal, 0, 1 * nrs->fieldOffset);
    mesh->o_z.copyFrom(o_xyz0, mesh->Nlocal, 0, 2 * nrs->fieldOffset);

    mesh->o_U.copyFrom(this->o_meshU, nrs->NVfields * nrs->fieldOffset);

    mesh->update();
  }

  computeUrst(nrs, false);

  nrs->p0the = p0theSave;

  for (int s = nrs->nEXT; s > 1; s--) {
    if (maskOffset) {
      o_maskValues.copyFrom(o_maskValues, maskOffset, (s - 1) * maskOffset, (s - 2) * maskOffset);
    }
  }

  // compute scalar boundary condition at time t1
  this->applyDirichlet(t1);

  if (detailedTimersEnabled) {
    platform->timer.toc(timerScope + "::restore");
  }
  platform->timer.toc(timerScope);
#endif
}

long cvode_t::numSteps() const
{
  return this->nsteps;
}

long cvode_t::numRHSEvals() const
{
  return this->nrhs; 
}

long cvode_t::numNonlinSolveIters() const
{
  return this->nni;
}

long cvode_t::numLinIters() const
{
  return this->nli;
}

#ifdef ENABLE_CVODE
void cvode_t::updateCounters()
{
  long int cnt;

  auto retval = CVodeGetNumSteps(cvodeMem, &cnt);
  check_retval(&retval, "CVodeGetNumSteps", 1);
  this->nsteps += (cnt - prevNsteps);

  retval = CVodeGetNumRhsEvals(cvodeMem, &cnt);
  check_retval(&retval, "CVodeGetNumRhsEvals", 1);
  this->nrhs += (cnt - prevNrhs);

  retval = CVodeGetNumNonlinSolvIters(cvodeMem, &cnt);
  check_retval(&retval, "CVodeGetNumNonlinSolvIters", 1);
  this->nni += (cnt - prevNni);

  retval = CVodeGetNumLinIters(cvodeMem, &cnt);
  check_retval(&retval, "CVodeGetNumLinIters", 1);
  this->nli += (cnt - prevNli);
}

void cvode_t::resetCounters()
{
  this->nsteps = 0;
  this->nrhs = 0;
  this->nni = 0;
  this->nli = 0;
}

#else

void cvode_t::updateCounters()
{
}

void cvode_t::resetCounters()
{
}
#endif

void cvode_t::printInfo(bool printVerboseInfo)
{
  auto cvodeMem = this->cvodeMem;

  constexpr auto lengthToColon = 9;
  std::string scalars;
  {
    std::ostringstream ss;
    ss << "S" << scalarDigitStr(minCvodeScalarId);
    if (maxCvodeScalarId > minCvodeScalarId) {
      ss << "-S" << scalarDigitStr(maxCvodeScalarId);
    }
    scalars = ss.str();
  }

  if (platform->comm.mpiRank == 0 && printVerboseInfo) {
    std::ostringstream ss;
    ss << "" << scalars;
    ss << std::setfill(' ') << std::setw(lengthToColon - ss.str().length()) << " ";

    // pad remaining space, lengthToColon long, with spaces
    printf("%s: nsteps %03ld  nRHS %03ld  nni %01ld  nni/nsteps %.1f  nli %03ld  nli/nni %.1f\n",
           ss.str().c_str(),
           nsteps,
           nrhs,
           nni,
           static_cast<float>(nni) / nsteps,
           nli,
           static_cast<float>(nli) / nni);
  } else if (platform->comm.mpiRank == 0) {
    std::ostringstream ss;
    ss << "  " << scalars;
    printf("%s: %ld %ld %ld %ldf", ss.str().c_str(), nsteps, nrhs, nni, nli);
  }

  resetCounters();
}

void cvode_t::printTimers()
{
  const auto timerTags = platform->timer.tags();

  // filter out tags that do not start with timerName
  std::vector<std::string> filteredTags;
  std::copy_if(timerTags.begin(),
               timerTags.end(),
               std::back_inserter(filteredTags),
               [&](const std::string &tag) { return tag.find(timerName) == 0; });

  // construct tree where parent entries are the portion of the tag left of the last '::'
  std::map<std::string, std::vector<std::string>> tree;

  for (auto &&tag : filteredTags) {
    auto pos = tag.rfind("::");
    if (pos == std::string::npos) {
      tree[""].push_back(tag);
    } else {
      auto parent = tag.substr(0, pos);
      tree[parent].push_back(tag);
    }
  }

  auto pos = timerName.rfind("::");
  const auto start = timerName.substr(0, pos);

  std::ios oldState(nullptr);
  oldState.copyfmt(std::cout);

  auto mesh = nrs->meshV;
  long long int NglobalElements = mesh->Nelements;
  MPI_Allreduce(MPI_IN_PLACE, &NglobalElements, 1, MPI_LONG_LONG_INT, MPI_SUM, platform->comm.mpiComm);
  const double GDOF = (NglobalElements * mesh->N * mesh->N * mesh->N * this->cvodeScalarIds.size()) /
                      (1e9 * platform->comm.mpiCommSize);

  // gather timer information from tree
  std::vector<std::string> operations;
  std::vector<std::string> times;
  std::vector<std::string> calls;
  std::vector<std::string> relPercentage;
  std::vector<std::string> absPercentage;
  std::vector<std::string> throughputs;

  std::cout.setf(std::ios::fixed);
  std::function<void(std::string, std::string, std::string, int)> gatherTreeStats;
  gatherTreeStats = [&](std::string tag, std::string rootTag, std::string parentTag, int level) {
    if (level > 0) {
      if (level == 1) {
        rootTag = tag; // set as root of timer tree
      }
      const auto tTag = platform->timer.query(tag, "DEVICE:MAX");
      const auto nCalls = platform->timer.count(tag);

      if(nCalls == 0) return; // nothing to print

      auto tParent = platform->timer.query(parentTag, "DEVICE:MAX");

      if (tParent < 0.0) {
        tParent = tTag;
      }

      const auto tRoot = platform->timer.query(rootTag, "DEVICE:MAX");

      const auto tCall = tTag / nCalls;

      // trim parentTag from the current tag
      auto pos = tag.rfind(parentTag);
      auto trimmedTag = tag.substr(pos + parentTag.length() + 2);

      if (platform->comm.mpiRank == 0) {
        std::ostringstream ss;
        for (int i = 0; i < level; ++i) {
          ss << "> ";
        }
        ss << trimmedTag;
        operations.push_back(ss.str());

        ss.str("");
        ss.clear();
        ss << std::setprecision(3) << std::scientific << tTag;
        times.push_back(ss.str());

        ss.str("");
        ss.clear();
        ss << std::setw(6) << nCalls;
        calls.push_back(ss.str());

        ss.str("");
        ss.clear();
        ss << std::setprecision(1) << std::fixed << 100.0 * tTag / tParent;
        relPercentage.push_back(level == 1 ? "" : ss.str());

        ss.str("");
        ss.clear();
        ss << std::setprecision(1) << std::fixed << 100.0 * tTag / tRoot;
        absPercentage.push_back(ss.str());

        ss.str("");
        ss.clear();
        ss << std::setprecision(3) << std::scientific << GDOF / tCall;
        throughputs.push_back(ss.str());
      }
    }

    std::vector<std::string> children;
    for (auto &&child : tree[tag]) {
      children.push_back(child);
    }

    // sort children by max time, from largest to smallest
    std::sort(children.begin(), children.end(), [&](const std::string &a, const std::string &b) {
      const auto ta = platform->timer.query(a, "DEVICE:MAX");
      const auto tb = platform->timer.query(b, "DEVICE:MAX");
      return ta > tb;
    });

    for (auto &&child : children) {
      gatherTreeStats(child, rootTag, tag, level + 1);
    }
  };

  gatherTreeStats(start, "", "", 0);

  std::map<int, std::vector<std::string>> table;
  table[0] = operations;
  table[1] = times;
  table[2] = calls;
  table[3] = relPercentage;
  table[4] = absPercentage;
  table[5] = throughputs;

  std::vector<std::string> headers = {"Operation", "time", "calls", "rel %", "abs %", "GDOF/s/rank"};

  if (platform->comm.mpiRank == 0) {
    std::cout << "\n";
    std::cout << "Timers for " << start << ":\n";
    printTable(table, headers, "    ");
    std::cout << "\n";
  }

  std::cout.copyfmt(oldState);
}

void cvode_t::resetTimers()
{
  const auto timerTags = platform->timer.tags();

  // filter out tags that do not start with timerName
  std::vector<std::string> filteredTags;
  std::copy_if(timerTags.begin(),
               timerTags.end(),
               std::back_inserter(filteredTags),
               [&](const std::string &tag) { return tag.find(timerName) == 0; });

  for (auto &&tag : filteredTags) {
    platform->timer.reset(tag);
  }
}

std::string cvode_t::rhsTagName() const
{
  return this->isJacobianEvaluation() ? timerScope + "::linearSolve::jtv::rhs" : timerScope + "::rhs";
}

void cvode_t::setLocalPointSource(userLocalPointSource_t _userLocalPointSource)
{
  userLocalPointSource = _userLocalPointSource;

  if (o_pointSource.size() == 0) {
    o_pointSource = platform->device.malloc<dfloat>(this->Nscalar * nrs->fieldOffset);
  }

  this->fusedAddRhoDivKernel = platform->kernels.get("cvode_t::fusedAddRhoDiv");
}
