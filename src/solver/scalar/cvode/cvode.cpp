#include "platform.hpp"
#include "cvode.hpp"
#include "elliptic.hpp"
#include "maskedFaceIds.hpp"
#include "inipp.hpp"
#include "nekInterfaceAdapter.hpp"
#include "udf.hpp"

#include "plugins/lowMach.hpp"
#include "tabularPrinter.hpp"

#ifdef ENABLE_CVODE

// cvode includes
#include "sunlinsol/sunlinsol_spgmr.h"
#include "sundials/sundials_types.h"
#include "sundials/sundials_math.h"
#include "cvode/cvode.h"

#include <sunnonlinsol/sunnonlinsol_fixedpoint.h>

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

namespace
{

void check_retval(void *returnvalue, const char *funcname, int opt)
{
  int *retval;

  /* Check if SUNDIALS function returned NULL pointer - no memory allocated */

  if (opt == 0 && returnvalue == NULL) {
    nekrsAbort(MPI_COMM_SELF,
               EXIT_FAILURE,
               "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
               funcname);
  }
  /* Check if retval < 0 */
  else if (opt == 1) {
    retval = (int *)returnvalue;
    if (*retval < 0) {
      nekrsAbort(MPI_COMM_SELF,
                 EXIT_FAILURE,
                 "\nSUNDIALS_ERROR: %s() failed with retval = %d\n\n",
                 funcname,
                 *retval);
    }
  }
  /* Check if function returned NULL pointer - no memory allocated */
  else if (opt == 2 && returnvalue == NULL) {
    nekrsAbort(MPI_COMM_SELF,
               EXIT_FAILURE,
               "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
               funcname);
  }
}

} // namespace

cvode_t::cvode_t(scalar_t *_scalar)
{
  this->scalar = _scalar;

  if (platform->comm.mpiRank() == 0) {
    std::cout << "initializing CVODE ...\n";
  }

  this->fieldOffset = scalar->fieldOffset(); // same for all scalars

  nekrsCheck(fieldOffset != scalar->vFieldOffset,
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "%s\n",
             "vFieldOffset offset does not match");

  o_coeffExt = platform->device.malloc<dfloat>(maxTimestepperOrder);

  o_U0 = platform->device.malloc<dfloat>(scalar->o_coeffEXT.size() * scalar->mesh(0)->dim * fieldOffset);

  if (scalar->geom) {
    nekrsCheck(fieldOffset != scalar->geom->fieldOffset,
               MPI_COMM_SELF,
               EXIT_FAILURE,
               "%s\n",
               "geom offset does not match");

    o_U0mesh = platform->device.malloc<dfloat>(scalar->o_coeffEXT.size() * scalar->geom->mesh->dim *
                                               scalar->geom->fieldOffset);
    o_xyz0 = platform->device.malloc<dfloat>(scalar->geom->mesh->dim * scalar->geom->fieldOffset);
  }

  if (platform->options.getArgs("CVODE JTV RECYCLE PROPERTIES").empty()) {
    platform->options.setArgs("CVODE JTV RECYCLE PROPERTIES", "TRUE");
  }
  recycleProperties = platform->options.compareArgs("CVODE JTV RECYCLE PROPERTIES", "TRUE");

  this->movingMesh = platform->options.compareArgs("MOVING MESH", "TRUE");

  this->scalarIds = std::vector<dlong>();
  this->cvodeScalarIds = std::vector<dlong>(scalar->NSfields, -1);

  Nscalar = 0;

  this->minCvodeScalarId = std::numeric_limits<dlong>::max();
  this->maxCvodeScalarId = -std::numeric_limits<dlong>::max();

  for (int is = 0; is < scalar->NSfields; is++) {
    if (!scalar->compute[is]) {
      continue;
    }
    if (!scalar->cvodeSolve[is]) {
      continue;
    }
    this->minCvodeScalarId = std::min(minCvodeScalarId, is);
    this->maxCvodeScalarId = std::max(maxCvodeScalarId, is);
  }

  // assumption: CVODE scalars are contiguous
  bool valid = true;
  for (int is = this->minCvodeScalarId; is <= this->maxCvodeScalarId; ++is) {
    valid &= scalar->compute[is];
    valid &= scalar->cvodeSolve[is];
  }

  nekrsCheck(!valid, platform->comm.mpiComm(), EXIT_FAILURE, "%s\n", "CVODE scalars must be contiguous");

  std::vector<mesh_t *> meshes;

  for (int is = 0; is < scalar->NSfields; is++) {
    if (!scalar->compute[is]) {
      continue;
    }
    if (!scalar->cvodeSolve[is]) {
      continue;
    }

    cvodeScalarIds[is] = Nscalar;
    scalarIds.push_back(is);

    Nscalar++;

    meshes.push_back(scalar->mesh(is));
  }

  YLVec = std::make_shared<LVector_t<dfloat>>(meshes, false);
  YdotLVec = std::make_shared<LVector_t<dfloat>>(meshes, false);

  std::vector<dlong> lengths_(this->Nscalar, 0);
  this->nEq = 0;
  for (int is = 0; is < Nscalar; ++is) {
    const auto Nlocal = YLVec->Nlocal(is);
    this->nEq += Nlocal;
    lengths_[is] = Nlocal;
  }

  // use compact L-vector layout for CVODE
  YLVec->fieldOffsets(lengths_);
  YdotLVec->fieldOffsets(lengths_);

  o_scalarIds = platform->device.malloc<dlong>(scalarIds.size(), this->scalarIds.data());
  o_cvodeScalarIds = platform->device.malloc<dlong>(cvodeScalarIds.size(), this->cvodeScalarIds.data());

  setupDirichletMask();

  isInitialized = false;

  verboseCVODE = platform->options.compareArgs("CVODE VERBOSE", "TRUE");
  sharedRho = platform->options.compareArgs("CVODE SHARED RHO", "TRUE");
  if (scalarIds.size() < 2) {
    sharedRho = false;
  }

  this->gsh = oogs::setup(scalar->meshV->ogs, this->Nscalar, fieldOffset, ogsDfloat, NULL, OOGS_AUTO);

  mixedPrecisionJtvEnabled = platform->options.compareArgs("CVODE MIXED PRECISION JTV", "TRUE");
  nekrsCheck(mixedPrecisionJtvEnabled,
             platform->comm.mpiComm(),
             EXIT_FAILURE,
             "%s\n",
             "CVODE MIXED PRECISION JTV = TRUE not supported yet");

  if (mixedPrecisionJtvEnabled) {
    auto mesh = scalar->mesh(0);
    o_vgeoPfloat = platform->device.malloc<pfloat>(mesh->Nelements * mesh->Np * mesh->Nvgeo);
    platform->copyDfloatToPfloatKernel(mesh->Nelements * mesh->Np * mesh->Nvgeo, mesh->o_vgeo, o_vgeoPfloat);
  }

  o_rhoCpAvg = platform->deviceMemoryPool.reserve<dfloat>(scalar->mesh(0)->Nlocal);

  if (platform->comm.mpiRank() == 0) {
    std::cout << "done\n";
  }
}

#ifdef ENABLE_CVODE
int cvode_t::cvodeRHS(realtype time, N_Vector Y, N_Vector Ydot)
{
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

// Jv = [f(y + v*sig) - f(y)]/sig, where sig = sigBar / ||v||_WRMS,
int cvode_t::jtv(double t,
                 const occa::memory &o_v,
                 const occa::memory &o_y,
                 const occa::memory &o_fy,
                 occa::memory &o_work,
                 occa::memory &o_Jv)
{
  if (detailedTimersEnabled) {
    platform->timer.tic(timerName + "solve::cvode::linearSolve::jtv", 0);
  }

  static dfloat sig;
  if (platform->options.compareArgs("CVODE UPDATE SIGMA", "TRUE")) {
    const auto v_wrms = sqrt(
        platform->linAlg->weightedSqrSum(this->nEq, o_ewt, o_v, platform->comm.mpiComm()) / this->nEqTotal);

    auto sigBar = [&]() {
      if (this->sigScale > 0) {
        return this->sigScale;
      }

      const auto y_wrms = sqrt(
          platform->linAlg->weightedSqrSum(this->nEq, o_ewt, o_y, platform->comm.mpiComm()) / this->nEqTotal);
      const auto sig = sqrt((1 + y_wrms) * std::numeric_limits<dfloat>::epsilon()) / v_wrms;
      if (platform->verbose() && platform->comm.mpiRank() == 0) {
        std::cout << "sigma= " << sig << std::endl;
      }
      return sig;
    };

    sig = sigBar() / v_wrms;
    platform->options.setArgs("CVODE UPDATE SIGMA", "FALSE");
  }

  // first-order DQ
  platform->linAlg->axpbyz(this->nEq, sig, o_v, 1.0, o_y, o_work);
  this->jtvRhs(t, o_work, o_Jv);
  platform->linAlg->axpbyz(this->nEq, 1 / sig, o_Jv, -1 / sig, o_fy, o_Jv);

  if (detailedTimersEnabled) {
    platform->timer.toc(timerName + "solve::cvode::linearSolve::jtv");
  }

  return 0;
}

int cvode_t::cvodeErrorWt(N_Vector y, N_Vector ewt)
{
  if (detailedTimersEnabled) {
    platform->timer.tic(timerName + "solve::cvode::errorWt", 0);
  }

  auto o_y = getN_VectorMemory(sunrealtype, y);
  o_ewt = getN_VectorMemory(sunrealtype, ewt);

  const auto cvodeCHT = scalar->cht && scalar->cvodeSolve[0];
  if (cvodeCHT) {
    launchKernel("cvode_t::errorWeight",
                 YLVec->Nlocal(0),
                 1,
                 this->relTol,
                 YLVec->invDegree(0),
                 this->o_absTol,
                 o_y,
                 o_ewt);
    if (this->Nscalar > 1) {
      auto NlocalL = YLVec->Nlocal(1);
      auto o_yV = o_y + NlocalL;
      auto o_ewtV = o_ewt + NlocalL;
      auto o_absV = this->o_absTol + 1;
      launchKernel("cvode_t::errorWeight",
                   NlocalL,
                   this->Nscalar - 1,
                   this->relTol,
                   YLVec->invDegree(1),
                   o_absV,
                   o_yV,
                   o_ewtV);
    }
  } else {
    launchKernel("cvode_t::errorWeight",
                 YLVec->Nlocal(0),
                 this->Nscalar,
                 this->relTol,
                 YLVec->invDegree(0),
                 this->o_absTol,
                 o_y,
                 o_ewt);
  }

  if (detailedTimersEnabled) {
    platform->timer.toc(timerName + "solve::cvode::errorWt");
  }

  return 0;
}
#endif

void cvode_t::initialize()
{

  if (isInitialized) {
    return;
  }
  isInitialized = true;

  int retval;

#ifdef ENABLE_CVODE
  auto fwdCvodeRHS = [](double time, N_Vector Y, N_Vector Ydot, void *user_data) {
    return static_cast<cvode_t *>(user_data)->cvodeRHS(time, Y, Ydot);
  };

  auto fwdLinearSolve = [](SUNLinearSolver S, SUNMatrix A, N_Vector x, N_Vector b, realtype tol) {
    const std::string timerName = "cvode_t::";
    std::string solverType;
    platform->options.setArgs("CVODE UPDATE SIGMA", "TRUE");
    platform->options.getArgs("CVODE SOLVER", solverType);

    platform->timer.tic(timerName + "solve::cvode::linearSolve", 0);
    int retVal;
    if (solverType == "CBGMRES") {
      retVal = cbGMRESSolve(S, x, b, tol);
    } else if (solverType == "GMRES") {
      retVal = SUNLinSolSolve_SPGMR(S, NULL, x, b, tol);
    }
    platform->timer.toc(timerName + "solve::cvode::linearSolve");
    return retVal;
  };

  // same as cvLsDQJtimes, but with scaling for sig
  auto fwdCvodeJtv =
      [](N_Vector v, N_Vector Jv, realtype t, N_Vector y, N_Vector fy, void *user_data, N_Vector work) {
        return static_cast<cvode_t *>(user_data)->cvodeJtv(v, Jv, t, y, fy, work);
      };

  SUNContext sunctx = nullptr;
  retval = SUNContext_Create((void *)&platform->comm.mpiComm(), &sunctx);
  check_retval(&retval, "SUNContext_Create", 1);

  const int blockSize = BLOCKSIZE;

  {
    this->y = nullptr;
    if (platform->device.mode() == "CUDA") {
#ifdef ENABLE_CUDA
      auto occaStream = platform->device.occaDevice().getStream();
      auto backendStreamNative = occa::unwrap(occaStream);
      SUNCudaThreadDirectExecPolicy stream_exec_policy(blockSize,
                                                       *static_cast<cudaStream_t *>(backendStreamNative));
      SUNCudaBlockReduceExecPolicy reduce_exec_policy(blockSize,
                                                      0,
                                                      *static_cast<cudaStream_t *>(backendStreamNative));
      this->y = N_VNew_Cuda(this->nEq, sunctx);
      check_retval((void *)this->y, "N_VNew_Cuda", 0);

      retval = N_VSetKernelExecPolicy_Cuda(this->y, &stream_exec_policy, &reduce_exec_policy);
      check_retval(&retval, "N_VSetKernelExecPolicy_Cuda", 0);

      retval = N_VEnableFusedOps_Cuda(this->y, SUNTRUE);
      check_retval(&retval, "N_VEnableFusedOps_Cuda", 1);
#else
      nekrsCheck(true,
                 platform->comm.mpiComm(),
                 EXIT_FAILURE,
                 "%s",
                 "CVODE ENABLE_CUDA not enabled, despite mode being CUDA!\n");
#endif
    } else if (platform->device.mode() == "HIP") {
#ifdef ENABLE_HIP
      auto occaStream = platform->device.occaDevice().getStream();
      auto backendStreamNative = occa::unwrap(occaStream);
      SUNHipThreadDirectExecPolicy stream_exec_policy(blockSize,
                                                      *static_cast<hipStream_t *>(backendStreamNative));
      SUNHupBlockReduceExecPolicy reduce_exec_policy(blockSize,
                                                     0,
                                                     *static_cast<hipStream_t *>(backendStreamNative));

      this->y = N_VNew_Hip(data->nEq, sunctx);
      check_retval((void *)this->y, "N_VNew_Hip", 0);

      retVal = N_VEnableFusedOps_Hip(this->y, SUNTRUE);
      check_retval(&retval, "N_VEnableFusedOps_Hip", 1);
#else
      nekrsCheck(true,
                 platform->comm.mpiComm(),
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
    this->cvodeY = N_VMake_MPIPlusX(platform->comm.mpiComm(), this->y, sunctx);
    this->nEqTotal = N_VGetLength(this->cvodeY);

    retval = N_VEnableFusedOps_MPIPlusX(this->cvodeY, SUNTRUE);
    check_retval(&retval, "N_VEnableFusedOps_MPIPlusX", 1);
  }

  o_cvodeY = getN_VectorMemory(sunrealtype, cvodeY);

  YLVec->optr(o_cvodeY);
  YdotLVec->optr(o_cvodeY);

  // set initial condition
  nrsToCv(scalar->o_S, *YLVec, false);

  if (platform->options.getArgs("CVODE ADVECTION TYPE").empty()) {
    const auto dealiasing = platform->options.compareArgs("OVERINTEGRATION", "TRUE") ? true : false;
    if (dealiasing) {
      platform->options.setArgs("CVODE ADVECTION TYPE", "CUBATURE+CONVECTIVE");
    } else {
      platform->options.setArgs("CVODE ADVECTION TYPE", "CONVECTIVE");
    }
  }

  if (platform->options.getArgs("CVODE INTEGRATOR").empty()) {
    platform->options.setArgs("CVODE INTEGRATOR", "BDF");
  }

  int integrator;
  if (platform->options.compareArgs("CVODE INTEGRATOR", "BDF")) {
    integrator = CV_BDF;
  } else if (platform->options.compareArgs("CVODE INTEGRATOR", "ADAMS")) {
    integrator = CV_ADAMS;
  } else {
    nekrsAbort(platform->comm.mpiComm(), EXIT_FAILURE, "%s", "Invalid CVODE INTEGRATOR!\n");
  }

  this->cvodeMem = CVodeCreate(integrator, sunctx);

  auto T0 = 0.0;
  platform->options.getArgs("START TIME", T0);

  this->relTol = 1e-4;
  platform->options.getArgs("CVODE RELATIVE TOLERANCE", this->relTol);

  double absTol = 1e-6;

  // populate absolute tolerance vector
  std::vector<dfloat> absTols(this->Nscalar, absTol);
  for (int is = 0; is < scalar->NSfields; is++) {
    if (!scalar->compute[is]) {
      continue;
    }
    if (!scalar->cvodeSolve[is]) {
      continue;
    }

    const auto cvodeScalarId = cvodeScalarIds[is];
    dfloat absTolScalar = absTol;
    platform->options.getArgs("SCALAR" + scalarDigitStr(is) + " CVODE ABSOLUTE TOLERANCE", absTolScalar);
    absTols[is] = absTolScalar;
  }

  this->o_absTol = platform->device.malloc<dfloat>(this->Nscalar);
  this->o_absTol.copyFrom(absTols.data());

  if (platform->options.getArgs("CVODE DQ SIGMA").empty()) {
    this->sigScale = -1;
  } else {
    platform->options.getArgs("CVODE DQ SIGMA", this->sigScale);
  }

  check_retval((void *)this->cvodeMem, "CVodeCreate", 0);
  retval = CVodeInit(this->cvodeMem, fwdCvodeRHS, T0, this->cvodeY);
  check_retval(&retval, "CVodeInit", 1);

  // provide function to compute weights to account for multiplicity
  // same as cvEwtSetSV, but with scaling for multiplicity
  CVEwtFn fwdCvodeErrorWt = [](N_Vector y, N_Vector ewt, void *user_data) {
    return static_cast<cvode_t *>(user_data)->cvodeErrorWt(y, ewt);
  };

  retval = CVodeWFtolerances(this->cvodeMem, fwdCvodeErrorWt);
  check_retval(&retval, "CVodeWFtolerances", 1);

  if (platform->options.getArgs("CVODE STOP TIME").empty()) {
    platform->options.setArgs("CVODE STOP TIME", "TRUE");
  }

#if 0
  auto nls = SUNNonlinSol_FixedPoint(this->cvodeY, 10, sunctx);
  SUNNonlinSolSetDamping_FixedPoint(nls, 0.1);
  CVodeSetNonlinearSolver(this->cvodeMem, nls);
#endif

  if (platform->options.getArgs("CVODE SOLVER").empty()) {
    platform->options.setArgs("CVODE SOLVER", "GMRES");
  }

  platform->options.getArgs("CVODE SOLVER", this->linearSolverType);

  SUNLinearSolver LS;

  if (this->linearSolverType.find("GMRES") != std::string::npos) {
    if (platform->options.getArgs("CVODE GMRES BASIS VECTORS").empty()) {
      platform->options.setArgs("CVODE GMRES BASIS VECTORS", "8");
    }

    int nVectors;
    platform->options.getArgs("CVODE GMRES BASIS VECTORS", nVectors);

    LS = SUNLinSol_SPGMR(cvodeY, PREC_NONE, nVectors, sunctx);
    check_retval(&retval, "SUNLinSol_SPFGMR", 1);
    LS->ops->solve = fwdLinearSolve;
  } else {
    nekrsCheck(true,
               platform->comm.mpiComm(),
               EXIT_FAILURE,
               "CVODE linear solver %s is not supported!\n",
               linearSolverType.c_str());
  }

  retval = CVodeSetLinearSolver(this->cvodeMem, LS, NULL);
  check_retval(&retval, "CVodeSetLinearSolver", 1);

  // custom settings
#if 1
  if (platform->options.getArgs("CVODE GS TYPE").empty()) {
    platform->options.setArgs("CVODE GS TYPE", "CLASSICAL");
  }

  if (linearSolverType.find("GMRES") != std::string::npos) {
    std::string gsType;
    platform->options.getArgs("CVODE GS TYPE", gsType);
    if (gsType.find("CLASSICAL") != std::string::npos) {
      SUNLinSol_SPGMRSetGSType(LS, SUN_CLASSICAL_GS);
      check_retval(&retval, "SUNLinSol_SPGMRSetGSType", 1);
    } else if (gsType == "MODIFIED") {
      nekrsCheck(this->linearSolverType == "CBGMRES",
                 platform->comm.mpiComm(),
                 EXIT_FAILURE,
                 "%s",
                 "CVODE GS TYPE MODIFIED not available for CBGMRES!\n");
    } else {
      nekrsAbort(platform->comm.mpiComm(), EXIT_FAILURE, "Invalid CVODE GS TYPE %s!\n", gsType.c_str());
    }
  }

  retval = CVodeSetJacTimes(this->cvodeMem, NULL, fwdCvodeJtv);
  check_retval(&retval, "CVodeSetJacTimes", 1);

  if (platform->options.getArgs("CVODE MAX STEPS").empty()) {
    platform->options.setArgs("CVODE MAX STEPS", "100");
  }

  int mxsteps;
  platform->options.getArgs("CVODE MAX STEPS", mxsteps);
  retval = CVodeSetMaxNumSteps(this->cvodeMem, mxsteps);

  if (platform->options.getArgs("CVODE MAX TIMESTEPPER ORDER").empty()) {
    platform->options.setArgs("CVODE MAX TIMESTEPPER ORDER", "3");
  }

  if (!platform->options.getArgs("CVODE MAX STEP SIZE").empty()) {
    dfloat hmax;
    platform->options.getArgs("CVODE MAX STEP SIZE", hmax);
    retval = CVodeSetMaxStep(this->cvodeMem, hmax);
    check_retval(&retval, "CVodeSetMaxStep", 1);
  }

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
  if (platform->options.getArgs("CVODE EPS LIN").empty()) {
    platform->options.setArgs("CVODE EPS LIN", "0.5");
  }

  double epsLin;
  platform->options.getArgs("CVODE EPS LIN", epsLin);
  retval = CVodeSetEpsLin(this->cvodeMem, epsLin);
  check_retval(&retval, "CVodeSetEpsLin", 1);
#endif

  // set user data as
  retval = CVodeSetUserData(this->cvodeMem, this);
  check_retval(&retval, "CVodeSetUserData", 1);

  if (platform->comm.mpiRank() != 0) {
    CVodeSetErrFile(this->cvodeMem, NULL);
  }

  resetCounters();

#else
  nekrsCheck(true, platform->comm.mpiComm(), EXIT_FAILURE, "%s\n", "cvode was not enabled");
#endif
}

cvode_t::~cvode_t()
{
  if (o_xyz0.byte_size()) {
    o_xyz0.free();
  }
  if (o_coeffExt.byte_size()) {
    o_coeffExt.free();
  }
  if (o_cvodeScalarIds.byte_size()) {
    o_cvodeScalarIds.free();
  }
  if (o_scalarIds.byte_size()) {
    o_scalarIds.free();
  }
  if (o_absTol.byte_size()) {
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
  auto mesh = scalar->mesh(0);
  int NelemT = mesh->Nelements;
  int NelemV = scalar->meshV->Nelements;
  std::vector<int> EToB(Nscalar * NelemT * mesh->Nfaces, 0);
  for (int cvodeScalar = 0; cvodeScalar < Nscalar; ++cvodeScalar) {
    const int fOffset = cvodeScalar * mesh->Nelements * mesh->Nfaces;
    const auto is = scalarIds.at(cvodeScalar);
    const auto sid = scalarDigitStr(is);
    for (dlong e = 0; e < NelemT; e++) {
      for (int f = 0; f < mesh->Nfaces; f++) {
        const int bID = mesh->EToB[f + e * mesh->Nfaces];
        EToB[f + e * mesh->Nfaces + fOffset] = platform->app->bc->typeElliptic(bID, "scalar" + sid);

        // Since EToB must include all of the faces on the T-mesh, we need to explicitly
        // mark the faces on the V-mesh as not having a boundary
        // This ensures that there is no effect in the resulting mask
        if (e >= NelemV && is != 0) {
          EToB[f + e * mesh->Nfaces + fOffset] = ellipticBcType::NO_OP;
        }
      }
    }
  }

  { // setup masked gs handle
    const auto [Nmasked_, o_maskIds_, NmaskedLocal, o_maskIdsLocal, NmaskedGlobal, o_maskIdsGlobal] =
        maskedFaceIds(mesh, mesh->Nlocal, Nscalar, fieldOffset, EToB, ellipticBcType::DIRICHLET);

    this->Nmasked = Nmasked_;
    this->o_maskIds = o_maskIds_;
  }

  maskOffset = alignStride<dfloat>(Nmasked);
  o_maskValues = platform->device.malloc<dfloat>(scalar->o_coeffEXT.size() * maskOffset);
}

void cvode_t::applyDirichlet(double time)
{
  // extrapolate masked Dirichlet values to current time state
  if (this->isRhsEvaluation() && (this->externalTStep > scalar->o_coeffEXT.size())) {

    if (this->Nmasked == 0) {
      return;
    }

    const int extOrder = std::min(this->externalTStep, static_cast<int>(scalar->o_coeffEXT.size()));
    launchKernel("cvode_t::extrapolateDirichlet",
                 this->Nmasked,
                 this->maskOffset,
                 extOrder,
                 this->o_maskIds,
                 o_coeffExt,
                 this->o_maskValues,
                 scalar->o_S + this->minCvodeScalarId * fieldOffset);
    return;
  }

  // lower than any other possible Dirichlet value
  static constexpr dfloat TINY = -1e30;

  auto o_S_start = platform->deviceMemoryPool.reserve<dfloat>(scalar->fieldOffsetSum);

  for (int is = 0; is < scalar->NSfields; is++) {
    if (!scalar->compute[is]) {
      continue;
    }
    if (!scalar->cvodeSolve[is]) {
      continue;
    }

    auto mesh = (is) ? scalar->meshV : scalar->mesh(0);

    const auto cvodeScalarId = cvodeScalarIds[is];

    auto o_diff_i = scalar->o_diff + scalar->fieldOffsetScan[is];
    auto o_rho_i = scalar->o_rho + scalar->fieldOffsetScan[is];
    auto o_Si = o_S_start + scalar->fieldOffsetScan[cvodeScalarId];

    platform->linAlg->fill(mesh->Nlocal, TINY, o_Si);

    for (int sweep = 0; sweep < 2; sweep++) {
      launchKernel("scalar-dirichletBC",
                   mesh->Nelements,
                   fieldOffset,
                   is,
                   time,
                   mesh->o_sgeo,
                   mesh->o_x,
                   mesh->o_y,
                   mesh->o_z,
                   mesh->o_vmapM,
                   mesh->o_EToB,
                   scalar->o_EToB + is * scalar->EToBOffset,
                   scalar->o_U,
                   o_diff_i,
                   o_rho_i,
                   0 /* neknekFieldOffset */,
                   o_NULL,
                   0,
                   o_NULL,
                   o_NULL,
                   o_NULL,
                   platform->app->bc->o_usrwrk,
                   o_Si);
      if (sweep == 0) {
        oogs::startFinish(o_Si, 1, 0, ogsDfloat, ogsMax, mesh->oogs);
      }
      if (sweep == 1) {
        oogs::startFinish(o_Si, 1, 0, ogsDfloat, ogsMin, mesh->oogs);
      }
    }
  }

  if (this->Nmasked == 0) {
    return;
  }

  launchKernel("scalar-maskCopy",
               this->Nmasked,
               0,
               scalar->fieldOffsetScan[minCvodeScalarId],
               o_maskIds,
               o_S_start,
               scalar->o_S);

  // o_maskValues must be at state t0 to be lagged by the subsequent CVODE solve call
  if (this->isRhsEvaluation()) {
    return;
  }

  launchKernel("cvode_t::mapToMaskedPoint", this->Nmasked, o_maskIds, o_S_start, o_maskValues);
}

void cvode_t::rhs(double time, const LVector_t<dfloat> &o_y, LVector_t<dfloat> &o_ydot)
{
  const auto tag = this->rhsTagName();
  const auto saveTimerScope = timerScope;
  timerScope = tag;
  if (detailedTimersEnabled) {
    platform->timer.tic(tag, 0);
  }
  this->setIsRhsEvaluation(true);

  if (userRHS) {
    userRHS(time, tExternal, o_y, o_ydot);
  } else {
    defaultRHS(time, tExternal, o_y, o_ydot);
  }

  this->setIsRhsEvaluation(false);
  if (detailedTimersEnabled) {
    platform->timer.toc(tag);
  }
  timerScope = saveTimerScope;
}

int cvode_t::jtvRhs(double time, const occa::memory &o_y, occa::memory &o_ydot)
{
  auto o_yy = o_y; /* kludge because optr takes occa::memory&*/
  this->YLVec->optr(o_yy);
  this->YdotLVec->optr(o_ydot);

  this->jtvRhs(time, *(this->YLVec), *(this->YdotLVec));

  return 0;
}

void cvode_t::jtvRhs(double time, const LVector_t<dfloat> &o_y, LVector_t<dfloat> &o_ydot)
{
  this->setIsJacobianEvaluation(true);

  if (userJacobian) {
    userJacobian(time, tExternal, o_y, o_ydot);
  } else {
    this->rhs(time, o_y, o_ydot);
  }

  this->setIsJacobianEvaluation(false);
}

void cvode_t::defaultRHS(double time, double t0, const LVector_t<dfloat> &o_y, LVector_t<dfloat> &o_ydot)
{

  if (time != tprev) {

    if (detailedTimersEnabled) {
      platform->timer.tic(timerScope + "::extrapolate", 0);
    }

    if (platform->comm.mpiRank() == 0 && time - tprev > 0 && verboseCVODE) {
      std::cout << "tCv= " << time << ", dt= " << time - tprev << std::endl;
    }

    const dfloat cvodeDt = time - t0;
    tprev = time;

    dtCvode[0] = cvodeDt;
    dtCvode[1] = scalar->dt[1];
    dtCvode[2] = scalar->dt[2];

    // evaluate external variables at integrator time using extrapolation
    const int bdfOrder = std::min(this->externalTStep, static_cast<int>(scalar->o_coeffBDF.size()));
    const int extOrder = std::min(this->externalTStep, static_cast<int>(scalar->o_coeffEXT.size()));
    nek::extCoeff(_coeffEXT.data(), dtCvode.data(), extOrder, bdfOrder);
    nek::bdfCoeff(&this->_g0, _coeffBDF.data(), dtCvode.data(), bdfOrder);
    for (int i = scalar->o_coeffEXT.size(); i > extOrder; i--) {
      _coeffEXT[i - 1] = 0.0;
    }
    for (int i = scalar->o_coeffBDF.size(); i > bdfOrder; i--) {
      _coeffBDF[i - 1] = 0.0;
    }

    o_coeffExt.copyFrom(_coeffEXT.data(), maxTimestepperOrder);

    launchKernel("cvode_t::extrapolate",
                 scalar->meshV->Nlocal,
                 scalar->meshV->dim,
                 extOrder,
                 fieldOffset,
                 o_coeffExt,
                 this->o_U0,
                 scalar->o_U);

    if (movingMesh) {
      auto mesh = scalar->geom->mesh;
      const int meshOrder = std::min(this->externalTStep, static_cast<int>(scalar->geom->o_coeffAB.size()));
      std::vector<dfloat> coeffAB(scalar->geom->o_coeffAB.size());
      nek::coeffAB(coeffAB.data(), dtCvode.data(), meshOrder);
      for (int i = 0; i < meshOrder; ++i) {
        coeffAB[i] *= dtCvode[0];
      }
      for (int i = scalar->geom->o_coeffAB.size(); i > meshOrder; i--) {
        coeffAB[i - 1] = 0.0;
      }
      scalar->geom->o_coeffAB.copyFrom(coeffAB.data());

      // restore prior to integration in move
      {
        mesh->o_x.copyFrom(o_xyz0, mesh->Nlocal, 0, 0 * fieldOffset);
        mesh->o_y.copyFrom(o_xyz0, mesh->Nlocal, 0, 1 * fieldOffset);
        mesh->o_z.copyFrom(o_xyz0, mesh->Nlocal, 0, 2 * fieldOffset);
        scalar->geom->o_U.copyFrom(o_U0mesh, mesh->dim * fieldOffset);
      }

      scalar->geom->integrate(false);
      launchKernel("cvode_t::extrapolate",
                   mesh->Nlocal,
                   mesh->dim,
                   extOrder,
                   fieldOffset,
                   o_coeffExt,
                   o_U0mesh,
                   scalar->geom->o_U);
    }

    computeUrst();

    if (detailedTimersEnabled) {
      platform->timer.toc(timerScope + "::extrapolate");
    }
  }

  cvToNrs(o_y, scalar->o_S, false);

  if (detailedTimersEnabled) {
    platform->timer.tic(timerScope + "::applyDirichlet", 0);
  }

  this->applyDirichlet(time);

  if (detailedTimersEnabled) {
    platform->timer.toc(timerScope + "::applyDirichlet");
  }

  const bool chtCVODE = platform->options.compareArgs("SCALAR00 MESH", "SOLID") && scalar->cvodeSolve[0];

  if (!(isJacobianEvaluation() && recycleProperties)) {
    platform->timer.tic(timerScope + "::evaluateProperties", 0);
    this->evaluateProperties(time);
    platform->timer.toc(timerScope + "::evaluateProperties");

    o_rhoCpAvg.copyFrom(scalar->o_rho);
    if (chtCVODE) {
      auto mesh = scalar->mesh(minCvodeScalarId);
      platform->linAlg->axmy(mesh->Nlocal, 1.0, mesh->o_LMM, o_rhoCpAvg);
      oogs::startFinish(o_rhoCpAvg, 1, fieldOffset, ogsDfloat, ogsAdd, scalar->mesh(0)->oogs);
      platform->linAlg->axmy(mesh->Nlocal, 1.0, mesh->o_invLMM, o_rhoCpAvg);
    }
  }

  platform->linAlg->fill(scalar->fieldOffsetSum, 0.0, scalar->o_EXT);
  if (userMakeq) {
    userMakeq(time);
  } else {
    makeq(time);
  }

  auto applyOgsOperation = [&](auto ogsFunc) {
    if (chtCVODE) {
      ogsFunc(scalar->o_EXT, 1, 0, ogsDfloat, ogsAdd, scalar->mesh(0)->oogs);
    }

    if (!chtCVODE || (chtCVODE && this->Nscalar > 1)) {
      dlong startScalar = minCvodeScalarId;
      dlong numScalars = this->Nscalar;
      if (chtCVODE) {
        startScalar++;
        numScalars--;
      }
      auto o_fld = scalar->o_EXT + fieldOffset;
      ogsFunc(o_fld, numScalars, fieldOffset, ogsDfloat, ogsAdd, this->gsh);
    }
  };

  if (detailedTimersEnabled) {
    platform->timer.tic(timerScope + "::gatherScatterAndLocalPointSource", 0);
  }

  applyOgsOperation(oogs::start);

  if (userLocalPointSourceE) {
    platform->timer.tic(timerScope + "::gatherScatterAndLocalPointSource::pointSourceE", 0);
    deviceMemory<dfloat> o_S(scalar->o_S);
    deviceMemory<dfloat> o_src(this->o_pointSource);
    userLocalPointSourceE(time, o_S, o_src);
    platform->timer.toc(timerScope + "::gatherScatterAndLocalPointSource::pointSourceE");
  } else if (userLocalPointSourceL) {
    platform->timer.tic(timerScope + "::gatherScatterAndLocalPointSource::pointSourceL", 0);
    userLocalPointSourceL(time, o_y, o_ydot);
    platform->timer.toc(timerScope + "::gatherScatterAndLocalPointSource::pointSourceL");
    cvToNrs(o_ydot, this->o_pointSource, true);
  }

  applyOgsOperation(oogs::finish);

  if (detailedTimersEnabled) {
    platform->timer.toc(timerScope + "::gatherScatterAndLocalPointSource");
  }

  // evaluate p0th + dp0thdt and update p-dependent quantities like "rho"
  // using evaluateDivergence as divergence is required to compute those quantities
  auto addDpdtTerm = false;
  if (scalar->dpdt) {
    addDpdtTerm = true;

    if (!(isJacobianEvaluation() && recycleProperties)) {
      if (detailedTimersEnabled) {
        platform->timer.tic(timerScope + "::dp0thdt", 0);
      }

      this->evaluateDivergence(time);

      o_rhoCpAvg.copyFrom(scalar->o_rho);
      if (chtCVODE) {
        auto mesh = scalar->mesh(minCvodeScalarId);
        platform->linAlg->axmy(mesh->Nlocal, 1.0, mesh->o_LMM, o_rhoCpAvg);
        oogs::startFinish(o_rhoCpAvg, 1, fieldOffset, ogsDfloat, ogsAdd, scalar->mesh(0)->oogs);
        platform->linAlg->axmy(mesh->Nlocal, 1.0, mesh->o_invLMM, o_rhoCpAvg);
      }

      // do not re-evaluate convection + filtering term (ignore updated rho)

      if (detailedTimersEnabled) {
        platform->timer.toc(timerScope + "::dp0thdt");
      }
    }
  }

  // weight FS by inverse assembled mass matrix, add pointSource and finally divide by "rho"
  {
    if (detailedTimersEnabled) {
      platform->timer.tic(timerScope + "::fusedAddRhoDiv", 0);
    }

    const auto startScalar = minCvodeScalarId;
    auto o_FS = scalar->o_EXT + scalar->fieldOffsetScan[startScalar];
    auto o_rho = scalar->o_rho + scalar->fieldOffsetScan[startScalar];

    occa::memory o_ptSource;
    if (this->o_pointSource.isInitialized()) {
      o_ptSource = this->o_pointSource + scalar->fieldOffsetScan[startScalar];
    }

    const int useFieldRho = !sharedRho;
    launchKernel("cvode_t::rhoDiv",
                 scalar->mesh(0)->Nlocal,
                 scalar->meshV->Nlocal,
                 this->Nscalar,
                 fieldOffset,
                 scalar->mesh(0)->o_invLMM,
                 scalar->meshV->o_invLMM,
                 useFieldRho,
                 o_rhoCpAvg,
                 o_rho,
                 o_ptSource,
                 (addDpdtTerm) ? (*scalar->dp0thdt) * (*scalar->alpha0Ref) : static_cast<dfloat>(0.0),
                 o_FS);

    if (detailedTimersEnabled) {
      platform->timer.toc(timerScope + "::fusedAddRhoDiv");
    }
  }

  // zero out RHS for unknowns using a Dirichlet BC
  if (detailedTimersEnabled) {
    platform->timer.tic(timerScope + "::maskDirichlet", 0);
  }
  if (this->Nmasked > 0) {
    auto o_FS = scalar->o_EXT + scalar->fieldOffsetScan[minCvodeScalarId];
    platform->linAlg->mask(this->Nmasked, this->o_maskIds, o_FS);
  }
  if (detailedTimersEnabled) {
    platform->timer.toc(timerScope + "::maskDirichlet");
  }

  nrsToCv(scalar->o_EXT, o_ydot, true);
}

void cvode_t::makeq(double time)
{
  const auto timerScopeSave = timerScope;
  timerScope = timerScopeSave + "::makeq";
  platform->timer.tic(timerScope, 0);

  auto applyTerms = [&](mesh_t *mesh, const dlong scalarStart, const dlong Nscalar) {
    if (detailedTimersEnabled) {
      platform->timer.tic(timerScope + "::weakLaplacian", 0);
    }

    // weight o_FS on input and add Laplacian term
    launchKernel("cvode_t::weakLaplacianHex3D",
                 mesh->Nelements,
                 Nscalar,
                 scalar->o_fieldOffsetScan + scalarStart,
                 mesh->o_ggeo,
                 mesh->o_D,
                 scalar->o_diff,
                 scalar->o_S,
                 scalar->o_EXT);

    if (detailedTimersEnabled) {
      platform->timer.toc(timerScope + "::weakLaplacian");
    }

    if (detailedTimersEnabled) {
      platform->timer.tic(timerScope + "::neumannBC", 0);
    }

    launchKernel("scalar-neumannBCHex3D",
                 mesh->Nelements,
                 Nscalar,
                 mesh->o_sgeo,
                 mesh->o_vmapM,
                 mesh->o_EToB,
                 scalarStart,
                 time,
                 scalar->vFieldOffset,
                 fieldOffset,
                 fieldOffset,
                 scalar->EToBOffset,
                 mesh->o_x,
                 mesh->o_y,
                 mesh->o_z,
                 scalar->o_U,
                 scalar->o_S,
                 scalar->o_EToB,
                 scalar->o_diff,
                 scalar->o_rho,
                 platform->app->bc->o_usrwrk,
                 scalar->o_EXT);

    if (detailedTimersEnabled) {
      platform->timer.toc(timerScope + "::neumannBC");
    }

    if (scalar->o_applyFilterRT.isInitialized()) {
      if (detailedTimersEnabled) {
        platform->timer.tic(timerScope + "::filterRT", 0);
      }

      launchKernel("scalar-filterRTHex3D",
                   scalar->meshV->Nelements,
                   scalarStart,
                   Nscalar,
                   scalar->o_fieldOffsetScan,
                   scalar->o_applyFilterRT,
                   scalar->o_filterRT,
                   scalar->o_filterS,
                   scalar->o_rho,
                   scalar->o_S,
                   scalar->o_EXT);

      if (detailedTimersEnabled) {
        platform->timer.toc(timerScope + "::filterRT");
      }
    }

    if (platform->options.compareArgs("ADVECTION", "TRUE")) {
      if (detailedTimersEnabled) {
        platform->timer.tic(timerScope + "::advection", 0);
      }
      const auto weighted = true;
      const bool useRelativeVelocity = platform->options.compareArgs("MOVING MESH", "TRUE");

      if (platform->options.compareArgs("CVODE ADVECTION TYPE", "CUBATURE")) {
        launchKernel("scalar->strongAdvectionCubatureVolumeHex3D",
                     scalar->meshV->Nelements,
                     Nscalar,
                     static_cast<int>(weighted),
                     static_cast<int>(false), // shared rho
                     mesh->o_vgeo,
                     mesh->o_cubDiffInterpT,
                     mesh->o_cubInterpT,
                     mesh->o_cubProjectT,
                     scalar->o_compute + scalarStart,
                     scalar->o_fieldOffsetScan + scalarStart,
                     scalar->vFieldOffset,
                     scalar->vCubatureOffset,
                     scalar->o_S,
                     o_Urst,
                     scalar->o_rho,
                     scalar->o_EXT);
      } else {
        launchKernel("scalar-strongAdvectionVolumeHex3D",
                     scalar->meshV->Nelements,
                     Nscalar,
                     weighted,
                     mesh->o_vgeo,
                     mesh->o_D,
                     scalar->o_compute + scalarStart,
                     scalar->o_fieldOffsetScan + scalarStart,
                     scalar->vFieldOffset,
                     scalar->o_S,
                     o_Urst,
                     scalar->o_rho,
                     scalar->o_EXT);
      }

      if (detailedTimersEnabled) {
        platform->timer.toc(timerScope + "::advection");
      }
    }
  };

  if (platform->app->userSource) {
    const auto makeQScope = timerScope;
    timerScope = makeQScope + "::udfSEqnSource";
    platform->callerScope = "cvode";
    platform->timer.tic(timerScope, 0);
    platform->app->userSource(time);
    platform->timer.toc(timerScope);
    platform->callerScope.clear();
    timerScope = makeQScope;
  }

  const bool chtCVODE = platform->options.compareArgs("SCALAR00 MESH", "SOLID") && scalar->cvodeSolve[0];
  if (chtCVODE) {
    applyTerms(scalar->mesh(0), 0, 1);
  }
  if (!chtCVODE || (chtCVODE && this->Nscalar > 1)) {
    dlong startScalar = minCvodeScalarId;
    dlong numScalars = this->Nscalar;
    if (chtCVODE) {
      startScalar++;
      numScalars--;
    }
    applyTerms(scalar->meshV, startScalar, numScalars);
  }

  platform->timer.toc(timerScope);
  timerScope = timerScopeSave;
}

void cvode_t::nrsToCv(occa::memory o_EField, LVector_t<dfloat> &o_LField, bool isYdot)
{
  if (detailedTimersEnabled) {
    platform->timer.tic(timerScope + "::nrsToCv", 0);
  }

  auto o_E = o_EField + scalar->fieldOffsetScan[this->minCvodeScalarId];
  o_LField.copyFromE(fieldOffset, o_E);

  if (userPostNrsToCv) {
    userPostNrsToCv(o_LField, isYdot);
  }
  if (detailedTimersEnabled) {
    platform->timer.toc(timerScope + "::nrsToCv");
  }
}

void cvode_t::cvToNrs(const LVector_t<dfloat> &o_LField, occa::memory o_EField, bool isYdot)
{
  if (detailedTimersEnabled) {
    platform->timer.tic(timerScope + "::cvToNrs", 0);
  }

  auto o_E = o_EField + scalar->fieldOffsetScan[this->minCvodeScalarId];
  o_LField.copyToE(fieldOffset, o_E);

  if (userPostCvToNrs) {
    deviceMemory<dfloat> o_in(o_EField);
    userPostCvToNrs(o_in, isYdot);
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

  auto mesh = (scalar->cht) ? scalar->mesh(0) : scalar->meshV;

  if (userPreSolve) {
    userPreSolve();
  }

  // lag solution state and update current state
  for (int s = scalar->o_coeffEXT.size(); s > 1; s--) {
    const auto N = mesh->dim * fieldOffset;
    o_U0.copyFrom(o_U0, N, (s - 1) * N, (s - 2) * N);
  }
  o_U0.copyFrom(scalar->o_U, mesh->dim * fieldOffset);

  if (movingMesh) {

    for (int s = scalar->o_coeffEXT.size(); s > 1; s--) {
      const auto N = mesh->dim * fieldOffset;
      o_U0mesh.copyFrom(o_U0mesh, N, (s - 1) * N, (s - 2) * N);
    }
    o_U0mesh.copyFrom(mesh->o_U, mesh->dim * fieldOffset);

    o_xyz0.copyFrom(mesh->o_x, mesh->Nlocal, 0 * fieldOffset, 0);
    o_xyz0.copyFrom(mesh->o_y, mesh->Nlocal, 1 * fieldOffset, 0);
    o_xyz0.copyFrom(mesh->o_z, mesh->Nlocal, 2 * fieldOffset, 0);
  }

  YLVec->optr(o_cvodeY);
  nrsToCv(scalar->o_S, *YLVec, false);

  this->tExternal = t0;
  this->externalTStep = tstep;

  double t;
  int retval = 0;

  // integrate only to time t1
  if (platform->options.compareArgs("CVODE STOP TIME", "TRUE")) {
    retval = CVodeSetStopTime(cvodeMem, t1);
    nekrsCheck(retval < 0, MPI_COMM_SELF, EXIT_FAILURE, "%s", "Error calling CVodeSetStopTime\n");
  }

  if (!platform->options.getArgs("CVODE HMAX RATIO").empty()) {
    double hmax;
    platform->options.getArgs("CVODE HMAX RATIO", hmax);
    hmax *= scalar->dt[0];
    retval = CVodeSetMaxStep(cvodeMem, hmax);
    nekrsCheck(retval < 0, MPI_COMM_SELF, EXIT_FAILURE, "%s", "Error calling CVodeSetMaxStep\n");
  }

  const auto oldScope = timerScope;
  timerScope = oldScope + "::cvode";
  if (detailedTimersEnabled) {
    platform->timer.tic(timerScope, 0);
  }

  // call cvode solver
  setPreviousCounters();
  retval = CVode(cvodeMem, t1, cvodeY, &t, CV_NORMAL);
  updateCounters();
  if (retval != CV_SUCCESS) {
    const auto maxRestarts = 5;
    int cnt = 0;
    while (retval != CV_SUCCESS) {
      cnt++;
      if (platform->comm.mpiRank() == 0) {
        std::cout << "restarting CVODE ...\n";
      }
      retval = CVodeReInit(cvodeMem, t, cvodeY);
      check_retval(&retval, "CVodeReInit", 1);

      this->tprev = std::numeric_limits<double>::max();
      setPreviousCounters(true);
      retval = CVode(cvodeMem, t1, cvodeY, &t, CV_NORMAL);
      updateCounters();

      nekrsCheck(cnt > maxRestarts,
                 platform->comm.mpiComm(),
                 EXIT_FAILURE,
                 "%s",
                 "Reached maximum number of allowed CVODE restarts! Giving up ...\n");
    }
  }
  nekrsCheck(retval < 0, MPI_COMM_SELF, EXIT_FAILURE, "%s", "CVODE failed after restart\n");

  if (detailedTimersEnabled) {
    platform->timer.toc(timerScope);
  }
  timerScope = oldScope;

  YLVec->optr(o_cvodeY);
  cvToNrs(*YLVec, scalar->o_S, false);

  if (detailedTimersEnabled) {
    platform->timer.tic(timerScope + "::restore", 0);
  }

  // restore previous state
  scalar->o_U.copyFrom(o_U0, scalar->meshV * scalar->vFieldOffset);

  if (movingMesh) {
    mesh->o_x.copyFrom(o_xyz0, mesh->Nlocal, 0, 0 * fieldOffset);
    mesh->o_y.copyFrom(o_xyz0, mesh->Nlocal, 0, 1 * fieldOffset);
    mesh->o_z.copyFrom(o_xyz0, mesh->Nlocal, 0, 2 * fieldOffset);

    scalar->geom->o_U.copyFrom(o_U0mesh, mesh->dim * fieldOffset);

    mesh->update();
  }

  o_Urst.free();

  for (int s = scalar->o_coeffEXT.size(); s > 1; s--) {
    if (maskOffset) {
      o_maskValues.copyFrom(o_maskValues, maskOffset, (s - 1) * maskOffset, (s - 2) * maskOffset);
    }
  }

  // enforce boundary condition on final solution
  this->applyDirichlet(t1);

  if (userPostSolve) {
    userPostSolve();
  }

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

void cvode_t::updateCounters() {}

void cvode_t::resetCounters() {}
#endif

void cvode_t::printInfo(bool printVerboseInfo)
{
  auto cvodeMem = this->cvodeMem;

  std::string scalars;
  {
    std::ostringstream ss;
    ss << "S" << scalarDigitStr(minCvodeScalarId);
    if (maxCvodeScalarId > minCvodeScalarId) {
      ss << "-" << scalarDigitStr(maxCvodeScalarId);
    }
    scalars = ss.str();
  }

  if (printVerboseInfo) {
    printf("%-20s: nsteps %03ld  nRHS %03ld  nni %01ld  nni/nsteps %.1f  nli %03ld  nli/nni %.1f\n",
           scalars.c_str(),
           nsteps,
           nrhs,
           nni,
           static_cast<float>(nni) / nsteps,
           nli,
           static_cast<float>(nli) / nni);
  } else {
    std::ostringstream ss;
    ss << "  " << scalars;
    printf("%s: %ld %ld %ld %ldf", ss.str().c_str(), nsteps, nrhs, nni, nli);
  }

  resetCounters();
}

void cvode_t::printTimers()
{
  auto mesh = scalar->meshV;
  long long int NglobalElements = mesh->Nelements;
  MPI_Allreduce(MPI_IN_PLACE, &NglobalElements, 1, MPI_LONG_LONG_INT, MPI_SUM, platform->comm.mpiComm());
  const long long int DOF = (NglobalElements * std::pow(mesh->N, 3) * this->cvodeScalarIds.size());
  platform->timer.print(timerName, DOF);
}

void cvode_t::resetTimers()
{
  platform->timer.reset(timerName);
}

std::string cvode_t::rhsTagName() const
{
  return this->isJacobianEvaluation() ? timerScope + "::linearSolve::jtv::rhs" : timerScope + "::rhs";
}

void cvode_t::setLocalPointSource(userLocalPointSourceE_t _userLocalPointSource)
{
  userLocalPointSourceE = _userLocalPointSource;

  nekrsCheck(!o_pointSource.isInitialized(),
             platform->comm.mpiComm(),
             EXIT_FAILURE,
             "%s",
             "o_pointSource not initialized!\n");
}

void cvode_t::setLocalPointSource(userLocalPointSourceL_t _userLocalPointSource)
{
  userLocalPointSourceL = _userLocalPointSource;

  nekrsCheck(!o_pointSource.isInitialized(),
             platform->comm.mpiComm(),
             EXIT_FAILURE,
             "%s",
             "o_pointSource not initialized!\n");
}

void cvode_t::computeUrst()
{
  const bool relative = movingMesh;
  auto mesh = scalar->meshV;

  const auto cubature = platform->options.compareArgs("CVODE ADVECTION TYPE", "CUBATURE");

  if (!o_Urst.isInitialized()) {
    o_Urst = platform->deviceMemoryPool.reserve<dfloat>(mesh->dim * scalar->vCubatureOffset);
  }

  if (cubature) {
    launchKernel("nrs-UrstCubatureHex3D",
                 mesh->Nelements,
                 static_cast<int>(relative),
                 mesh->o_cubvgeo,
                 mesh->o_cubInterpT,
                 fieldOffset,
                 scalar->vCubatureOffset,
                 scalar->o_U,
                 scalar->geom->o_U,
                 o_Urst);
  } else {
    launchKernel("nrs-UrstHex3D",
                 mesh->Nelements,
                 static_cast<int>(relative),
                 mesh->o_vgeo,
                 fieldOffset,
                 scalar->o_U,
                 scalar->geom->o_U,
                 o_Urst);
  }
}
