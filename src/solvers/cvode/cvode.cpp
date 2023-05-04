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
#include <sunlinsol/sunlinsol_spgmr.h>
#include <sundials/sundials_types.h>
#include <sundials/sundials_math.h>
#include <cvode/cvode.h>
#include <nvector/nvector_serial.h>
#include <nvector/nvector_mpiplusx.h>
#ifdef ENABLE_CUDA
#include <nvector/nvector_cuda.h>
#endif
#ifdef ENABLE_HIP
#include <nvector/nvector_hip.h>
#endif
#endif

//#define USE_E_VECTOR_LAYOUT 1

namespace {

#ifdef ENABLE_CVODE
sunrealtype *__N_VGetDeviceArrayPointer(N_Vector u)
{
  bool useDevice = false;
  useDevice |= platform->device.mode() == "CUDA";
  useDevice |= platform->device.mode() == "HIP";
  useDevice |= platform->device.mode() == "OPENCL";

  if (useDevice) {
    return N_VGetDeviceArrayPointer(u);
  }
  else {
    return N_VGetArrayPointer_Serial(u);
  }
}
#endif

void check_retval(void *returnvalue, const char *funcname, int opt)
{
  int *retval;

  /* Check if SUNDIALS function returned NULL pointer - no memory allocated */

  if (opt == 0 && returnvalue == NULL) {
    nrsAbort(MPI_COMM_SELF, EXIT_FAILURE, 
             "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n", funcname);
  }
  /* Check if retval < 0 */
  else if (opt == 1) {
    retval = (int *)returnvalue;
    if (*retval < 0) {
      nrsAbort(MPI_COMM_SELF, EXIT_FAILURE, 
      "\nSUNDIALS_ERROR: %s() failed with retval = %d\n\n", funcname, *retval);
    }
  }
  /* Check if function returned NULL pointer - no memory allocated */
  else if (opt == 2 && returnvalue == NULL) {
    nrsAbort(MPI_COMM_SELF, EXIT_FAILURE, 
             "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n", funcname);
  }
}

} // namespace

cvode_t::cvode_t(nrs_t *nrs)
{
  auto cds = nrs->cds;

  o_coeffExt = platform->device.malloc(maxTimestepperOrder * sizeof(dfloat));

  o_U = platform->device.malloc((nrs->nEXT * nrs->NVfields * sizeof(dfloat)) * nrs->fieldOffset);

  if (platform->options.compareArgs("MOVING MESH", "TRUE")) {
    o_meshU = platform->device.malloc((nrs->nEXT * nrs->NVfields * sizeof(dfloat)) * nrs->fieldOffset);
    o_xyz0 = platform->device.malloc((nrs->NVfields * sizeof(dfloat)) * nrs->fieldOffset);
  }

  if (platform->options.getArgs("CVODE RECYCLE PROPERTIES").empty()) {
    platform->options.setArgs("CVODE RECYCLE PROPERTIES", "TRUE");
  }
  recycleProperties = platform->options.compareArgs("CVODE RECYCLE PROPERTIES", "TRUE");

  setupEToLMapping(nrs);

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
  }

  o_scalarIds = platform->device.malloc(scalarIds.size() * sizeof(dlong), scalarIds.data());
  o_cvodeScalarIds = platform->device.malloc(cvodeScalarIds.size() * sizeof(dlong), cvodeScalarIds.data());

  setupDirichletMask(nrs);

  this->weakLaplacianKernel = platform->kernels.get("cvode_t::weakLaplacianHex3D");
  this->nrsToCvKernel = platform->kernels.get("cvode_t::nrsToCv");
  this->cvToNrsKernel = platform->kernels.get("cvode_t::cvToNrs");
  this->extrapolateDirichletKernel = platform->kernels.get("cvode_t::extrapolateDirichlet");
  this->mapToMaskedPointKernel = platform->kernels.get("cvode_t::mapToMaskedPoint");
  this->errorWeightKernel = platform->kernels.get("cvode_t::errorWeight");
  this->fusedAddRhoDivKernel = platform->kernels.get("cvode_t::rhoDiv");

  nEq = Nscalar * LFieldOffset;

  isInitialized = false;

  verboseCVODE = platform->options.compareArgs("CVODE VERBOSE", "TRUE");
  sharedRho = platform->options.compareArgs("CVODE SHARED RHO", "TRUE");
  if(scalarIds.size() < 2) sharedRho = false;
 
  mixedPrecisionJtvEnabled = platform->options.compareArgs("CVODE MIXED PRECISION JTV", "TRUE");
  nrsCheck(mixedPrecisionJtvEnabled, platform->comm.mpiComm, EXIT_FAILURE, "%s\n", "CVODE MIXED PRECISION JTV = TRUE not supported yet");

  if(mixedPrecisionJtvEnabled){
    auto mesh = cds->mesh[0];
    o_vgeoPfloat = platform->device.malloc(mesh->Nelements * mesh->Np * mesh->Nvgeo, sizeof(pfloat));
    platform->copyDfloatToPfloatKernel(mesh->Nelements * mesh->Np * mesh->Nvgeo,
                                       mesh->o_vgeo,
                                       o_vgeoPfloat);
  }

  _nrs = nrs;

  const auto NbyteCvode = (this->Nscalar + 1) * nrs->fieldOffset * sizeof(dfloat);
  if(NbyteCvode > platform->o_mempool.bytesAllocated){
    platform->create_mempool(nrs->fieldOffset, this->Nscalar + 1);
  }

}

void cvode_t::initialize(nrs_t *nrs)
{

  if (isInitialized)
    return;
  isInitialized = true;

  auto *cds = nrs->cds;

  int retval;

#ifdef ENABLE_CVODE
  // wrap RHS function into type expected by CVODE
  CVRhsFn cvodeRHS = [](double time, N_Vector Y, N_Vector Ydot, void *user_data) {
    auto data = static_cast<userData_t *>(user_data);
    auto nrs = data->nrs;
    auto platform = data->platform;
    auto cvode = data->cvode;

    occa::memory o_y = platform->device.occaDevice().wrapMemory<sunrealtype>(
        __N_VGetDeviceArrayPointer(N_VGetLocalVector_MPIPlusX(Y)),
        cvode->numEquations());

    occa::memory o_ydot = platform->device.occaDevice().wrapMemory<sunrealtype>(
        __N_VGetDeviceArrayPointer(N_VGetLocalVector_MPIPlusX(Ydot)),
        cvode->numEquations());

    cvode->rhs(nrs, time, o_y, o_ydot);

    return 0;
  };

  CVRhsFn cvodeJtvRHS = [](double time, N_Vector Y, N_Vector Ydot, void *user_data) {
    auto data = static_cast<userData_t *>(user_data);
    auto nrs = data->nrs;
    auto platform = data->platform;
    auto cvode = data->cvode;

    occa::memory o_y = platform->device.occaDevice().wrapMemory<sunrealtype>(
        __N_VGetDeviceArrayPointer(N_VGetLocalVector_MPIPlusX(Y)),
        cvode->numEquations());

    occa::memory o_ydot = platform->device.occaDevice().wrapMemory<sunrealtype>(
        __N_VGetDeviceArrayPointer(N_VGetLocalVector_MPIPlusX(Ydot)),
        cvode->numEquations());

    cvode->jtvRHS(nrs, time, o_y, o_ydot);

    return 0;
  };

  // same as cvLsDQJtimes, but with scaling for sig
  CVLsJacTimesVecFn cvodeJtv =
      [](N_Vector v, N_Vector Jv, realtype t, N_Vector y, N_Vector fy, void *user_data, N_Vector work) {
        auto data = static_cast<userData_t *>(user_data);
        auto nrs = data->nrs;
        auto platform = data->platform;
        auto cvode = data->cvode;
        const auto sigScale = cvode->sigmaScale();

        void *cvode_mem = cvode->getCvodeMem();

        realtype sig, siginv;
        int iter, retval;

        retval = CVodeGetErrWeights(cvode_mem, work);
        if (retval != CV_SUCCESS)
          return (retval);

        /* Initialize perturbation to 1/||v|| */
        sig = sigScale / N_VWrmsNorm(v, work);

        constexpr int maxDQIters{3};

        occa::memory o_work = platform->device.occaDevice().wrapMemory<sunrealtype>(
            __N_VGetDeviceArrayPointer(N_VGetLocalVector_MPIPlusX(work)),
            cvode->numEquations());

        occa::memory o_Jv = platform->device.occaDevice().wrapMemory<sunrealtype>(
            __N_VGetDeviceArrayPointer(N_VGetLocalVector_MPIPlusX(Jv)),
            cvode->numEquations());

        for (iter = 0; iter < maxDQIters; iter++) {

          /* Set work = y + sig*v */
          N_VLinearSum(sig, v, 1.0, y, work);

          /* Set Jv = f(tn, y+sig*v) */
          cvode->jtvRHS(nrs, t, o_work, o_Jv);
          retval = 0; // currently we don't do any error checking in the RHS
          if (retval == 0)
            break;
          if (retval < 0)
            return (-1);

          /* If f failed recoverably, shrink sig and retry */
          sig *= 0.25;
        }

        /* If retval still isn't 0, return with a recoverable failure */
        if (retval > 0)
          return (+1);

        /* Replace Jv by (Jv - fy)/sig */
        siginv = 1.0 / sig;
        N_VLinearSum(siginv, Jv, -siginv, fy, Jv);

        return (0);
      };

  SUNContext sunctx = nullptr;
  retval = SUNContext_Create((void *)&platform->comm.mpiComm, &sunctx);
  check_retval(&retval, "SUNContext_Create", 1);

  int blockSize = BLOCKSIZE;

  {
    this->y = nullptr;
    if (platform->device.mode() == "CUDA") {
#ifdef ENABLE_CUDA
      SUNCudaThreadDirectExecPolicy stream_exec_policy(blockSize);
      SUNCudaBlockReduceExecPolicy reduce_exec_policy(blockSize, 0);

      this->y = N_VNew_Cuda(this->nEq, sunctx);
      check_retval((void *)this->y, "N_VNew_Cuda", 0);

      retval = N_VSetKernelExecPolicy_Cuda(this->y, &stream_exec_policy, &reduce_exec_policy);
      check_retval(&retval, "N_VSetKernelExecPolicy_Cuda", 0);
#else
      nrsCheck(true,
               platform->comm.mpiComm,
               EXIT_FAILURE,
               "%s", "CVODE ENABLE_CUDA not enabled, despite mode being CUDA!\n");
#endif
    }
    else if (platform->device.mode() == "HIP") {
#ifdef ENABLE_HIP
      this->y = N_VNew_Hip(data->nEq, sunctx);
      check_retval((void *)this->y, "N_VNew_Hip", 0);
#else
      nrsCheck(true,
               platform->comm.mpiComm,
               EXIT_FAILURE,
               "%s", "CVODE ENABLE_HIP not enabled, despite mode being HIP!\n");
#endif
    }
    else if (platform->device.mode() == "Serial") {
      this->y = N_VNew_Serial(this->nEq, sunctx);
      check_retval((void *)this->y, "N_VNew_Serial", 0);
    }
    this->cvodeY = N_VMake_MPIPlusX(platform->comm.mpiComm, this->y, sunctx);
    this->nEqTotal = N_VGetLength(this->cvodeY);
  }

  o_cvodeY = platform->device.occaDevice().wrapMemory<sunrealtype>(
      __N_VGetDeviceArrayPointer(N_VGetLocalVector_MPIPlusX(cvodeY)),
      this->numEquations());

  // set initial condition
  nrsToCv(nrs, nrs->cds->o_S, o_cvodeY);

  auto integrator = CV_BDF;
  if (platform->options.compareArgs("CVODE INTEGRATOR", "ADAMS")) {
    integrator = CV_ADAMS;
  }

  this->cvodeMem = CVodeCreate(integrator, sunctx);

  auto T0 = 0.0;
  platform->options.getArgs("START TIME", T0);

  this->relTol = 1e-4;
  platform->options.getArgs("CVODE RELATIVE TOLERANCE", this->relTol);

  double absTol = 1e-6;
  platform->options.getArgs("CVODE ABSOLUTE TOLERANCE", absTol);

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

  this->o_absTol = platform->device.malloc(this->Nscalar * sizeof(dfloat), absTols.data());

  this->sigScale = 1.0;
  platform->options.getArgs("CVODE SIGMA SCALE", this->sigScale);

  check_retval((void *)this->cvodeMem, "CVodeCreate", 0);
  retval = CVodeInit(this->cvodeMem, cvodeRHS, T0, this->cvodeY);
  check_retval(&retval, "CVodeInit", 1);

  // provide function to compute weights to account for multiplicity
  // same as cvEwtSetSV, but with scaling for multiplicity
  CVEwtFn cvodeErrorWt = [](N_Vector y, N_Vector ewt, void *user_data) {
    auto data = static_cast<userData_t *>(user_data);
    auto nrs = data->nrs;
    auto platform = data->platform;
    auto cvode = data->cvode;

    occa::memory o_y = platform->device.occaDevice().wrapMemory<sunrealtype>(
        __N_VGetDeviceArrayPointer(N_VGetLocalVector_MPIPlusX(y)),
        cvode->numEquations());
    occa::memory o_ewt = platform->device.occaDevice().wrapMemory<sunrealtype>(
        __N_VGetDeviceArrayPointer(N_VGetLocalVector_MPIPlusX(ewt)),
        cvode->numEquations());

    cvode->computeErrorWeight(o_y, o_ewt);

    return 0;
  };

  retval = CVodeWFtolerances(this->cvodeMem, cvodeErrorWt);
  check_retval(&retval, "CVodeWFtolerances", 1);

  int nVectors = 10;
  platform->options.getArgs("CVODE PGMRES RESTART", nVectors);

  std::string linearSolver = "GMRES";
  platform->options.getArgs("CVODE SOLVER", linearSolver);

  SUNLinearSolver LS;

  if(linearSolver == "GMRES"){
    LS = SUNLinSol_SPGMR(cvodeY, PREC_NONE, nVectors, sunctx);
    check_retval(&retval, "SUNLinSol_SPFGMR", 1);
  } else {
    nrsCheck(true,
             platform->comm.mpiComm,
             EXIT_FAILURE,
             "CVODE SOLVER %s not supported!\n",
             linearSolver.c_str());
  }

  retval = CVodeSetLinearSolver(this->cvodeMem, LS, NULL);
  check_retval(&retval, "CVodeSetLinearSolver", 1);

  retval = CVodeSetJacTimesRhsFn(this->cvodeMem, cvodeJtvRHS);
  check_retval(&retval, "CVodeSetJacTimesRhsFn", 1);

  retval = CVodeSetJacTimes(this->cvodeMem, NULL, cvodeJtv);
  check_retval(&retval, "CVodeSetJacTimes", 1);

  // custom settings
  int mxsteps = 500;
  platform->options.getArgs("CVODE MAX STEPS", mxsteps);
  retval = CVodeSetMaxNumSteps(this->cvodeMem, mxsteps);

  int maxOrder = 3;
  platform->options.getArgs("CVODE MAX TIMESTEPPER ORDER", maxOrder);
  retval = CVodeSetMaxOrd(this->cvodeMem, maxOrder);

  double epsLin = 0.1;
  platform->options.getArgs("CVODE EPS LIN", epsLin);
  retval = CVodeSetEpsLin(this->cvodeMem, epsLin);

  userdata = std::make_shared<userData_t>(platform, nrs, this);

  // set user data as
  retval = CVodeSetUserData(this->cvodeMem, userdata.get());
  check_retval(&retval, "CVodeSetUserData", 1);

#else
  nrsCheck(true, platform->comm.mpiComm, EXIT_FAILURE, "%s\n", "No cvode installation found");

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
  if (o_EToL.size()) {
    o_EToL.free();
  }
  if (o_EToLUnique.size()) {
    o_EToLUnique.free();
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
  if (o_invDegree.size()) {
    o_invDegree.free();
  }

#ifdef ENABLE_CVODE
  // despite documentation, this function does not exist?
  // N_VDestroy_MPIPlusX(cvodeY);

  if (platform->device.mode() == "CUDA") {
#ifdef ENABLE_CUDA
    N_VDestroy_Cuda(y);
#endif
  }
  else if (platform->device.mode() == "HIP") {
#ifdef ENABLE_HIP
    N_VDestroy_HIP(y);
#endif
  }
  else if (platform->device.mode() == "Serial") {
    N_VDestroy_Serial(y);
  }

  CVodeFree(&cvodeMem);
#endif
}

void cvode_t::setupEToLMapping(nrs_t *nrs)
{
  auto *mesh = nrs->meshV;
  if (nrs->cht)
    mesh = nrs->cds->mesh[0];

#ifdef USE_E_VECTOR_LAYOUT
  std::vector<dlong> Eids(mesh->Nlocal);
  std::iota(Eids.begin(), Eids.end(), 0);
  this->o_EToL = platform->device.malloc(mesh->Nlocal * sizeof(dlong), Eids.data());
  this->o_EToLUnique = platform->device.malloc(mesh->Nlocal * sizeof(dlong), Eids.data());
  this->LFieldOffset = mesh->Nlocal;
#else

  auto o_Lids = platform->device.malloc(mesh->Nlocal * sizeof(dlong));
  std::vector<dlong> Eids(mesh->Nlocal);
  std::iota(Eids.begin(), Eids.end(), 0);
  o_Lids.copyFrom(Eids.data(), mesh->Nlocal * sizeof(dlong));

  {
    const auto saveNhaloGather = mesh->ogs->NhaloGather;
    mesh->ogs->NhaloGather = 0;
    ogsGatherScatter(o_Lids, "int", "ogsMin", mesh->ogs);
    mesh->ogs->NhaloGather = saveNhaloGather;
  }

  std::vector<dlong> Lids(mesh->Nlocal);
  o_Lids.copyTo(Lids.data(), mesh->Nlocal * sizeof(dlong));

  std::set<dlong> uniqueIds;
  for (auto &&id : Lids) {
    uniqueIds.insert(id);
  }

  const auto NL = uniqueIds.size();

  this->LFieldOffset = NL;

  std::vector<dlong> EToLUnique(mesh->Nlocal, -1);
  unsigned ctr = 0;
  for (auto &&uniqueEid : uniqueIds) {
    EToLUnique[uniqueEid] = ctr;
    ctr++;
  }
  this->o_EToL = platform->device.malloc(mesh->Nlocal * sizeof(dlong), EToLUnique.data());
  {
    const auto saveNhaloGather = mesh->ogs->NhaloGather;
    mesh->ogs->NhaloGather = 0;
    ogsGatherScatter(o_Lids, "int", "ogsMin", mesh->ogs);
    mesh->ogs->NhaloGather = saveNhaloGather;
  }

  this->o_EToLUnique = platform->device.malloc(mesh->Nlocal * sizeof(dlong), EToLUnique.data());

  // construct L-vector version of inv degree, based on duplicated points in L-vector
  {
    auto *mesh = nrs->meshV;
    if (nrs->cht)
      mesh = nrs->cds->mesh[0];

    std::vector<dfloat> degree(mesh->Nlocal, 0.0);

    for (int n = 0; n < mesh->Nlocal; ++n) {
      const auto lid = EToLUnique[n];
      if (lid > -1) {
        degree[lid] = 1.0;
      }
    }

    ogsGatherScatter(degree.data(), dfloatString, "ogsSum", mesh->ogs);

    std::vector<dfloat> invDegreeL(LFieldOffset, 1.0);
    
    for (int n = 0; n < mesh->Nlocal; ++n) {
      const auto lid = EToLUnique[n];
      if (lid > -1) {
        invDegreeL[lid] = 1.0 / degree[n];
      }
    }

    this->o_invDegree = platform->device.malloc(LFieldOffset * sizeof(dfloat), invDegreeL.data());
  }

  // a few sanity checks:
  // EToL has non-negative values
  std::vector<dlong> EToL(mesh->Nlocal);
  this->o_EToL.copyTo(EToL.data(), mesh->Nlocal * sizeof(dlong));
  bool allNonNegative = std::all_of(EToL.begin(), EToL.end(), [](auto &&val) { return val >= 0; });
  int err = allNonNegative ? 0 : 1;
  MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_INT, MPI_MAX, platform->comm.mpiComm);

  nrsCheck(err, platform->comm.mpiComm, EXIT_FAILURE, "%s\n", "Encountered negative value in EToL mapping");

  o_Lids.free();
#endif
}

void cvode_t::setupDirichletMask(nrs_t *nrs)
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
        if(e >= NelemV && is != 0) EToB[f + e * mesh->Nfaces + fOffset] = NO_OP;
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

  maskOffset = Nmasked;
  const int pageW = ALIGN_SIZE / sizeof(dfloat);
  if (maskOffset % pageW)
    maskOffset = (maskOffset / pageW + 1) * pageW;

  o_maskValues = platform->device.malloc(nrs->nEXT * maskOffset * sizeof(dfloat));
}

void cvode_t::applyDirichlet(nrs_t *nrs, dfloat time)
{
  // extrapolate masked Dirichlet values to current time state
  // NOTE: this can only be applied after the extrapolation order is reached
  // to avoid introducing CVODE convergence issues in the first few time steps
  if(this->isRhsEvaluation() && (this->externalTStep > nrs->nEXT)){
    
    if(this->Nmasked == 0) return;

    auto cds = nrs->cds;
    const int extOrder = std::min(this->externalTStep, nrs->nEXT);
    this->extrapolateDirichletKernel(this->Nmasked,
                                     this->maskOffset,
                                     extOrder,
                                     this->o_maskIds,
                                     o_coeffExt,
                                     this->o_maskValues,
                                     cds->o_S + this->minCvodeScalarId * nrs->fieldOffset * sizeof(dfloat));
    return;
  }

  // lower than any other possible Dirichlet value
  static constexpr dfloat TINY = -1e30;
  cds_t *cds = nrs->cds;
  
  auto o_S_start = platform->o_mempool.slice0;

  for (int is = 0; is < cds->NSfields; is++) {
    if (!cds->compute[is])
      continue;
    if (!cds->cvodeSolve[is])
      continue;

    mesh_t *mesh = cds->mesh[0];
    oogs_t *gsh = cds->gshT;
    if (is) {
      mesh = cds->meshV;
      gsh = cds->gsh;
    }

    const auto cvodeScalarId = cvodeScalarIds[is];

    auto o_diff_i = cds->o_diff + cds->fieldOffsetScan[is] * sizeof(dfloat);
    auto o_rho_i = cds->o_rho + cds->fieldOffsetScan[is] * sizeof(dfloat);
    auto o_Si = o_S_start + cds->fieldOffsetScan[cvodeScalarId] * sizeof(dfloat);

    platform->linAlg->fill(cds->fieldOffset[is], TINY, o_Si);
    
    for(int sweep = 0; sweep < 2; sweep++){
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
                             cds->o_EToB + is * cds->EToBOffset * sizeof(int),
                             cds->o_Ue,
                             o_diff_i,
                             o_rho_i,
                             cds->neknek ? cds->neknek->o_pointMap : o_NULL,
                             cds->neknek ? cds->neknek->o_U : o_NULL,
                             cds->neknek ? cds->neknek->o_S : o_NULL,
                             *(cds->o_usrwrk),
                             o_Si);
      if (sweep == 0)
        oogs::startFinish(o_Si, 1, cds->fieldOffset[is], ogsDfloat, ogsMax, gsh);
      if (sweep == 1)
        oogs::startFinish(o_Si, 1, cds->fieldOffset[is], ogsDfloat, ogsMin, gsh);
    }
  }

  if(this->Nmasked == 0) return;

  cds->maskCopyKernel(this->Nmasked,
                      0,
                      cds->fieldOffsetScan[minCvodeScalarId],
                      o_maskIds,
                      o_S_start,
                      cds->o_S);
  
  // o_maskValues must be at state t0 to be lagged by the subsequent CVODE solve call
  if(this->isRhsEvaluation())
    return;

  this->mapToMaskedPointKernel(this->Nmasked,
                               o_maskIds,
                               o_S_start,
                               o_maskValues);
}

void cvode_t::computeErrorWeight(occa::memory o_y, occa::memory o_ewt)
{
  this->errorWeightKernel(this->LFieldOffset,
                          this->Nscalar,
                          this->relTol,
                          this->o_invDegree,
                          this->o_absTol,
                          o_y,
                          o_ewt);
}

void cvode_t::rhs(nrs_t *nrs, dfloat time, occa::memory o_y, occa::memory o_ydot)
{
  const auto tag = this->rhsTagName();
  const auto saveTimerScope = timerScope;
  timerScope = tag;
  platform->timer.tic(tag, 1);
  this->setIsRhsEvaluation(true);

  if (userRHS) {
    userRHS(nrs, time, tnekRS, o_y, o_ydot);
  }
  else {
    defaultRHS(nrs, time, tnekRS, o_y, o_ydot);
  }

  this->setIsRhsEvaluation(false);
  platform->timer.toc(tag);
  timerScope = saveTimerScope;
}

void cvode_t::jtvRHS(nrs_t *nrs, dfloat time, occa::memory o_y, occa::memory o_ydot)
{
  this->setIsJacobianEvaluation(true);

  if (userJacobian) {
    userJacobian(nrs, time, tnekRS, o_y, o_ydot);
  }
  else {
    this->rhs(nrs, time, o_y, o_ydot);
  }

  this->setIsJacobianEvaluation(false);
}

void cvode_t::defaultRHS(nrs_t *nrs, dfloat time, dfloat t0, occa::memory o_y, occa::memory o_ydot)
{
  const bool movingMesh = platform->options.compareArgs("MOVING MESH", "TRUE");
  mesh_t *mesh = nrs->meshV;
  if (nrs->cht)
    mesh = nrs->cds->mesh[0];

  auto *cds = nrs->cds;

  if (time != tprev) {

    if (detailedTimersEnabled) {
      platform->timer.tic(timerScope + "::extrapolate", 1);
    }

    if (platform->comm.mpiRank == 0 && verboseCVODE) {
      std::cout << "t = " << time << ", stepsize = " << time - tprev << std::endl;
    }

    const auto cvodeDt = time - t0;
    tprev = time;

    dtCvode[0] = cvodeDt;
    dtCvode[1] = nrs->dt[1];
    dtCvode[2] = nrs->dt[2];

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

    if (nrs->pSolver->allNeumann && platform->options.compareArgs("LOWMACH", "TRUE")) {
      nrs->p0the = 0.0;
      for (int ext = 0; ext < extOrder; ++ext) {
        nrs->p0the += _coeffEXT[ext] * nrs->p0th[ext];
      }
    }

    o_coeffExt.copyFrom(_coeffEXT.data(), maxTimestepperOrder * sizeof(dfloat));

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
      for (int i = 0; i < meshOrder; ++i)
        mesh->coeffAB[i] *= dtCvode[0];
      for (int i = mesh->nAB; i > meshOrder; i--)
        mesh->coeffAB[i - 1] = 0.0;
      mesh->o_coeffAB.copyFrom(mesh->coeffAB, mesh->nAB * sizeof(dfloat));

      // restore mesh coordinates prior to integration
      {
        mesh->o_x.copyFrom(this->o_xyz0,
                           mesh->Nlocal * sizeof(dfloat),
                           0,
                           (0 * sizeof(dfloat)) * nrs->fieldOffset);
        mesh->o_y.copyFrom(this->o_xyz0,
                           mesh->Nlocal * sizeof(dfloat),
                           0,
                           (1 * sizeof(dfloat)) * nrs->fieldOffset);
        mesh->o_z.copyFrom(this->o_xyz0,
                           mesh->Nlocal * sizeof(dfloat),
                           0,
                           (2 * sizeof(dfloat)) * nrs->fieldOffset);
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

  cvToNrs(nrs, o_y, cds->o_S);

  if (detailedTimersEnabled) {
    platform->timer.tic(timerScope + "::applyDirichlet", 1);
  }

  this->applyDirichlet(nrs, time);

  if (detailedTimersEnabled) {
    platform->timer.toc(timerScope + "::applyDirichlet");
  }

  if (!(isJacobianEvaluation() && recycleProperties)) {
    platform->timer.tic(timerScope + "::evaluateProperties", 1);
    evaluateProperties(nrs, time);
    platform->timer.toc(timerScope + "::evaluateProperties");
  }

  // terms to include: user source, advection, filtering, add weak Laplacian
  platform->linAlg->fillKernel(cds->fieldOffsetSum, 0.0, cds->o_FS);
  makeq(nrs, time);

  bool chtCVODE = nrs->cht && cds->cvodeSolve[0];

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
      auto o_fld = cds->o_FS + nrs->cds->fieldOffsetScan[startScalar] * sizeof(dfloat);
      ogsFunc(o_fld, numScalars, cds->fieldOffset[startScalar], ogsDfloat, ogsAdd, cds->gsh);
    }
  };

  if (detailedTimersEnabled) {
    platform->timer.tic(timerScope + "::gatherScatterAndLocalPoint", 1);
  }

  applyOgsOperation(oogs::start);

  // input and output are assumed to be L-vectors
  if (userLocalPointSource) {
    platform->timer.tic(timerScope + "::gatherScatterAndLocalPoint::localPointSource", 1);
    // o_ydot is simply used as scratch space to store the L-vector source term
    userLocalPointSource(nrs, LFieldOffset, o_y, o_ydot);
    platform->timer.toc(timerScope + "::gatherScatterAndLocalPoint::localPointSource");

    cvToNrs(nrs, o_ydot, this->o_pointSource);
  }

  applyOgsOperation(oogs::finish);

  if (detailedTimersEnabled) {
    platform->timer.toc(timerScope + "::gatherScatterAndLocalPoint");
    platform->timer.tic(timerScope + "::fusedAddRhoDiv", 1);
  }

  if (chtCVODE) {

    auto mesh = cds->mesh[0];
    auto gsh = cds->mesh[0]->oogs;

    if (userLocalPointSource) {
      // o_FS += o_ptSource
      platform->linAlg->axpby(cds->mesh[0]->Nlocal, 1.0, this->o_pointSource, 1.0, cds->o_FS);
    }

    // o_FS /= invLMM(LMM*gs(o_rho)), i.e.
    // average interface values for o_rho on the boundary between the CHT/non-CHT regions
    platform->o_mempool.slice0.copyFrom(cds->o_rho, cds->fieldOffset[0] * sizeof(dfloat));
    platform->linAlg->axmy(mesh->Nlocal, 1.0, mesh->o_LMM, platform->o_mempool.slice0);
    oogs::startFinish(platform->o_mempool.slice0, 1, nrs->fieldOffset, ogsDfloat, ogsAdd, gsh);
    platform->linAlg->axmy(mesh->Nlocal, 1.0, mesh->o_invLMM, platform->o_mempool.slice0);

    platform->linAlg->aydx(mesh->Nlocal, 1.0, platform->o_mempool.slice0, cds->o_FS);
  }
  if (!chtCVODE || (chtCVODE && this->Nscalar > 1)) {
    dlong startScalar = minCvodeScalarId;
    dlong numScalars = this->Nscalar;
    if (chtCVODE) {
      startScalar++;
      numScalars--;
    }
    auto o_FS_start = cds->o_FS + cds->fieldOffsetScan[startScalar] * sizeof(dfloat);
    auto o_rho_start = cds->o_rho + cds->fieldOffsetScan[startScalar] * sizeof(dfloat);
    occa::memory o_ptSource_start;

    int addPointSourceContrib = 0;
    if (userLocalPointSource) {
      addPointSourceContrib = 1;
      o_ptSource_start = this->o_pointSource + cds->fieldOffsetScan[startScalar] * sizeof(dfloat);
    }
    
    this->fusedAddRhoDivKernel(cds->meshV->Nlocal,
                               numScalars,
                               nrs->fieldOffset,
                               (int) !sharedRho, // fieldRho
                               o_rho_start,
                               o_ptSource_start,
                               o_FS_start);

  }

  if (detailedTimersEnabled) {
    platform->timer.toc(timerScope + "::fusedAddRhoDiv");
    platform->timer.tic(timerScope + "::dp0thdt", 1);
  }

  // TODO: add dpdt term to userq?
  if (platform->options.compareArgs("LOWMACH", "TRUE") && nrs->pSolver->allNeumann) {

    // call is only used to evaluate dp0thdt, not divergence
    platform->linAlg->fill(mesh->Nlocal, 0.0, nrs->o_div);
    udf.div(nrs, time, nrs->o_div);

    // RHS += 1/vtrans * dp0thdt * alpha0
    // for ideal gas, alpha = (gamma-1)/gamma
    const auto alpha0 = nrs->alpha0Ref;
    platform->o_mempool.slice0.copyFrom(cds->o_rho, mesh->Nlocal * sizeof(dfloat));
    platform->linAlg->ady(mesh->Nlocal, nrs->dp0thdt * alpha0, platform->o_mempool.slice0);
    platform->linAlg->axpby(mesh->Nlocal, 1.0, platform->o_mempool.slice0, 1.0, cds->o_FS);
  }

  if (detailedTimersEnabled) {
    platform->timer.toc(timerScope + "::dp0thdt");
    platform->timer.tic(timerScope + "::maskDirichlet", 1);
  }

  auto o_FS_start = cds->o_FS + cds->fieldOffsetScan[minCvodeScalarId] * sizeof(dfloat);
  if(this->Nmasked > 0){
    nrs->maskKernel(this->Nmasked, this->o_maskIds, o_FS_start);
  }

  if (detailedTimersEnabled) {
    platform->timer.toc(timerScope + "::maskDirichlet");
  }

  nrsToCv(nrs, cds->o_FS, o_ydot);
}

void cvode_t::makeq(nrs_t *nrs, dfloat time)
{

  const auto timerScopeSave = timerScope;
  timerScope = timerScopeSave + "::makeq";
  platform->timer.tic(timerScope, 1);

  auto *cds = nrs->cds;
  auto o_FS = nrs->cds->o_FS;

  bool useRelativeVelocity = platform->options.compareArgs("MOVING MESH", "TRUE");
  auto &o_Urst = useRelativeVelocity ? cds->o_relUrst : cds->o_Urst;

  if (udf.sEqnSource) {
    const auto makeQScope = timerScope;
    // ensure that user/application forward the correct base timer
    timerScope = makeQScope + "::udfSEqnSource";
    platform->timer.tic(timerScope, 1);
    udf.sEqnSource(nrs, time, cds->o_S, o_FS);
    platform->timer.toc(timerScope);
    timerScope = makeQScope;
  }

  auto applyTerms = [&](mesh_t *mesh, dlong scalarStart, dlong Nscalar, bool chtPass) {

    if(cds->applyFilter){
      cds->filterRTKernel(cds->meshV->Nelements,
                          scalarStart,
                          Nscalar,
                          nrs->fieldOffset,
                          cds->o_applyFilterRT,
                          cds->o_filterMT,
                          cds->o_filterS,
                          cds->o_rho,
                          cds->o_S,
                          o_FS);
    }

    double flops = 6 * mesh->Np * mesh->Nq + 4 * mesh->Np;
    flops *= static_cast<double>(mesh->Nelements);
    flops *= Nscalar;
    platform->flopCounter->add("scalarFilterRT", flops);

    int applyLMM = 1;
    if(chtPass){
      applyLMM = 0;
    }

    if (detailedTimersEnabled) {
      platform->timer.tic(timerScope + "::advection", 1);
    }

    if (platform->options.compareArgs("ADVECTION", "TRUE")) {
      if (platform->options.compareArgs("ADVECTION TYPE", "CUBATURE")) {
        cds->strongAdvectionCubatureVolumeKernel(cds->meshV->Nelements,
                                                 Nscalar,
                                                 applyLMM,
                                                 mesh->o_LMM,
                                                 mesh->o_vgeo,
                                                 mesh->o_cubDiffInterpT,
                                                 mesh->o_cubInterpT,
                                                 mesh->o_cubProjectT,
                                                 cds->o_compute + scalarStart * sizeof(dlong),
                                                 cds->o_fieldOffsetScan + scalarStart * sizeof(dlong),
                                                 cds->vFieldOffset,
                                                 cds->vCubatureOffset,
                                                 cds->o_S,
                                                 o_Urst,
                                                 cds->o_rho,
                                                 o_FS);
      }
      else {
        cds->strongAdvectionVolumeKernel(cds->meshV->Nelements,
                                         Nscalar,
                                         applyLMM,
                                         mesh->o_LMM,
                                         mesh->o_vgeo,
                                         mesh->o_D,
                                         cds->o_compute + scalarStart * sizeof(dlong),
                                         cds->o_fieldOffsetScan + scalarStart * sizeof(dlong),
                                         cds->vFieldOffset,
                                         cds->o_S,
                                         o_Urst,
                                         cds->o_rho,
                                         o_FS);
      }

      // During a CHT pass, the LMM term is not applied in the advection kernels as
      // the advection term is only defined in the V-mesh portion.
      // Therefore, apply the LMM term here over the entire T-mesh.
      if(chtPass){
        auto o_FS_start = o_FS + cds->fieldOffsetScan[scalarStart] * sizeof(dfloat);
        platform->linAlg->axmyMany(mesh->Nlocal, Nscalar, nrs->fieldOffset, 0, 1.0, mesh->o_LMM, o_FS_start);
      }

      timeStepper::advectionFlops(cds->mesh[scalarStart], Nscalar);
    }

    if (detailedTimersEnabled) {
      platform->timer.toc(timerScope + "::advection");
      platform->timer.tic(timerScope + "::neumannBC", 1);
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
      platform->timer.tic(timerScope + "::weakLaplacian",1);
    }

    weakLaplacianKernel(mesh->Nelements,
                        Nscalar,
                        cds->o_cvodeSolve + scalarStart * sizeof(dlong),
                        cds->o_fieldOffsetScan + scalarStart * sizeof(dlong),
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

void cvode_t::nrsToCv(nrs_t *nrs, occa::memory o_EField, occa::memory o_LField)
{
  if (detailedTimersEnabled) {
    platform->timer.tic(timerScope + "::nrsToCv", 1);
  }
  nrsToCvKernel(nrs->cds->mesh[0]->Nlocal,
                nrs->meshV->Nlocal,
                nrs->Nscalar,
                nrs->fieldOffset,
                this->LFieldOffset,
                this->o_cvodeScalarIds,
                this->o_EToLUnique,
                o_EField,
                o_LField);
  if (userPostNrsToCv) {
    userPostNrsToCv(nrs, o_LField);
  }
  if (detailedTimersEnabled) {
    platform->timer.toc(timerScope + "::nrsToCv");
  }
}

void cvode_t::cvToNrs(nrs_t *nrs, occa::memory o_LField, occa::memory o_EField)
{
  if (detailedTimersEnabled) {
    platform->timer.tic(timerScope + "::cvToNrs", 1);
  }

  cvToNrsKernel(nrs->cds->mesh[0]->Nlocal,
                nrs->meshV->Nlocal,
                this->Nscalar,
                nrs->fieldOffset,
                this->LFieldOffset,
                this->o_scalarIds,
                this->o_EToL,
                o_LField,
                o_EField);
  if (userPostCvToNrs) {
    userPostCvToNrs(nrs, o_EField);
  }

  if (detailedTimersEnabled) {
    platform->timer.toc(timerScope + "::cvToNrs");
  }
}

void cvode_t::solve(nrs_t *nrs, double t0, double t1, int tstep)
{
#ifdef ENABLE_CVODE
  platform->timer.tic(timerName + "solve", 1);
  timerScope = timerName + "solve";

  // update solver statistics from previous state
  {
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

  mesh_t *mesh = nrs->meshV;
  if (nrs->cht)
    mesh = nrs->cds->mesh[0];

  bool movingMesh = platform->options.compareArgs("MOVING MESH", "TRUE");

  // lag solution state and update current state
  for (int s = nrs->nEXT; s > 1; s--) {
    const auto Nbyte = (nrs->NVfields * sizeof(dfloat)) * nrs->fieldOffset;
    o_U.copyFrom(o_U, Nbyte, (s - 1) * Nbyte, (s - 2) * Nbyte);
  }
  o_U.copyFrom(nrs->o_U, nrs->NVfields * nrs->fieldOffset * sizeof(dfloat));

  const auto p0theSave = nrs->p0the;

  if (movingMesh) {

    for (int s = nrs->nEXT; s > 1; s--) {
      const auto Nbyte = (nrs->NVfields * sizeof(dfloat)) * nrs->fieldOffset;
      o_meshU.copyFrom(o_meshU, Nbyte, (s - 1) * Nbyte, (s - 2) * Nbyte);
    }
    o_meshU.copyFrom(mesh->o_U, nrs->NVfields * nrs->fieldOffset * sizeof(dfloat));

    o_xyz0.copyFrom(mesh->o_x, mesh->Nlocal * sizeof(dfloat), (0 * sizeof(dfloat)) * nrs->fieldOffset, 0);
    o_xyz0.copyFrom(mesh->o_y, mesh->Nlocal * sizeof(dfloat), (1 * sizeof(dfloat)) * nrs->fieldOffset, 0);
    o_xyz0.copyFrom(mesh->o_z, mesh->Nlocal * sizeof(dfloat), (2 * sizeof(dfloat)) * nrs->fieldOffset, 0);
  }

  nrsToCv(nrs, nrs->cds->o_S, o_cvodeY);

  this->tnekRS = t0;
  this->externalTStep = tstep;

  double t;
  int retval = 0;

  // integrate only to time t1
  if (platform->options.compareArgs("CVODE STOP TIME", "TRUE")) {
    retval = CVodeSetStopTime(cvodeMem, t1);
    nrsCheck(retval < 0, MPI_COMM_SELF, EXIT_FAILURE, "%s", "Error calling CVodeSetStopTime\n");
  }

  if(!platform->options.getArgs("CVODE HMAX RATIO").empty()) {
    double hmax;
    platform->options.getArgs("CVODE HMAX RATIO", hmax);
    hmax *= nrs->dt[0];
    retval = CVodeSetMaxStep(cvodeMem, hmax);
    nrsCheck(retval < 0, MPI_COMM_SELF, EXIT_FAILURE, "%s", "Error calling CVodeSetMaxStep\n");
  }

  platform->device.finish();

  const auto oldScope = timerScope;
  timerScope = oldScope + "::cvode";
  if (detailedTimersEnabled) {
    platform->timer.tic(timerScope, 1);
  }
  // call cvode solver
  retval = CVode(cvodeMem, t1, cvodeY, &t, CV_NORMAL);
  if (retval < 0) {
    if (platform->comm.mpiRank == 0) {
      std::cout << "... Restarting CVODE integrator\n";
    }
    nrsToCv(nrs, nrs->cds->o_S, o_cvodeY);
    retval = CVodeReInit(cvodeMem, t0, cvodeY);
    check_retval(&retval, "CVodeReInit", 1);
    this->tprev = std::numeric_limits<dfloat>::max();

    platform->device.finish();
    retval = CVode(cvodeMem, t1, cvodeY, &t, CV_NORMAL);
  }

  nrsCheck(retval < 0,
           MPI_COMM_SELF,
           EXIT_FAILURE,
           "%s", "CVODE failed after restart. Ending simulation.\n");
  platform->device.finish();

  if (detailedTimersEnabled) {
    platform->timer.toc(timerScope);
  }
  timerScope = oldScope;

  cvToNrs(nrs, o_cvodeY, nrs->cds->o_S);

  if (detailedTimersEnabled) {
    platform->timer.tic(timerScope + "::restore", 1);
  }

  // restore previous state
  nrs->o_U.copyFrom(this->o_U, (nrs->NVfields * sizeof(dfloat)) * nrs->fieldOffset);

  if (movingMesh) {
    mesh->o_x.copyFrom(o_xyz0, mesh->Nlocal * sizeof(dfloat), 0, (0 * sizeof(dfloat)) * nrs->fieldOffset);
    mesh->o_y.copyFrom(o_xyz0, mesh->Nlocal * sizeof(dfloat), 0, (1 * sizeof(dfloat)) * nrs->fieldOffset);
    mesh->o_z.copyFrom(o_xyz0, mesh->Nlocal * sizeof(dfloat), 0, (2 * sizeof(dfloat)) * nrs->fieldOffset);

    mesh->o_U.copyFrom(this->o_meshU, (nrs->NVfields * sizeof(dfloat)) * nrs->fieldOffset);

    mesh->update();
  }

  computeUrst(nrs, false);

  nrs->p0the = p0theSave;

  for (int s = nrs->nEXT; s > 1; s--) {
    const auto NbyteMasked = maskOffset * sizeof(dfloat);
    if (NbyteMasked)
      o_maskValues.copyFrom(o_maskValues, NbyteMasked, (s - 1) * NbyteMasked, (s - 2) * NbyteMasked);
  }

  // compute scalar boundary condition at time t1
  this->applyDirichlet(nrs, t1);

  if (detailedTimersEnabled) {
    platform->timer.toc(timerScope + "::restore");
  }
  platform->timer.toc(timerScope);
#endif
}

#ifdef ENABLE_CVODE
long cvode_t::numSteps() const
{
  long int nsteps;
  auto retval = CVodeGetNumSteps(cvodeMem, &nsteps);
  check_retval(&retval, "CVodeGetNumSteps", 1);
  return nsteps - prevNsteps;
}

long cvode_t::numRHSEvals() const
{
  long int nrhs;
  auto retval = CVodeGetNumRhsEvals(cvodeMem, &nrhs);
  check_retval(&retval, "CVodeGetNumRhsEvals", 1);
  return nrhs - prevNrhs;
}

long cvode_t::numNonlinSolveIters() const
{
  long int nni;
  auto retval = CVodeGetNumNonlinSolvIters(cvodeMem, &nni);
  check_retval(&retval, "CVodeGetNumNonlinSolvIters", 1);
  return nni - prevNni;
}

long cvode_t::numLinIters() const
{
  long int nli;
  auto retval = CVodeGetNumLinIters(cvodeMem, &nli);
  check_retval(&retval, "CVodeGetNumLinIters", 1);
  return nli - prevNli;
}
#else
long cvode_t::numSteps() const { return 0; }

long cvode_t::numRHSEvals() const { return 0; }

long cvode_t::numNonlinSolveIters() const { return 0; }

long cvode_t::numLinIters() const { return 0; }
#endif

void cvode_t::printInfo(bool printVerboseInfo) const
{
  auto cvodeMem = this->cvodeMem;

  const auto nsteps = numSteps();
  const auto nrhs = numRHSEvals();
  const auto nni = numNonlinSolveIters();
  const auto nli = numLinIters();

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
    printf("%s: nsteps %03ld  nRHS %03ld  nni %01ld  nli %03ld\n",
           ss.str().c_str(),
           nsteps,
           nrhs,
           nni,
           nli);
  }
  else if (platform->comm.mpiRank == 0) {
    std::ostringstream ss;
    ss << "  " << scalars;
    printf("%s: %ld %ld %ld %ldf", ss.str().c_str(), nsteps, nrhs, nni, nli);
  }
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
    }
    else {
      auto parent = tag.substr(0, pos);
      tree[parent].push_back(tag);
    }
  }

  auto pos = timerName.rfind("::");
  const auto start = timerName.substr(0, pos);

  std::ios oldState(nullptr);
  oldState.copyfmt(std::cout);

  auto mesh = _nrs->meshV;
  long long int NglobalElements = mesh->Nelements;
  MPI_Allreduce(MPI_IN_PLACE, &NglobalElements, 1, MPI_LONG_LONG_INT, MPI_SUM, platform->comm.mpiComm);
  const double GDOF = (NglobalElements * mesh->N * mesh->N * mesh->N * this->cvodeScalarIds.size())
                      / (1e9 * platform->comm.mpiCommSize);

  // gather timer information from tree
  std::vector<std::string> operations;
  std::vector<std::string> timesPerCall;
  std::vector<std::string> relPercentage;
  std::vector<std::string> absPercentage;
  std::vector<std::string> throughputs;

  std::cout.setf(std::ios::fixed);
  std::function<void(std::string, std::string, std::string, int)> gatherTreeStats;
  gatherTreeStats = [&](std::string tag, std::string rootTag, std::string parentTag, int level) {
    if (level > 0) {
      if(level == 1) rootTag = tag; // set as root of timer tree
      const auto tTag = platform->timer.query(tag, "DEVICE:MAX");
      const auto nCalls = platform->timer.count(tag);
      auto tParent = platform->timer.query(parentTag, "DEVICE:MAX");

      if(tParent < 0.0)
        tParent = tTag;

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
        ss << std::setprecision(3)
           << std::scientific
           << tCall;
        timesPerCall.push_back(ss.str());
        
        ss.str("");
        ss.clear();
        ss << std::setprecision(1)
           << std::fixed
           << 100.0 * tTag/tParent
           << "%";
        relPercentage.push_back(level == 1 ? "" : ss.str());
        
        ss.str("");
        ss.clear();
        ss << std::setprecision(1)
           << std::fixed
           << 100.0 * tTag/tRoot
           << "%";
        absPercentage.push_back(ss.str());

        ss.str("");
        ss.clear();
        ss << std::setprecision(3)
           << std::scientific
           << GDOF/tCall;
        throughputs.push_back(ss.str());
      }
    }
    
    std::vector<std::string> children;
    for (auto &&child : tree[tag]) {
      children.push_back(child);
    }

    // sort children by max time, from largest to smallest
    std::sort(children.begin(),
              children.end(),
              [&](const std::string &a, const std::string &b) {
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
  table[1] = timesPerCall;
  table[2] = relPercentage;
  table[3] = absPercentage;
  table[4] = throughputs;

  std::vector<std::string> headers = {"Operation", "time/call", "rel %", "abs %", "GDOF/s/rank"};

  if(platform->comm.mpiRank == 0){
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
  return this->isJacobianEvaluation() ? timerScope + "::jtv" : timerScope + "::rhs";
}


void cvode_t::setLocalPointSource(userLocalPointSource_t _userLocalPointSource)
{
  userLocalPointSource = _userLocalPointSource;

  if(o_pointSource.size() == 0){
    o_pointSource = platform->device.malloc(this->Nscalar * _nrs->fieldOffset * sizeof(dfloat));
  }

  this->fusedAddRhoDivKernel = platform->kernels.get("cvode_t::fusedAddRhoDiv");
}
