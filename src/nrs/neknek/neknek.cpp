#include <cfloat>
#include "platform.hpp"
#include "bcMap.hpp"
#include "findpts.hpp"
#include "neknek.hpp"
#include "nrs.hpp"
#include "nekInterfaceAdapter.hpp"
#include "pointInterpolation.hpp"

#include "sha1.hpp"

namespace
{

bool isIntBc(int bcType, std::string field)
{
  bool isInt = bcType == bcMap::bcTypeINT;

  if (field.find("scalar") != std::string::npos) {
    isInt = bcType == bcMap::bcTypeINTS;
  }

  return isInt;
}

} // namespace

bool neknekCoupled()
{
  auto fields = nrsFieldsToSolve(platform->options);

  int intFound = 0;
  for (auto &&field : fields) {
    for (int bID = 1; bID <= bcMap::size(field); ++bID) {
      auto bcType = bcMap::id(bID, field);
      bool isInt = isIntBc(bcType, field);

      if (isInt) {
        intFound = 1;
      }

    }
  }

  // findpts functions have to be called collectively across all sessions
  MPI_Allreduce(MPI_IN_PLACE, &intFound, 1, MPI_INT, MPI_MAX, platform->comm.mpiCommParent);
  return intFound > 0;
}

void neknek_t::reserveAllocation()
{
  this->fieldOffset_ = alignStride<dfloat>(this->npt_);
  this->o_pointMap_ = platform->device.malloc<dlong>(nrs->_mesh->Nlocal);

  if (this->npt_) {
    if (std::find(this->fields.begin(), this->fields.end(), "velocity") != this->fields.end()) {
      this->o_U_ = platform->device.malloc<dfloat>(nrs->NVfields * this->fieldOffset_ * (this->nEXT_ + 1));
    }
    if (this->Nscalar_) {
      this->o_S_ = platform->device.malloc<dfloat>(this->Nscalar_ * this->fieldOffset_ * (this->nEXT_ + 1));
    }
  }
}

void neknek_t::updateInterpPoints()
{
  // called in case of moving mesh ONLY
  if (!this->globalMovingMesh) {
    return;
  }

  auto neknek = nrs->neknek;
  const dlong nsessions = this->nsessions_;
  const dlong sessionID = this->sessionID_;

  auto mesh = nrs->_mesh;

  // Setup findpts
  const dfloat tol = (sizeof(dfloat) == sizeof(double)) ? 5e-13 : 1e-6;
  constexpr dlong npt_max = 1;
  const dfloat bb_tol = 0.01;

  auto &device = platform->device.occaDevice();

  this->interpolator.reset();
  this->interpolator = std::make_shared<pointInterpolation_t>(mesh, bb_tol, tol, true, sessionID_, true);
  this->interpolator->setTimerLevel(TimerLevel::Basic);
  this->interpolator->setTimerName("neknek_t::");

  // neknekX[:] = mesh->x[pointMap[:]]
  this->copyNekNekPointsKernel(mesh->Nlocal,
                               this->o_pointMap_,
                               mesh->o_x,
                               mesh->o_y,
                               mesh->o_z,
                               this->o_x_,
                               this->o_y_,
                               this->o_z_);

  this->interpolator->setPoints(this->npt_, this->o_x_, this->o_y_, this->o_z_, this->o_session_);

  const auto verboseLevel = pointInterpolation_t::VerbosityLevel::Detailed;
  this->interpolator->find(verboseLevel);
}

void neknek_t::findIntPoints()
{
  const dlong nsessions = this->nsessions_;
  const dlong sessionID = this->sessionID_;

  auto mesh = nrs->_mesh;

  // Setup findpts
  const dfloat tol = (sizeof(dfloat) == sizeof(double)) ? 5e-13 : 1e-6;
  constexpr dlong npt_max = 1;
  const dfloat bb_tol = 0.01;

  auto &device = platform->device.occaDevice();

  this->interpolator.reset();
  this->interpolator = std::make_shared<pointInterpolation_t>(mesh, bb_tol, tol, true, sessionID_, true);
  this->interpolator->setTimerLevel(TimerLevel::Basic);
  this->interpolator->setTimerName("neknek_t::");

  // remove fields with no INT boundaries
  for (auto &&field : this->fields) {
    int intFound = 0;
    for (dlong e = 0; e < mesh->Nelements; ++e) {
      for (dlong f = 0; f < mesh->Nfaces; ++f) {
        auto bID = mesh->EToB[f + mesh->Nfaces * e];
        auto bcType = bcMap::id(bID, field);
        if(isIntBc(bcType, field)) intFound = 1;
      }
    }
    MPI_Allreduce(MPI_IN_PLACE, &intFound, 1, MPI_INT, MPI_MAX, platform->comm.mpiComm);
    if (!intFound) {
     this->fields.erase(std::remove(this->fields.begin(), this->fields.end(), field), this->fields.end());
    }
  }

  // check if all remaining fields have all the same INT boundaries
  std::ostringstream errorLogger;
  std::set<int> intBIDs;
  for (auto &&field : this->fields) {
    for (int bID = 1; bID <= bcMap::size(field); ++bID) {
      auto bcType = bcMap::id(bID, field);
      bool isInt = isIntBc(bcType, field);

      if (isInt) {
        intBIDs.insert(bID);
      }

      if ((intBIDs.find(bID) != intBIDs.end()) && !isInt) {
        errorLogger << "ERROR: expected INT boundary condition on boundary id " << bID << " for field "
                    << field << "\n";
      }
    }
  }
  int errorLength = errorLogger.str().length();
  MPI_Allreduce(MPI_IN_PLACE, &errorLength, 1, MPI_INT, MPI_MAX, platform->comm.mpiCommParent);
  nekrsCheck(errorLength > 0, platform->comm.mpiCommParent, EXIT_FAILURE, "%s\n", errorLogger.str().c_str());

  // int points are the same for all neknek fields
  dlong numInterpFaces = 0;
  for (dlong e = 0; e < mesh->Nelements; ++e) {
    for (dlong f = 0; f < mesh->Nfaces; ++f) {
      auto bID = mesh->EToB[f + mesh->Nfaces * e];
      auto bcType = bcMap::id(bID, this->fields[0]);
      numInterpFaces += (isIntBc(bcType, this->fields[0]));
    }
  }
  const auto numPoints = numInterpFaces * mesh->Nfp;

  this->npt_ = numPoints;
  this->reserveAllocation();

  std::vector<dfloat> neknekX(numPoints, 0.0);
  std::vector<dfloat> neknekY(numPoints, 0.0);
  std::vector<dfloat> neknekZ(numPoints, 0.0);
  std::vector<dlong> session(numPoints, 0.0);

  std::vector<dlong> pointMap(mesh->Nlocal, -1);

  if (this->fields.size()) {
    dlong ip = 0;
    for (dlong e = 0; e < mesh->Nelements; ++e) {
      for (dlong f = 0; f < mesh->Nfaces; ++f) {

        for (dlong m = 0; m < mesh->Nfp; ++m) {
          dlong id = mesh->Nfaces * mesh->Nfp * e + mesh->Nfp * f + m;
          dlong idM = mesh->vmapM[id];

          auto bID = mesh->EToB[f + mesh->Nfaces * e];
          auto bcType = bcMap::id(bID, this->fields[0]);

          if (isIntBc(bcType, this->fields[0])) {
            neknekX[ip] = mesh->x[idM];
            neknekY[ip] = mesh->y[idM];
            neknekZ[ip] = mesh->z[idM];
            session[ip] = sessionID;

            pointMap[idM] = ip;
            ++ip;
          }
        }
      }
    }
  }

  this->o_pointMap_.copyFrom(pointMap.data());

  this->interpolator->setPoints(numPoints, neknekX.data(), neknekY.data(), neknekZ.data(), session.data());

  const auto verboseLevel = pointInterpolation_t::VerbosityLevel::Detailed;
  this->interpolator->find(verboseLevel);

  this->o_x_ = platform->device.malloc<dfloat>(this->npt_, neknekX.data());
  this->o_y_ = platform->device.malloc<dfloat>(this->npt_, neknekY.data());
  this->o_z_ = platform->device.malloc<dfloat>(this->npt_, neknekZ.data());
  this->o_session_ = platform->device.malloc<dlong>(this->npt_, session.data());
}

void neknek_t::setup()
{
  dlong globalRank;
  MPI_Comm_rank(platform->comm.mpiCommParent, &globalRank);

  const int nsessions = this->nsessions_;
  if (platform->comm.mpiRank == 0) {
    printf("initializing neknek with %d sessions\n", nsessions);
    std::fflush(stdout);
  }

  nekrsCheck(static_cast<bool>(platform->options.compareArgs("CONSTANT FLOW RATE", "TRUE")),
             platform->comm.mpiComm,
             EXIT_FAILURE,
             "%s\n",
             "constant flow rate support not supported");

  if (nrs->pSolver) {
    nekrsCheck(
        static_cast<bool>(nrs->pSolver->nullSpace() && platform->options.compareArgs("LOWMACH", "TRUE")),
        platform->comm.mpiComm,
        EXIT_FAILURE,
        "%s\n",
        "variable p0th is not supported!");
  }

  std::vector<int> Nscalars(nsessions, 0);
  Nscalars[this->sessionID_] = nrs->Nscalar;

  MPI_Allreduce(MPI_IN_PLACE, Nscalars.data(), nsessions, MPI_INT, MPI_MAX, platform->comm.mpiCommParent);

  auto minNscalar = *std::min_element(Nscalars.begin(), Nscalars.end());

  bool allSame =
      std::all_of(Nscalars.begin(), Nscalars.end(), [minNscalar](auto v) { return v == minNscalar; });

  if (platform->comm.mpiRank == 0 && !allSame) {
    std::cout << "WARNING: Nscalar is not the same across all sessions -> using the minimum value: "
              << minNscalar << "\n";
  }

  this->Nscalar_ = minNscalar;

  int movingMesh = platform->options.compareArgs("MOVING MESH", "TRUE");
  MPI_Allreduce(MPI_IN_PLACE, &movingMesh, 1, MPI_INT, MPI_MAX, platform->comm.mpiCommParent);
  this->globalMovingMesh = movingMesh;

  this->fields = nrsFieldsToSolve(platform->options);
  for (auto &&field : this->fields) {
    if (field.find("scalar") != std::string::npos) {
      const auto id = std::stoi(field.substr(std::string("scalar").length()));
      if(id+1 > this->Nscalar_) {
        this->fields.erase(std::remove(this->fields.begin(), this->fields.end(), field), this->fields.end());
      }
    }
  }

  this->findIntPoints();

  if (platform->comm.mpiRank == 0) {
    std::cout << "exchanged fields: ";
    for (auto &&field : this->fields) {
      std::cout << field << "  ";
    }
    std::cout << "\n";
  }
  std::fflush(stdout);

  // check if fields across all sessions match
  {
    std::string s;
    for (auto &&field : this->fields) s += field;

    SHA1 sha;
    sha.update(s);
    const auto hash = sha.final();
    const auto hashTruncated = hash.substr(hash.length() - 8);
    unsigned long intHash = std::stoul("0x" + hashTruncated, nullptr, 0);

    unsigned long intHashMin;
    unsigned long intHashMax;

    MPI_Allreduce(&intHash, &intHashMin,1, MPI_UNSIGNED_LONG, MPI_MIN, platform->comm.mpiCommParent);
    MPI_Allreduce(&intHash, &intHashMax,1, MPI_UNSIGNED_LONG, MPI_MAX, platform->comm.mpiCommParent);

    nekrsCheck(intHashMin != intHashMax, platform->comm.mpiCommParent, EXIT_FAILURE, "%s\n", "neknek fields do not match across all sessions");
  }
  
  if (platform->comm.mpiRank == 0) {
    std::cout << "done\n";
  }
}

neknek_t::neknek_t(nrs_t *nrs_, dlong nsessions, dlong sessionID)
    : nsessions_(nsessions), sessionID_(sessionID), nrs(nrs_)
{
  nrs->neknek = this;

  this->nEXT_ = 1;
  if (!platform->options.getArgs("NEKNEK BOUNDARY EXT ORDER").empty()) {
    platform->options.getArgs("NEKNEK BOUNDARY EXT ORDER", this->nEXT_);
  }

  // set boundary ext order to report to user, if not specified
  platform->options.setArgs("NEKNEK BOUNDARY EXT ORDER", std::to_string(this->nEXT_));

  this->coeffEXT.resize(this->nEXT_);
  this->o_coeffEXT = platform->device.malloc<dfloat>(this->nEXT_);

  this->setup();

  this->copyNekNekPointsKernel = platform->kernels.get("copyNekNekPoints");
  this->computeFluxKernel = platform->kernels.get("computeFlux");
  this->fixSurfaceFluxKernel = platform->kernels.get("fixSurfaceFlux");
}

void neknek_t::updateBoundary(int tstep, int stage)
{
  // do not invoke barrier -- this is performed later
  platform->timer.tic("neknek update boundary", 0);

  // do not invoke barrier in timer_t::tic
  platform->timer.tic("neknek sync", 0);
  MPI_Barrier(platform->comm.mpiCommParent);
  platform->timer.toc("neknek sync");
  this->tSync_ = platform->timer.query("neknek sync", "HOST:MAX");

  if (this->globalMovingMesh) {
    platform->timer.tic("neknek updateInterpPoints", 1);
    this->updateInterpPoints();
    platform->timer.toc("neknek updateInterpPoints");

    this->recomputePartition = true;
  }

  platform->timer.tic("neknek exchange", 1);

  if (std::find(this->fields.begin(), this->fields.end(), "velocity") != this->fields.end()) {
    this->interpolator->eval(nrs->NVfields, nrs->fieldOffset, nrs->o_U, this->fieldOffset_, this->o_U_);
  }

  if (this->Nscalar_) {
    this->interpolator->eval(this->Nscalar_, nrs->fieldOffset, nrs->cds->o_S, this->fieldOffset_, this->o_S_);
  }

  // lag state, update timestepper coefficients and compute extrapolated state
  if (stage == 1) {
    auto *mesh = nrs->_mesh;
    int extOrder = std::min(tstep, this->nEXT_);
    int bdfOrder = std::min(tstep, nrs->nBDF);
    nek::extCoeff(this->coeffEXT.data(), nrs->dt, extOrder, bdfOrder);

    for (int i = this->nEXT_; i > extOrder; i--) {
      this->coeffEXT[i - 1] = 0.0;
    }

    this->o_coeffEXT.copyFrom(this->coeffEXT.data(), this->nEXT_);

    for (int s = this->nEXT_ + 1; s > 1; s--) {
      auto N = nrs->NVfields * this->fieldOffset_;
      if (std::find(this->fields.begin(), this->fields.end(), "velocity") != this->fields.end()) {
        this->o_U_.copyFrom(this->o_U_, N, (s - 1) * N, (s - 2) * N);
      }

      N = this->Nscalar_ * this->fieldOffset_;
      this->o_S_.copyFrom(this->o_S_, N, (s - 1) * N, (s - 2) * N);
    }


    if (this->npt_) {
      if (std::find(this->fields.begin(), this->fields.end(), "velocity") != this->fields.end()) {
        auto o_Uold = this->o_U_ + this->fieldOffset_ * nrs->NVfields;
        nrs->extrapolateKernel(this->npt_,
                               nrs->NVfields,
                               this->nEXT_,
                               this->fieldOffset_,
                               this->o_coeffEXT,
                               o_Uold,
                               this->o_U_);
      }

      if (this->Nscalar_) {
        auto o_Sold = this->o_S_ + this->fieldOffset_ * this->Nscalar_;
        nrs->extrapolateKernel(this->npt_,
                               this->Nscalar_,
                               this->nEXT_,
                               this->fieldOffset_,
                               this->o_coeffEXT,
                               o_Sold,
                               this->o_S_);
      }
    }
  }

  platform->timer.toc("neknek exchange");

  this->tExch_ = platform->timer.query("neknek exchange", "DEVICE:MAX");
  this->ratio_ = this->tSync_ / this->tExch_;

  platform->timer.toc("neknek update boundary");
}

occa::memory neknek_t::partitionOfUnity()
{
  if (!this->o_partition_.isInitialized()) {
    this->o_partition_ = platform->device.malloc<dfloat>(nrs->fieldOffset);
  }

  if (!recomputePartition) {
    return this->o_partition_;
  }
  recomputePartition = false;

  const dfloat tol = (sizeof(dfloat) == sizeof(double)) ? 5e-13 : 1e-6;
  constexpr dlong npt_max = 1;
  const dfloat bb_tol = 0.01;
  auto mesh = nrs->_mesh;

  auto pointInterp = pointInterpolation_t(mesh, bb_tol, tol, true, sessionID_, true);
  auto o_dist = pointInterp.distance();

  auto o_sess = platform->o_memPool.reserve<dlong>(nrs->fieldOffset);
  auto o_sumDist = platform->o_memPool.reserve<dfloat>(nrs->fieldOffset);
  auto o_found = platform->o_memPool.reserve<dfloat>(nrs->fieldOffset);
  auto o_interpDist = platform->o_memPool.reserve<dfloat>(nrs->fieldOffset);
  o_sumDist.copyFrom(o_dist, mesh->Nlocal);

  std::vector<dfloat> found(mesh->Nlocal);
  std::vector<dlong> sessions(mesh->Nlocal);

  for (int sess = 0; sess < this->nsessions_; ++sess) {
    auto id = (sess + this->sessionID_) % this->nsessions_;
    if (id == this->sessionID_) {
      continue;
    }
    std::fill(sessions.begin(), sessions.end(), id);
    o_sess.copyFrom(sessions.data(), mesh->Nlocal);

    pointInterp.setPoints(mesh->Nlocal, mesh->o_x, mesh->o_y, mesh->o_z, o_sess);
    pointInterp.find(pointInterpolation_t::VerbosityLevel::None, true);

    auto &data = pointInterp.data();
    for (int n = 0; n < mesh->Nlocal; ++n) {
      found[n] = (data.code[n] == findpts::CODE_NOT_FOUND) ? 0.0 : 1.0;
    }

    o_found.copyFrom(found.data());
    pointInterp.eval(1, nrs->fieldOffset, o_dist, nrs->fieldOffset, o_interpDist);

    platform->linAlg->axmy(mesh->Nlocal, 1.0, o_found, o_interpDist);
    platform->linAlg->axpby(mesh->Nlocal, 1.0, o_interpDist, 1.0, o_sumDist);
  }

  // \Xi(x) = \dfrac{\delta^s(x)}{\sum_{i=1}^S \delta^s(x_i)}
  this->o_partition_.copyFrom(o_dist, mesh->Nlocal);
  platform->linAlg->aydx(mesh->Nlocal, 1.0, o_sumDist, this->o_partition_);

  o_sess.free();
  o_sumDist.free();
  o_found.free();
  o_interpDist.free();

  return this->o_partition_;
}
