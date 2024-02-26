#include "platform.hpp"
#include "nrs.hpp"
#include "nekInterfaceAdapter.hpp"
#include "printHeader.hpp"
#include "udf.hpp"
#include "bcMap.hpp"
#include "parsePar.hpp"
#include "re2Reader.hpp"
#include "configReader.hpp"
#include "re2Reader.hpp"
#include "platform.hpp"
#include "linAlg.hpp"
#include "AMGX.hpp"
#include "hypreWrapper.hpp"
#include "hypreWrapperDevice.hpp"

// define extern variable from nekrsSys.hpp
platform_t *platform;

namespace
{

static nrs_t *nrs = nullptr;

static int rank, size;
static MPI_Comm commg, comm;

static double currDt;
static double lastOutputTime = 0;
static int firstOutfld = 1;
static int enforceLastStep = 0;
static int enforceOutputStep = 0;
static bool initialized = false;

void printFileStdout(std::string file)
{
  std::ifstream f(file);
  std::string text;
  std::cout << std::endl;
  while (!f.eof()) {
    getline(f, text);
    std::cout << "<<< " << text << "\n";
  }
  std::cout << std::endl;
  f.close();
}

setupAide *setDefaultSettings(std::string casename)
{
  auto options = new setupAide();

  options->setArgs("FORMAT", std::string("1.0"));

  options->setArgs("EQUATION TYPE", "NAVIERSTOKES");

  options->setArgs("ELEMENT TYPE", std::string("12")); /* HEX */
  options->setArgs("ELEMENT MAP", std::string("ISOPARAMETRIC"));
  options->setArgs("MESH DIMENSION", std::string("3"));

  options->setArgs("CASENAME", casename);
  options->setArgs("UDF OKL FILE", casename + ".oudf");
  options->setArgs("UDF FILE", casename + ".udf");
  options->setArgs("NEK USR FILE", casename + ".usr");
  options->setArgs("MESH FILE", casename + ".re2");

  options->setArgs("DEVICE NUMBER", "LOCAL-RANK");
  options->setArgs("PLATFORM NUMBER", "0");

  options->setArgs("VERBOSE", "FALSE");

  options->setArgs("STDOUT PAR", "TRUE");
  options->setArgs("STDOUT UDF", "TRUE");

  options->setArgs("SOLUTION OUTPUT INTERVAL", "-1");
  options->setArgs("SOLUTION OUTPUT CONTROL", "STEPS");

  options->setArgs("START TIME", "0.0");

  options->setArgs("ENABLE GS COMM OVERLAP", "TRUE");

  options->setArgs("CHECKPOINT OUTPUT MESH", "FALSE");

  const auto dropTol = 5.0 * std::numeric_limits<pfloat>::epsilon();
  options->setArgs("AMG DROP TOLERANCE", to_string_f(setPrecision(dropTol,2)));

  if (options->compareArgs("EQUATION TYPE", "NAVIERSTOKES") || options->compareArgs("EQUATION TYPE", "STOKES")) {
    nrsSetDefaultSettings(options);
  }

  return options;
}

} // namespace

namespace nekrs
{

void reset()
{
  lastOutputTime = 0;
  firstOutfld = 1;
  enforceLastStep = 0;
  enforceOutputStep = 0;
}

double startTime(void)
{
  double val = 0;
  platform->options.getArgs("START TIME", val);
  return val;
}

void setup(MPI_Comm commg_in,
           MPI_Comm comm_in,
           int buildOnly,
           int commSizeTarget,
           int ciMode,
           const std::map<std::string, std::map<std::string, std::string>> &parKeyValuePairs,
           std::string casename,
           std::string _backend,
           std::string _deviceID,
           int nSessions,
           int sessionID,
           int debug)
{
  nekrsCheck(initialized, comm_in, EXIT_FAILURE, "%s\n", "Calling setup twice is erroneous!");

  commg = commg_in;
  comm = comm_in;

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  if (rank == 0) {
    printHeader();
    if (sizeof(dfloat) == sizeof(float)) {
      std::cout << "FP precision: 32-bit" << std::endl;
    }
    std::cout << "MPI tasks: " << size << std::endl << std::endl;
  }

  configRead(comm);

  auto options = setDefaultSettings(casename);
  static auto par = new inipp::Ini();
  par->sections = parKeyValuePairs;
  parsePar(par, comm, *options);

  if (nSessions > 1) {
    options->setArgs("NEKNEK NUMBER OF SESSIONS", std::to_string(nSessions));
    options->setArgs("NEKNEK SESSION ID", std::to_string(sessionID));
  }

  options->setArgs("BUILD ONLY", "FALSE");
  if (buildOnly) {
    options->setArgs("BUILD ONLY", "TRUE");
    options->setArgs("NP TARGET", std::to_string(commSizeTarget));
    if (rank == 0) {
      std::cout << "jit-compiling for >=" << commSizeTarget << " MPI tasks ...\n" << std::endl;
    }
    fflush(stdout);
  }

  // precedence: cmd arg, par, env-var
  if (options->getArgs("THREAD MODEL").length() == 0) {
    options->setArgs("THREAD MODEL", getenv("NEKRS_OCCA_MODE_DEFAULT"));
  }
  if (!_backend.empty()) {
    options->setArgs("THREAD MODEL", _backend);
  }
  if (!_deviceID.empty()) {
    options->setArgs("DEVICE NUMBER", _deviceID);
  }

  if (debug) {
    options->setArgs("VERBOSE", "TRUE");
  }

  // setup platform (requires THREAD MODEL)
  platform = platform_t::getInstance(*options, commg, comm);
  platform->par = par;

  if (options->compareArgs("EQUATION TYPE", "NAVIERSTOKES") || options->compareArgs("EQUATION TYPE", "STOKES")) {
    nrs = new nrs_t();
  }

  int buildRank = rank;
  if (platform->cacheLocal) {
    MPI_Comm_rank(platform->comm.mpiCommLocal, &buildRank);
  }

  if (rank == 0) {
    std::cout << "using NEKRS_HOME: " << getenv("NEKRS_HOME") << std::endl;
    std::cout << "using NEKRS_CACHE_DIR: " << getenv("NEKRS_CACHE_DIR") << std::endl;
    std::cout << "using OCCA_CACHE_DIR: " << occa::env::OCCA_CACHE_DIR << std::endl << std::endl;
  }

  platform->options.setArgs("CI-MODE", std::to_string(ciMode));
  if (rank == 0 && ciMode) {
    std::cout << "enabling continous integration mode " << ciMode << "\n";
  }

  {
    int nelgt, nelgv;
    re2::nelg(platform->options.getArgs("MESH FILE"), nelgt, nelgv, comm);
    nekrsCheck(size > nelgv, platform->comm.mpiComm, EXIT_FAILURE, "%s\n", "MPI tasks > number of elements!");
  }

  bcMap::setup();

  nek::bootstrap();

  // jit compile udf
  std::string udfFile;
  platform->options.getArgs("UDF FILE", udfFile);
  if (!udfFile.empty()) {
    udfBuild(udfFile, platform->options);
  }

  if (platform->cacheBcast || platform->cacheLocal)
    platform->bcastKernelSources();

  if (!udfFile.empty()) {
    udfLoad();
  }

  // here we might access nek variables
  if (udf.setup0) {
    udf.setup0(comm, platform->options);
  }

  if (rank == 0) {
    if (!buildOnly && platform->options.compareArgs("STDOUT PAR", "TRUE")) {
      printFileStdout(casename + ".par");
    }
    if (!buildOnly && platform->options.compareArgs("STDOUT UDF", "TRUE")) {
      udfEcho();
    }
  }

  platform->compileKernels();

  oogs::overlap(platform->options.compareArgs("ENABLE GS COMM OVERLAP", "FALSE") ? 0 : 1);

  if (buildOnly) {
    MPI_Barrier(platform->comm.mpiComm);
    if (buildRank == 0) {
      std::string cache_dir;
      cache_dir.assign(getenv("NEKRS_CACHE_DIR"));
      std::string file = cache_dir + "/build-only.timestamp";
      remove(file.c_str());
      std::ofstream ofs;
      ofs.open(file, std::ofstream::out);
      ofs.close();
      if (rank == 0) {
        std::cout << "\nBuild successful." << std::endl;
      }
    }
    return;
  }

  platform->linAlg = linAlg_t::getInstance();

  if (nrs) {
    nrs->init();
  }

  if (neknekCoupled()) {
    new neknek_t(nrs, nSessions, sessionID);
  }

  const double setupTime = platform->timer.query("setup", "DEVICE:MAX");
  if (rank == 0) {
    std::cout << "\nsettings:\n" << std::endl << platform->options << std::endl;

    std::cout << "memoryPool size: " << platform->o_memPool.size() / 1e9 << " GB" << std::endl;
    std::cout << "occa memory usage: " << platform->device.occaDevice().memoryAllocated() / 1e9 << " GB"
              << std::endl;
  }
  fflush(stdout);

  platform->flopCounter->clear();

#if 1
  if (platform->cacheBcast) {
    MPI_Barrier(platform->comm.mpiComm);

    int rankLocal;
    MPI_Comm_rank(platform->comm.mpiCommLocal, &rankLocal);
    if (rankLocal == 0) {
      for (auto &entry : fs::directory_iterator(platform->tmpDir)) {
        fs::remove_all(entry.path());
      }
    }
  }
#endif

  initialized = true;
}

void copyFromNek(double time, int tstep)
{
  nek::ocopyToNek(time, tstep);
}

void udfExecuteStep(double time, int tstep, int isOutputStep)
{
  nrs->isOutputStep = isOutputStep;
  if (nrs->isOutputStep) {
    nek::ifoutfld(1);
  }

  platform->timer.tic("udfExecuteStep", 1);
  if (udf.executeStep) {
    udf.executeStep(time, tstep);
  }
  platform->timer.toc("udfExecuteStep");

  // reset
  nek::ifoutfld(0);
  nrs->isOutputStep = 0;
}

void nekUserchk(void)
{
  nek::userchk();
}

double dt(int tstep)
{
  dfloat dt_ = -1;
  platform->options.getArgs("DT", dt_);

  if (platform->options.compareArgs("VARIABLE DT", "TRUE")) {
    if (platform->options.getArgs("DT").empty() && tstep == 1 || tstep > 1) {
      dt_ = nrs->adjustDt(tstep);

      // limit dt change
      dfloat maxAdjustDtRatio = 1;
      dfloat minAdjustDtRatio = 1;
      platform->options.getArgs("MAX ADJUST DT RATIO", maxAdjustDtRatio);
      platform->options.getArgs("MIN ADJUST DT RATIO", minAdjustDtRatio);
 
      if (tstep > 1) {
        dt_ = std::max(dt_, minAdjustDtRatio * nrs->dt[0]);
      }
      if (tstep > 1) {
        dt_ = std::min(dt_, maxAdjustDtRatio * nrs->dt[0]);
      }
 
      dfloat maxDt = 0;
      platform->options.getArgs("MAX DT", maxDt);
      if (maxDt > 0) {
        dt_ = std::min(nrs->dt[0], dt_);
      }
    }
  }

  nekrsCheck(dt_ < 1e-10 || std::isnan(dt_) || std::isinf(dt_),
             platform->comm.mpiComm,
             EXIT_FAILURE,
             "Invalid time step size %.2e\n",
             dt_);

  // during a neknek simulation, sync dt across all ranks
  if (nrs->neknek) {
    MPI_Allreduce(MPI_IN_PLACE, &dt_, 1, MPI_DFLOAT, MPI_MIN, platform->comm.mpiCommParent);
  }

  // limit dt to 5 significant digits
  dt_ = setPrecision(dt_, 5);
  return dt_;
}

double writeInterval(void)
{
  double val = -1;
  platform->options.getArgs("SOLUTION OUTPUT INTERVAL", val);
  return (val > 0) ? val : -1;
}

int writeControlRunTime(void)
{
  return platform->options.compareArgs("SOLUTION OUTPUT CONTROL", "SIMULATIONTIME");
}

int outputStep(double time, int tStep)
{
  int outputStep = 0;
  if (writeControlRunTime()) {
    double val;
    platform->options.getArgs("START TIME", val);
    if (lastOutputTime == 0 && val > 0) {
      lastOutputTime = val;
    }
    outputStep = ((time - lastOutputTime) + 1e-10) > nekrs::writeInterval();
  } else {
    if (writeInterval() > 0) {
      outputStep = (tStep % (int)writeInterval() == 0);
    }
  }

  if (enforceOutputStep) {
    enforceOutputStep = 0;
    return 1;
  }
  return outputStep;
}

void outputStep(int val)
{
  nrs->isOutputStep = val;
}

void outfld(double time, int step, std::string suffix)
{
  std::string oldValue;
  platform->options.getArgs("CHECKPOINT OUTPUT MESH", oldValue);

  if (firstOutfld) {
    platform->options.setArgs("CHECKPOINT OUTPUT MESH", "TRUE");
  }

  if (platform->options.compareArgs("MOVING MESH", "TRUE")) {
    platform->options.setArgs("CHECKPOINT OUTPUT MESH", "TRUE");
  }

  nrs->writeFld(time, step, suffix);
  lastOutputTime = time;
  firstOutfld = 0;

  platform->options.setArgs("CHECKPOINT OUTPUT MESH", oldValue);
}

void outfld(double time, int step)
{
  outfld(time, step, "");
}

double endTime(void)
{
  double endTime = -1;
  platform->options.getArgs("END TIME", endTime);
  return endTime;
}

int numSteps(void)
{
  int numSteps = -1;
  platform->options.getArgs("NUMBER TIMESTEPS", numSteps);
  return numSteps;
}

void lastStep(int val)
{
  nrs->lastStep = val;
}

int lastStep(double timeNew, int tstep, double elapsedTime)
{
  if (!platform->options.getArgs("STOP AT ELAPSED TIME").empty()) {
    double maxElaspedTime;
    platform->options.getArgs("STOP AT ELAPSED TIME", maxElaspedTime);
    if (elapsedTime > 60.0 * maxElaspedTime) {
      nrs->lastStep = 1;
    }
  } else if (endTime() >= 0) {
    const double eps = 1e-10;
    nrs->lastStep = fabs(timeNew - endTime()) < eps || timeNew > endTime();
  } else {
    nrs->lastStep = (tstep == numSteps());
  }

  if (enforceLastStep) {
    return 1;
  }
  return nrs->lastStep;
}

void *platformPtr(void)
{
  return platform;
}

int runTimeStatFreq()
{
  int freq = 500;
  platform->options.getArgs("RUNTIME STATISTICS FREQUENCY", freq);
  return freq;
}

int printInfoFreq()
{
  int freq = 1;
  platform->options.getArgs("PRINT INFO FREQUENCY", freq);
  return freq;
}

int updateFileCheckFreq()
{
  int freq = 20;
  platform->options.getArgs("UPDATE FILE CHECK FREQUENCY", freq);
  return freq;
}

void printRuntimeStatistics(int step)
{
  platform->solver->printRunStat(step);
}

void processUpdFile()
{
  char *rbuf = nullptr;
  long long int fsize = 0;
  const std::string updFile = "nekrs.upd";

  if (rank == 0) {
    if (fs::exists(updFile)) {
      FILE *f = fopen(updFile.c_str(), "r");
      fseek(f, 0, SEEK_END);
      fsize = ftell(f);
      fseek(f, 0, SEEK_SET);
      rbuf = new char[fsize];
      const auto readCnt = fread(rbuf, 1, fsize, f);
      fclose(f);
    }
  }

  MPI_Bcast(&fsize, 1, MPI_LONG_LONG_INT, 0, comm);

  if (fsize) {
    std::string txt = "processing " + updFile + " ...\n";

    if (rank != 0) {
      rbuf = new char[fsize];
    }

    MPI_Bcast(rbuf, fsize, MPI_CHAR, 0, comm);
    std::stringstream is;
    is.write(rbuf, fsize);
    inipp::Ini ini;
    ini.parse(is, false);

    std::string checkpoint;
    ini.extract("", "checkpoint", checkpoint);
    if (checkpoint == "true") {
      enforceOutputStep = 1;
    }

    std::string endTime;
    ini.extract("general", "endtime", endTime);
    if (!endTime.empty()) {
      if (rank == 0) {
        txt += "  set endTime = " + endTime + "\n";
      }
      platform->options.setArgs("END TIME", endTime);
    }

    std::string numSteps;
    ini.extract("general", "numsteps", numSteps);
    if (!numSteps.empty()) {
      if (rank == 0) {
        txt += "  set numSteps = " + numSteps + "\n";
      }
      platform->options.setArgs("NUMBER TIMESTEPS", numSteps);
    }

    std::string writeInterval;
    ini.extract("general", "writeinterval", writeInterval);
    if (!writeInterval.empty()) {
      if (rank == 0) {
        txt += "  set writeInterval = " + writeInterval + "\n";
      }
      platform->options.setArgs("SOLUTION OUTPUT INTERVAL", writeInterval);
    }

    if (rank == 0) {
      std::cout << txt;
    }

    delete[] rbuf;
  }
}

void printInfo(double time, int tstep, bool printStepInfo, bool printVerboseInfo)
{
  nrs->printInfo(time, tstep, printStepInfo, printVerboseInfo);
}

void verboseInfo(bool enabled)
{
  platform->options.setArgs("VERBOSE SOLVER INFO", "FALSE");
  if (enabled) {
    platform->options.setArgs("VERBOSE SOLVER INFO", "TRUE");
  }
}

void updateTimer(const std::string &key, double time)
{
  platform->timer.set(key, time);
}

void resetTimer(const std::string &key)
{
  platform->timer.reset(key);
}

int exitValue()
{
  return platform->exitValue;
}

void initStep(double time, double dt, int tstep)
{
  nrs->initStep(time, dt, tstep);
  currDt = dt;
}

bool runStep(std::function<bool(int)> convergenceCheck, int corrector)
{
  return nrs->runStep(convergenceCheck, corrector);
}

bool runStep(int corrector)
{

  auto _nrs = &nrs;
  auto _udf = &udf;

  std::function<bool(int)> check = [](int corrector) -> bool {
    if (nrs->userConvergenceCheck) {
      return nrs->userConvergenceCheck(corrector);
    } else {
      return true;
    }
  };

  return nrs->runStep(check, corrector);
}

double finishStep()
{
  nrs->finishStep();
  return nrs->timePrevious + setPrecision(currDt, 5);
}

bool stepConverged()
{
  return nrs->timeStepConverged;
}

int finalize()
{
  int exitValue = nekrs::exitValue();
  if (platform->options.compareArgs("BUILD ONLY", "FALSE")) {
    nrs->finalize();

    hypreWrapper::finalize();
    hypreWrapperDevice::finalize();
    AMGXfinalize();
    nek::finalize();
  }

  MPI_Allreduce(MPI_IN_PLACE, &exitValue, 1, MPI_INT, MPI_MAX, platform->comm.mpiCommParent);
  if (platform->comm.mpiRank == 0) {
    std::cout << "finished with exit code " << exitValue << std::endl;
  }

  return exitValue;
}

} // namespace nekrs
