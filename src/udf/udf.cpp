#include <iostream>
#include <unistd.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <regex>
#include "unifdef.h"

#include "udf.hpp"
#include "fileUtils.hpp"
#include "fileBcast.hpp"
#include "platform.hpp"
#include "bcMap.hpp"
#include "bcType.h"
#include "sha1.hpp"

#include "udfMake.hpp"

UDF udf = {NULL, NULL, NULL, NULL};

static int velocityDirichletConditions = 0;
static int meshVelocityDirichletConditions = 0;
static int velocityNeumannConditions = 0;
static int pressureDirichletConditions = 0;
static int scalarDirichletConditions = 0;
static int scalarNeumannConditions = 0;

static void *libudfHandle = nullptr;
static std::string udfFile;

static void verifyOudf()
{
  for (auto &[key, value] : bcMap::map()) {
    auto field = key.first;
    const int bcID = value;

    if (field.compare("velocity") == 0 && (bcID == bcMap::bcTypeV || bcID == bcMap::bcTypeINT)) {
      oudfFindDirichlet(field);
    }
    if (field.compare("mesh") == 0 && bcID == bcMap::bcTypeV) {
      oudfFindDirichlet(field);
    }
    if (field.compare("pressure") == 0 &&
        (bcID == bcMap::bcTypeONX || bcID == bcMap::bcTypeONY || bcID == bcMap::bcTypeONZ ||
         bcID == bcMap::bcTypeON || bcID == bcMap::bcTypeO)) {
      oudfFindDirichlet(field);
    }
    if (field.compare(0, 6, "scalar") == 0 && (bcID == bcMap::bcTypeS || bcID == bcMap::bcTypeINTS)) {
      oudfFindDirichlet(field);
    }

    if (field.compare("velocity") == 0 && (bcID == bcMap::bcTypeSHLX || bcID == bcMap::bcTypeSHLY ||
                                           bcID == bcMap::bcTypeSHLZ || bcID == bcMap::bcTypeSHL)) {
      oudfFindNeumann(field);
    }
    if (field.compare("mesh") == 0 && bcID == bcMap::bcTypeSHL) {
      oudfFindNeumann(field);
    }
    if (field.compare(0, 6, "scalar") == 0 && bcID == bcMap::bcTypeF) {
      oudfFindNeumann(field);
    }
  }
}

void oudfFindDirichlet(std::string &field)
{
  nekrsCheck(field.find("velocity") != std::string::npos && !velocityDirichletConditions,
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "%s\n",
             "Cannot find okl function codedFixedValueVelocity!");

  nekrsCheck(field.find("scalar") != std::string::npos && !scalarDirichletConditions,
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "%s\n",
             "Cannot find okl function codedFixedValueScalar!");

  if (field == "pressure" && !pressureDirichletConditions) {
    if (platform->comm.mpiRank == 0) {
      std::cout << "WARNING: Cannot find okl function codedFixedValuePressure => fallback to zero value!\n";
    }
  }

  if (field.find("mesh") != std::string::npos) {
    if (bcMap::useDerivedMeshBoundaryConditions()) {
      nekrsCheck(
          meshVelocityDirichletConditions,
          MPI_COMM_SELF,
          EXIT_FAILURE,
          "%s\n",
          "okl function codedFixedValueMesh is defined although derived mesh boundary conditions are used!");
    } else {
      nekrsCheck(!meshVelocityDirichletConditions,
                 MPI_COMM_SELF,
                 EXIT_FAILURE,
                 "%s\n",
                 "Cannot find okl function codedFixedValueMesh!");
    }
  }
}

void oudfFindNeumann(std::string &field)
{
  nekrsCheck(field.find("velocity") != std::string::npos && !velocityNeumannConditions,
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "%s\n",
             "Cannot find codedFixedGradientVelocity!");
  nekrsCheck(field.find("scalar") != std::string::npos && !scalarNeumannConditions,
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "%s\n",
             "Cannot find codedFixedGradientScalar!");
}

void adjustOudf(bool buildRequired, const std::string &postOklSource, const std::string &filePath)
{
  std::stringstream buffer;
  {
    std::ifstream f;
    f.open(postOklSource);
    buffer << f.rdbuf();
    f.close();
  }

  std::ofstream f;
  f.open(filePath, std::ios_base::app);

  if (buildRequired) {
    f << "#ifdef __okl__\n";
  }

  bool found = std::regex_search(buffer.str(), std::regex(R"(\s*void\s+codedFixedValueVelocity)"));
  velocityDirichletConditions = found;
  if (!found && buildRequired) {
    f << "void codedFixedValueVelocity(bcData *bc){}\n";
  }

  found = std::regex_search(buffer.str(), std::regex(R"(\s*void\s+codedFixedValueMesh)"));
  meshVelocityDirichletConditions = found;

  if (buildRequired) {
    if (bcMap::useDerivedMeshBoundaryConditions()) {
      f << "void codedFixedValueMesh(bcData *bc){\n"
           "  codedFixedValueVelocity(bc);\n"
           "}\n";
    } else if (!meshVelocityDirichletConditions) {
      if (platform->options.getArgs("MESH SOLVER").empty() ||
          platform->options.compareArgs("MESH SOLVER", "NONE")) {
        f << "void codedFixedValueMesh(bcData *bc){}\n";
      }
    }
  }

  found = std::regex_search(buffer.str(), std::regex(R"(\s*void\s+codedFixedGradientVelocity)"));
  velocityNeumannConditions = found;
  if (!found && buildRequired) {
    f << "void codedFixedGradientVelocity(bcData *bc){}\n";
  }

  found = std::regex_search(buffer.str(), std::regex(R"(\s*void\s+codedFixedValuePressure)"));
  pressureDirichletConditions = found;
  if (!found && buildRequired) {
    f << "void codedFixedValuePressure(bcData *bc){}\n";
  }

  found = std::regex_search(buffer.str(), std::regex(R"(\s*void\s+codedFixedGradientScalar)"));
  scalarNeumannConditions = found;
  if (!found && buildRequired) {
    f << "void codedFixedGradientScalar(bcData *bc){}\n";
  }

  found = std::regex_search(buffer.str(), std::regex(R"(\s*void\s+codedFixedValueScalar)"));
  scalarDirichletConditions = found;

  if (!found && buildRequired) {
    f << "void codedFixedValueScalar(bcData *bc){}\n";
  }

  if (buildRequired) {
    f << "#endif\n";
  }

  f.close();
}

void udfBuild(setupAide &options)
{
  options.getArgs("UDF FILE", udfFile);

  udfFile = fs::absolute(udfFile);
  if (platform->comm.mpiRank == 0) {
    nekrsCheck(!fs::exists(udfFile), MPI_COMM_SELF, EXIT_FAILURE, "Cannot find %s!\n", udfFile.c_str());
  }

  const int verbose = options.compareArgs("VERBOSE", "TRUE") ? 1 : 0;
  const std::string installDir(getenv("NEKRS_HOME"));
  const std::string cache_dir(getenv("NEKRS_CACHE_DIR"));
  const std::string udfLib = cache_dir + "/udf/libudf.so";
  const std::string udfFileCache = cache_dir + "/udf/udf.cpp";
  const std::string udfHashFile = cache_dir + "/udf/udf.hash";
  const std::string oudfFileCache = cache_dir + "/udf/udf.okl";
  const std::string case_dir(fs::current_path());
  const std::string casename = options.getArgs("CASENAME");

  const std::string cmakeBuildDir = cache_dir + "/udf";
  const std::string postOklSource = cmakeBuildDir + "/CMakeFiles/OKL.dir/okl.cpp.i";

  const std::string libnekrsFile = (sizeof(dfloat) == sizeof(float)) ? installDir + "/lib/libnekrs-fp32.so"
                                                                     : installDir + "/lib/libnekrs.so";
  const std::string libnekrsHashFile = cache_dir + "/udf/libnekrs.hash";

  std::string oudfFile;
  options.getArgs("UDF OKL FILE", oudfFile);
  oudfFile = fs::absolute(oudfFile);

  MPI_Comm comm = (platform->cacheLocal) ? platform->comm.mpiCommLocal : platform->comm.mpiComm;
  int buildRank;
  MPI_Comm_rank(comm, &buildRank);

  int buildRequired = 0;
  if (platform->comm.mpiRank == 0) {

    auto getHash = [&](const std::string &fname) {
      std::ifstream f(fname);
      if (!f.is_open()) {
        return std::string("");
      }
      std::stringstream buffer;
      buffer << f.rdbuf();
      f.close();

      return buffer.str();
    };

    // changes in udf include files + env-vars are currently not detected
    // note, we want to avoid calling system()
    if (options.compareArgs("BUILD ONLY", "TRUE")) {
      buildRequired = 1;
    } else if (!fs::exists(udfLib) || !fs::exists(oudfFileCache)) {
      buildRequired = 1;
    } else if (SHA1::from_file(udfFile) != getHash(udfHashFile)) {
      buildRequired = 1;
    } else if (SHA1::from_file(libnekrsFile) != getHash(libnekrsHashFile)) {
      buildRequired = 1;
    }

    if (fs::exists(std::string(case_dir + "/ci.inc"))) {
      if (isFileNewer(std::string(case_dir + "/ci.inc").c_str(), udfFileCache.c_str())) {
        buildRequired = 1;
      }
    }

    if (fs::exists(oudfFile)) {
      if (isFileNewer(oudfFile.c_str(), oudfFileCache.c_str())) {
        buildRequired = 1;
      }
    }

    // check for a typical include file
    if (fs::exists(std::string(case_dir + "/" + casename + ".okl"))) {
      if (isFileNewer(std::string(case_dir + "/" + casename + ".okl").c_str(), oudfFileCache.c_str())) {
        buildRequired = 1;
      }
    }
  }
  MPI_Bcast(&buildRequired, 1, MPI_INT, 0, comm);

  int oudfFileExists;
  if (platform->comm.mpiRank == 0) {
    oudfFileExists = fs::exists(oudfFile);
  }
  MPI_Bcast(&oudfFileExists, 1, MPI_INT, 0, comm);
  if (!oudfFileExists) {
    options.removeArgs("UDF OKL FILE");
  }

  if (platform->cacheBcast || platform->cacheLocal) {
    if (oudfFileExists) {
      fileBcast(oudfFile, platform->tmpDir, comm, platform->verbose);
      oudfFile = platform->tmpDir / fs::path(oudfFile).filename();
      options.setArgs("UDF OKL FILE", std::string(oudfFile));
    }
  }

  const auto err = (buildRank == 0 && buildRequired) ? udfMake(options, platform->solver->id(), platform->comm.mpiRank) : 0;
  nekrsCheck(err, platform->comm.mpiComm, EXIT_FAILURE, "%s\n", "see above and cmake.log for more details");

  if (platform->cacheBcast || platform->cacheLocal) {
    const auto dst = fs::path(platform->tmpDir) / "udf";
    fileBcast(fs::path(udfLib), dst, comm, platform->verbose);
    fileBcast(fs::path(oudfFileCache), dst, comm, platform->verbose);
  }

  if (buildRank == 0) {
    if (fs::exists(cache_dir + "/udf/okl.cpp")) {
      fs::rename(cache_dir + "/udf/okl.cpp", oudfFileCache);
    }

    adjustOudf(buildRequired, postOklSource, oudfFileCache); // call every time for verifyOudf
    verifyOudf();

    fileSync(oudfFileCache.c_str());
  }

  // some BC kernels will include this file
  options.setArgs("OKL FILE CACHE", oudfFileCache);
  if (platform->cacheBcast || platform->cacheLocal) {
    options.setArgs("OKL FILE CACHE", std::string(platform->tmpDir + "/udf/udf.okl"));
  }

  MPI_Bcast(&velocityDirichletConditions, 1, MPI_INT, 0, comm);
  MPI_Bcast(&meshVelocityDirichletConditions, 1, MPI_INT, 0, comm);
  MPI_Bcast(&velocityNeumannConditions, 1, MPI_INT, 0, comm);
  MPI_Bcast(&pressureDirichletConditions, 1, MPI_INT, 0, comm);
  MPI_Bcast(&scalarNeumannConditions, 1, MPI_INT, 0, comm);
  MPI_Bcast(&scalarDirichletConditions, 1, MPI_INT, 0, comm);
}

void *udfLoadFunction(const char *fname, int errchk)
{
  if (!libudfHandle) {
    std::string cache_dir(getenv("NEKRS_CACHE_DIR"));
    if (platform->cacheBcast) {
      cache_dir = fs::path(platform->tmpDir);
    }

    const auto udfLib = std::string(fs::path(cache_dir) / "udf/libudf.so");

    if (platform->comm.mpiRank == 0 && platform->verbose) {
      std::cout << "loading " << udfLib << std::endl;
    }

    libudfHandle = dlopen(udfLib.c_str(), RTLD_NOW | RTLD_GLOBAL);
    nekrsCheck(!libudfHandle, MPI_COMM_SELF, EXIT_FAILURE, "%s\n", dlerror());
  }

  void *fptr = dlsym(libudfHandle, fname);
  nekrsCheck(!fptr && errchk, MPI_COMM_SELF, EXIT_FAILURE, "%s\n", dlerror());

  dlerror();

  return fptr;
}

void udfUnload()
{
  dlclose(libudfHandle);
}

void udfLoad()
{
  *(void **)(&udf.setup0) = udfLoadFunction("UDF_Setup0", 0);
  *(void **)(&udf.setup) = udfLoadFunction("UDF_Setup", 1);
  *(void **)(&udf.loadKernels) = udfLoadFunction("UDF_LoadKernels", 0);
  *(void **)(&udf.autoloadKernels) = udfLoadFunction("UDF_AutoLoadKernels", 0);
  *(void **)(&udf.autoloadPlugins) = udfLoadFunction("UDF_AutoLoadPlugins", 1);
  *(void **)(&udf.executeStep) = udfLoadFunction("UDF_ExecuteStep", 1);
}

void udfEcho()
{
  const std::string cache_dir(getenv("NEKRS_CACHE_DIR"));
  const std::string oudfFileCache = cache_dir + "/udf/udf.okl";

  const auto tmpFile = udfFile + ".unifdef";
  unifdef("__okl__", udfFile.c_str(), tmpFile.c_str());

  std::ifstream fudf(tmpFile);
  std::string text;

  std::cout << std::endl;
  while (!fudf.eof()) {
    getline(fudf, text);
    std::cout << "<<< " << text << "\n";
  }
  fudf.close();
  fs::remove(tmpFile);

  std::ifstream foudf(oudfFileCache);

  std::cout << std::endl;
  while (!foudf.eof()) {
    getline(foudf, text);
    std::cout << "<<< " << text << "\n";
  }
  std::cout << std::endl;

  foudf.close();
}
