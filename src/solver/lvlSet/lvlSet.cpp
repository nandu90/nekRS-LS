#include "lvlSet.hpp"
#include "nrs.hpp"
#include "platform.hpp"
#include "linAlg.hpp"
#include "solver.hpp"
#include "bdryBase.hpp"
#include "advectionSubCycling.hpp"
#include "nekInterfaceAdapter.hpp"
#include "iofldFactory.hpp"
#include <stdexcept>
#include <algorithm>
#include "par.hpp"
#include "avm.hpp"
#include "lowPassFilter.hpp"
#include "gjp.hpp"

// private members
namespace
{
static std::ostringstream errorLogger;
static std::ostringstream valueErrorLogger;
nrs_t *nrs;
occa::kernel signlsKernel;
occa::kernel normalVectorKernel;
bool buildKernelCalled = false;
std::unique_ptr<lvlSet_t> tlsr = nullptr;

static double elapsedTime;
static int stepsMax;

static double tlsrTimer = 0.0;
static double clsrTimer = 0.0;
static double fluidStartTime = -1.0;

// common keys
static std::vector<std::string> commonKeys = {
    {"solver"},
    {"residualTol"},
    {"initialGuess"},
    {"preconditioner"},
    {"pMGSchedule"},
    {"smootherType"},
    {"coarseSolver"},
    {"semfemSolver"},
    {"coarseGridDiscretization"},
    {"boundaryTypeMap"},
    {"regularization"},
    {"checkpointing"},

    // deprecated filter params
    {"filterWeight"},
    {"filterModes"},
    {"filterCutoffRatio"},
};

static std::vector<std::string> lvlSetKeys = {
  {"solveFrequency"},
};

static std::vector<std::string> scalarKeys = {
  {"boundaryTypeMap"},
  {"absoluteTol"},
};

static std::vector<std::string> validSections = {
  {"lvlset"},
  {"tlsr"},
  {"clsr"},
};

template <typename Printable> void append_error(Printable message)
{
  errorLogger << "\t" << message << "\n";
}

template <typename Printable> void append_value_error(Printable message)
{
  valueErrorLogger << "\t" << message << "\n";
}

void processError()
{
  const std::string valueErrors = valueErrorLogger.str();
  errorLogger << valueErrors;
  const std::string errorMessage = errorLogger.str();
  int length = errorMessage.size();
  MPI_Bcast(&length, 1, MPI_INT, 0, platform->comm.mpiComm());

  auto errTxt = [&]() {
    std::stringstream txt;
    txt << std::endl;
    txt << errorMessage;
    txt << "\nrun with `--help par` for more details\n";

    return txt.str();
  };

  nekrsCheck(length > 0, platform->comm.mpiComm(), EXIT_FAILURE, "%s\n", errTxt().c_str());
}

} // namespace

void lvlSet::buildKernel(occa::properties _kernelInfo)
{
  occa::properties kernelInfo;
  kernelInfo += _kernelInfo;

  auto buildKernel = [&kernelInfo](const std::string &kernelName) {
    const auto path = getenv("NEKRS_KERNEL_DIR") + std::string("/solver/lvlSet/");
    const auto fileName = path + "lvlSet.okl";
    const auto reqName = "lvlSet::";
    if (platform->options.compareArgs("REGISTER ONLY", "TRUE")) {
      platform->kernelRequests.add(reqName, fileName, kernelInfo);
      return occa::kernel();
    } else {
      buildKernelCalled = 1;
      return platform->kernelRequests.load(reqName, kernelName);
    }
  };

  signlsKernel = buildKernel("signls");
  normalVectorKernel = buildKernel("normalVector");
}

void validateLvlSetSections()
{
  auto sections = platform->par->ini->sections;

  auto sectionExists = [&](const std::string& name) -> void {
    for (const auto &sec : sections) {
      if(sec.first.find(name) != std::string::npos) {
        return;
      }
    }
    std::ostringstream error;
    error << "mandatory section for Level-Set ["<< upperCase(name) <<"] not found!\n";
    append_error(error.str());
  };

  sectionExists("scalar tls");
  sectionExists("scalar cls");

  for (auto const &sec : sections) {
    if (std::find(validSections.begin(), validSections.end(), sec.first) != validSections.end()) {
      std::vector<std::string> validKeys;
      if(sec.first == "lvlset") {
        validKeys = lvlSetKeys;
      } else {
        validKeys = scalarKeys;
      }

      for (auto const &val : sec.second) {
        const auto &key = val.first;

        if (std::find(validKeys.begin(), validKeys.end(), key) == validKeys.end()) {
          if (std::find(commonKeys.begin(), commonKeys.end(), key) == commonKeys.end()) {
            std::ostringstream error;
            error << "unknown key: " << sec.first << "::" << key << "\n";
            append_error(error.str());
          }
        }
      }
    }
  }
}

void cleanupStaleKeys(const int rank, setupAide &options, inipp::Ini *ini)
{
  std::vector<std::string> sections = {"TLSR", "CLSR", "LVLSET DEFAULT"};

  auto cleanSection = [&](const std::string &section, const std::vector<std::string> &staleKeys) {
    std::vector<std::string> staleOptions;
    for (auto const &option : options) {
      if (option.first.find(section) == 0) {
        for (auto const &key : staleKeys) {
          if (option.first.find(key) != std::string::npos) {
            staleOptions.push_back(option.first);
          }
        }
      }
    }
    for (auto const &key : staleOptions) {
      options.removeArgs(key);
    }
  };

  const std::vector<std::string> staleKeys = {"RESIDUAL PROJECTION",
                                              "INITIAL GUESS",
                                              "ELLIPTIC COEFF FIELD",
                                              "REGULARIZATION",
                                              "BOUNDARY TYPE MAP",
                                              "SOLVER MAXIMUM ITERATIONS",
                                              "BLOCK SOLVER",
                                              "PRECONDITIONER",
                                              "ELLIPTIC",
                                              "CVODE",
                                              "TOLERANCE",
                                              "MULTIGRID",
                                              "MGSOLVER"};

  const std::vector<std::string> invalidKeysCvode = {"RESIDUAL PROJECTION",
                                                     "INITIAL GUESS",
                                                     "MAXIMUM ITERATIONS",
                                                     "PRECONDITIONER",
                                                     "ELLIPTIC",
                                                     "MULTIGRID",
                                                     "MGSOLVER"};

  for (const auto &section : sections) {
    if (options.compareArgs(section + " SOLVER", "NONE")) {
      cleanSection(section, staleKeys);
    }

    if (options.compareArgs(section + " SOLVER", "CVODE")) {
      cleanSection(section, invalidKeysCvode);
    }
  }

  std::vector<std::string> staleOptions;
  for (auto const &option : options) {
    if (option.first.find("LVLSET DEFAULT") == 0) {
      staleOptions.push_back(option.first);
    }
  }
  for (auto const &key : staleOptions) {
    options.removeArgs(key);
  }
}

void parseLvlSetSections()
{
  const auto &ini = platform->par->ini;
  const auto &sections = ini->sections;
  const auto &rank = platform->comm.mpiRank();
  auto &options = platform->options;

  auto parseLvlSetSection = [&](const auto &sec) {
    std::istringstream stream(sec.first);
    std::string firstWord;
    stream >> firstWord;
    if (firstWord != "default" && firstWord != "tlsr" && firstWord != "clsr") {
      return;
    }

    auto parScope = sec.first;
    if(firstWord == "default") 
      parScope = "lvlset " + parScope;
    auto parPrefix = upperCase(parScope) + " ";

    {
      std::string val = "false";
      if (ini->extract(parScope, "checkpointing", val)) {
        if(val == "true") {
          val = "true";
        } else {
          val = "false";
        }
      }
      options.setArgs(parPrefix + "CHECKPOINTING", upperCase(val));
    }

    parseRegularization(rank, options, ini, parScope);
    if(options.compareArgs(parPrefix + "REGULARIZATION METHOD","NONE")) {
      if(firstWord == "default") {
        options.setArgs(parPrefix + "REGULARIZATION METHOD","SVV");
        options.setArgs("TLSR REGULARIZATION SVV SCALING COEFF", "2.0");
        options.setArgs("TLSR REGULARIZATION SVV FILTER POWER", "6.0");
        options.setArgs("CLSR REGULARIZATION SVV SCALING COEFF", "1.0");
        options.setArgs("CLSR REGULARIZATION SVV FILTER POWER", "4.0");
      }
    }

    std::string solver;
    ini->extract(parScope, "solver", solver);

    if (solver == "cvode") {
      options.setArgs(parPrefix + "SOLVER", "CVODE");
    }

    options.setArgs(parPrefix + "ELLIPTIC COEFF FIELD", "TRUE");

    parseInitialGuess(rank, options, ini, parScope);

    parsePreconditioner(rank, options, ini, parScope);

    parseLinearSolver(rank, options, ini, parScope);

    parseSolverTolerance(rank, options, ini, parScope);
    if(firstWord == "default")
      options.setArgs(parPrefix + "SOLVER TOLERANCE", to_string_f(1e-8));

    std::string sbuf;

    //lvlset works only on fluid mesh
    options.setArgs(parPrefix + "MESH", "FLUID");

    if(firstWord == "default") {
      options.setArgs("TLSR DIFFUSIONCOEFF", to_string_f(1e-12));
      options.setArgs("TLSR TRANSPORTCOEFF", to_string_f(1.0));
      options.setArgs("CLSR DIFFUSIONCOEFF", to_string_f(1.0));
    }

    if(firstWord == "default") {
      double dt = 0.0;
      options.getArgs("DT", dt);
      options.setArgs("TLSR FREQUENCY", to_string_f(dt * 100.0));
      options.setArgs("CLSR FREQUENCY", to_string_f(dt * 10.0));
    }

    {
      double freq;
      if(ini->extract(parScope, "solveFrequency", freq))
        options.setArgs(parPrefix + "FREQUENCY", to_string_f(freq));
    }

    std::string s_bcMap;
    if (ini->extract(parScope, "boundarytypemap", s_bcMap)) {
      options.setArgs(parPrefix + "BOUNDARY TYPE MAP", s_bcMap);
    }
  };

  // apply default lvlSet section arguments
  auto defaultSection = std::make_pair(std::string("default"), sections.at("scalar"));
  parseLvlSetSection(defaultSection);

  // initialize with default settings
  const std::string defaultSettingStr = "LVLSET DEFAULT";
  const auto options_ = options;
  for (auto [keyWord, value] : options_) {
    auto delPos = keyWord.find(defaultSettingStr);
    if (delPos != std::string::npos) {
      auto newKey = keyWord;
      newKey.erase(delPos, defaultSettingStr.size());
      options.setArgs("TLSR" + newKey, value);
      options.setArgs("CLSR" + newKey, value);
    }
  }

  // override default settings if specified explicitly
  for (auto &&sec : sections) {
    parseLvlSetSection(sec);
  }

  // set boundarytypemap from tls field if not specified explicitly
  std::string s_bcMap;
  if(!ini->extract("tlsr", "boundarytypemap", s_bcMap)){
    ini->extract("scalar tls", "boundarytypemap", s_bcMap);
    if(s_bcMap.size() > 0) {
      const auto list = serializeString(s_bcMap, ',');

      std::string s_newMap = "";
      for (int i = 0; i < list.size(); i++) {
        if (i > 0) 
          s_newMap = s_newMap + ", ";
        s_newMap = s_newMap + "zeroNeumann";
      }

      options.setArgs("TLSR BOUNDARY TYPE MAP", s_newMap);
    }
  }

  if(!ini->extract("clsr", "boundarytypemap", s_bcMap)){
    ini->extract("scalar cls", "boundarytypemap", s_bcMap);
    if(s_bcMap.size() > 0) {
      const auto list = serializeString(s_bcMap, ',');

      std::string s_newMap = "";
      for (int i = 0; i < list.size(); i++) {
        if (i > 0) 
          s_newMap = s_newMap + ", ";
        if(list[i] == "inlet" || list[i] == "v" || list[i] == "udfDirichlet") {
          s_newMap = s_newMap + list[i];
        } 
        else {
          s_newMap = s_newMap + "zeroNeumann";
        }
      }

      options.setArgs("CLSR BOUNDARY TYPE MAP", s_newMap);
    }
  }

  //TODO: remove this
  options.setArgs("TLSR SOLVER","NONE");

  cleanupStaleKeys(rank, options, ini);
}

void bdrySetupFromPar()
{
  std::vector<std::string> sectionsPar = {"TLSR","CLSR"};

  int count = 0;
  int expectedCount = 0;

  auto process = [&](const std::string &sectionPar) {
    std::vector<std::string> staleOptions;
    for (auto const &option : platform->options) {
      if (option.first.find(sectionPar) != std::string::npos) {
        if (option.first.compare(sectionPar + " SOLVER") == 0 &&
            option.second.find("NONE") == std::string::npos) {
          expectedCount++;
        }

        if (option.first.find("BOUNDARY TYPE MAP") != std::string::npos) {
          count++;

          platform->app->bc->setupField(serializeString(option.second, ','), sectionPar, false);

          staleOptions.push_back(option.first);
        }
      }
    }
    for (auto const &key : staleOptions) {
      platform->options.removeArgs(key);
    }
  };

  for (auto &sectionPar : sectionsPar) {
    process(sectionPar);
  }

  nekrsCheck(count > 0 && count != expectedCount,
             platform->comm.mpiComm(),
             EXIT_FAILURE,
             "boundaryTypeMap specfied for %d fields but not all %d fields!",
             count,
             expectedCount);
}

void lvlSet::setup()
{
  static bool isInitialized = false;
  if (isInitialized) {
    return;
  }
  isInitialized = true;

  nrs = dynamic_cast<nrs_t *>(platform->app);
  if (!nrs || !nrs->meshV || !nrs->scalar) {
    throw std::runtime_error("lvlSet::setup: nrs/nrs->meshV and/or nrs->scalar is null (mesh not initialized)");
  }

  if(platform->comm.mpiRank() == 0)
    validateLvlSetSections();
  processError();

  parseLvlSetSections();
  processError();

  bdrySetupFromPar();

  //bc from nek. TODO - need to check this. Also add CLSR
  if(platform->app->bc->useNek()) {
    int nIDs = nekData.NboundaryID; //cannot be Tmesh
    std::vector<int> map(nIDs);

    {
      auto sIndex = nrs->scalar->nameToIndex.find("tls")->second;
      for (int id = 0; id < map.size(); id++) {
        map[id] = nek::bcmap(id + 1, sIndex + 2, 0);
        map[id] = bdryBase::bcType_zeroNeumann;      //all zero flux BCs
      }
      platform->app->bc->setBcMap("tlsr", false, map);
    }
    {
      auto sIndex = nrs->scalar->nameToIndex.find("cls")->second;
      for (int id = 0; id < map.size(); id++) {
        map[id] = nek::bcmap(id + 1, sIndex + 2, 0);
        if(map[id] != bdryBase::bcType_udfDirichlet) 
          map[id] = bdryBase::bcType_zeroNeumann;      //zero flux BC on all non-Dirichlet faces
      }
      platform->app->bc->setBcMap("clsr", false, map);
    }
  }

  // currently just holds TLSR solver --> TODO: add in CLSR solver (separate object or hold both in one LS object?)
  tlsr = [&]() {
    lvlSetConfig_t cfg;
    cfg.name = "tlsr";
    cfg.g0 = &nrs->g0;
    cfg.dt = nrs->dt;
    cfg.fieldOffset = nrs->meshV->fieldOffset;
    cfg.vFieldOffset = nrs->scalar->vFieldOffset;       // TODO: is it safe to assume nrs->scalar is always accessible?
    cfg.vCubatureOffset = nrs->scalar->vCubatureOffset; // TODO: is it safe to assume nrs->scalar is always accessible?
    cfg.mesh = nrs->meshV;
    cfg.meshV = nrs->meshV;
    return std::make_unique<lvlSet_t>(cfg, nrs->geom);
  }();
  tlsr->setupEllipticSolver();
}

void lvlSet_t::pseudoStepper(const double &fluidTime)
{
  auto mesh = this->meshV;

  double time = 0.0;
  int tstep = 1;
  stepsMax = 1000;
  int innerIterMax = 100;

  // set constant dt --> TODO: need to change this if we want variable dt
  this->dt[1] = this->dt[0];
  this->dt[2] = this->dt[0];

  elapsedTime = 0.0;

  while(tstep <= stepsMax) {
    MPI_Barrier(platform->comm.mpiComm());
    const double timeStartStep = MPI_Wtime();

    time += this->dt[0];
    this->setTimeIntegrationCoeffs(tstep);
    this->extrapolateSolution();
    platform->linAlg->fill(this->fieldOffset(), 0.0, this->o_EXT);
    this->computeAdvectionCoeff();
    this->computeWrst();
    this->makeAdvection(time, tstep);
    this->makeExplicit(time, tstep);
    this->makeForcing();
    this->mueSVV();
    this->mueAVM();
    this->lagSolution();
    this->applyDirichlet(time);
    this->solve(time, 1);

    MPI_Barrier(platform->comm.mpiComm());
    const double elapsedStep = MPI_Wtime() - timeStartStep;
    elapsedTime += elapsedStep;

    this->printStepInfo(time, tstep, true, true);
    this->writeFile(fluidTime, tstep);
    tstep += 1;
  }

  // copy the TLSR solution back to the scalar S00
  //nrs->scalar->o_S.copyFrom(tlsr->o_S);
}

void lvlSet::solve(const double &fluidTime)
{
  if(fluidStartTime < 0.0) fluidStartTime = fluidTime;

  double tlsrFreq;
  platform->options.getArgs("TLSR FREQUENCY", tlsrFreq);

  const double totalTime = fluidTime - fluidStartTime + 1e-12;

  if(totalTime > tlsrTimer  && tlsrFreq > 1e-12) {
    tlsrTimer += tlsrFreq;

    tlsr->o_S.copyFrom(nrs->scalar->o_solution("tls"), tlsr->fieldOffset());

    tlsr->pseudoStepper(fluidTime);
  }
}


lvlSet_t* lvlSet::getLS()
{
  return tlsr.get();
}

lvlSet_t::lvlSet_t(lvlSetConfig_t &cfg, const std::unique_ptr<geomSolver_t> &_geom) : geom(_geom)
{
  this->name = cfg.name;

  if (platform->comm.mpiRank() == 0) {
  std::cout << "================ " << "SETUP " << upperCase(this->name) << " ===============\n";
  }

  auto &options = platform->options;

  this->Nsubsteps = 0;
  platform->options.getArgs("SUBCYCLING STEPS", Nsubsteps);

  this->ellipticSolver.resize(1);

  this->meshV = cfg.meshV;

  this->g0 = cfg.g0;
  this->dt = cfg.dt;

  int nBDF;
  int nEXT;
  options.getArgs("BDF ORDER", nBDF);
  options.getArgs("EXT ORDER", nEXT);
  if (this->Nsubsteps) {
    nEXT = nBDF;
    platform->options.setArgs("EXT ORDER", std::to_string(nEXT));
  }
  nekrsCheck(nEXT < nBDF,
             platform->comm.mpiComm(),
             EXIT_FAILURE,
             "%s\n",
             "EXT order needs to be >= BDF order!");

  this->o_coeffBDF = platform->device.malloc<dfloat>(nBDF);
  this->o_coeffEXT = platform->device.malloc<dfloat>(nEXT);

  this->_fieldOffset = cfg.fieldOffset; 
  this->vFieldOffset = cfg.vFieldOffset; // does not operate on solid mesh
  this->vCubatureOffset = cfg.vCubatureOffset;

  this->_mesh = cfg.mesh;

  this->fieldOffsetScan = 0;

  this->o_fieldOffsetScan = platform->device.malloc<dlong>(1, &fieldOffsetScan);

  this->o_prop = platform->device.malloc<dfloat>(2 * this->_fieldOffset);
  this->o_diff = this->o_prop.slice(0 * this->_fieldOffset, this->_fieldOffset);
  this->o_rho = this->o_prop.slice(1 * this->_fieldOffset, this->_fieldOffset);
  
  auto o_tmp = [&]() {
    const std::string prefixedName = this->name;
    auto tmp = platform->device.malloc<char>(prefixedName.size() + 1);
    tmp.copyFrom(prefixedName.data());
    const char nullChar[] = {'\0'};
    tmp.copyFrom(nullChar, 1, prefixedName.size());
    return tmp;
  }();
  this->o_name = o_tmp;

  if (!options.compareArgs(upperCase(this->name) + " SOLVER", "NONE")) {

    nekrsCheck(options.compareArgs(upperCase(this->name) + " SOLVER", "BLOCK"),
        platform->comm.mpiComm(),
        EXIT_FAILURE,
        "%s\n",
        "level-set does not support BLOCK solver!");

    platform->app->bc->printBcTypeMapping(this->name);
    if (platform->comm.mpiRank() == 0) {
      std::cout << std::endl;
    }

    dfloat diff = 1;
    dfloat rho = 1;
    options.getArgs(upperCase(this->name) + " DIFFUSIONCOEFF", diff);
    options.getArgs(upperCase(this->name) + " TRANSPORTCOEFF", rho);

    std::vector<dfloat> diffTmp(this->_mesh->Nlocal, diff);
    std::vector<dfloat> rhoTmp(this->_mesh->Nlocal, rho);

    this->o_diff.copyFrom(diffTmp.data(), diffTmp.size());
    this->o_rho.copyFrom(rhoTmp.data(), rhoTmp.size());
  }

  this->EToBOffset = this->_mesh->Nelements * this->_mesh->Nfaces;

  {
    std::vector<int> EToB(EToBOffset);
    this->compute = 1;
    if (options.compareArgs(upperCase(this->name) + " SOLVER", "NONE")) {
      this->compute = 0;
    } 
    else {
      auto mesh = this->_mesh;
      int cnt = 0;
      for (int e = 0; e < mesh->Nelements; e++) {
        for (int f = 0; f < mesh->Nfaces; f++) {
          EToB[cnt] =
            platform->app->bc->typeId(mesh->EToB[f + e * mesh->Nfaces], this->name);
          cnt++;
        }
      }
    }

    this->o_EToB = platform->device.malloc<int>(EToB.size());
    this->o_EToB.copyFrom(EToB.data());
  }

  this->o_compute = platform->device.malloc<dlong>(1, &compute);

  {
    int nFieldsAlloc = std::max(this->o_coeffBDF.size(), this->o_coeffEXT.size());
    this->o_S = platform->device.malloc<dfloat>(nFieldsAlloc * this->_fieldOffset);
    this->o_W = platform->device.malloc<dfloat>(this->meshV->dim * std::max(this->o_coeffBDF.size(), this->o_coeffEXT.size()) * this->vFieldOffset);
    const dlong Nstates = this->Nsubsteps ? std::max(this->o_coeffBDF.size(), this->o_coeffEXT.size()) : 1;
    this->o_relWrst = platform->device.malloc<dfloat>(Nstates * this->meshV->dim * this->vCubatureOffset);

    this->o_signls = platform->device.malloc<dfloat>(nFieldsAlloc * this->_fieldOffset);

    nFieldsAlloc = this->o_coeffEXT.size();
    this->o_ADV = platform->device.malloc<dfloat>(nFieldsAlloc * this->_fieldOffset);
    this->o_EXT = platform->device.malloc<dfloat>(nFieldsAlloc * this->_fieldOffset);

    this->o_Se = platform->device.malloc<dfloat>(this->_fieldOffset);
    this->o_JwF = platform->device.malloc<dfloat>(this->_fieldOffset);
  }

  bool filteringEnabled = false;
  bool avmEnabled = false;
  bool svvEnabled = false;

  if (options.compareArgs(upperCase(this->name) + " REGULARIZATION METHOD", "HPFRT")) {
    filteringEnabled = true;
  }

  if (options.compareArgs(upperCase(this->name) + " REGULARIZATION METHOD", "AVM_AVERAGED_MODAL_DECAY")) {
    avmEnabled = true;
  }

  if (options.compareArgs(upperCase(this->name) + " REGULARIZATION METHOD", "SVV")) {
    svvEnabled = true;
  }

  if (filteringEnabled) {
    dlong applyFilterRT = 0;
    const dlong Nmodes = this->meshV->N + 1; // assumed to be the same for all fields
    o_filterRT = platform->device.malloc<dfloat>(Nmodes * Nmodes);
    dfloat filterS = 0;

    if (this->compute) {
      int filterNc = -1;
      options.getArgs(upperCase(this->name) + " HPFRT MODES", filterNc);
      dfloat strength = NAN;
      options.getArgs(upperCase(this->name) + " HPFRT STRENGTH", strength);
      filterS = strength;
      this->o_filterRT.copyFrom(lowPassFilterSetup(this->_mesh, filterNc), Nmodes * Nmodes);

      applyFilterRT = 1;
    }

    o_filterS = platform->device.malloc<dfloat>(1, &filterS);
    o_applyFilterRT = platform->device.malloc<dlong>(1, &applyFilterRT);
  }

  if (avmEnabled) {
    bool avmEnabledScalar = false;
    for (int is = 0; is < nrs->scalar->NSfields; is++) {
      const auto sid = scalarDigitStr(is);

      if (options.compareArgs("SCALAR" + sid + " REGULARIZATION METHOD", "AVM_AVERAGED_MODAL_DECAY")) {
        avmEnabledScalar = true;
      }
    }
    if(!avmEnabledScalar) avm::setup(this->meshV);  //only initialize if not done in scalar
  }

  if(svvEnabled) {
    this->o_svvf = platform->device.malloc<dfloat>(this->_fieldOffset);
    this->o_svvmu = platform->device.malloc<dfloat>(this->_fieldOffset);
  }

  auto verifyBC = [&]() {
    if (this->compute) {
      const std::string field = this->name;
      nekrsCheck(this->_mesh->Nbid != platform->app->bc->size(field),
                 platform->comm.mpiComm(),
                 EXIT_FAILURE,
                 "Size of %s boundaryTypeMap (%d) does not match number of boundary IDs in mesh (%d)!\n",
                 field.c_str(),
                 platform->app->bc->size(field),
                 this->_mesh->Nbid);
    }
  };

  verifyBC();
}

void lvlSet_t::setTimeIntegrationCoeffs(int tstep)
{
  const auto bdfOrder = std::min(tstep, static_cast<int>(this->o_coeffBDF.size()));
  const auto extOrder = std::min(tstep, static_cast<int>(this->o_coeffEXT.size()));

  {
    std::vector<dfloat> coeff(this->o_coeffBDF.size());
    nek::bdfCoeff(this->g0, coeff.data(), this->dt, bdfOrder);
    for (int i = coeff.size(); i > bdfOrder; i--) {
      coeff[i - 1] = 0;
    }
    this->o_coeffBDF.copyFrom(coeff.data());
  }

  {
    std::vector<dfloat> coeff(this->o_coeffEXT.size());
    nek::extCoeff(coeff.data(), this->dt, extOrder, bdfOrder);
    for (int i = coeff.size(); i > extOrder; i--) {
      coeff[i - 1] = 0;
    }
    this->o_coeffEXT.copyFrom(coeff.data());
  }
}

void lvlSet_t::computeAdvectionCoeff()
{
  auto meshV = this->meshV;
  // a. compute interface normals
  launchKernel("core-gradientVolumeHex3D",
               meshV->Nelements,
               meshV->o_vgeo,
               meshV->o_D,
               this->vFieldOffset,
               this->o_S,
               this->o_W);

  oogs::startFinish(this->o_W, meshV->dim, this->vFieldOffset, ogsDfloat, ogsAdd, meshV->oogs);
  platform->linAlg->axmyVector(meshV->Nlocal, this->vFieldOffset, 0, 1.0, meshV->o_invLMM, this->o_W);

  signlsKernel(meshV->Nlocal, this->o_S, this->o_signls);

  normalVectorKernel(meshV->Nlocal, this->vFieldOffset, this->o_signls, this->o_W);
}

void lvlSet_t::makeAdvection(double time, int tstep)
{
  if (this->Nsubsteps) {
    advectionSubcycling(std::min(tstep, static_cast<int>(this->o_coeffEXT.size())), time);
  } else {
    auto mesh = this->meshV;

    if (platform->options.compareArgs("ADVECTION TYPE", "CUBATURE")) {
      launchKernel("core-strongAdvectionCubatureVolumeScalarHex3D",
                   mesh->Nelements,
                   1, /* nScalars */
                   0, /* weighted */
                   0, /* sharedRho */
                   mesh->o_vgeo,
                   mesh->o_cubDiffInterpT,
                   mesh->o_cubInterpT,
                   mesh->o_cubProjectT,
                   this->o_compute,
                   this->o_fieldOffsetScan,
                   this->vFieldOffset,
                   this->vCubatureOffset,
                   this->o_S,
                   this->o_relWrst,   // --> TODO change to w
                   this->o_rho, // --> set to 1
                   this->o_ADV);
    } else {
      launchKernel("core-strongAdvectionVolumeScalarHex3D",
                   mesh->Nelements,
                   1, /* nScalars */
                   0, /* weighted */
                   mesh->o_vgeo,
                   mesh->o_D,
                   this->o_compute,
                   this->o_fieldOffsetScan,
                   this->vFieldOffset,
                   this->o_S,
                   this->o_relWrst, // --> TODO change to w
                   this->o_rho, // --> set to 1
                   this->o_ADV);
    }
  }
}

void lvlSet_t::advectionSubcycling(int nEXT, double time)
{
  const auto mesh = this->_mesh;
  const auto meshV = this->meshV;

  static occa::kernel kernel;
  if (!kernel.isInitialized()) {
    if (platform->options.compareArgs("ADVECTION TYPE", "CUBATURE")) {
      kernel = platform->kernelRequests.load("core-subCycleStrongCubatureVolumeScalarHex3D");
    } else {
      kernel = platform->kernelRequests.load("core-subCycleStrongVolumeScalarHex3D");
    }
  }

  platform->linAlg->fill(this->o_JwF.size(), 0, this->o_JwF);

  advectionSubcyclingRK(mesh,
                        meshV,
                        time,
                        this->dt,
                        this->Nsubsteps,
                        this->o_coeffBDF,
                        nEXT,
                        1,    //nFields
                        kernel,
                        meshV->oogs,
                        mesh->fieldOffset,
                        this->_fieldOffset,
                        this->vCubatureOffset,
                        this->_fieldOffset,
                        (this->geom) ? this->geom->o_div : o_NULL,
                        this->o_relWrst,
                        this->o_S,
                        this->o_JwF);

  if (platform->verbose()) {
    const dfloat debugNorm = platform->linAlg->weightedNorm2Many(mesh->Nlocal,
                                                                 1,
                                                                 0,
                                                                 mesh->ogs->o_invDegree,
                                                                 this->o_JwF,
                                                                 platform->comm.mpiComm());
    if (platform->comm.mpiRank() == 0) {
      printf("%s advSub norm: %.15e\n", this->name.c_str(), debugNorm);
    }
  }
}

void lvlSet_t::makeExplicit(double time, int tstep)
{
  auto mesh = this->_mesh;

  const auto parPrefix = upperCase(this->name);

  if(this->name == "tlsr")
    this->o_EXT.copyFrom(this->o_signls); // TODO: shouldn't use tlsr here but will use this hack for now
                                                          
  if (platform->options.compareArgs(parPrefix + " REGULARIZATION METHOD", "HPFRT")) {
    launchKernel("core-filterRTHex3D",
                 this->meshV->Nelements,
                 0,
                 1,
                 this->o_fieldOffsetScan,
                 this->o_applyFilterRT,
                 this->o_filterRT,
                 this->o_filterS,
                 this->o_rho,
                 this->o_S,
                 this->o_EXT);
  }

  if (platform->options.compareArgs(parPrefix + " REGULARIZATION METHOD", "GJP")) {
    dfloat tauFactor;
    platform->options.getArgs(parPrefix + " REGULARIZATION GJP SCALING COEFF", tauFactor);

    addGJP(mesh, this->ellipticSolver[0]->o_EToB(), this->o_rho, this->vFieldOffset, this->o_W, this->o_S, this->o_EXT, tauFactor);
  }

  const int movingMesh = platform->options.compareArgs("MOVING MESH", "TRUE");
  if (movingMesh && !this->Nsubsteps) {
    launchKernel("scalar_t::advectMeshVelocityHex3D",
                 this->meshV->Nelements,
                 mesh->o_vgeo,
                 mesh->o_D,
                 0,
                 (this->geom) ? this->geom->fieldOffset : 0,
                 this->o_rho,
                 (this->geom) ? this->geom->o_U : o_NULL,
                 this->o_S,
                 this->o_EXT);
  }
}

void lvlSet_t::makeForcing()
{
  auto mesh = this->_mesh;

  if (this->compute) {
    launchKernel("scalar_t::sumMakef",
                 mesh->Nlocal,
                 mesh->o_LMM,
                 1 / this->dt[0],
                 this->o_coeffEXT,
                 this->o_coeffBDF,
                 this->fieldOffsetScan,
                 this->_fieldOffset,  //offset sum
                 mesh->fieldOffset,
                 this->o_rho,
                 this->o_S,
                 this->o_ADV,
                 this->o_EXT,
                 this->o_JwF);
  }

  const auto n = std::max(this->o_coeffEXT.size(), this->o_coeffBDF.size());
  for (int s = n; s > 1; s--) {
    this->o_EXT.copyFrom(this->o_EXT, this->_fieldOffset, (s - 1) * this->_fieldOffset, (s - 2) * this->_fieldOffset);
    if (this->o_ADV.isInitialized()) {
      this->o_ADV.copyFrom(this->o_ADV, this->_fieldOffset, (s - 1) * this->_fieldOffset, (s - 2) * this->_fieldOffset);
    }
  }
}

void lvlSet_t::solve(double time, int stage)
{
  if (!this->compute) {
    return;
  }

  auto mesh = this->_mesh;

  auto o_rhs = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
  o_rhs.copyFrom(this->o_JwF, mesh->Nlocal, 0, this->fieldOffsetScan);

  auto o_lhs = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);

  launchKernel("scalar_t::neumannBCHex3D",
      this->o_name,
      mesh->Nelements,
      1,
      mesh->o_sgeo,
      mesh->o_vmapM,
      mesh->o_EToB,
      0, //scalar index
      time,
      this->vFieldOffset,
      this->_fieldOffset,
      0,
      this->EToBOffset,
      mesh->o_x,
      mesh->o_y,
      mesh->o_z,
      this->o_W, // changed from o_Ue --> TODO: think about this
      this->o_S,
      this->o_EToB,
      this->o_diff,
      this->o_rho,
      platform->app->bc->o_usrwrk,
      o_lhs,
      o_rhs);

  const auto o_lambda0 = this->o_diff;
  const auto o_lambda1 = [&]() {
    auto o_l = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
    platform->linAlg->axpby(mesh->Nlocal, *this->g0 / this->dt[0], this->o_rho, 0.0, o_l);

    if (this->userImplicitLinearTerm) {
      auto o_implicitLT = this->userImplicitLinearTerm(time, 0);
      if (o_implicitLT.isInitialized()) {
        platform->linAlg->axpby(mesh->Nlocal, 1.0, o_implicitLT, 1.0, o_l);
      }
    }

    return o_l;
  }();

  auto o_Si = [&]() {
    auto o_S0 = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
    if (platform->options.compareArgs(upperCase(this->name) + " INITIAL GUESS", "EXTRAPOLATION") && stage == 1) {
      o_S0.copyFrom(this->o_Se, o_S0.size(), 0, this->fieldOffsetScan);
    } else {
      o_S0.copyFrom(this->o_S, o_S0.size(), 0, this->fieldOffsetScan);
    }

    return o_S0;
  }();

  this->ellipticSolver[0]->coeff0HLM(o_lambda0);
  this->ellipticSolver[0]->coeff1HLM(o_lambda1);
  this->ellipticSolver[0]->solve(o_rhs, o_Si);
  o_Si.copyTo(this->o_S, o_Si.size(), this->fieldOffsetScan);
}

void lvlSet_t::saveSolutionState() { //dormant. needed for neknek
  if(!this->o_S0.isInitialized()) {
    this->o_S0 = platform->device.malloc<dfloat>(this->o_S.length());
    this->o_EXT0 = platform->device.malloc<dfloat>(this->o_EXT.length());
    this->o_ADV0 = platform->device.malloc<dfloat>(this->o_ADV.length());
    this->o_prop0 = platform->device.malloc<dfloat>(this->o_prop.length());
  }
}

void lvlSet_t::restoreSolutionState() {
  this->o_S0.copyTo(this->o_S, this->o_S.length());
  this->o_EXT0.copyTo(this->o_EXT, this->o_EXT.length());
  this->o_ADV0.copyTo(this->o_ADV, this->o_ADV.length());
  this->o_prop0.copyTo(this->o_prop, this->o_prop.length());
}

void lvlSet_t::lagSolution()
{
  const auto n = std::max(this->o_coeffEXT.size(), this->o_coeffBDF.size());
  for (int s = n; s > 1; s--) {
    o_S.copyFrom(o_S, this->_fieldOffset, (s - 1) * this->_fieldOffset, (s - 2) * this->_fieldOffset);
  }
}

void lvlSet_t::extrapolateSolution()
{
  if (!this->o_Se.isInitialized()) {
    return;
  }
  const auto Nlocal = this->_fieldOffset; // assumed to be the same for all fields
  launchKernel("core-extrapolate",
               Nlocal,
               1,       //only 1 field
               static_cast<int>(o_coeffEXT.size()),
               this->_fieldOffset,
               this->o_coeffEXT,
               this->o_S,
               this->o_Se);
}

void lvlSet_t::applyDirichlet(double time) 
{
  if (this->compute) {
    auto mesh = this->_mesh;

    // lower than any other possible Dirichlet value
    static constexpr dfloat TINY = -1e30;
    occa::memory o_SiDirichlet = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
    platform->linAlg->fill(o_SiDirichlet.size(), TINY, o_SiDirichlet);

    auto &neknek = platform->app->neknek;

    //TODO: check for neknek later
    auto o_intValU = [&]() {
      if (neknek) {
        if (neknek->hasField("fluid velocity")) {
          return neknek->getField("fluid velocity").o_intVal;
        }
      }
      return o_NULL;
    }();

    auto o_intVal = [&]() {
      if (neknek) {
        if (neknek->hasField("tlsr")) {
          return neknek->getField("tlsr").o_intVal;
        }
      }
      return o_NULL;
    }();

    for (int sweep = 0; sweep < 2; sweep++) {
      launchKernel("scalar_t::dirichletBC",
                   this->o_name,
                   mesh->Nelements,
                   this->_fieldOffset,
                   0,
                   time,
                   mesh->o_sgeo,
                   mesh->o_x,
                   mesh->o_y,
                   mesh->o_z,
                   mesh->o_vmapM,
                   mesh->o_EToB,
                   this->o_EToB,
                   this->o_W,
                   this->o_diff,
                   this->o_rho,
                   neknek ? neknek->intValOffset() : 0,
                   neknek ? neknek->o_pointMap() : o_NULL,
                   static_cast<int>(o_intValU.isInitialized()),
                   o_intValU,
                   o_intVal,
                   0,
                   platform->app->bc->o_usrwrk,
                   o_SiDirichlet);

      oogs::startFinish(o_SiDirichlet,
                        1,
                        this->_fieldOffset,
                        ogsDfloat,
                        (sweep == 0) ? ogsMax : ogsMin,
                        mesh->oogs);
    }

    if (this->o_Se.isInitialized()) {
      if (this->ellipticSolver[0]->Nmasked()) {
        launchKernel("core-maskCopy2",
                     ellipticSolver[0]->Nmasked(),
                     0,
                     0,
                     ellipticSolver[0]->o_maskIds(),
                     o_SiDirichlet,
                     this->o_S,
                     this->o_Se);
      }
    } else {
      if (this->ellipticSolver[0]->Nmasked()) {
        launchKernel("core-maskCopy",
                     this->ellipticSolver[0]->Nmasked(),
                     0,
                     0,
                     this->ellipticSolver[0]->o_maskIds(),
                     o_SiDirichlet,
                     this->o_S);
      }
    }
  }
}

void lvlSet_t::setupEllipticSolver()
{
  auto mesh = this->_mesh;
  
  if (this->compute) {
    auto o_lambda0 = this->o_diff;
    auto o_lambda1 = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
    platform->linAlg->axpby(mesh->Nlocal, *this->g0 / this->dt[0], this->o_rho, 0.0, o_lambda1);

    this->ellipticSolver[0] = new elliptic(this->name, mesh, this->_fieldOffset, o_lambda0, o_lambda1);

    if (platform->options.compareArgs(upperCase(this->name) + " REGULARIZATION METHOD", "SVV")) {
      this->ellipticSolver[0]->mueSVV(this->o_svvmu);
      this->ellipticSolver[0]->setupSVV();
    }
  }
}

void lvlSet_t::finalize() {
  if(this->ellipticSolver[0])
    delete this->ellipticSolver[0];
}

void lvlSet_t::mueSVV()
{
  auto mesh = this->meshV;

  static auto initialized = false;

  auto umagInitialized = false;

  auto o_umag = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);

  if(platform->options.compareArgs(upperCase(this->name) + " REGULARIZATION METHOD", "SVV")) {
    if(!initialized) {
      if(!platform->options.compareArgs("MOVING MESH","TRUE"))
        launchKernel("core-svv::svvMeshScale", mesh->Nelements, mesh->o_vgeo, this->o_svvf);
      initialized = true;
    }

    if(platform->options.compareArgs("MOVING MESH","TRUE"))
      launchKernel("core-svv::svvMeshScale", mesh->Nelements, mesh->o_vgeo, this->o_svvf);

    if(!umagInitialized) {
      platform->linAlg->magVector(mesh->Nlocal, this->vFieldOffset, this->o_W, o_umag); 
      umagInitialized = true;
    }

    dfloat scale = 0.1;
    platform->options.getArgs(upperCase(this->name) + " REGULARIZATION SVV SCALING COEFF", scale);

    platform->linAlg->axmyz(mesh->Nlocal, scale, this->o_svvf, o_umag, this->o_svvmu);
  }
}

void lvlSet_t::computeWrst()
{
  auto mesh = this->meshV;

  if (this->Nsubsteps) {
    for (int s = std::max(this->o_coeffBDF.size(), this->o_coeffEXT.size()); s > 1; s--) {
      auto lagOffset = mesh->dim * this->vCubatureOffset;
      this->o_relWrst.copyFrom(this->o_relWrst, lagOffset, (s - 1) * lagOffset, (s - 2) * lagOffset);
    }
  }

  const auto relative = this->geom && this->Nsubsteps; // const auto relative = geom && Nsubsteps;
  if (platform->options.compareArgs("ADVECTION TYPE", "CUBATURE")) {
    launchKernel("nrs-UrstCubatureHex3D",
                 mesh->Nelements,
                 static_cast<int>(relative),
                 mesh->o_cubvgeo,
                 mesh->o_cubInterpT,
                 this->vFieldOffset,
                 (this->geom) ? this->geom->fieldOffset : 0,
                 this->vCubatureOffset,
                 this->o_W,
                 (this->geom) ? this->geom->o_U : o_NULL,
                 this->o_relWrst);
  } else {
    launchKernel("nrs-UrstHex3D",
                 mesh->Nelements,
                 static_cast<int>(relative),
                 mesh->o_vgeo,
                 this->vFieldOffset,
                 (this->geom) ? this->geom->fieldOffset : 0,
                 this->o_W,
                 (this->geom) ? this->geom->o_U : o_NULL,
                 this->o_relWrst);
  }
}

void lvlSet_t::mueAVM()
{
  auto verbose = platform->verbose();
  auto mesh = this->meshV; // assumes mesh is the same for all scalars
  static occa::memory o_diff0;

  static occa::memory o_nuAVM;
  static auto initialized = false;

  auto parPrefix = upperCase(this->name);

  if (!initialized) {
    if (platform->options.compareArgs(parPrefix + " REGULARIZATION METHOD", "AVM_AVERAGED_MODAL_DECAY")) {
      nekrsCheck(mesh->N < 5,
          platform->comm.mpiComm(),
          EXIT_FAILURE,
          "%s\n",
          "AVM requires polynomialOrder >= 5!");

      o_diff0 = platform->device.malloc<dfloat>(mesh->Nlocal);
      o_diff0.copyFrom(this->o_diff, mesh->Nlocal);
    }
    initialized = true;
  }

  if (platform->options.compareArgs(parPrefix + " REGULARIZATION METHOD", "AVM_AVERAGED_MODAL_DECAY")) {
    // restore inital viscosity
    this->o_diff.copyFrom(o_diff0, mesh->Nlocal);

    dfloat kappa = 1.0;
    platform->options.getArgs(parPrefix + " REGULARIZATION AVM ACTIVATION WIDTH", kappa);

    dfloat logS0 = 2.0; // threshold smoothness exponent (activate for logSk > logS0 - kappa)
    platform->options.getArgs(parPrefix + " REGULARIZATION AVM DECAY THRESHOLD", logS0);

    dfloat scalingCoeff = 1.0;
    platform->options.getArgs(parPrefix + " REGULARIZATION AVM SCALING COEFF", scalingCoeff);

    dfloat absTol = 0;
    platform->options.getArgs(parPrefix + " REGULARIZATION AVM ABSOLUTE TOL", absTol);

    const bool makeCont = platform->options.compareArgs(parPrefix + " REGULARIZATION AVM C0", "TRUE");

    auto o_eps = avm::viscosity(vFieldOffset, this->o_W, this->o_S, absTol, scalingCoeff, logS0, kappa, makeCont);

    if (verbose) {
      const dfloat maxEps = platform->linAlg->max(mesh->Nlocal, o_eps, platform->comm.mpiComm());
      const dfloat minEps = platform->linAlg->min(mesh->Nlocal, o_eps, platform->comm.mpiComm());

      const dfloat maxDiff = platform->linAlg->max(mesh->Nlocal, this->o_diff, platform->comm.mpiComm());
      const dfloat minDiff = platform->linAlg->min(mesh->Nlocal, this->o_diff, platform->comm.mpiComm());

      if (platform->comm.mpiRank() == 0) {
        printf("applying a min/max artificial viscosity of (%f,%f) to %s with min/max visc (%f,%f)\n",
               minEps,
               maxEps,
               this->name.c_str(),
               minDiff,
               maxDiff);
      }
    }

    platform->linAlg->axpby(mesh->Nlocal, 1.0, o_eps, 1.0, this->o_diff, 0, 0);

    if (verbose) {
      const dfloat maxDiff = platform->linAlg->max(mesh->Nlocal, this->o_diff, platform->comm.mpiComm());
      const dfloat minDiff = platform->linAlg->min(mesh->Nlocal, this->o_diff, platform->comm.mpiComm());

      if (platform->comm.mpiRank() == 0) {
        printf("%s now has a min/max visc: (%f,%f)\n", this->name.c_str(), minDiff, maxDiff);
      }
    }
  }
}

void lvlSet_t::writeFile(double time, int tstep)
{
  if(platform->options.compareArgs(upperCase(this->name) + " CHECKPOINTING", "TRUE")) {
    if(!this->fieldWriter) {
      this->fieldWriter = iofldFactory::create();

      this->fieldWriter->open(this->meshV, iofld::mode::write, this->name);

      if (platform->options.compareArgs("CHECKPOINT PRECISION", "FP32")) {
        this->fieldWriter->writeAttribute("precision", "32");
      } else {
        this->fieldWriter->writeAttribute("precision", "64");
      }

      auto o_Si = this->o_S.slice(this->fieldOffsetScan, this->meshV->Nlocal);
      this->fieldWriter->addVariable("scalar00", o_Si);

      int N;
      platform->options.getArgs("POLYNOMIAL DEGREE", N);
      this->fieldWriter->writeAttribute("polynomialOrder", std::to_string(N));
    }

    if(tstep == stepsMax) {
      this->fieldWriter->writeAttribute("outputmesh", (!outfldCounter) ? "true" : "false");
      this->fieldWriter->addVariable("time", const_cast<double &>(time));
      this->fieldWriter->process();
      this->outfldCounter++;
    }
  }
}

void lvlSet_t::printStepInfo(double time, int tstep, bool printStepInfo, bool solverInfo)
{
  const auto cfl = nrs->computeCFL(this->meshV, this->o_W, this->dt[0]);

  auto printSolverInfo = [tstep](elliptic *solver, const std::string &name) {
    if (!solver) {
      return;
    }
    const auto [prevProjVecs, nProjVecs] = solver->projectionCounters();
    if (nProjVecs > 0) {
      if (prevProjVecs > 0) {
        printf("Pseudo-step=%-8d %-20s: resNorm0 %.2e  resNorm %.2e  ratio = %.3e  %d/%d\n",
               tstep,
               std::string("proj " + name).c_str(),
               solver->initialResidual(),
               solver->initialGuessResidual(),
               solver->initialResidual() / solver->initialGuessResidual(),
               prevProjVecs,
               nProjVecs);
      }
    }
    printf("Pseudo-step=%-8d %-20s: iter %03d  resNorm0 %.2e  resNorm %.2e\n",
           tstep,
           name.c_str(),
           solver->Niter(),
           solver->initialGuessResidual(),
           solver->finalResidual());
  };

  if (platform->comm.mpiRank() == 0) {
    if (solverInfo) {
      if (this->compute) {
        printSolverInfo(this->ellipticSolver.at(0), upperCase(this->name));
      }
    }

    if (printStepInfo) {
      printf("Pseudo-step=%-8d tau= %.8e  dtau=%.1e  CFL= %.3f\n", tstep, time, this->dt[0], cfl);
    }

    if (tstep == stepsMax) {
      printf("Pseudo-step=%-8d elapsedTime= %.2es  elapsedTimePerStep= %.5es\n", tstep, elapsedTime, elapsedTime/tstep);
    }
  }
}
