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

static void printOccaArray(const occa::memory &o_mem, const std::string &name, size_t maxPrint = 0)
{
  const size_t N = o_mem.size();                  // element count (dtype entries)
  const size_t elemBytes = o_mem.dtype().bytes(); // bytes per entry
  const size_t bytes = N * elemBytes;

  if (elemBytes != sizeof(dfloat)) {
    std::cout << name << ": dtype bytes=" << elemBytes
              << " but sizeof(dfloat)=" << sizeof(dfloat) << "\n";
    return;
  }

  std::vector<dfloat> h(N);

  // In datatype-aware OCCA, this is element count, not bytes.
  o_mem.copyTo(h.data(), N);

  std::cout << name << " (N=" << N << ", bytes=" << bytes << ")\n";
  const size_t nOut = (maxPrint > 0) ? std::min(maxPrint, N) : N;

  for (size_t i = 0; i < nOut; ++i)
    std::cout << "  [" << i << "] " << h[i] << "\n";

  if (nOut < N)
    std::cout << "  ... (" << (N - nOut) << " more entries)\n";
}

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
  {"freqTLSR"},
  {"freqCLSR"},
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

  bool tlsExists = false;
  for (auto const &sec : sections) {
    if(sec.first.find("scalar tls") != std::string::npos) {
      tlsExists = true;
    }
  }

  if(!tlsExists) {
    std::ostringstream error;
    error << "mandatory section for Level-Set [SCALAR TLS] not found!\n";
    append_error(error.str());
  }

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

  //TODO: remove this
  options.setArgs("CLSR SOLVER","NONE");

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

    auto sIndex = nrs->scalar->nameToIndex.find("tls")->second;
    for (int id = 0; id < map.size(); id++) {
      map[id] = nek::bcmap(id + 1, sIndex + 2, 0);
      map[id] = bdryBase::bcType_zeroNeumann;      //all zero flux BCs
    }
    platform->app->bc->setBcMap("tlsr", false, map);
  }

  // currently just holds TLSR solver --> TODO: add in CLSR solver (separate object or hold both in one LS object?)
  tlsr = [&]() {
    lvlSetConfig_t cfg;
    cfg.g0 = &nrs->g0;
    cfg.dt = nrs->dt;
    cfg.fieldOffset = nrs->meshV->fieldOffset;
    cfg.vFieldOffset = nrs->scalar->vFieldOffset;       // TODO: is it safe to assume nrs->scalar is always accessible?
    cfg.vCubatureOffset = nrs->scalar->vCubatureOffset; // TODO: is it safe to assume nrs->scalar is always accessible?
    cfg.mesh.resize(1); // currently hard-coded for one equation (e.g. TLSR) --> TODO: handle TLSR and CLSR
    cfg.mesh[0] = nrs->meshV;
    cfg.meshV = nrs->meshV;
    return std::make_unique<lvlSet_t>(cfg);
  }();
  tlsr->mueSVV(); // needs to be called before setupEllipticSolver() otherwise o_svvmue in elliptic solver will not be initialized --> TODO: handle AVM?
  tlsr->setupEllipticSolver();
}

void lvlSet::solveLSR()
{
  auto mesh = tlsr->meshV;

  double time = 0.0;
  int outerIter = 1;
  int outerIterMax = 1000;
  platform->options.getArgs("NUMBER TIMESTEPS",outerIterMax);
  int innerIterMax = 100;

  // set constant dt --> TODO: need to change this if we want variable dt
  tlsr->dt[1] = tlsr->dt[0];
  tlsr->dt[2] = tlsr->dt[0];

  // set the TLSR initial condition -- currently just copy the scalar S00 --> TODO: handle the correct initial condition
  if (tlsr->fieldOffsetSum != nrs->scalar->fieldOffsetSum) {
    throw std::runtime_error("LS::solveLSR: tlsr and nrs->scalar fieldOffsetSum are not equal.");
  }
  tlsr->o_S.copyFrom(nrs->scalar->o_S);

  while(outerIter <= outerIterMax) {
    std::cout << "ITER: " << outerIter << std::endl;
    time += tlsr->dt[0];
    tlsr->setTimeIntegrationCoeffs(outerIter);
    tlsr->extrapolateSolution();
    if (tlsr->anyEllipticSolver) { platform->linAlg->fill(tlsr->fieldOffsetSum, 0.0, tlsr->o_EXT); }
    tlsr->computeAdvectionCoeff();
    tlsr->computeWrst();
    tlsr->makeAdvection(0, time, outerIter); // currently assume 1 LS equation -->  is = 1
    tlsr->makeExplicit(0, time, outerIter);
    tlsr->makeForcing();
    tlsr->mueSVV();
    tlsr->lagSolution();
    // tlsr->applyDirichlet(time);
    tlsr->solve(time, 1);
    tlsr->writeFile(time);
    outerIter += 1;
  }

  // copy the TLSR solution back to the scalar S00
  //nrs->scalar->o_S.copyFrom(tlsr->o_S);
}


lvlSet_t* lvlSet::getLS()
{
  return tlsr.get();
}

lvlSet_t::lvlSet_t(lvlSetConfig_t &cfg)
{
  if (platform->comm.mpiRank() == 0) {
    std::cout << "================ " << "SETUP LEVEL-SET" << " ===============\n";
  }

  auto &options = platform->options;
  platform_t *platform = platform_t::getInstance();

  Nsubsteps = 0;
  platform->options.getArgs("SUBCYCLING STEPS", Nsubsteps);

  fieldOffsetScan.resize(1);
  ellipticSolver.resize(1);
  compute.resize(1);

  meshV = cfg.meshV;

  g0 = cfg.g0;
  dt = cfg.dt;

  int nBDF;
  int nEXT;
  platform->options.getArgs("BDF ORDER", nBDF);
  platform->options.getArgs("EXT ORDER", nEXT);
  if (Nsubsteps) {
    nEXT = nBDF;
    platform->options.setArgs("EXT ORDER", std::to_string(nEXT));
  }
  nekrsCheck(nEXT < nBDF,
             platform->comm.mpiComm(),
             EXIT_FAILURE,
             "%s\n",
             "EXT order needs to be >= BDF order!");

  printf("here\n");
  o_coeffBDF = platform->device.malloc<dfloat>(nBDF);
  o_coeffEXT = platform->device.malloc<dfloat>(nEXT);

  this->_fieldOffset = cfg.fieldOffset; // for now same for all scalars
  this->vFieldOffset = cfg.vFieldOffset; // TODO: check if this is correct
  this->vCubatureOffset = cfg.vCubatureOffset;

  dlong sum = 0;
  for (int s = 0; s < 1; ++s) {
    fieldOffsetScan[s] = (s > 0) ? sum : 0;
    sum += this->_fieldOffset;
    this->_mesh.push_back(cfg.mesh[s]);
  }
  fieldOffsetSum = sum;
  o_fieldOffsetScan = platform->device.malloc<dlong>(1, fieldOffsetScan.data());

  o_prop = platform->device.malloc<dfloat>(2 * fieldOffsetSum);
  o_diff = o_prop.slice(0 * fieldOffsetSum, fieldOffsetSum);
  o_rho = o_prop.slice(1 * fieldOffsetSum, fieldOffsetSum);
  
  printf("here\n");
  for (int is = 0; is < 1; is++) {
    const std::string sid = scalarDigitStr(is);

    const auto _name = "tlsr" ;//lowerCase(options.getArgs("LS" + sid + " NAME"));
    name.push_back(_name);
    nameToIndex[_name] = is;

    auto o_tmp = [&]() {
      const std::string prefixedName = _name;
      auto tmp = platform->device.malloc<char>(prefixedName.size() + 1);
      tmp.copyFrom(prefixedName.data());
      const char nullChar[] = {'\0'};
      tmp.copyFrom(nullChar, 1, prefixedName.size());
      return tmp;
    }();
    o_name.push_back(o_tmp);

    if (options.compareArgs("TLSR SOLVER", "NONE")) {
      continue;
    }

    nekrsCheck(options.compareArgs("TLSR SOLVER", "BLOCK"),
               platform->comm.mpiComm(),
               EXIT_FAILURE,
               "%s\n",
               "level-set does not support BLOCK solver!");

    if (platform->comm.mpiRank() == 0) {
      std::cout << "LS" << sid << ": " << name[is] << std::endl;
    }
    platform->app->bc->printBcTypeMapping("tlsr");
    if (platform->comm.mpiRank() == 0) {
      std::cout << std::endl;
    }

    dfloat diff = 1;
    dfloat rho = 1;
    options.getArgs("TLSR DIFFUSIONCOEFF", diff);
    options.getArgs("TLSR TRANSPORTCOEFF", rho);

    auto o_diff_i = o_diff + fieldOffsetScan[is];
    auto o_rho_i = o_rho + fieldOffsetScan[is];

    std::vector<dfloat> diffTmp(this->_mesh[is]->Nlocal, diff);
    std::vector<dfloat> rhoTmp(this->_mesh[is]->Nlocal, rho);

    /* dfloat diffSolid = diff; */
    /* dfloat rhoSolid = rho; */
    /* options.getArgs("LS" + sid + " DIFFUSIONCOEFF SOLID", diffSolid); */
    /* options.getArgs("LS" + sid + " TRANSPORTCOEFF SOLID", rhoSolid); */
    /* for (int i = meshV->Nlocal; i < this->_mesh[is]->Nlocal; i++) { */
    /*   diffTmp[i] = diffSolid; */
    /*   rhoTmp[i] = rhoSolid; */
    /* } */

    o_diff_i.copyFrom(diffTmp.data(), diffTmp.size());
    o_rho_i.copyFrom(rhoTmp.data(), rhoTmp.size());
  }

  printf("here3\n");
  anyEllipticSolver = false;

  EToBOffset = [&]() {
    dlong NelementsMax = 0;
    for (int is = 0; is < 1; is++) {
      NelementsMax = std::max(this->_mesh[is]->Nelements, NelementsMax);
    }
    return NelementsMax * meshV->Nfaces;
  }();

  std::vector<int> EToB(EToBOffset);

  for (int is = 0; is < 1; is++) {
    std::string sid = scalarDigitStr(is);

    compute[is] = 1;
    if (options.compareArgs("TLSR SOLVER", "NONE")) {
      compute[is] = 0;
      continue;
    }

    anyEllipticSolver |= (compute[is]);

    auto mesh = this->_mesh[is];
    int cnt = 0;
    for (int e = 0; e < mesh->Nelements; e++) {
      for (int f = 0; f < mesh->Nfaces; f++) {
        EToB[cnt + EToBOffset * is] =
            platform->app->bc->typeId(mesh->EToB[f + e * mesh->Nfaces], "tlsr");
        cnt++;
      }
    }
  }

  o_EToB = platform->device.malloc<int>(EToB.size());
  o_EToB.copyFrom(EToB.data());

  o_compute = platform->device.malloc<dlong>(1, compute.data());

  int nFieldsAlloc = anyEllipticSolver ? std::max(o_coeffBDF.size(), o_coeffEXT.size()) : 1;
  o_S = platform->device.malloc<dfloat>(nFieldsAlloc * fieldOffsetSum);
  o_W = platform->device.malloc<dfloat>(meshV->dim * std::max(o_coeffBDF.size(), o_coeffEXT.size()) * this->vFieldOffset);
  const dlong Nstates = Nsubsteps ? std::max(o_coeffBDF.size(), o_coeffEXT.size()) : 1;
  o_relWrst = platform->device.malloc<dfloat>(Nstates * meshV->dim * this->vCubatureOffset);

  o_signls = platform->device.malloc<dfloat>(nFieldsAlloc * fieldOffsetSum);

  nFieldsAlloc = anyEllipticSolver ? o_coeffEXT.size() : 1;
  o_ADV = platform->device.malloc<dfloat>(nFieldsAlloc * fieldOffsetSum);
  o_EXT = platform->device.malloc<dfloat>(nFieldsAlloc * fieldOffsetSum);

  if (anyEllipticSolver) {
    o_Se = platform->device.malloc<dfloat>(fieldOffsetSum);
    o_JwF = platform->device.malloc<dfloat>(fieldOffsetSum);
  }

  // TODO consider adding these options in later
  bool filteringEnabled = false;
  bool avmEnabled = false;

  printf("here\n");

  auto verifyBC = [&]() {
    for (int is = 0; is < 1; is++) {
      if (!compute[is]) {
        continue;
      }

      const std::string field = "tlsr";//"ls" + scalarDigitStr(is);
      nekrsCheck(_mesh[is]->Nbid != platform->app->bc->size(field),
                 platform->comm.mpiComm(),
                 EXIT_FAILURE,
                 "Size of %s boundaryTypeMap (%d) does not match number of boundary IDs in mesh (%d)!\n",
                 field.c_str(),
                 platform->app->bc->size(field),
                 _mesh[is]->Nbid);
    }
  };

  verifyBC();
}

void lvlSet_t::setTimeIntegrationCoeffs(int tstep)
{
  const auto bdfOrder = std::min(tstep, static_cast<int>(o_coeffBDF.size()));
  const auto extOrder = std::min(tstep, static_cast<int>(o_coeffEXT.size()));

  {
    std::vector<dfloat> coeff(o_coeffBDF.size());
    nek::bdfCoeff(g0, coeff.data(), dt, bdfOrder);
    for (int i = coeff.size(); i > bdfOrder; i--) {
      coeff[i - 1] = 0;
    }
    o_coeffBDF.copyFrom(coeff.data());
  }

  {
    std::vector<dfloat> coeff(o_coeffEXT.size());
    nek::extCoeff(coeff.data(), dt, extOrder, bdfOrder);
    for (int i = coeff.size(); i > extOrder; i--) {
      coeff[i - 1] = 0;
    }
    o_coeffEXT.copyFrom(coeff.data());
  }
}

void lvlSet_t::computeAdvectionCoeff()
{
  // a. compute interface normals
  launchKernel("core-gradientVolumeHex3D",
               meshV->Nelements,
               meshV->o_vgeo,
               meshV->o_D,
               this->vFieldOffset,
               o_solution("tlsr"),
               o_W);
  oogs::startFinish(o_W, meshV->dim, this->vFieldOffset, ogsDfloat, ogsAdd, meshV->oogs);
  platform->linAlg->axmyVector(meshV->Nlocal, this->vFieldOffset, 0, 1.0, meshV->o_invLMM, o_W);
  // b. compute sign function
  signlsKernel(meshV->Nlocal, o_solution("tlsr"), o_signls);
  // c. compute w = sign(phi) * n
  normalVectorKernel(meshV->Nlocal, this->vFieldOffset, o_signls, o_W);
}

void lvlSet_t::makeAdvection(int is, double time, int tstep)
{
  if (Nsubsteps) {
    advectionSubcycling(std::min(tstep, static_cast<int>(o_coeffEXT.size())), time, is);
  } else {
    auto mesh = meshV;

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
                   o_compute + is,
                   o_fieldOffsetScan + is,
                   this->vFieldOffset,
                   this->vCubatureOffset,
                   o_S,
                   o_relWrst,   // --> TODO change to w
                   o_rho, // --> set to 1
                   o_ADV);
    } else {
      launchKernel("core-strongAdvectionVolumeScalarHex3D",
                   mesh->Nelements,
                   1, /* nScalars */
                   0, /* weighted */
                   mesh->o_vgeo,
                   mesh->o_D,
                   o_compute + is,
                   o_fieldOffsetScan + is,
                   this->vFieldOffset,
                   o_S,
                   o_relWrst, // --> TODO change to w
                   o_rho, // --> set to 1
                   o_ADV);
    }
  }
}

void lvlSet_t::advectionSubcycling(int nEXT, double time, int is)
{
  const auto mesh = this->_mesh[is];

  const auto nFields = 1;
  
  auto o_Si = o_S.slice(fieldOffsetScan[is], mesh->Nlocal);
  auto o_JwFi = o_JwF.slice(fieldOffsetScan[is], mesh->Nlocal);

  static occa::kernel kernel;
  if (!kernel.isInitialized()) {
    if (platform->options.compareArgs("ADVECTION TYPE", "CUBATURE")) {
      kernel = platform->kernelRequests.load("core-subCycleStrongCubatureVolumeScalarHex3D");
    } else {
      kernel = platform->kernelRequests.load("core-subCycleStrongVolumeScalarHex3D");
    }
  }

  platform->linAlg->fill(o_JwFi.size(), 0, o_JwFi);

  advectionSubcyclingRK(mesh,
                        meshV,
                        time,
                        dt,
                        Nsubsteps,
                        o_coeffBDF,
                        nEXT,
                        nFields,
                        kernel,
                        meshV->oogs,
                        mesh->fieldOffset,
                        this->_fieldOffset,
                        this->vCubatureOffset,
                        fieldOffsetSum,
                        o_NULL, // (geom) ? geom->o_div : o_NULL,
                        o_W,
                        o_Si,
                        o_JwFi);

  if (platform->verbose()) {
    const dfloat debugNorm = platform->linAlg->weightedNorm2Many(mesh->Nlocal,
                                                                 1,
                                                                 0,
                                                                 mesh->ogs->o_invDegree,
                                                                 o_JwFi,
                                                                 platform->comm.mpiComm());
    if (platform->comm.mpiRank() == 0) {
      printf("%s%s advSub norm: %.15e\n", "tlsr", scalarDigitStr(is).c_str(), debugNorm);
    }
  }
}

void lvlSet_t::makeExplicit(int is, double time, int tstep)
{
  const std::string sid = scalarDigitStr(is);

  auto mesh = this->_mesh[is];
  const dlong isOffset = fieldOffsetScan[is];

  o_explicitTerms("tlsr").copyFrom(o_signls); // TODO: shouldn't use tlsr here but will use this hack for now
}

void lvlSet_t::makeForcing()
{
  for (int is = 0; is < 1; is++) {
    if (!compute[is]) {
      continue;
    }

    launchKernel("scalar_t::sumMakef",
                 _mesh[is]->Nlocal,
                 _mesh[is]->o_LMM,
                 1 / dt[0],
                 o_coeffEXT,
                 o_coeffBDF,
                 fieldOffsetScan[is],
                 fieldOffsetSum,
                 _mesh[is]->fieldOffset,
                 o_rho,
                 o_S,
                 o_ADV,
                 o_EXT,
                 o_JwF);
  }

  const auto n = std::max(o_coeffEXT.size(), o_coeffBDF.size());
  for (int s = n; s > 1; s--) {
    o_EXT.copyFrom(o_EXT, fieldOffsetSum, (s - 1) * fieldOffsetSum, (s - 2) * fieldOffsetSum);
    if (o_ADV.isInitialized()) {
      o_ADV.copyFrom(o_ADV, fieldOffsetSum, (s - 1) * fieldOffsetSum, (s - 2) * fieldOffsetSum);
    }
  }
}

void lvlSet_t::solve(double time, int stage)
{
  for (int is = 0; is < 1; is++) {
    if (!compute[is]) {
      continue;
    }

    const std::string sid = scalarDigitStr(is);
    auto mesh = this->_mesh[is];

    auto o_rhs = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
    o_rhs.copyFrom(o_JwF, mesh->Nlocal, 0, fieldOffsetScan[is]);

    auto o_lhs = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);

    launchKernel("scalar_t::neumannBCHex3D",
                 o_name[is],
                 mesh->Nelements,
                 1,
                 mesh->o_sgeo,
                 mesh->o_vmapM,
                 mesh->o_EToB,
                 is,
                 time,
                 this->vFieldOffset,
                 this->_fieldOffset,
                 0,
                 EToBOffset,
                 mesh->o_x,
                 mesh->o_y,
                 mesh->o_z,
                 o_W, // changed from o_Ue --> TODO: think about this
                 o_S,
                 o_EToB,
                 o_diff,
                 o_rho,
                 platform->app->bc->o_usrwrk,
                 o_lhs,
                 o_rhs);

    const auto o_diff_i = o_diff.slice(fieldOffsetScan[is], mesh->Nlocal);

    const auto o_lambda0 = o_diff_i;
    const auto o_lambda1 = [&]() {
      const auto o_rho_i = o_rho.slice(fieldOffsetScan[is], mesh->Nlocal);
      auto o_l = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
      platform->linAlg->axpby(mesh->Nlocal, *g0 / dt[0], o_rho_i, 0.0, o_l);

      if (userImplicitLinearTerm) {
        auto o_implicitLT = userImplicitLinearTerm(time, is);
        if (o_implicitLT.isInitialized()) {
          platform->linAlg->axpby(mesh->Nlocal, 1.0, o_implicitLT, 1.0, o_l);
        }
      }

      //if (platform->app->bc->hasRobin("SCALAR" + sid)) {
      //  platform->linAlg->axpby(mesh->Nlocal, 1.0, o_lhs, 1.0, o_l);
      //}

      return o_l;
    }();

    auto o_Si = [&]() {
      auto o_S0 = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
      if (platform->options.compareArgs("TLSR INITIAL GUESS", "EXTRAPOLATION") && stage == 1) {
        o_S0.copyFrom(o_Se, o_S0.size(), 0, fieldOffsetScan[is]);
      } else {
        o_S0.copyFrom(o_S, o_S0.size(), 0, fieldOffsetScan[is]);
      }

      return o_S0;
    }();

    this->ellipticSolver[is]->coeff0HLM(o_lambda0);
    this->ellipticSolver[is]->coeff1HLM(o_lambda1);
    this->ellipticSolver[is]->solve(o_rhs, o_Si);
    o_Si.copyTo(o_S, o_Si.size(), fieldOffsetScan[is]);
  }
}

void lvlSet_t::saveSolutionState() {}

void lvlSet_t::restoreSolutionState() {}

void lvlSet_t::lagSolution()
{
  if (!anyEllipticSolver) {
    return;
  }

  const auto n = std::max(o_coeffEXT.size(), o_coeffBDF.size());
  for (int s = n; s > 1; s--) {
    o_S.copyFrom(o_S, fieldOffsetSum, (s - 1) * fieldOffsetSum, (s - 2) * fieldOffsetSum);
  }
}

void lvlSet_t::extrapolateSolution()
{
  if (!o_Se.isInitialized()) {
    return;
  }
  const auto Nlocal = this->_fieldOffset; // assumed to be the same for all fields
  launchKernel("core-extrapolate",
               Nlocal,
               1,       //only 1 field
               static_cast<int>(o_coeffEXT.size()),
               this->_fieldOffset,
               o_coeffEXT,
               o_S,
               o_Se);
}

void lvlSet_t::applyDirichlet(double time) {}

void lvlSet_t::setupEllipticSolver()
{
  for (int is = 0; is < 1; is++) {
    std::string sid = scalarDigitStr(is);

    if (!compute[is]) {
      continue;
    }

    auto o_rho_i = o_rho.slice(fieldOffsetScan[is], _mesh[is]->Nlocal);
    auto o_lambda0 = o_diff.slice(fieldOffsetScan[is], _mesh[is]->Nlocal);
    auto o_lambda1 = platform->deviceMemoryPool.reserve<dfloat>(_mesh[is]->Nlocal);
    platform->linAlg->axpby(_mesh[is]->Nlocal, *g0 / dt[0], o_rho_i, 0.0, o_lambda1);

    ellipticSolver[is] = new elliptic("tlsr", _mesh[is], this->_fieldOffset, o_lambda0, o_lambda1);

    if (platform->options.compareArgs("TLSR REGULARIZATION METHOD", "SVV")) {
      auto o_svvmu = this->o_svvmu.slice(is * this->_fieldOffset, this->_fieldOffset);
      ellipticSolver[is]->mueSVV(o_svvmu);
      ellipticSolver[is]->setupSVV();
    }
  }
}

void lvlSet_t::finalize() {}

void lvlSet_t::mueSVV()
{
  auto mesh = this->meshV;

  static auto initialized = false;

  auto umagInitialized = false;

  auto o_umag = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);

  for (int is = 0; is < 1; is++) {
    const auto sid = scalarDigitStr(is);

    if(platform->options.compareArgs("TLSR REGULARIZATION METHOD", "SVV")) {
      if(!initialized) {
        this->o_svvf = platform->device.malloc<dfloat>(this->_fieldOffset);
        this->o_svvmu = platform->device.malloc<dfloat>(this->_fieldOffset);

        if(!platform->options.compareArgs("MOVING MESH","TRUE"))
          launchKernel("core-svv::svvMeshScale", mesh->Nelements, mesh->o_vgeo, this->o_svvf);

        initialized = true;
      }

      if(platform->options.compareArgs("MOVING MESH","TRUE"))
        launchKernel("core-svv::svvMeshScale", mesh->Nelements, mesh->o_vgeo, this->o_svvf);

      if(!umagInitialized) {
        platform->linAlg->magVector(mesh->Nlocal, this->vFieldOffset, o_W, o_umag); // changed o_U to o_W --> TODO: think if this is the right thing to do
        umagInitialized = true;
      }

      dfloat scale = 0.1;
      platform->options.getArgs("TLSR REGULARIZATION SVV SCALING COEFF", scale);

      auto o_svvmu = this->o_svvmu.slice(is * this->_fieldOffset, this->_fieldOffset);
      platform->linAlg->axmyz(mesh->Nlocal, scale, this->o_svvf, o_umag, o_svvmu);
    }
  }
}

void lvlSet_t::computeWrst()
{
  auto mesh = meshV;

  if (Nsubsteps) {
    for (int s = std::max(o_coeffBDF.size(), o_coeffEXT.size()); s > 1; s--) {
      auto lagOffset = mesh->dim * this->vCubatureOffset;
      o_relWrst.copyFrom(o_relWrst, lagOffset, (s - 1) * lagOffset, (s - 2) * lagOffset);
    }
  }

  const auto relative = Nsubsteps; // const auto relative = geom && Nsubsteps;
  if (platform->options.compareArgs("ADVECTION TYPE", "CUBATURE")) {
    launchKernel("nrs-UrstCubatureHex3D",
                 mesh->Nelements,
                 relative,
                 mesh->o_cubvgeo,
                 mesh->o_cubInterpT,
                 this->vFieldOffset,
                 0, // (geom) ? geom->fieldOffset : 0,
                 this->vCubatureOffset,
                 o_W,
                 o_NULL, // (geom) ? geom->o_U : o_NULL,
                 o_relWrst);
  } else {
    launchKernel("nrs-UrstHex3D",
                 mesh->Nelements,
                 relative,
                 mesh->o_vgeo,
                 this->vFieldOffset,
                 0, // (geom) ? geom->fieldOffset : 0,
                 o_W,
                 o_NULL, // (geom) ? geom->o_U : o_NULL,
                 o_relWrst);
  }
}

void lvlSet_t::writeFile(double time)
{
  if(!fieldWriter) {
    fieldWriter = iofldFactory::create();

    fieldWriter->open(meshV, iofld::mode::write, "tlsr");

    if (platform->options.compareArgs("CHECKPOINT PRECISION", "FP32")) {
      fieldWriter->writeAttribute("precision", "32");
    } else {
      fieldWriter->writeAttribute("precision", "64");
    }

    auto o_Si = o_S.slice(fieldOffsetScan[0], meshV->Nlocal);
    fieldWriter->addVariable("scalar00", o_Si);
  }

  fieldWriter->writeAttribute("outputmesh", (!outfldCounter) ? "true" : "false");
  fieldWriter->addVariable("time", const_cast<double &>(time));
  fieldWriter->process();
  outfldCounter++;
}
