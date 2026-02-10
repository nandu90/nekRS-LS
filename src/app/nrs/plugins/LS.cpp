#include "app.hpp"
#include "nrs.hpp"
#include "platform.hpp"
#include "LS.hpp"
#include "linAlg.hpp"
#include "solver.hpp"
#include <stdexcept>

// private members
namespace
{
nrs_t *nrs;
occa::kernel signlsKernel;
occa::kernel normalVectorKernel;
bool buildKernelCalled = false;
std::unique_ptr<ls_t> tlsr = nullptr;
occa::memory o_coeffEXT, o_coeffBDF;
int advectionSubcycingSteps = 0;
};

void LS::buildKernel(occa::properties _kernelInfo)
{
  std::cout << "ENTERING LS::buildKernel()...\n";

  occa::properties kernelInfo;
  kernelInfo += _kernelInfo;

  auto buildKernel = [&kernelInfo](const std::string &kernelName) {
    const auto path = getenv("NEKRS_KERNEL_DIR") + std::string("/app/nrs/plugins/");
    const auto fileName = path + "LS.okl";
    const auto reqName = "LS::";
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

  std::cout << "EXITING LS::buildKernel()...\n";
}

void LS::updateSourceTerms()
{
  std::cout << "ENTERING LS::updateSourceTerms()...\n";

  auto mesh = nrs->meshV;

  // compute interface normals - currently we store this in nrs->fluid->o_U
  launchKernel("core-gradientVolumeHex3D",
               mesh->Nelements,
               mesh->o_vgeo,
               mesh->o_D,
               nrs->fluid->fieldOffset,
               nrs->scalar->o_solution("phi"),
               nrs->fluid->o_U);
  oogs::startFinish(nrs->fluid->o_U, mesh->dim, nrs->fluid->fieldOffset, ogsDfloat, ogsAdd, mesh->oogs);
  platform->linAlg->axmyVector(mesh->Nlocal, nrs->fluid->fieldOffset, 0, 1.0, mesh->o_invLMM, nrs->fluid->o_U);

  // compute sign function
  auto o_signls = platform->deviceMemoryPool.reserve<dfloat>(nrs->fluid->fieldOffset);
  signlsKernel(mesh->Nlocal, nrs->scalar->o_solution("phi"), o_signls);

  normalVectorKernel(mesh->Nlocal, nrs->fluid->fieldOffset, o_signls, nrs->fluid->o_U);

  // update source term
  nrs->scalar->o_explicitTerms("phi").copyFrom(o_signls);

  std::cout << "EXITING LS::updateSourceTerms()...\n";
}

void LS::setup()
{
  std::cout << "ENTERING LS::setup()...\n";

  static bool isInitialized = false;
  if (isInitialized) {
    return;
  }
  isInitialized = true;

  // we set these here for now --> TODO add LEVELSET section in par file
  std::string lid = "00";
  platform->options.setArgs("LS" + lid + " CHECKPOINTING", upperCase("false"));
  platform->options.setArgs("LS" + lid + " DIFFUSIONCOEFF", to_string_f(1.0e-14));
  platform->options.setArgs("LS" + lid + " ELLIPTIC COEFF FIELD", upperCase("true"));
  platform->options.setArgs("LS" + lid + " ELLIPTIC PRECO COEFF FIELD", upperCase("true"));
  platform->options.setArgs("LS" + lid + " INITIAL GUESS", "EXTRAPOLATION");
  platform->options.setArgs("LS" + lid + " MESH", "FLUID");
  platform->options.setArgs("LS" + lid + " NAME", "TLSR");
  platform->options.setArgs("LS" + lid + " PRECONDITIONER", "JACOBI");
  platform->options.setArgs("LS" + lid + " REGULARIZATION METHOD", "SVV");
  platform->options.setArgs("LS" + lid + " REGULARIZATION SVV FILTER POWER", to_string_f(6.0));
  platform->options.setArgs("LS" + lid + " REGULARIZATION SVV SCALING COEFF", to_string_f(2.0));
  platform->options.setArgs("LS" + lid + " SOLVER", "CG");
  platform->options.setArgs("LS" + lid + " SOLVER TOLERANCE", to_string_f(1.0e-08));
  platform->options.setArgs("LS" + lid + " TRANSPORTCOEFF", to_string_f(1.0));

  nrs = dynamic_cast<nrs_t *>(platform->app);
  if (!nrs || !nrs->meshV) {
    throw std::runtime_error("LS::setup: nrs or nrs->meshV is null (mesh not initialized)");
  }

  int nBDF;
  int nEXT;
  platform->options.getArgs("BDF ORDER", nBDF);
  platform->options.getArgs("EXT ORDER", nEXT);
  platform->options.getArgs("SUBCYCLING STEPS", advectionSubcycingSteps);
  if (advectionSubcycingSteps) {
    nEXT = nBDF;
    platform->options.setArgs("EXT ORDER", std::to_string(nEXT));
  }
  nekrsCheck(nEXT < nBDF,
             platform->comm.mpiComm(),
             EXIT_FAILURE,
             "%s\n",
             "EXT order needs to be >= BDF order!");
  o_coeffEXT = platform->device.malloc<dfloat>(nEXT);
  o_coeffBDF = platform->device.malloc<dfloat>(nBDF);

  tlsr = [&]() {
    lsConfig_t cfg;
    cfg.g0 = &nrs->g0;
    cfg.dt = nrs->dt;
    cfg.o_coeffBDF = o_coeffBDF;
    cfg.o_coeffEXT = o_coeffEXT;
    cfg.fieldOffset = nrs->meshV->fieldOffset;
    cfg.mesh.resize(1);
    cfg.mesh[0] = nrs->meshV;
    cfg.meshV = nrs->meshV;
    return std::make_unique<ls_t>(cfg);
  }();
  tlsr->setupEllipticSolver();

  std::cout << "EXITING LS::setup()...\n";
}

ls_t::ls_t(lsConfig_t &cfg)
{
  std::cout << "ENTERING LS::init()...\n";
  if (platform->comm.mpiRank() == 0) {
    std::cout << "================ " << "SETUP LEVEL-SET" << " ===============\n";
  }

  auto &options = platform->options;
  platform_t *platform = platform_t::getInstance();

  Nsubsteps = 0;
  platform->options.getArgs("SUBCYCLING STEPS", Nsubsteps);

  int NLSfields = 1;

  qqt.resize(NLSfields);
  fieldOffsetScan.resize(NLSfields);
  ellipticSolver.resize(NLSfields);
  compute.resize(NLSfields);

  meshV = cfg.meshV;

  g0 = cfg.g0;
  dt = cfg.dt;
  o_coeffBDF = cfg.o_coeffBDF;
  o_coeffEXT = cfg.o_coeffEXT;

  _fieldOffset = cfg.fieldOffset; // for now same for all scalars

  dlong sum = 0;
  for (int s = 0; s < NLSfields; ++s) {
    fieldOffsetScan[s] = (s > 0) ? sum : 0;
    sum += _fieldOffset;
    this->_mesh.push_back(cfg.mesh[s]);
    qqt[s] = new QQt(this->_mesh[s]->oogs);
  }
  fieldOffsetSum = sum;
  o_fieldOffsetScan = platform->device.malloc<dlong>(NLSfields, fieldOffsetScan.data());

  o_prop = platform->device.malloc<dfloat>(2 * fieldOffsetSum);
  o_diff = o_prop.slice(0 * fieldOffsetSum, fieldOffsetSum);
  o_rho = o_prop.slice(1 * fieldOffsetSum, fieldOffsetSum);

  for (int is = 0; is < NLSfields; is++) {
    const std::string sid = scalarDigitStr(is);

    const auto _name = lowerCase(options.getArgs("LS" + sid + " NAME"));
    name.push_back(_name);
    nameToIndex[_name] = is;

    auto o_tmp = [&]() {
      const auto prefixedName = "ls " + _name;
      auto tmp = platform->device.malloc<char>(prefixedName.size() + 1);
      tmp.copyFrom(prefixedName.data());
      const char nullChar[] = {'\0'};
      tmp.copyFrom(nullChar, 1, prefixedName.size());
      return tmp;
    }();
    o_name.push_back(o_tmp);

    if (options.compareArgs("LS" + sid + " SOLVER", "NONE")) {
      continue;
    }

    nekrsCheck(options.compareArgs("LS" + sid + " SOLVER", "BLOCK"),
               platform->comm.mpiComm(),
               EXIT_FAILURE,
               "%s\n",
               "level-set does not support BLOCK solver!");

    if (platform->comm.mpiRank() == 0) {
      std::cout << "LS" << sid << ": " << name[is] << std::endl;
    }
    platform->app->bc->printBcTypeMapping("ls" + sid);
    if (platform->comm.mpiRank() == 0) {
      std::cout << std::endl;
    }

    dfloat diff = 1;
    dfloat rho = 1;
    options.getArgs("LS" + sid + " DIFFUSIONCOEFF", diff);
    options.getArgs("LS" + sid + " TRANSPORTCOEFF", rho);

    auto o_diff_i = o_diff + fieldOffsetScan[is];
    auto o_rho_i = o_rho + fieldOffsetScan[is];

    std::vector<dfloat> diffTmp(this->_mesh[is]->Nlocal, diff);
    std::vector<dfloat> rhoTmp(this->_mesh[is]->Nlocal, rho);

    dfloat diffSolid = diff;
    dfloat rhoSolid = rho;
    options.getArgs("LS" + sid + " DIFFUSIONCOEFF SOLID", diffSolid);
    options.getArgs("LS" + sid + " TRANSPORTCOEFF SOLID", rhoSolid);
    for (int i = meshV->Nlocal; i < this->_mesh[is]->Nlocal; i++) {
      diffTmp[i] = diffSolid;
      rhoTmp[i] = rhoSolid;
    }

    o_diff_i.copyFrom(diffTmp.data(), diffTmp.size());
    o_rho_i.copyFrom(rhoTmp.data(), rhoTmp.size());
  }

  anyEllipticSolver = false;

  EToBOffset = [&]() {
    dlong NelementsMax = 0;
    for (int is = 0; is < NLSfields; is++) {
      NelementsMax = std::max(this->_mesh[is]->Nelements, NelementsMax);
    }
    return NelementsMax * meshV->Nfaces;
  }();

  std::vector<int> EToB(EToBOffset * NLSfields);

  for (int is = 0; is < NLSfields; is++) {
    std::string sid = scalarDigitStr(is);

    compute[is] = 1;
    if (options.compareArgs("LS" + sid + " SOLVER", "NONE")) {
      compute[is] = 0;
      continue;
    }

    anyEllipticSolver |= (compute[is]);

    auto mesh = this->_mesh[is];

    int cnt = 0;
    for (int e = 0; e < mesh->Nelements; e++) {
      for (int f = 0; f < mesh->Nfaces; f++) {
        EToB[cnt + EToBOffset * is] =
            platform->app->bc->typeId(mesh->EToB[f + e * mesh->Nfaces], "ls" + sid);
        cnt++;
      }
    }
  }

  o_EToB = platform->device.malloc<int>(EToB.size());
  o_EToB.copyFrom(EToB.data());

  o_compute = platform->device.malloc<dlong>(NLSfields, compute.data());

  int nFieldsAlloc = anyEllipticSolver ? std::max(o_coeffBDF.size(), o_coeffEXT.size()) : 1;
  o_S = platform->device.malloc<dfloat>(nFieldsAlloc * fieldOffsetSum);

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

  auto verifyBC = [&]() {
    for (int is = 0; is < NLSfields; is++) {
      if (!compute[is]) {
        continue;
      }

      const std::string field = "ls" + scalarDigitStr(is);
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

  std::cout << "EXITING LS::init()...\n";
}

void ls_t::solve(double time, int stage) {}

void ls_t::saveSolutionState() {}
void ls_t::restoreSolutionState() {}
void ls_t::lagSolution() {}

void ls_t::setTimeIntegrationCoeffs(int tstep) {}

void ls_t::extrapolateSolution() {}

void ls_t::applyDirichlet(double time) {}
void ls_t::setupEllipticSolver()
{
  int NLSFields = 1;
  for (int is = 0; is < NLSFields; is++) {
    std::string sid = scalarDigitStr(is);

    if (!compute[is]) {
      continue;
    }

    auto o_rho_i = o_rho.slice(fieldOffsetScan[is], _mesh[is]->Nlocal);
    auto o_lambda0 = o_diff.slice(fieldOffsetScan[is], _mesh[is]->Nlocal);
    auto o_lambda1 = platform->deviceMemoryPool.reserve<dfloat>(_mesh[is]->Nlocal);
    platform->linAlg->axpby(_mesh[is]->Nlocal, *g0 / dt[0], o_rho_i, 0.0, o_lambda1);

    ellipticSolver[is] = new elliptic("ls" + sid, _mesh[is], _fieldOffset, o_lambda0, o_lambda1);

    if (platform->options.compareArgs("LS" + sid + " REGULARIZATION METHOD", "SVV")) {
      auto o_svvmu = this->o_svvmu.slice(is * _fieldOffset, _fieldOffset);
      ellipticSolver[is]->mueSVV(o_svvmu);
      ellipticSolver[is]->setupSVV();
    }
  }
}

void ls_t::finalize() {}
