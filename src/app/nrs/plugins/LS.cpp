#include "app.hpp"
#include "nrs.hpp"
#include "platform.hpp"
#include "LS.hpp"
#include "linAlg.hpp"
#include "solver.hpp"
#include "bdryBase.hpp"
#include "advectionSubCycling.hpp"
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
class bdry : public bdryBase
{
  public:
    bdry(){};
    void setup() override;
};
bdry bc;
//std::unique_ptr<bdry> bc = std::make_unique<bdry>();
};

void bdry::setup() {
  std::string field = "LS00";
  std::string key = "zeroNeumann";
  int bid = 0;
  bToBc[make_pair(lowerCase(field), bid)] = vBcTextToID.at(lowerCase(key));
  // for (const auto& [key, value] : bToBc) {
  //   std::cout
  //     << "(" << key.first << ", " << key.second << ") -> "
  //     << value
  //     << '\n';
  // }
}

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

  bc.setup();

  nrs = dynamic_cast<nrs_t *>(platform->app);
  if (!nrs || !nrs->meshV || !nrs->scalar) {
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
    cfg.vFieldOffset = nrs->scalar->vFieldOffset;       // TODO: is this a safe access?
    cfg.vCubatureOffset = nrs->scalar->vCubatureOffset; // TODO: is this a safe access?
    cfg.mesh.resize(1);
    cfg.mesh[0] = nrs->meshV;
    cfg.meshV = nrs->meshV;
    return std::make_unique<ls_t>(cfg);
  }();
  tlsr->setupEllipticSolver();

  std::cout << "EXITING LS::setup()...\n";
}

void LS::solveLSR()
{
  std::cout << "ENTERING LS::solveLSR()...\n";

  if (tlsr->fieldOffsetSum != nrs->scalar->fieldOffsetSum) {
    throw std::runtime_error("LS::solveLSR: tlsr and nrs->scalar fieldOffsetSum are not equal.");
  }

  auto mesh = tlsr->meshV;

  double time = 0.0;
  double dt = 4.0e-03;
  int outerIter = 0;
  int outerIterMax = 10000;
  int innerIterMax = 100;

  // set the TLSR initial condition -- currently just copy the scalar S00. TODO: handle the correct initial condition
  tlsr->o_S.copyFrom(nrs->scalar->o_S);

  while(outerIter < outerIterMax) {
    std::cout << "ITER: " << outerIter << std::endl;
    tlsr->computeAdvectionCoeff();
    tlsr->makeAdvection(0, time, outerIter); // currently assume 1 LS equation -->  is = 1
    // tlsr->makeExplicit();
    tlsr->makeForcing();
    tlsr->mueSVV();
    int innerIter = 1;
    bool stepConverged = false;
    while(!stepConverged) {
      if (innerIter == 1) { tlsr->lagSolution(); }
      // tlsr->applyDirichlet(timeNew);
      tlsr->solve(time, innerIter);
      stepConverged = true;
    }
    outerIter += 1;
    time += dt;
  }

  // copy the TLSR solution back to the scalar S00
  nrs->scalar->o_S.copyFrom(tlsr->o_S);

  std::cout << "EXITING LS::solveLSR()...\n";
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

  int NSfields = 1;

  qqt.resize(NSfields);
  fieldOffsetScan.resize(NSfields);
  ellipticSolver.resize(NSfields);
  compute.resize(NSfields);

  meshV = cfg.meshV;

  g0 = cfg.g0;
  dt = cfg.dt;
  o_coeffBDF = cfg.o_coeffBDF;
  o_coeffEXT = cfg.o_coeffEXT;

  _fieldOffset = cfg.fieldOffset; // for now same for all scalars
  vFieldOffset = cfg.vFieldOffset; // TODO: check if this is correct
  vCubatureOffset = cfg.vCubatureOffset;

  dlong sum = 0;
  for (int s = 0; s < NSfields; ++s) {
    fieldOffsetScan[s] = (s > 0) ? sum : 0;
    sum += _fieldOffset;
    this->_mesh.push_back(cfg.mesh[s]);
    qqt[s] = new QQt(this->_mesh[s]->oogs);
  }
  fieldOffsetSum = sum;
  o_fieldOffsetScan = platform->device.malloc<dlong>(NSfields, fieldOffsetScan.data());

  o_prop = platform->device.malloc<dfloat>(2 * fieldOffsetSum);
  o_diff = o_prop.slice(0 * fieldOffsetSum, fieldOffsetSum);
  
  for (int is = 0; is < NSfields; is++) {
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
    bc.printBcTypeMapping("ls" + sid);
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
    for (int is = 0; is < NSfields; is++) {
      NelementsMax = std::max(this->_mesh[is]->Nelements, NelementsMax);
    }
    return NelementsMax * meshV->Nfaces;
  }();

  std::vector<int> EToB(EToBOffset * NSfields);

  for (int is = 0; is < NSfields; is++) {
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
            bc.typeId(mesh->EToB[f + e * mesh->Nfaces], "ls" + sid);
        cnt++;
      }
    }
  }

  o_EToB = platform->device.malloc<int>(EToB.size());
  o_EToB.copyFrom(EToB.data());

  o_compute = platform->device.malloc<dlong>(NSfields, compute.data());

  int nFieldsAlloc = anyEllipticSolver ? std::max(o_coeffBDF.size(), o_coeffEXT.size()) : 1;
  o_S = platform->device.malloc<dfloat>(nFieldsAlloc * fieldOffsetSum);
  o_W = platform->device.malloc<dfloat>(nFieldsAlloc * fieldOffsetSum);
  o_signls = platform->device.malloc<dfloat>(nFieldsAlloc * fieldOffsetSum);
  std::vector<dfloat> rho(nFieldsAlloc * fieldOffsetSum, (dfloat)1.0);
  o_rho = platform->device.malloc<dfloat>(nFieldsAlloc * fieldOffsetSum, rho.data());

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
    for (int is = 0; is < NSfields; is++) {
      if (!compute[is]) {
        continue;
      }

      const std::string field = "ls" + scalarDigitStr(is);
      nekrsCheck(_mesh[is]->Nbid != bc.size(field),
                 platform->comm.mpiComm(),
                 EXIT_FAILURE,
                 "Size of %s boundaryTypeMap (%d) does not match number of boundary IDs in mesh (%d)!\n",
                 field.c_str(),
                 bc.size(field),
                 _mesh[is]->Nbid);
    }
  };

  verifyBC();

  std::cout << "EXITING LS::init()...\n";
}

void ls_t::computeAdvectionCoeff()
{
  // a. compute interface normals
  launchKernel("core-gradientVolumeHex3D",
               meshV->Nelements,
               meshV->o_vgeo,
               meshV->o_D,
               _fieldOffset,
               o_solution("tlsr"),
               o_W);
  oogs::startFinish(o_W, meshV->dim, _fieldOffset, ogsDfloat, ogsAdd, meshV->oogs);
  platform->linAlg->axmyVector(meshV->Nlocal, _fieldOffset, 0, 1.0, meshV->o_invLMM, o_W);
  // b. compute sign function
  signlsKernel(meshV->Nlocal, o_solution("tlsr"), o_signls);
  // c. compute w = sign(phi) * n
  normalVectorKernel(meshV->Nlocal, _fieldOffset, o_signls, o_W);
}

void ls_t::makeAdvection(int is, double time, int tstep)
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
                   vFieldOffset,
                   vCubatureOffset,
                   o_S,
                   o_W,   // --> TODO change to w
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
                   vFieldOffset,
                   o_S,
                   o_W, // --> TODO change to w
                   o_rho, // --> set to 1
                   o_ADV);
    }
  }
}

void ls_t::advectionSubcycling(int nEXT, double time, int is)
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
                        fieldOffset(),
                        vCubatureOffset,
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
      printf("%s%s advSub norm: %.15e\n", "scalar", scalarDigitStr(is).c_str(), debugNorm);
    }
  }
}

void ls_t::makeForcing()
{
  for (int is = 0; is < this->NSfields; is++) {
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

void ls_t::solve(double time, int stage)
{
  for (int is = 0; is < NSfields; is++) {
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
                 vFieldOffset,
                 _fieldOffset,
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
                 bc.o_usrwrk,
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
      if (platform->options.compareArgs("LS" + sid + " INITIAL GUESS", "EXTRAPOLATION") && stage == 1) {
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

void ls_t::saveSolutionState() {}

void ls_t::restoreSolutionState() {}

void ls_t::lagSolution()
{
  if (!anyEllipticSolver) {
    return;
  }

  const auto n = std::max(o_coeffEXT.size(), o_coeffBDF.size());
  for (int s = n; s > 1; s--) {
    o_S.copyFrom(o_S, fieldOffsetSum, (s - 1) * fieldOffsetSum, (s - 2) * fieldOffsetSum);
  }
}

void ls_t::setTimeIntegrationCoeffs(int tstep) {}

void ls_t::extrapolateSolution()
{
  if (!o_Se.isInitialized()) {
    return;
  }
  const auto Nlocal = _fieldOffset; // assumed to be the same for all fields
  launchKernel("core-extrapolate",
               Nlocal,
               NSfields,
               static_cast<int>(o_coeffEXT.size()),
               _fieldOffset,
               o_coeffEXT,
               o_S,
               o_Se);
}

void ls_t::applyDirichlet(double time) {}

void ls_t::setupEllipticSolver()
{
  for (int is = 0; is < NSfields; is++) {
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

void ls_t::mueSVV()
{
  auto mesh = this->meshV;

  static auto initialized = false;

  auto umagInitialized = false;

  auto o_umag = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);

  for (int is = 0; is < NSfields; is++) {
    const auto sid = scalarDigitStr(is);

    if(platform->options.compareArgs("LS" + sid + " REGULARIZATION METHOD", "SVV")) {
      if(!initialized) {
        this->o_svvf = platform->device.malloc<dfloat>(_fieldOffset);
        this->o_svvmu = platform->device.malloc<dfloat>(NSfields * _fieldOffset);

        if(!platform->options.compareArgs("MOVING MESH","TRUE"))
          launchKernel("core-svv::svvMeshScale", mesh->Nelements, mesh->o_vgeo, this->o_svvf);

        initialized = true;
      }

      if(platform->options.compareArgs("MOVING MESH","TRUE"))
        launchKernel("core-svv::svvMeshScale", mesh->Nelements, mesh->o_vgeo, this->o_svvf);

      if(!umagInitialized) {
        platform->linAlg->magVector(mesh->Nlocal, vFieldOffset, o_W, o_umag); // changed o_U to o_W --> TODO: think if this is the right thing to do
        umagInitialized = true;
      }

      dfloat scale = 0.1;
      platform->options.getArgs("LS" + sid + " REGULARIZATION SVV SCALING COEFF", scale);

      auto o_svvmu = this->o_svvmu.slice(is * _fieldOffset, _fieldOffset);
      platform->linAlg->axmyz(mesh->Nlocal, scale, this->o_svvf, o_umag, o_svvmu);
    }
  }
}
