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
nrs_t *nrs;
occa::kernel signlsKernel;
occa::kernel normalVectorKernel;
bool buildKernelCalled = false;
std::unique_ptr<ls_t> tlsr = nullptr;
} // namespace

void LS::buildKernel(occa::properties _kernelInfo)
{
  occa::properties kernelInfo;
  kernelInfo += _kernelInfo;

  auto buildKernel = [&kernelInfo](const std::string &kernelName) {
    const auto path = getenv("NEKRS_KERNEL_DIR") + std::string("/solver/lvlSet/");
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
}

void LS::setup()
{
  static bool isInitialized = false;
  if (isInitialized) {
    return;
  }
  isInitialized = true;

  nrs = dynamic_cast<nrs_t *>(platform->app);
  if (!nrs || !nrs->meshV || !nrs->scalar) {
    throw std::runtime_error("LS::setup: nrs/nrs->meshV and/or nrs->scalar is null (mesh not initialized)");
  }

  // we hard-code these options here for now --> TODO: add LEVELSET section in par file
  std::string sid = "00";
  platform->options.setArgs("LS" + sid + " CHECKPOINTING", upperCase("false"));
  platform->options.setArgs("LS" + sid + " DIFFUSIONCOEFF", to_string_f(1.0e-14));
  platform->options.setArgs("LS" + sid + " ELLIPTIC COEFF FIELD", upperCase("true"));
  platform->options.setArgs("LS" + sid + " ELLIPTIC PRECO COEFF FIELD", upperCase("true"));
  platform->options.setArgs("LS" + sid + " INITIAL GUESS", "EXTRAPOLATION");
  platform->options.setArgs("LS" + sid + " MESH", "FLUID");
  platform->options.setArgs("LS" + sid + " NAME", "TLSR");
  platform->options.setArgs("LS" + sid + " PRECONDITIONER", "JACOBI");
  platform->options.setArgs("LS" + sid + " REGULARIZATION METHOD", "SVV");
  platform->options.setArgs("LS" + sid + " REGULARIZATION SVV FILTER POWER", to_string_f(6.0));
  platform->options.setArgs("LS" + sid + " REGULARIZATION SVV SCALING COEFF", to_string_f(2.0));
  platform->options.setArgs("LS" + sid + " SOLVER", "CG");
  platform->options.setArgs("LS" + sid + " SOLVER TOLERANCE", to_string_f(1.0e-14));
  platform->options.setArgs("LS" + sid + " TRANSPORTCOEFF", to_string_f(1.0));

  // add in the TLSR boundary condition per boundary
  std::string field = "ls00";
  int nBCs = platform->app->bc->size("scalar00");
  std::vector<int> bcTypes(nBCs, bdryBase::bcType_zeroNeumann);
  platform->app->bc->setBcMap(field, false, bcTypes);
  //auto bcMap = platform->app->bc->bIdToTypeId();
  // for (const auto& [key, value] : bcMap) {
  //   std::cout
  //     << "(" << key.first << ", " << key.second << ") -> "
  //     << value
  //     << '\n';
  // }

  // currently just holds TLSR solver --> TODO: add in CLSR solver (separate object or hold both in one LS object?)
  tlsr = [&]() {
    lsConfig_t cfg;
    cfg.g0 = &nrs->g0;
    cfg.dt = nrs->dt;
    cfg.fieldOffset = nrs->meshV->fieldOffset;
    cfg.vFieldOffset = nrs->scalar->vFieldOffset;       // TODO: is it safe to assume nrs->scalar is always accessible?
    cfg.vCubatureOffset = nrs->scalar->vCubatureOffset; // TODO: is it safe to assume nrs->scalar is always accessible?
    cfg.mesh.resize(1); // currently hard-coded for one equation (e.g. TLSR) --> TODO: handle TLSR and CLSR
    cfg.mesh[0] = nrs->meshV;
    cfg.meshV = nrs->meshV;
    return std::make_unique<ls_t>(cfg);
  }();
  tlsr->mueSVV(); // needs to be called before setupEllipticSolver() otherwise o_svvmue in elliptic solver will not be initialized --> TODO: handle AVM?
  tlsr->setupEllipticSolver();
}

void LS::solveLSR()
{
  auto mesh = tlsr->meshV;

  double time = 0.0;
  int outerIter = 1;
  int outerIterMax = 1000;
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


ls_t* LS::getLS()
{
  return tlsr.get();
}

ls_t::ls_t(lsConfig_t &cfg)
{
  if (platform->comm.mpiRank() == 0) {
    std::cout << "================ " << "SETUP LEVEL-SET" << " ===============\n";
  }

  auto &options = platform->options;
  platform_t *platform = platform_t::getInstance();

  Nsubsteps = 0;
  platform->options.getArgs("SUBCYCLING STEPS", Nsubsteps);

  NSfields = 1; // currently hard-coded for TLSR only --> TODO: add in CLSR support

  qqt.resize(NSfields);
  fieldOffsetScan.resize(NSfields);
  ellipticSolver.resize(NSfields);
  compute.resize(NSfields);

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

  o_coeffBDF = platform->device.malloc<dfloat>(nBDF);
  o_coeffEXT = platform->device.malloc<dfloat>(nEXT);

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
  o_rho = o_prop.slice(1 * fieldOffsetSum, fieldOffsetSum);
  
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
            platform->app->bc->typeId(mesh->EToB[f + e * mesh->Nfaces], "ls" + sid);
        cnt++;
      }
    }
  }

  o_EToB = platform->device.malloc<int>(EToB.size());
  o_EToB.copyFrom(EToB.data());

  o_compute = platform->device.malloc<dlong>(NSfields, compute.data());

  int nFieldsAlloc = anyEllipticSolver ? std::max(o_coeffBDF.size(), o_coeffEXT.size()) : 1;
  o_S = platform->device.malloc<dfloat>(nFieldsAlloc * fieldOffsetSum);
  o_W = platform->device.malloc<dfloat>(meshV->dim * std::max(o_coeffBDF.size(), o_coeffEXT.size()) * vFieldOffset);
  const dlong Nstates = Nsubsteps ? std::max(o_coeffBDF.size(), o_coeffEXT.size()) : 1;
  o_relWrst = platform->device.malloc<dfloat>(Nstates * meshV->dim * vCubatureOffset);

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

  auto verifyBC = [&]() {
    for (int is = 0; is < NSfields; is++) {
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
}

void ls_t::setTimeIntegrationCoeffs(int tstep)
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

void ls_t::computeAdvectionCoeff()
{
  // a. compute interface normals
  launchKernel("core-gradientVolumeHex3D",
               meshV->Nelements,
               meshV->o_vgeo,
               meshV->o_D,
               vFieldOffset,
               o_solution("tlsr"),
               o_W);
  oogs::startFinish(o_W, meshV->dim, vFieldOffset, ogsDfloat, ogsAdd, meshV->oogs);
  platform->linAlg->axmyVector(meshV->Nlocal, vFieldOffset, 0, 1.0, meshV->o_invLMM, o_W);
  // b. compute sign function
  signlsKernel(meshV->Nlocal, o_solution("tlsr"), o_signls);
  // c. compute w = sign(phi) * n
  normalVectorKernel(meshV->Nlocal, vFieldOffset, o_signls, o_W);
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
                   vFieldOffset,
                   o_S,
                   o_relWrst, // --> TODO change to w
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

void ls_t::makeExplicit(int is, double time, int tstep)
{
  const std::string sid = scalarDigitStr(is);

  auto mesh = this->_mesh[is];
  const dlong isOffset = fieldOffsetScan[is];

  o_explicitTerms("tlsr").copyFrom(o_signls); // TODO: shouldn't use tlsr here but will use this hack for now
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

void ls_t::computeWrst()
{
  auto mesh = meshV;

  if (Nsubsteps) {
    for (int s = std::max(o_coeffBDF.size(), o_coeffEXT.size()); s > 1; s--) {
      auto lagOffset = mesh->dim * vCubatureOffset;
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
                 vFieldOffset,
                 0, // (geom) ? geom->fieldOffset : 0,
                 vCubatureOffset,
                 o_W,
                 o_NULL, // (geom) ? geom->o_U : o_NULL,
                 o_relWrst);
  } else {
    launchKernel("nrs-UrstHex3D",
                 mesh->Nelements,
                 relative,
                 mesh->o_vgeo,
                 vFieldOffset,
                 0, // (geom) ? geom->fieldOffset : 0,
                 o_W,
                 o_NULL, // (geom) ? geom->o_U : o_NULL,
                 o_relWrst);
  }
}

void ls_t::writeFile(double time)
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
