#if !defined(nekrs_lvlSet_hpp_)
#define nekrs_lvlSet_hpp_

#include "platform.hpp"
#include "solver.hpp"
#include "geomSolver.hpp"
#include "iofldFactory.hpp"

class lvlSet_t;

namespace lvlSet
{
void buildKernel(occa::properties kernelInfo);
void setup();
void solve(const double &fluidTime);
lvlSet_t* getTLSR();
lvlSet_t* getCLSR();

void clsrAx(elliptic_t* elliptic,
            dlong NelementsList,
            const occa::memory &o_elementsList,
            const occa::memory &o_x,
            occa::memory &o_Ax);

void clsrPreconditioner(elliptic_t* elliptic, const occa::memory &o_r, occa::memory &o_z);

void normalVector(const occa::memory &o_phi, occa::memory &o_normals, bool avg = true);
void initHeaviside(const occa::memory& o_phi, occa::memory& o_psi, const dfloat epsin = -1.0);

void applySurfaceTensionAcc(const dfloat& We, occa::memory &o_stForce);
const occa::memory& getSignField(const occa::memory &o_phi);
const occa::memory& getDeltaFunction();
const occa::memory& getCurvature(const occa::memory &o_normals);

void updateProperties(const dfloat &rhoRatio, const dfloat &muRatio, const dfloat &Re);
void updateProperties(const dfloat &rhog, const dfloat &rhol, const dfloat &mug, const dfloat &mul);
}

struct lvlSetConfig_t : public solverCfg_t {
public:
  mesh_t *mesh;
  mesh_t *meshV;
  dlong fieldOffset;
  dlong vFieldOffset;
  dlong vCubatureOffset;
};

class lvlSet_t : public solver_t
{
public:

  lvlSet_t(lvlSetConfig_t &cfg, const std::unique_ptr<geomSolver_t> &geom);

  void computeAdvectionCoeff(int tstep);
  void makeExplicit(double time, int tstep);
  void makeAdvection(double time, int tstep);
  void makeForcing();
  void mueAVM();
  void mueSVV(int tstep);
  void computeWrst();

  void solve(double time, int stage) override;

  void saveSolutionState() override;
  void restoreSolutionState() override;
  void lagSolution() override;

  void extrapolateSolution() override;

  void applyDirichlet(double time) override;
  void setupEllipticSolver() override;

  void finalize() override;

  void setTimeIntegrationCoeffs(int tstep) override;

  void printStepInfo(double time, dfloat cfl, int tstep, bool printStepInfo, bool solverInfo);

  std::function<occa::memory(double, int)> userImplicitLinearTerm = nullptr;

  void pseudoStepper(const double &fluidTime);

  dlong fieldOffset() const
  {
    return _fieldOffset;
  };

  mesh_t* mesh()
  {
    return _mesh;
  };

  deviceMemory<dfloat> o_solution(std::string key = "") override
  {
    return deviceMemory<dfloat>(o_S);
  };

  deviceMemory<dfloat> o_explicitTerms(std::string key = "") override
  {
    return deviceMemory<dfloat>(o_EXT);
  };

  deviceMemory<dfloat> o_diffusionCoeff(std::string key = "") override
  {
    return deviceMemory<dfloat>(o_prop.slice(0, _fieldOffset));
  };

  deviceMemory<dfloat> o_transportCoeff(std::string key = "") override
  {
    return deviceMemory<dfloat>(o_prop.slice(_fieldOffset, _fieldOffset));
  }

  mesh_t *meshV;

  occa::memory o_fieldOffsetScan;
  dlong fieldOffsetScan; /* exclusive */

  dlong vFieldOffset;
  dlong vCubatureOffset;

  int compute;
  occa::memory o_compute;

  occa::memory o_applyFilterRT;
  occa::memory o_filterS;
  occa::memory o_filterRT;

  occa::memory o_svvmu;

  int Nsubsteps = 0;

  dlong EToBOffset = -1;

  occa::memory o_S;
  occa::memory o_Se;

  occa::memory o_W;
  occa::memory o_relWrst;
  occa::memory o_rho;
  occa::memory o_diff;

  std::string name;

  int stepsMax;
  dfloat distanceFactor;
  dfloat targetCFL;

  const std::unique_ptr<geomSolver_t> &geom;

  void writeFile(double fluidTime, int tstep);
  std::unique_ptr<iofld> fieldWriter;
  int outfldCounter = 0;

  std::tuple<dfloat, dfloat, int> computeFixedDistanceAdvectionParams();

private:
  void advectionSubcycling(int nEXT, double time);
  mesh_t * _mesh;

  occa::memory o_name;

  dlong _fieldOffset = -1; // all scalar fields share the same offset

  occa::memory o_ADV;

  occa::memory o_S0;
  occa::memory o_EXT0;
  occa::memory o_ADV0;
  occa::memory o_prop0;

};

#endif
