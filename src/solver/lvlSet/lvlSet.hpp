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
lvlSet_t* getLS();
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
  void mueSVV();
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

  void printStepInfo(double time, int tstep, bool printStepInfo, bool solverInfo);

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
  occa::memory o_svvf;

  int Nsubsteps = 0;

  dlong EToBOffset = -1;

  occa::memory o_S;
  occa::memory o_Se;

  occa::memory o_W;
  occa::memory o_relWrst;
  occa::memory o_signls;
  occa::memory o_rho;
  occa::memory o_diff;

  std::string name;

  const std::unique_ptr<geomSolver_t> &geom;

  void writeFile(double fluidTime, int tstep);
  std::unique_ptr<iofld> fieldWriter;
  int outfldCounter = 0;

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
