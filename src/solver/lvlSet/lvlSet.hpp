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
void solveLSR();
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

  void computeAdvectionCoeff();
  void makeExplicit(int is, double time, int tstep);
  void makeAdvection(int is, double time, int tstep);
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

  std::function<occa::memory(double, int)> userImplicitLinearTerm = nullptr;

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
    if (key.empty()) return deviceMemory<dfloat>(o_S);
    auto it = nameToIndex.find(lowerCase(key));
    const auto idx = (it != nameToIndex.end()) ? it->second : -1;
    return (idx >= 0) ? deviceMemory<dfloat>(o_S.slice(fieldOffsetScan, _fieldOffset)) : deviceMemory<dfloat>(o_NULL);
  };

  deviceMemory<dfloat> o_explicitTerms(std::string key = "") override
  {
    if (key.empty()) return deviceMemory<dfloat>(o_EXT);
    auto it = nameToIndex.find(lowerCase(key));
    const auto idx = (it != nameToIndex.end()) ? it->second : -1;
    return (idx >= 0) ? deviceMemory<dfloat>(o_EXT.slice(fieldOffsetScan, _fieldOffset)) : deviceMemory<dfloat>(o_NULL);
  };

  deviceMemory<dfloat> o_diffusionCoeff(std::string key = "") override
  {
    if (key.empty()) return deviceMemory<dfloat>(o_prop.slice(0, _fieldOffset));
    auto it = nameToIndex.find(lowerCase(key));
    const auto idx = (it != nameToIndex.end()) ? it->second : -1;
    return (idx >= 0) ? deviceMemory<dfloat>(o_prop.slice(fieldOffsetScan, _fieldOffset)) : deviceMemory<dfloat>(o_NULL);
  };

  deviceMemory<dfloat> o_transportCoeff(std::string key = "") override
  {
    if (key.empty()) return deviceMemory<dfloat>(o_prop.slice(_fieldOffset, _fieldOffset));
    auto it = nameToIndex.find(lowerCase(key));
    const auto idx = (it != nameToIndex.end()) ? it->second : -1;
    return (idx >= 0) ? deviceMemory<dfloat>(o_prop.slice(_fieldOffset + fieldOffsetScan, _fieldOffset)) : deviceMemory<dfloat>(o_NULL);
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

  void writeFile(double time);
  std::unique_ptr<iofld> fieldWriter;
  int outfldCounter = 0;

private:
  void advectionSubcycling(int nEXT, double time, int scalarIdx);
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
