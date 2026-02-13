#if !defined(nekrs_LS_hpp_)
#define nekrs_LS_hpp_

#include "platform.hpp"
#include "solver.hpp"
#include "geomSolver.hpp"

namespace LS
{
void buildKernel(occa::properties kernelInfo);
void updateSourceTerms();
void setup();
void solveLSR();
}

struct lsConfig_t : public solverCfg_t {
public:
  std::vector<mesh_t *> mesh;
  mesh_t *meshV;
  dlong fieldOffset;
  dlong vFieldOffset;
  dlong vCubatureOffset;
  dfloat *g0;
  dfloat *dt;
};

class ls_t : public solver_t
{
public:

  ls_t(lsConfig_t &cfg);

  void computeAdvectionCoeff();
  void makeExplicit(int is, double time, int tstep);
  void makeAdvection(int is, double time, int tstep);
  void makeForcing();

  void solve(double time, int stage) override;

  void saveSolutionState() override;
  void restoreSolutionState() override;
  void lagSolution() override;

  void setTimeIntegrationCoeffs(int tstep) override;

  void extrapolateSolution() override;

  void applyDirichlet(double time) override;
  void setupEllipticSolver() override;

  void finalize() override;

  void mueAVM();
  void mueSVV();

  std::function<occa::memory(double, int)> userImplicitLinearTerm = nullptr;

  dlong fieldOffset() const
  {
    return _fieldOffset;
  };

  mesh_t* mesh(std::string key)
  {
    auto it = nameToIndex.find(lowerCase(key));
    const auto idx = (it != nameToIndex.end()) ? it->second : -1;
    return (idx >= 0) ? _mesh[idx] : nullptr;
  };

  mesh_t* mesh(int idx)
  {
    return _mesh.at(idx);
  };

  deviceMemory<dfloat> o_solution(std::string key = "") override
  {
    if (key.empty()) return deviceMemory<dfloat>(o_S);
    auto it = nameToIndex.find(lowerCase(key));
    const auto idx = (it != nameToIndex.end()) ? it->second : -1;
    return (idx >= 0) ? deviceMemory<dfloat>(o_S.slice(fieldOffsetScan[idx], _fieldOffset)) : deviceMemory<dfloat>(o_NULL);
  };

  deviceMemory<dfloat> o_explicitTerms(std::string key = "") override
  {
    if (key.empty()) return deviceMemory<dfloat>(o_EXT);
    auto it = nameToIndex.find(lowerCase(key));
    const auto idx = (it != nameToIndex.end()) ? it->second : -1;
    return (idx >= 0) ? deviceMemory<dfloat>(o_EXT.slice(fieldOffsetScan[idx], _fieldOffset)) : deviceMemory<dfloat>(o_NULL);
  };

  deviceMemory<dfloat> o_diffusionCoeff(std::string key = "") override
  {
    if (key.empty()) return deviceMemory<dfloat>(o_prop.slice(0, fieldOffsetSum));
    auto it = nameToIndex.find(lowerCase(key));
    const auto idx = (it != nameToIndex.end()) ? it->second : -1;
    return (idx >= 0) ? deviceMemory<dfloat>(o_prop.slice(fieldOffsetScan[idx], _fieldOffset)) : deviceMemory<dfloat>(o_NULL);
  };

  deviceMemory<dfloat> o_transportCoeff(std::string key = "") override
  {
    if (key.empty()) return deviceMemory<dfloat>(o_prop.slice(fieldOffsetSum, fieldOffsetSum));
    auto it = nameToIndex.find(lowerCase(key));
    const auto idx = (it != nameToIndex.end()) ? it->second : -1;
    return (idx >= 0) ? deviceMemory<dfloat>(o_prop.slice(fieldOffsetSum + fieldOffsetScan[idx], _fieldOffset)) : deviceMemory<dfloat>(o_NULL);
  }

  mesh_t *meshV;

  occa::memory o_fieldOffsetScan;

  dlong vFieldOffset;
  dlong vCubatureOffset;

  std::vector<dlong> fieldOffsetScan; /* exclusive */

  bool anyEllipticSolver = false;

  int NSfields = 0;

  std::vector<QQt *> qqt;

  std::vector<int> compute;
  occa::memory o_compute;

  occa::memory o_applyFilterRT;
  occa::memory o_filterS;
  occa::memory o_filterRT;

  occa::memory o_svvmu;
  occa::memory o_svvf;

  int Nsubsteps = 0;

  dfloat *dp0thdt = nullptr;
  dfloat *alpha0Ref = nullptr;

  dlong EToBOffset = -1;

  occa::memory o_S;
  occa::memory o_Se;

  occa::memory o_W;
  occa::memory o_signls;
  occa::memory o_rho;
  occa::memory o_diff;

  std::vector<std::string> name;

private:
  void advectionSubcycling(int nEXT, double time, int scalarIdx);
  std::vector<mesh_t *> _mesh;

  std::vector<occa::memory> o_name;

  dlong _fieldOffset = -1; // all scalar fields share the same offset

  occa::memory o_ADV;

  occa::memory o_S0;
  occa::memory o_EXT0;
  occa::memory o_ADV0;
  occa::memory o_prop0;

};

#endif
