#include "nrs.hpp"
#include "udf.hpp"
#include "avm.hpp"

void nrs_t::evaluateProperties(const double timeNew) 
{

  bool rhsCVODE = false;
  if(this->cvode)
    rhsCVODE = this->cvode->isRhsEvaluation();

  const std::string tag = rhsCVODE ? "udfPropertiesCVODE" : "udfProperties";

  platform->timer.tic(tag, 1);
  cds_t *cds = this->cds;

  if (udf.properties) {
    occa::memory o_S = (this->Nscalar) ? cds->o_S : o_NULL;
    occa::memory o_SProp = (this->Nscalar) ? cds->o_prop : o_NULL;
    udf.properties(this, timeNew, this->o_U, o_S, this->o_prop, o_SProp);
  }

  if (this->Nscalar) {
    cds_t *cds = this->cds;
    for (int is = 0; is < cds->NSfields; ++is) {
      std::string sid = scalarDigitStr(is);

      std::string regularizationMethod;
      platform->options.getArgs("SCALAR" + sid + " REGULARIZATION METHOD", regularizationMethod);
      const bool applyAVM = regularizationMethod.find("AVM_RESIDUAL") != std::string::npos ||
                            regularizationMethod.find("AVM_HIGHEST_MODAL_DECAY") != std::string::npos;
      if (applyAVM)
        avm::apply(this, timeNew, is, cds->o_S);
    }
  }

  platform->timer.toc(tag);
}
