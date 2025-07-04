#include <optional>

#include "nekrsSys.hpp"
#include "par.hpp"

#include "inipp.hpp"
#include "tinyexpr.h"

#include "ellipticParseMultigridSchedule.hpp"
#include "hypreWrapperDevice.hpp"

#include "AMGX.hpp"

namespace
{
static std::ostringstream errorLogger;
static std::ostringstream valueErrorLogger;
std::string setupFile;
int nscal = 0;
bool cvodeRequested = false;
MPI_Comm comm;
std::map<std::string, int> scalarMap;

static std::string mapTemperatureToScalarString()
{
  std::string sid = scalarDigitStr(0);
  return "scalar" + sid;
}

std::optional<int> parseScalarIntegerFromString(const std::string &scalarString)
{
  if (scalarString == std::string("scalar")) {
    return {};
  }

  if (scalarString.length() > std::string("scalar").length()) {
    try {
        std::istringstream iss(scalarString);
        std::string firstWord, secondWord;
        iss >> firstWord >> secondWord;
        return scalarMap.at(secondWord);
    } catch (std::invalid_argument &e) {
      std::cout << "Hit an invalid_argument error for scalarString=\"" << scalarString << "\". It said\n"
                << e.what() << "\n";
      nekrsAbort(MPI_COMM_SELF, EXIT_FAILURE, "%s\n", "");
      return {};
    }
  } else {
    nekrsAbort(MPI_COMM_SELF, EXIT_FAILURE, "%s\n", "");
    return {};
  }
}

// option prefix
std::string parPrefixFromParSection(const std::string &parSection)
{
  if (parSection.find("general") != std::string::npos) {
    return std::string("");
  }
  if (parSection.find("scalar") != std::string::npos) {
    if (parSection == std::string("scalar")) return "scalar default ";
    const auto is = parseScalarIntegerFromString(parSection);
    return "scalar" +  scalarDigitStr(is.value()) + " ";
  }
  return parSection + std::string(" ");
}
} // namespace

bool checkForTrue(const std::string &s)
{
  return (s.find("true") != std::string::npos) || (s.find("yes") != std::string::npos) ||
         (s.find("1") != std::string::npos);
}

bool checkForFalse(const std::string &s)
{
  return (s.find("false") != std::string::npos) || (s.find("no ") != std::string::npos) ||
         (s.find("0") != std::string::npos);
}

template <typename Printable> void append_error(Printable message)
{
  errorLogger << "\t" << message << "\n";
}

template <typename Printable> void append_value_error(Printable message)
{
  valueErrorLogger << "\t" << message << "\n";
}

namespace
{

static std::string parseValueForKey(std::string token, std::string key)
{
  if (token.find(key) != std::string::npos) {
    std::vector<std::string> params = serializeString(token, '=');
    if (params.size() != 2) {
      std::ostringstream error;
      error << "could not parse " << key << " " << token << "!\n";
      append_error(error.str());
    }
    return params[1];
  }

  return "";
}

static bool enforceLowerCase = false;

static std::vector<std::string> nothing = {};

static std::vector<std::string> noSectionKeys = {{"userSections"}};

static std::vector<std::string> generalKeys = {
    {"dt"},
    {"endTime"},
    {"numSteps"},
    {"polynomialOrder"},
    {"dealiasing"},
    {"cubaturePolynomialOrder"},
    {"startFrom"},
    {"stopAt"},
    {"elapsedtime"},
    {"timestepper"},
    {"advectionSubCyclingSteps"},
    {"redirectOutputTo"},
    {"writeControl"},
    {"checkpointEngine"},
    {"checkpointControl"},
    {"writeInterval"},
    {"checkpointInterval"},
    {"constFlowRate"},
    {"verbose"},
    {"variableDT"},
    {"checkpointprecision"},
    {"scalars"},
    {"oudf"},
    {"udf"},
    {"usr"},
};

static std::vector<std::string> neknekKeys = {
    {"boundaryextorder"},
    {"multiratetimestepping"},
};

static std::vector<std::string> problemTypeKeys = {
    {"stressFormulation"},
    {"equation"},
};

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
    {"maxIterations"},
    {"regularization"},
    {"checkpointing"},

    // deprecated filter params
    {"filterWeight"},
    {"filterModes"},
    {"filterCutoffRatio"},
};

static std::vector<std::string> meshKeys = {
    {"partitioner"},
    {"file"},
    {"connectivitytol"},
    {"boundaryidmapV"},
    {"boundaryidmap"},
};

static std::vector<std::string> velocityKeys = {
    {"density"},
    {"rho"},
    {"viscosity"},
    {"mu"},
};

static std::vector<std::string> scalarDefaultKeys = {
    {"transportCoeff"},
    {"transportCoeffSolid"},
    {"diffusionCoeff"},
    {"diffusionCoeffSolid"},
    {"absolutetol"},
    {"mesh"},
};

static std::vector<std::string> scalarKeys = {
    {"transportCoeff"},
    {"rhocp"},
    {"transportCoeffSolid"},
    {"rhocpSolid"},
    {"diffusionCoeff"},
    {"conductivity"},
    {"diffusionCoeffSolid"},
    {"conductivitySolid"},
    {"absolutetol"},
    {"mesh"},
};

static std::vector<std::string> cvodeKeys = {
    {"relativetol"},
    {"epslin"},
    {"gstype"},
    {"dqsigma"},
    {"maxOrder"},
    {"maxSteps"},
    {"jtvrecycleproperties"},
    {"sharedrho"},
    {"dealiasing"},
    {"regularization"},
    {"solver"},
};

static std::vector<std::string> boomeramgKeys = {
    {"coarsenType"},
    {"interpolationType"},
    {"smootherType"},
    {"iterations"},
    {"coarseSmootherType"},
    {"strongThreshold"},
    {"nonGalerkinTol"},
    {"aggressiveCoarseningLevels"},
    {"chebyshevRelaxOrder"},
    {"chebyshevFraction"},
};

static std::vector<std::string> amgxKeys = {
    {"configFile"},
};
static std::vector<std::string> occaKeys = {{"backend"}, {"deviceNumber"}, {"platformNumber"}};

static std::vector<std::string> pressureKeys = {};

static std::vector<std::string> geomKeys = {};

static std::vector<std::string> deprecatedKeys = {
    // deprecated filter params
    {"filterWeight"},
    {"filterModes"},
    {"filterCutoffRatio"},
    {"writeControl"},
    {"writeInterval"},
    {"stressFormulation"},
};

static std::vector<std::string> validSections = {
    {""},
    {"general"},
    {"neknek"},
    {"fluid pressure"},
    {"fluid velocity"},
    {"problemtype"},
    {"amgx"},
    {"boomeramg"},
    {"occa"},
    {"mesh"},
    {"geom"},
    {"scalar"},
    {"cvode"},
};

void makeStringsLowerCase()
{
  noSectionKeys = lowerCase(noSectionKeys);
  generalKeys = lowerCase(generalKeys);
  neknekKeys = lowerCase(neknekKeys);
  problemTypeKeys = lowerCase(problemTypeKeys);
  commonKeys = lowerCase(commonKeys);
  meshKeys = lowerCase(meshKeys);
  geomKeys = lowerCase(geomKeys);
  scalarDefaultKeys = lowerCase(scalarDefaultKeys);
  scalarKeys = lowerCase(scalarKeys);
  deprecatedKeys = lowerCase(deprecatedKeys);
  amgxKeys = lowerCase(amgxKeys);
  boomeramgKeys = lowerCase(boomeramgKeys);
  pressureKeys = lowerCase(pressureKeys);
  occaKeys = lowerCase(occaKeys);
  cvodeKeys = lowerCase(cvodeKeys);
  validSections = lowerCase(validSections);
}

void processError()
{
  const std::string valueErrors = valueErrorLogger.str();
  errorLogger << valueErrors;
  const std::string errorMessage = errorLogger.str();
  int length = errorMessage.size();
  MPI_Bcast(&length, 1, MPI_INT, 0, comm);

  auto errTxt = [&]() {
    std::stringstream txt;
    txt << std::endl;
    txt << errorMessage;
    txt << "\nrun with `--help par` for more details\n";

    return txt.str();
  };

  nekrsCheck(length > 0, comm, EXIT_FAILURE, "%s\n", errTxt().c_str());
}

const std::vector<std::string> &getValidKeys(const std::string &section)
{
  if (!enforceLowerCase) {
    makeStringsLowerCase();
    enforceLowerCase = true;
  }

  if (section == "") {
    return noSectionKeys;
  }

  if (section == "general") {
    return generalKeys;
  }
  if (section == "neknek") {
    return neknekKeys;
  }
  if (section == "problemtype") {
    return problemTypeKeys;
  }
  if (section == "mesh") {
    return meshKeys;
  }
  if (section == "geom") {
    return geomKeys;
  }

  if (section == "fluid pressure") {
    return pressureKeys;
  }
  if (section == "scalar") {
    return scalarDefaultKeys;
  }
  if (section.find("scalar ") != std::string::npos) {
    return scalarKeys;
  }
  if (section == "amgx") {
    return amgxKeys;
  }
  if (section == "boomeramg") {
    return boomeramgKeys;
  }
  if (section == "occa") {
    return occaKeys;
  }
  if (section == "fluid velocity") {
    return velocityKeys;
  }
  if (section == "cvode") {
    return cvodeKeys;
  } else {
    return nothing;
  }
}

void validate(inipp::Ini *ini, const std::vector<std::string> &userSections)
{
  auto sections = ini->sections;

  int err = 0;
  bool generalExists = false;
  for (auto const &sec : sections) {
    if (sec.first.find("general") != std::string::npos) {
      generalExists = true;
    }
  }

  if (!generalExists) {
    std::ostringstream error;
    error << "mandatory section [GENERAL] not found!\n";
    append_error(error.str());
    err++;
  }

  bool defaultScalarExists = false;
  for (auto const &sec : sections) {
    if (sec.first == "scalar") {
      defaultScalarExists = true;
    }
  }

  for (auto const &sec : sections) {
    const auto isScalar = sec.first.find("scalar") != std::string::npos;

    std::istringstream iss(sec.first);
    std::string firstWord, secondWord;
    iss >> firstWord >> secondWord;

    if (isScalar) {
      if (firstWord != "scalar") {
        std::ostringstream error;
        error << "invalid scalar section " << sec.first << "\n"; 
        append_error(error.str());
        err++;
      }

      if (sec.first != "scalar" && scalarMap.find(secondWord) == scalarMap.end()) {
        std::ostringstream error;
        error << "scalar section defined for unknown scalar " << secondWord << "\n"; 
        append_error(error.str());
        err++;
      }
    }

    // check that section exists
    if (std::find(validSections.begin(), validSections.end(), sec.first) == validSections.end() &&
        !isScalar) {
      std::ostringstream error;
      error << "Invalid section name: " << sec.first << std::endl;
      append_error(error.str());
      err++;
    } else {
      const auto &validKeys = getValidKeys(sec.first);
      for (auto const &val : sec.second) {
        if (std::find(userSections.begin(), userSections.end(), sec.first) != userSections.end()) {
          continue;
        }

        if (std::find(validKeys.begin(), validKeys.end(), val.first) == validKeys.end()) {
          if (std::find(commonKeys.begin(), commonKeys.end(), val.first) == commonKeys.end()) {
            std::ostringstream error;
            error << "unknown key: " << sec.first << "::" << val.first << "\n";
            append_error(error.str());
            err++;
          }
        }
      }
    }
  }

  if (scalarMap.size() >= NSCALAR_MAX) {
    std::ostringstream error;
    error << "specified " << scalarMap.size() << " scalars, while the maximum allowed is "
          << NSCALAR_MAX << "\n";
    append_error(error.str());
    err++;
  }
}

void printDeprecation(const inipp::Ini::Sections &sections)
{
  for (auto const &sec : sections) {
    for (auto const &val : sec.second) {
      if (std::find(deprecatedKeys.begin(), deprecatedKeys.end(), val.first) != deprecatedKeys.end()) {
        std::cout << sec.first << "::" << val.first << " is deprecated and might be removed in the future!\n";
      }
    }
  }
}

} // namespace

void checkValidity(const int rank, const std::vector<std::string> &validValues, const std::string &entry)
{
  bool valid = false;
  for (auto &&v : validValues) {
    valid |= (entry.find(v) == 0);
  }

  if (!valid) {
    std::ostringstream ss;
    ss << "Value " << entry << " is not recognized!\n";
    ss << "\t\tValid values are:\n";
    for (auto &&v : validValues) {
      ss << "\t\t\t" << v << "\n";
    }
    append_value_error(ss.str());
  }
}

void parseConstFlowRate(const int rank, setupAide &options, inipp::Ini *ini)
{
  const std::vector<std::string> validValues = {
      {"constflowrate"},
      {"meanvelocity"},
      {"meanvolumetricflow"},
      {"bid"},
      {"direction"},
  };

  options.setArgs("CONSTANT FLOW RATE", "FALSE");

  std::string flowRateDescription;
  if (ini->extract("general", "constflowrate", flowRateDescription)) {
    options.setArgs("CONSTANT FLOW RATE", "TRUE");
    bool flowRateSet = false;
    bool flowDirectionSet = false;
    bool issueError = false;
    const std::vector<std::string> list = serializeString(flowRateDescription, '+');
    for (std::string s : list) {
      checkValidity(rank, validValues, s);

      const auto meanVelocityStr = parseValueForKey(s, "meanvelocity");
      if (!meanVelocityStr.empty()) {
        flowRateSet = true;
        options.setArgs("FLOW RATE", meanVelocityStr);
        options.setArgs("CONSTANT FLOW RATE TYPE", "BULK");
      }

      const auto meanVolumetricFlowStr = parseValueForKey(s, "meanvolumetricflow");
      if (!meanVolumetricFlowStr.empty()) {
        flowRateSet = true;
        options.setArgs("FLOW RATE", meanVolumetricFlowStr);
        options.setArgs("CONSTANT FLOW RATE TYPE", "VOLUMETRIC");
      }

      if (s.find("bid") == 0) {
        if (flowDirectionSet) {
          issueError = true;
        }
        flowDirectionSet = true;
        std::vector<std::string> items = serializeString(s, '=');

        std::string bidStr;
        if (items.size() == 2) {
          bidStr = items[1];
        } else {
          std::ostringstream error;
          error << "could not parse " << s << "!\n";
          append_error(error.str());
        }
        std::vector<std::string> bids = serializeString(items[1], ',');
        if (bids.size() == 2) {
          const int fromBID = std::stoi(bids[0]);
          const int toBID = std::stoi(bids[1]);
          options.setArgs("CONSTANT FLOW FROM BID", std::to_string(fromBID));
          options.setArgs("CONSTANT FLOW TO BID", std::to_string(toBID));
        } else {
          std::ostringstream error;
          error << "could not parse " << s << "!\n";
          append_error(error.str());
        }

        append_error(
            "Specifying a constant flow direction with a pair of BIDs is currently not supported.\n");
      }

      if (s.find("direction") == 0) {
        if (flowDirectionSet) {
          issueError = true;
        }
        flowDirectionSet = true;
        std::vector<std::string> items = serializeString(s, '=');
        if (items.size() == 2) {
          std::string direction = items[1];
          issueError = (direction.find("x") == std::string::npos &&
                        direction.find("y") == std::string::npos && direction.find("z") == std::string::npos);
          options.setArgs("CONSTANT FLOW DIRECTION", upperCase(direction));
        } else {
          std::ostringstream error;
          error << "could not parse " << s << "!\n";
          append_error(error.str());
        }
      }
    }
    if (!flowDirectionSet) {
      append_error("Flow direction has not been set in GENERAL:constFlowRate!\n");
    }
    if (!flowRateSet) {
      append_error("Flow rate has not been set in GENERAL:constFlowRate!\n");
    }
    if (issueError) {
      append_error("Error parsing GENERAL:constFlowRate!\n");
    }
  }
}

void parseCvodeSolver(const int rank, setupAide &options, inipp::Ini *ini)
{
#ifndef ENABLE_CVODE
  append_error("ERROR: CVODE not enabled! Recompile with -DENABLE_CVODE=ON\n");
  return;
#endif

  // default values
  double relativeTol;
  double absoluteTol;
  double hmax;
  int maxSteps = 500;
  double epsLin;
  double sigScale;
  bool recycleProps;
  bool mixedPrecisionJtv;

  std::string integrator = "bdf";

  const std::string parScope = "cvode";

  if (ini->extract(parScope, "relativetol", relativeTol)) {
    options.setArgs("CVODE RELATIVE TOLERANCE", to_string_f(relativeTol));
  }

  options.setArgs("CVODE GMRES BASIS VECTORS", "10");
  options.setArgs("CVODE SOLVER", "CBGMRES");

  // parse cvode linear solver
  [&]() {
    std::string p_solver;

    if (!ini->extract("cvode", "solver", p_solver)) {
      return;
    }

    const std::vector<std::string> validValues = {
        {"gmres"},
        {"cbgmres"},
        {"nvector"},
    };

    std::vector<std::string> list = serializeString(p_solver, '+');
    for (const std::string s : list) {
      checkValidity(rank, validValues, s);
    }

    if (p_solver.find("gmres") != std::string::npos) {
      std::vector<std::string> list;
      list = serializeString(p_solver, '+');

      std::string n = "10";

      for (std::string s : list) {
        const auto nvectorStr = parseValueForKey(s, "nvector");
        if (!nvectorStr.empty()) {
          n = nvectorStr;
        }
      }
      options.setArgs("CVODE GMRES BASIS VECTORS", n);

      if (p_solver.find("cb") != std::string::npos) {
        options.setArgs("CVODE SOLVER", "CBGMRES");
      } else {
        options.setArgs("CVODE SOLVER", "GMRES");
      }
    }
  }();

  if (options.compareArgs("VERBOSE", "TRUE")) {
    options.setArgs("CVODE VERBOSE", "TRUE");
  }

  options.setArgs("CVODE STOP TIME", "TRUE");

  if (ini->extract(parScope, "hmaxratio", hmax)) {
    options.setArgs("CVODE HMAX RATIO", std::to_string(hmax));
    options.setArgs("CVODE STOP TIME", "FALSE");
  }

  if (ini->extract(parScope, "epslin", epsLin)) {
    options.setArgs("CVODE EPS LIN", std::to_string(epsLin));
  }

  if (ini->extract(parScope, "maxSteps", maxSteps)) {
    options.setArgs("CVODE MAX STEPS", std::to_string(maxSteps));
  }

  int maxOrder;
  if (ini->extract(parScope, "maxOrder", maxOrder)) {
    options.setArgs("CVODE MAX TIMESTEPPER ORDER", std::to_string(maxOrder));
  }

  options.setArgs("CVODE GS TYPE", "CLASSICAL");
  std::string gstype;
  if (ini->extract(parScope, "gstype", gstype)) {
    if (gstype == "classical") {
      options.setArgs("CVODE GS TYPE", "CLASSICAL");
    } else if (gstype == "modified") {
      options.setArgs("CVODE GS TYPE", "MODIFIED");
    } else {
      append_error("Invalid gsType for " + parScope);
    }
  }

  options.setArgs("CVODE INTEGRATOR", upperCase(integrator));

  if (ini->extract(parScope, "dqsigma", sigScale)) {
    options.setArgs("CVODE DQ SIGMA", to_string_f(sigScale));
  }

  bool dealiasing ;
  ini->extract(parScope, "dealiasing", dealiasing);
  if (dealiasing && !options.compareArgs("OVERINTEGRATION", "TRUE")) {
    append_error("dealiasing for CVODE only is not supported!" + parScope);
  }

  std::string recyclePropsStr;
  if (ini->extract(parScope, "jtvrecycleproperties", recyclePropsStr)) {
    recycleProps = checkForTrue(recyclePropsStr);
    if (recycleProps) {
      options.setArgs("CVODE JTV RECYCLE PROPERTIES", "TRUE");
    } else {
      options.setArgs("CVODE JTV RECYCLE PROPERTIES", "FALSE");
    }
  }

  options.setArgs("CVODE SHARED RHO", "FALSE");
  std::string sharedRhoStr;
  if (ini->extract(parScope, "sharedrho", sharedRhoStr)) {
    bool sharedRho = checkForTrue(sharedRhoStr);
    if (sharedRho) {
      options.setArgs("CVODE SHARED RHO", "TRUE");
    } else {
      options.setArgs("CVODE SHARED RHO", "FALSE");
    }
  }

  std::string mixedPrecisionStr;
  if (ini->extract(parScope, "jtvmixedprecision", mixedPrecisionStr)) {
    mixedPrecisionJtv = checkForTrue(mixedPrecisionStr);
    if (mixedPrecisionJtv) {
      options.setArgs("CVODE MIXED PRECISION JTV", "TRUE");
    } else {
      options.setArgs("CVODE MIXED PRECISION JTV", "FALSE");
    }
  }

  if (mixedPrecisionJtv) {
    append_error("cvode::jtvmixedprecision is not supported yet!\n");
  }
}

void parseSolverTolerance(const int rank, setupAide &options, inipp::Ini *ini, std::string parScope)
{

  std::string parSectionName = upperCase(parPrefixFromParSection(parScope));

  const std::vector<std::string> validValues = {
      {"relative"},
  };

  std::string residualTol;
  if (ini->extract(parScope, "residualtol", residualTol) ||
      ini->extract(parScope, "residualtolerance", residualTol)) {
    if (residualTol.find("relative") != std::string::npos) {
      options.setArgs(parSectionName + "LINEAR SOLVER STOPPING CRITERION", "RELATIVE");

      const auto subStr = residualTol.substr(residualTol.find("relative"));
      if (subStr != "relative") {
        auto relTolerance = std::strtod(parseValueForKey(subStr, "relative").c_str(), nullptr);
        if (relTolerance > 0) {
          options.setArgs(parSectionName + "SOLVER RELATIVE TOLERANCE", to_string_f(relTolerance));
        }
      }
    }

    std::vector<std::string> entries = serializeString(residualTol, '+');
    for (std::string entry : entries) {
      double tolerance = std::strtod(entry.c_str(), nullptr);
      if (tolerance > 0) {
        options.setArgs(parSectionName + "SOLVER TOLERANCE", entry);
      } else {
        checkValidity(rank, validValues, entry);
      }
    }
  }

  std::string absoluteTol;
  if (ini->extract(parScope, "absolutetol", absoluteTol)) {
    bool issueError = false;
    std::string solver;
    if (ini->extract(parScope, "solver", solver)) {
      issueError |= (solver != "cvode" && solver != "none");
    } else {
      solver = options.getArgs(parSectionName + "SOLVER");
      issueError |= !options.compareArgs(parSectionName + "SOLVER", "CVODE");
    }
    if (issueError) {
      append_error("absoluteTol is only supported for solver=cvode");
    }
    options.setArgs(parSectionName + "CVODE ABSOLUTE TOLERANCE", absoluteTol);
  }
}

void parseCoarseGridDiscretization(const int rank, setupAide &options, inipp::Ini *ini, std::string parScope)
{
  std::string parSectionName = upperCase(parPrefixFromParSection(parScope));
  std::string p_coarseGridDiscretization;
  const bool continueParsing = ini->extract(parScope, "coarsegriddiscretization", p_coarseGridDiscretization);
  if (!continueParsing) {
    return;
  }

  const std::vector<std::string> validValues = {
      {"semfem"},
      {"fem"},
      {"galerkin"},
  };

  const auto entries = serializeString(p_coarseGridDiscretization, '+');
  for (auto &&s : entries) {
    checkValidity(rank, validValues, s);
  }

  // exit early if not using multigrid as preconditioner
  if (!options.compareArgs(parSectionName + "PRECONDITIONER", "MULTIGRID")) {
    return;
  }

  options.setArgs(parSectionName + "MULTIGRID SEMFEM", "FALSE");

  if (p_coarseGridDiscretization.find("semfem") != std::string::npos) {
    options.setArgs(parSectionName + "MULTIGRID SEMFEM", "TRUE");
  } else if (p_coarseGridDiscretization.find("fem") != std::string::npos) {
    options.setArgs(parSectionName + "GALERKIN COARSE OPERATOR", "FALSE");
    if (p_coarseGridDiscretization.find("galerkin") != std::string::npos) {
      options.setArgs(parSectionName + "GALERKIN COARSE OPERATOR", "TRUE");
    }
  }
}

void parseCoarseSolver(const int rank, setupAide &options, inipp::Ini *ini, std::string parScope)
{
  std::string parSectionName = upperCase(parPrefixFromParSection(parScope));

  std::string p_coarseSolver;
  const bool keyExist = ini->extract(parScope, "coarsesolver", p_coarseSolver) ||
                        ini->extract(parScope, "semfemsolver", p_coarseSolver);
  if (!keyExist) {
    if (parScope == "fluid pressure") {
      p_coarseSolver = "boomeramg";
    } else {
      p_coarseSolver = "smoother";
    }
  }

  const std::vector<std::string> validValues = {
      {"smoother"},
      {"boomeramg"},
      {"amgx"},
      {"cpu"},
      {"device"},
      {"overlap"},
  };

  std::vector<std::string> entries = serializeString(p_coarseSolver, '+');
  for (std::string entry : entries) {
    checkValidity(rank, validValues, entry);
  }

  const int smoother = p_coarseSolver.find("smoother") != std::string::npos;
  const int amgx = p_coarseSolver.find("amgx") != std::string::npos;
  const int boomer = p_coarseSolver.find("boomeramg") != std::string::npos;
  if (amgx + boomer > 1) {
    append_error("Conflicting solver types in coarseSolver!\n");
  }

  if (boomer) {
    const auto MGsemfem = options.compareArgs(parSectionName + "MULTIGRID SEMFEM", "TRUE");

    std::string smoother;
    options.getArgs(parSectionName + "MULTIGRID SMOOTHER", smoother);

    if ((smoother.find("DAMPEDJACOBI") != std::string::npos) && !MGsemfem) {
      options.setArgs("BOOMERAMG ITERATIONS", "2");
    }
  }

  if (boomer || amgx) {
    options.setArgs(parSectionName + "MULTIGRID COARSE SOLVE", "TRUE");
    options.setArgs(parSectionName + "COARSE SOLVER", "BOOMERAMG");
    if (amgx) {
      options.setArgs(parSectionName + "COARSE SOLVER", "AMGX");
      if (!AMGXenabled()) {
        append_error("AMGX was requested but is not enabled!\n");
      }
    }

    options.setArgs(parSectionName + "COARSE SOLVER PRECISION", "FP32");
    if (options.compareArgs(parSectionName + "PRECONDITIONER", "SEMFEM")) {
      options.setArgs(parSectionName + "COARSE SOLVER LOCATION", "DEVICE");
    } else {
      options.setArgs(parSectionName + "COARSE SOLVER LOCATION", "CPU");
      if (options.compareArgs(parSectionName + "MULTIGRID SEMFEM", "TRUE")) {
        options.setArgs(parSectionName + "COARSE SOLVER LOCATION", "DEVICE");
      }
    }

    for (std::string entry : entries) {
      if (entry.find("smoother") != std::string::npos) {
        options.setArgs(parSectionName + "MULTIGRID COARSE SOLVE AND SMOOTH", "TRUE");
      } else if (entry.find("cpu") != std::string::npos) {
        options.setArgs(parSectionName + "COARSE SOLVER LOCATION", "CPU");
      } else if (entry.find("device") != std::string::npos) {
        options.setArgs(parSectionName + "COARSE SOLVER LOCATION", "DEVICE");
      } else if (entry.find("overlap") != std::string::npos) {
        std::string currentSettings = options.getArgs(parSectionName + "MGSOLVER CYCLE");
        options.setArgs(parSectionName + "MGSOLVER CYCLE", currentSettings + "+OVERLAPCRS");

        if (!options.compareArgs(parSectionName + "MGSOLVER CYCLE", "ADDITIVE")) {
          append_error("Overlapping coarse solve requires additive multigrid!\n");
        }
      }
    }
  } else {
    options.setArgs(parSectionName + "COARSE SOLVER", "SMOOTHER");
    options.removeArgs(parSectionName + "COARSE SOLVER PRECISION");
    options.removeArgs(parSectionName + "COARSE SOLVER LOCATION");
    options.setArgs(parSectionName + "MULTIGRID COARSE SOLVE", "FALSE");
    if (options.compareArgs(parSectionName + "MGSOLVER CYCLE", "OVERLAPCRS")) {
      append_error("Overlap qualifier invalid if coarse solver is smoother!\n");
    }
  }

  {
    std::string preconditioner;
    options.getArgs(parSectionName + "MGSOLVER CYCLE", preconditioner);
    if (preconditioner.find("additive") != std::string::npos) {
      append_error("additive V-cycle does not support coarseSolver = smoother!\n");
    }
  }

  if (amgx && options.compareArgs(parSectionName + "COARSE SOLVER LOCATION", "CPU")) {
    append_error("AMGX on CPU is not supported!\n");
  }

  if (boomer && options.compareArgs(parSectionName + "COARSE SOLVER LOCATION", "GPU")) {
    if (hypreWrapperDevice::enabled()) {
      append_error("HYPRE is not configured to run on the GPU!\n");
    }
  }

  const bool runSolverOnDevice = options.compareArgs(parSectionName + "COARSE SOLVER LOCATION", "DEVICE");
  const bool overlapCrsSolve = options.compareArgs(parSectionName + "MGSOLVER CYCLE", "OVERLAPCRS");
  if (overlapCrsSolve && runSolverOnDevice) {
    append_error("Cannot overlap coarse grid solve when running coarse solver on the GPU!\n");
  }
}

bool is_number(const std::string &s)
{
  return !s.empty() &&
         std::find_if(s.begin(), s.end(), [](unsigned char c) { return !std::isdigit(c); }) == s.end();
}

std::vector<int> checkForIntInInputs(const std::vector<std::string> &inputs)
{
  std::vector<int> values;
  for (std::string s : inputs) {
    if (is_number(s)) {
      values.emplace_back(std::stoi(s));
    }
  }
  return values;
}

void parseSmoother(const int rank, setupAide &options, inipp::Ini *ini, std::string parScope)
{
  std::string p_smoother;

  std::string parSection = upperCase(parPrefixFromParSection(parScope));

  if (options.compareArgs(parSection + "PRECONDITIONER", "MULTIGRID")) {
    options.setArgs(parSection + "MULTIGRID SMOOTHER", "FOURTHOPTCHEBYSHEV+DAMPEDJACOBI");
    options.setArgs(parSection + "MULTIGRID CHEBYSHEV DEGREE", "1");
    options.setArgs(parSection + "MULTIGRID CHEBYSHEV MAX EIGENVALUE BOUND FACTOR", "1.1");
    if (parScope == "fluid pressure") {
      if (options.compareArgs(parSection + "SMOOTHED SEMFEM", "TRUE")) {
        options.setArgs(parSection + "MULTIGRID CHEBYSHEV DEGREE", "2");
      } else {
        options.setArgs(parSection + "MULTIGRID SMOOTHER", "FOURTHOPTCHEBYSHEV+ASM");
        options.setArgs(parSection + "MULTIGRID CHEBYSHEV DEGREE", "3");
      }
    }
  }

  if (!ini->extract(parScope, "smoothertype", p_smoother)) {
    return;
  }

  std::string p_preconditioner;
  ini->extract(parScope, "preconditioner", p_preconditioner);

  const std::vector<std::string> validValues = {
      {"asm"},
      {"ras"},
      {"cheby"},
      {"fourthcheby"},
      {"fourthoptcheby"},
      {"jac"},
      {"degree"},
      {"mineigenvalueboundfactor"},
      {"maxeigenvalueboundfactor"},
  };

  {
    const std::vector<std::string> list = serializeString(p_smoother, '+');
    for (const std::string s : list) {
      checkValidity(rank, validValues, s);
    }
  }

  if (options.compareArgs(parSection + "PRECONDITIONER", "MULTIGRID")) {
    std::vector<std::string> list;
    list = serializeString(p_smoother, '+');

    if (p_smoother.find("cheb") != std::string::npos) {
      bool surrogateSmootherSet = false;
      std::string chebyshevType = "";
      if (p_smoother.find("fourthopt") != std::string::npos) {
        chebyshevType = "FOURTHOPTCHEBYSHEV";
      } else if (p_smoother.find("fourth") != std::string::npos) {
        chebyshevType = "FOURTHCHEBYSHEV";
      } else {
        // using 1st-kind Chebyshev, so set a reasonable lmin multiplier
        chebyshevType = "CHEBYSHEV";
        options.setArgs(parSection + "MULTIGRID CHEBYSHEV MIN EIGENVALUE BOUND FACTOR", "0.1");
      }

      for (std::string s : list) {
        const auto degreeStr = parseValueForKey(s, "degree");
        if (!degreeStr.empty()) {
          options.setArgs(parSection + "MULTIGRID CHEBYSHEV DEGREE", degreeStr);
        }

        const auto minEigBoundStr = parseValueForKey(s, "mineigenvalueboundfactor");
        if (!minEigBoundStr.empty()) {
          if (chebyshevType.find("FOURTH") != std::string::npos) {
            append_error(
                "minEigenvalueBoundFactor not supported for 4th kind or Opt. 4th kind Chebyshev smoother!\n");
          }
          options.setArgs(parSection + "MULTIGRID CHEBYSHEV MIN EIGENVALUE BOUND FACTOR", minEigBoundStr);
        }

        const auto maxEigBoundStr = parseValueForKey(s, "maxeigenvalueboundfactor");
        if (!maxEigBoundStr.empty()) {
          options.setArgs(parSection + "MULTIGRID CHEBYSHEV MAX EIGENVALUE BOUND FACTOR", maxEigBoundStr);
        }

        if (s.find("jac") != std::string::npos) {
          surrogateSmootherSet = true;
          options.setArgs(parSection + "MULTIGRID SMOOTHER", chebyshevType + "+DAMPEDJACOBI");
          if (p_preconditioner.find("additive") != std::string::npos) {
            append_error("Additive vcycle is not supported for Chebyshev smoother");
          }
        } else if (s.find("asm") != std::string::npos) {
          surrogateSmootherSet = true;
          options.setArgs(parSection + "MULTIGRID SMOOTHER", chebyshevType + "+ASM");
          if (p_preconditioner.find("additive") != std::string::npos) {
            append_error("Additive vcycle is not supported for hybrid Schwarz/Chebyshev smoother");
          }
        } else if (s.find("ras") != std::string::npos) {
          surrogateSmootherSet = true;
          options.setArgs(parSection + "MULTIGRID SMOOTHER", chebyshevType + "+RAS");
          if (p_preconditioner.find("additive") != std::string::npos) {
            append_error("Additive vcycle is not supported for hybrid Schwarz/Chebyshev smoother");
          }
        }
      }

      if (!surrogateSmootherSet) {
        append_error("Inner Chebyshev smoother not set");
      }
      return;
    }

    // Non-Chebyshev smoothers
    options.removeArgs(parSection + "MULTIGRID CHEBYSHEV DEGREE");
    options.removeArgs(parSection + "MULTIGRID CHEBYSHEV MAX EIGENVALUE BOUND FACTOR");
    if (p_smoother.find("asm") == 0) {
      options.setArgs(parSection + "MULTIGRID SMOOTHER", "ASM");
      if (p_preconditioner.find("multigrid") != std::string::npos) {
        if (p_preconditioner.find("additive") == std::string::npos) {
          append_error("ASM smoother only supported for additive V-cycle");
        }
      } else {
        options.setArgs(parSection + "MGSOLVER CYCLE", "VCYCLE+ADDITIVE+OVERLAPCRS");
      }
    } else if (p_smoother.find("ras") == 0) {
      options.setArgs(parSection + "MULTIGRID SMOOTHER", "RAS");
      if (p_preconditioner.find("multigrid") != std::string::npos) {
        if (p_preconditioner.find("additive") == std::string::npos) {
          append_error("RAS smoother only supported for additive V-cycle");
        }
      } else {
        options.setArgs(parSection + "MGSOLVER CYCLE", "VCYCLE+ADDITIVE+OVERLAPCRS");
      }
    } else if (p_smoother.find("jac") == 0) {
      append_error("Jacobi smoother requires Chebyshev");
      options.setArgs(parSection + "MULTIGRID SMOOTHER", "DAMPEDJACOBI");
      if (p_preconditioner.find("additive") != std::string::npos) {
        append_error("Additive vcycle is not supported for Jacobi smoother");
      }
    } else {
      append_error("Unknown ::smootherType");
    }
  }
}

void parsePreconditioner(const int rank, setupAide &options, inipp::Ini *ini, std::string parScope)
{
  const std::vector<std::string> validValues = {
      {"none"},
      {"jac"},
      {"semfem"},
      {"nonsmoothed"},
      {"femsem"},
      {"pmg"},
      {"multigrid"},
      {"additive"},
      {"multiplicative"},
  };

  std::string parSection = upperCase(parPrefixFromParSection(parScope));

  std::string p_preconditioner;
  if (!ini->extract(parScope, "preconditioner", p_preconditioner)) {
    p_preconditioner = "jac";
    if (parScope.find("fluid pressure") != std::string::npos) {
      p_preconditioner = "multigrid";
    }
  }

  const std::vector<std::string> list = serializeString(p_preconditioner, '+');
  for (std::string s : list) {
    checkValidity(rank, validValues, s);
  }

  const bool mg = p_preconditioner.find("pmg") != std::string::npos ||
                  p_preconditioner.find("multigrid") != std::string::npos;

  if (p_preconditioner.find("none") != std::string::npos) {
    options.setArgs(parSection + "PRECONDITIONER", "NONE");
    return;
  } else if (p_preconditioner.find("jac") != std::string::npos) {
    options.setArgs(parSection + "PRECONDITIONER", "JACOBI");
    options.setArgs(parSection + "ELLIPTIC PRECO COEFF FIELD", "TRUE");
  } else if (mg) {
    options.setArgs(parSection + "PRECONDITIONER", "MULTIGRID");
    options.setArgs(parSection + "ELLIPTIC PRECO COEFF FIELD", "FALSE");
    std::string key = "VCYCLE+MULTIPLICATIVE";
    if (p_preconditioner.find("additive") != std::string::npos) {
      key = "VCYCLE+ADDITIVE";
    } else if (p_preconditioner.find("multiplicative") != std::string::npos) {
      key = "VCYCLE+MULTIPLICATIVE";
    }
    options.setArgs(parSection + "MGSOLVER CYCLE", key);
  } else if (p_preconditioner.find("semfem") != std::string::npos ||
             p_preconditioner.find("femsem") != std::string::npos) {
    auto smoothed = (p_preconditioner.find("nonsmoothed") != std::string::npos) ? false : true;

    std::string p_coarseGridDiscretization;
    if (ini->extract(parScope, "coarsegriddiscretization", p_coarseGridDiscretization)) {
      if (p_coarseGridDiscretization.find("semfem") != std::string::npos) {
        smoothed = false;
      }
    }

    options.setArgs(parSection + "PRECONDITIONER", "SEMFEM");
    options.setArgs(parSection + "SMOOTHED SEMFEM", "FALSE");
    if (smoothed) {
      options.setArgs(parSection + "SMOOTHED SEMFEM", "TRUE");
      options.setArgs(parSection + "PRECONDITIONER", "MULTIGRID");
      options.setArgs(parSection + "MGSOLVER CYCLE", "VCYCLE+MULTIPLICATIVE");
      options.setArgs(parSection + "MULTIGRID COARSE SOLVE AND SMOOTH", "TRUE");
      options.setArgs(parSection + "MULTIGRID SEMFEM", "TRUE");
      options.setArgs(parSection + "ELLIPTIC PRECO COEFF FIELD", "FALSE");
    }
  }

  parseSmoother(rank, options, ini, parScope);

  parseCoarseGridDiscretization(rank, options, ini, parScope);

  if (options.compareArgs(parSection + "PRECONDITIONER", "MULTIGRID") ||
      options.compareArgs(parSection + "PRECONDITIONER", "SEMFEM")) {
    parseCoarseSolver(rank, options, ini, parScope);
  }

  if (options.compareArgs(parSection + "PRECONDITIONER", "MULTIGRID")) {
    std::string p_mgschedule;
    if (ini->extract(parScope, "pmgschedule", p_mgschedule)) {
      const auto semfem = options.compareArgs(parSection + "SMOOTHED SEMFEM", "TRUE");
      if (semfem) {
        append_error("pMGSchedule not supported for preconditioner = semfem.\n");
      }

      options.setArgs(parSection + "MULTIGRID SCHEDULE", p_mgschedule);

      options.removeArgs(parSection + "MULTIGRID CHEBYSHEV DEGREE");

      // validate multigrid schedule
      // note: default order here is not actually required
      auto [scheduleMap, errorString] = ellipticParseMultigridSchedule(p_mgschedule, options, 3);
      if (!errorString.empty()) {
        append_error(errorString);
      }

      int minDegree = std::numeric_limits<int>::max();
      for (auto &&[cyclePosition, smootherOrder] : scheduleMap) {
        auto [polyOrder, isDownLeg] = cyclePosition;
        minDegree = std::min(minDegree, polyOrder);
      }

      const auto INVALID = std::numeric_limits<int>::lowest();

      // bail if degree is set _and_ it conflicts
      std::string p_smoother;
      if (ini->extract(parScope, "smoothertype", p_smoother)) {
        for (auto &&s : serializeString(p_smoother, '+')) {
          if (s.find("degree") != std::string::npos) {
            const auto degreeStr = parseValueForKey(s, "degree");
            if (!degreeStr.empty()) {
              const auto specifiedDegree = std::stoi(degreeStr);
              for (auto &&[cyclePosition, smootherOrder] : scheduleMap) {
                auto [polyOrder, isDownLeg] = cyclePosition;
                const bool degreeConflicts = smootherOrder != specifiedDegree;
                const bool isMinOrder = polyOrder == minDegree;
                const bool minOrderInvalid = smootherOrder == INVALID;

                if (isMinOrder && minOrderInvalid) {
                  continue;
                }

                if (degreeConflicts) {
                  append_error(
                      "order specified in pMGSchedule conflicts with that specified in smootherType!\n");
                }
              }
            }
          }
        }
      }

      // bail if coarse degree is set, but we're not smoothing on the coarsest level
      if (scheduleMap[{minDegree, true}] > 0) {

        const bool smoothCrs = options.compareArgs(parSection + "COARSE SOLVER", "SMOOTHER") ||
                               options.compareArgs(parSection + "MULTIGRID COARSE SOLVE AND SMOOTH", "TRUE");
        if (!smoothCrs && !semfem) {
          append_error("specified coarse Chebyshev degree, but coarseSolver=smoother is not set.\n");
        }
      }
    }
  }
}

void parseLinearSolver(const int rank, setupAide &options, inipp::Ini *ini, std::string parScope)
{
  std::string parSectionName = upperCase(parPrefixFromParSection(parScope));

  int maxIter = 500;
  ini->extract(parScope, "maxiterations", maxIter);

  options.setArgs(parSectionName + "MAXIMUM ITERATIONS", std::to_string(maxIter));

  std::string noop;
  bool applyDefault = (options.getArgs(parSectionName + "SOLVER", noop) == 0);

  if (applyDefault) {
    options.setArgs(parSectionName + "SOLVER", "PCG");
    if (options.compareArgs(parSectionName + "PRECONDITIONER", "JACOBI")) {
#if 0
      options.setArgs(parSectionName + "SOLVER", "PCG+COMBINED");
#else
      options.setArgs(parSectionName + "SOLVER", "PCG");
#endif
    }

    if (parScope == "fluid pressure") {
      options.setArgs(parSectionName + "SOLVER", "PGMRES+FLEXIBLE");
      options.setArgs(parSectionName + "PGMRES RESTART", "15");
    }
    if (parScope == "fluid mesh") {
      options.setArgs(parSectionName + "SOLVER", "NONE");
    }

    if (parScope == "fluid velocity" || parScope == "geom") {
      options.setArgs(parSectionName + "BLOCK SOLVER", "TRUE");
    }
  }

  std::string p_solver;
  if (!ini->extract(parScope, "solver", p_solver)) {
    return;
  }

  const std::vector<std::string> validValues = {
      {"user"},
      {"cvode"},
      {"none"},
      {"nvector"},
      {"pfgmres"},
      {"pfcg"},
      {"flexible"},
      {"pgmres"},
      {"pcg"},
      {"combined"},
      {"block"},
  };
  std::vector<std::string> list = serializeString(p_solver, '+');
  for (const std::string s : list) {
    checkValidity(rank, validValues, s);
  }

  if (p_solver.find("gmres") != std::string::npos) {
    std::vector<std::string> list;
    list = serializeString(p_solver, '+');
    std::string n = "15";
    for (std::string s : list) {
      const auto nvectorStr = parseValueForKey(s, "nvector");
      if (!nvectorStr.empty()) {
        n = nvectorStr;
      }
    }
    options.setArgs(parSectionName + "PGMRES RESTART", n);
    if (p_solver.find("fgmres") != std::string::npos || p_solver.find("flexible") != std::string::npos) {
      p_solver = "PGMRES+FLEXIBLE";
    } else {
      p_solver = "PGMRES";
    }
  } else if (p_solver.find("cg") != std::string::npos) {
    if (p_solver.find("block") != std::string::npos) {
      options.setArgs(parSectionName + "BLOCK SOLVER", "TRUE");
    } else {
      options.setArgs(parSectionName + "BLOCK SOLVER", "FALSE");
    }

    if (p_solver.find("fcg") != std::string::npos || p_solver.find("flexible") != std::string::npos) {
      p_solver = "PCG+FLEXIBLE";
      if (p_solver.find("combined") != std::string::npos) {
        std::ostringstream ss;
        ss << "combined PCG solver not supported with flexible preconditioner!\n";
        append_value_error(ss.str());
      }
    } else {
      if (p_solver.find("combined") != std::string::npos) {
        if (!options.compareArgs(parSectionName + "PRECONDITIONER", "JACOBI")) {
          std::ostringstream ss;
          ss << "combined PCG solver only supported with Jacobi preconditioner!\n";
          append_value_error(ss.str());
        }
        p_solver = "PCG+COMBINED";
      } else {
        p_solver = "PCG";
      }
    }
  } else if (p_solver.find("user") != std::string::npos) {
    p_solver = "USER";
  } else if (p_solver.find("cvode") != std::string::npos) {
    p_solver = "CVODE";
  } else if (p_solver.find("none") != std::string::npos) {
    p_solver = "NONE";
  } else {
    append_error("Invalid solver for " + parScope);
  }
  options.setArgs(parSectionName + "SOLVER", p_solver);
}

void parseInitialGuess(const int rank, setupAide &options, inipp::Ini *ini, std::string parScope)
{
  std::string parSectionName = upperCase(parPrefixFromParSection(parScope));

  std::string initialGuess;

  const std::vector<std::string> validValues = {
      {"projectionaconj"},
      {"projection"},
      {"extrapolation"},
      {"previous"},
      // settings
      {"nvector"},
      {"start"},
  };

  options.setArgs(parSectionName + "INITIAL GUESS", "EXTRAPOLATION");
  if (parScope == "fluid pressure") {
    options.setArgs(parSectionName + "INITIAL GUESS", "PROJECTION-ACONJ");
  }

  if (ini->extract(parScope, "initialguess", initialGuess)) {
    if (initialGuess.find("extrapolation") != std::string::npos) {
      options.setArgs(parSectionName + "INITIAL GUESS", "EXTRAPOLATION");

      if (parScope == "fluid pressure") {
        append_error("initialGuess = extrapolation not supported for pressure!\n");
      }
      return;
    }

    const int defaultNumVectors = (parScope == "fluid pressure") ? 10 : 5;
    int proj = false;

    if (initialGuess.find("projectionaconj") != std::string::npos) {
      options.setArgs(parSectionName + "INITIAL GUESS", "PROJECTION-ACONJ");
      proj = true;
    } else if (initialGuess.find("projection") != std::string::npos) {
      options.setArgs(parSectionName + "INITIAL GUESS", "PROJECTION");
      proj = true;
    } else if (initialGuess.find("previous") != std::string::npos) {
      options.setArgs(parSectionName + "INITIAL GUESS", "PREVIOUS");
    } else {
      std::ostringstream error;
      error << "Could not parse initialGuess = " << initialGuess << "!\n";
      append_error(error.str());
    }

    if (proj) {
      options.setArgs(parSectionName + "RESIDUAL PROJECTION VECTORS", std::to_string(defaultNumVectors));
      options.setArgs(parSectionName + "RESIDUAL PROJECTION START", "5");
    }

    const std::vector<std::string> list = serializeString(initialGuess, '+');

    for (std::string s : list) {
      checkValidity(rank, validValues, s);

      const auto nVectorStr = parseValueForKey(s, "nvector");
      if (!nVectorStr.empty() && proj) {
        options.setArgs(parSectionName + "RESIDUAL PROJECTION VECTORS", nVectorStr);
      }

      const auto startStr = parseValueForKey(s, "start");
      if (!startStr.empty() && proj) {
        options.setArgs(parSectionName + "RESIDUAL PROJECTION START", startStr);
      }
    }
    return;
  }
}

void parseCheckpointing(const int rank, setupAide &options, inipp::Ini *ini, std::string parSection)
{
  std::string val = "true";
  if (ini->extract(parSection, "checkpointing", val)) {
     if (val == "true") {
       val = "true"; 
     } else {
       val = "false";
     }
  }

  std::string parPrefix = upperCase(parPrefixFromParSection(parSection));

  options.setArgs(parPrefix + "CHECKPOINTING", upperCase(val));
}

void parseRegularization(const int rank, setupAide &options, inipp::Ini *ini, std::string parSection)
{
  int N;
  options.getArgs("POLYNOMIAL DEGREE", N);
  const bool isScalar = (parSection.find("scalar") != std::string::npos); 
  const bool isVelocity = parSection.find("fluid velocity") != std::string::npos;
  std::string sbuf;

  std::string parPrefix = upperCase(parPrefixFromParSection(parSection));

  [&]() {
    std::string regularization;
    if (ini->extract(parSection, "regularization", regularization)) {
      const std::vector<std::string> validValues = {
          {"none"},
          {"hpfrt"},
          {"gjp"},
          {"avm"},
          {"c0"},
          {"nmodes"},
          {"cutoffratio"},
          {"penaltyfactor"},
          {"scalingcoeff"},
          {"activationwidth"},
          {"decaythreshold"},
          {"noisethreshold"},

      };
      const std::vector<std::string> list = serializeString(regularization, '+');
      for (const std::string s : list) {
        checkValidity(rank, validValues, s);
      }
      if (regularization.find("none") != std::string::npos) {
        options.setArgs(parPrefix + "REGULARIZATION METHOD", "NONE");
        return;
      }
      const bool usesAVM = std::find(list.begin(), list.end(), "avm") != list.end();
      const bool usesGJP = std::find(list.begin(), list.end(), "gjp") != list.end();
      const bool usesHPFRT = std::find(list.begin(), list.end(), "hpfrt") != list.end();

      if (!usesAVM && !usesHPFRT && !usesGJP) {
        append_error("unknown regularization!\n");
      }

      if (usesAVM && isVelocity) {
        append_error("avm regularization is only enabled for scalars!\n");
      }

      if (usesGJP) {
        options.setArgs(parPrefix + "REGULARIZATION METHOD", "GJP");
        options.setArgs(parPrefix + "REGULARIZATION GJP PENALTY FACTOR", "0.8");
        for (std::string s : list) {
          const auto penaltyStr = parseValueForKey(s, "penaltyfactor");
          if (!penaltyStr.empty()) {
            options.setArgs(parPrefix + "REGULARIZATION GJP PENALTY FACTOR", penaltyStr);
          }
        }
      }

      if (usesHPFRT) {
        options.setArgs(parPrefix + "HPFRT MODES", "1");
        options.setArgs(parPrefix + "REGULARIZATION METHOD", "HPFRT");
        if (usesGJP) options.setArgs(parPrefix + "REGULARIZATION METHOD", "GJP+HPFRT");

      }

      if (usesAVM) {
        options.setArgs(parPrefix + "REGULARIZATION METHOD", "AVM_AVERAGED_MODAL_DECAY");
        if (usesGJP) options.setArgs(parPrefix + "REGULARIZATION METHOD", "GJP+AVM_AVERAGED_MODAL_DECAY");
        options.setArgs(parPrefix + "REGULARIZATION AVM ACTIVATION WIDTH", to_string_f(1.0));
        options.setArgs(parPrefix + "REGULARIZATION AVM DECAY THRESHOLD", to_string_f(2.0));
        options.setArgs(parPrefix + "REGULARIZATION AVM C0", "FALSE");

        for (std::string s : list) {

          const auto nmodeStr = parseValueForKey(s, "nmodes");
          if (!nmodeStr.empty()) {
            append_error("nModes qualifier is invalid for avm!\n");
          }
          const auto cutoffratioStr = parseValueForKey(s, "cutoffratio");
          if (!cutoffratioStr.empty()) {
            append_error("cutoffRatio qualifier is invalid for avm!\n");
          }

          const auto absTolStr = parseValueForKey(s, "noisethreshold");
          if (!absTolStr.empty()) {
            options.setArgs(parPrefix + "REGULARIZATION AVM ABSOLUTE TOL", absTolStr);
          }

          const auto scalingcoeffStr = parseValueForKey(s, "scalingcoeff");
          if (!scalingcoeffStr.empty()) {
            options.setArgs(parPrefix + "REGULARIZATION AVM SCALING COEFF", scalingcoeffStr);
          }

          if (s.find("c0") != std::string::npos) {
            options.setArgs(parPrefix + "REGULARIZATION AVM C0", "TRUE");
          }

          const auto rampConstantStr = parseValueForKey(s, "activationwidth");
          if (!rampConstantStr.empty()) {
            options.setArgs(parPrefix + "REGULARIZATION AVM ACTIVATION WIDTH", rampConstantStr);
          }
          const auto thresholdStr = parseValueForKey(s, "decaythreshold");
          if (!thresholdStr.empty()) {
            options.setArgs(parPrefix + "REGULARIZATION AVM DECAY THRESHOLD", thresholdStr);
          }
        }

        if (options.getArgs(parPrefix + "REGULARIZATION AVM ABSOLUTE TOL").empty()) {
          append_error("absoluteTol qualifier required for avm!\n");
        }

      }

      if (usesHPFRT) {
        bool setsStrength = false;
        for (std::string s : list) {
          const auto nmodeStr = parseValueForKey(s, "nmodes");
          if (!nmodeStr.empty()) {
            double value = std::stod(nmodeStr);
            value = round(value);
            options.setArgs(parPrefix + "HPFRT MODES", to_string_f(value));
          }
          const auto cutoffRatioStr = parseValueForKey(s, "cutoffratio");
          if (!cutoffRatioStr.empty()) {
            double filterCutoffRatio = std::stod(cutoffRatioStr);
            double NFilterModes = round((N + 1) * (1 - filterCutoffRatio));
            options.setArgs(parPrefix + "HPFRT MODES", to_string_f(NFilterModes));
          }

          const auto scalingCoeffStr = parseValueForKey(s, "scalingcoeff");
          if (!scalingCoeffStr.empty()) {
            setsStrength = true;
            int err = 0;
            double weight = parseFormula(scalingCoeffStr.c_str(), &err);
            if (err) {
              append_error("Invalid expression for scalingCoeff");
            }
            options.setArgs(parPrefix + "HPFRT STRENGTH", to_string_f(weight));
          }
        }
        if (!setsStrength) {
          append_error("required parameter scalingCoeff for hpfrt regularization is not "
                       "set!\n");
        }
      }
      return;
    } else {
      // if options already exist, don't overwrite them with defaults from general
      std::string regularizationMethod;
      if (options.getArgs(parPrefix + "REGULARIZATION METHOD", regularizationMethod)) {
        return;
      }

      // use default settings, if applicable
      std::string defaultSettings;
      if (ini->extract("general", "regularization", defaultSettings)) {
        options.setArgs(parPrefix + "REGULARIZATION METHOD", options.getArgs("REGULARIZATION METHOD"));
        options.setArgs(parPrefix + "HPFRT MODES", options.getArgs("HPFRT MODES"));

        if (defaultSettings.find("hpfrt") != std::string::npos) {
          options.setArgs(parPrefix + "HPFRT STRENGTH", options.getArgs("HPFRT STRENGTH"));
        }

        if (defaultSettings.find("gjp") != std::string::npos) {
          options.setArgs(parPrefix + "REGULARIZATION GJP PENALTY FACTOR", options.getArgs("REGULARIZATION GJP PENALTY FACTOR"));
        }

        if (defaultSettings.find("avm") != std::string::npos) {
          if (isVelocity) {
            // Catch if the general block is using AVM + no [VELOCITY] specification
            append_error("avm regularization is only enabled for scalars!\n");
          }
          options.setArgs(parPrefix + "REGULARIZATION VISMAX COEFF",
                          options.getArgs("REGULARIZATION VISMAX COEFF"));
          options.setArgs(parPrefix + "REGULARIZATION SCALING COEFF",
                          options.getArgs("REGULARIZATION SCALING COEFF"));
          options.setArgs(parPrefix + "REGULARIZATION MDH ACTIVATION WIDTH",
                          options.getArgs("REGULARIZATION MDH ACTIVATION WIDTH"));
          options.setArgs(parPrefix + "REGULARIZATION MDH THRESHOLD",
                          options.getArgs("REGULARIZATION MDH THRESHOLD"));
          options.setArgs(parPrefix + "REGULARIZATION AVM C0", options.getArgs("REGULARIZATION AVM C0"));
          options.setArgs(parPrefix + "REGULARIZATION HPF MODES",
                          options.getArgs("REGULARIZATION HPF MODES"));
        }
      }
    }
  }();

  // if regularization method has not been set, fall back to none
  std::string regularizationMethod;
  if (options.getArgs(parPrefix + "REGULARIZATION METHOD", regularizationMethod) == 0) {
    options.setArgs(parPrefix + "REGULARIZATION METHOD", "NONE");
  }
}

void parseBoomerAmgSection(const int rank, setupAide &options, inipp::Ini *ini)
{
  if (ini->sections.count("boomeramg")) {
    int coarsenType;
    if (ini->extract("boomeramg", "coarsentype", coarsenType)) {
      options.setArgs("BOOMERAMG COARSEN TYPE", std::to_string(coarsenType));
    }
    int interpolationType;
    if (ini->extract("boomeramg", "interpolationtype", interpolationType)) {
      options.setArgs("BOOMERAMG INTERPOLATION TYPE", std::to_string(interpolationType));
    }
    int smootherType;
    if (ini->extract("boomeramg", "smoothertype", smootherType)) {
      options.setArgs("BOOMERAMG SMOOTHER TYPE", std::to_string(smootherType));
    }
    int coarseSmootherType;
    if (ini->extract("boomeramg", "coarsesmoothertype", coarseSmootherType)) {
      options.setArgs("BOOMERAMG COARSE SMOOTHER TYPE", std::to_string(coarseSmootherType));
    }
    int numCycles;
    if (ini->extract("boomeramg", "iterations", numCycles)) {
      options.setArgs("BOOMERAMG ITERATIONS", std::to_string(numCycles));
    }
    double strongThres;
    if (ini->extract("boomeramg", "strongthreshold", strongThres)) {
      options.setArgs("BOOMERAMG STRONG THRESHOLD", to_string_f(strongThres));
    }
    double nonGalerkinTol;
    if (ini->extract("boomeramg", "nongalerkintol", nonGalerkinTol)) {
      options.setArgs("BOOMERAMG NONGALERKIN TOLERANCE", to_string_f(nonGalerkinTol));
    }
    int aggLevels;
    if (ini->extract("boomeramg", "aggressivecoarseninglevels", aggLevels)) {
      options.setArgs("BOOMERAMG AGGRESSIVE COARSENING LEVELS", std::to_string(aggLevels));
    }
    int chebyRelaxOrder;
    if (ini->extract("boomeramg", "chebyshevrelaxorder", chebyRelaxOrder)) {
      options.setArgs("BOOMERAMG CHEBYSHEV RELAX ORDER", std::to_string(chebyRelaxOrder));
    }
    double chebyFraction;
    if (ini->extract("boomeramg", "chebyshevfraction", chebyFraction)) {
      options.setArgs("BOOMERAMG CHEBYSHEV FRACTION", std::to_string(chebyFraction));
    }
  }
}

void parseOccaSection(const int rank, setupAide &options, inipp::Ini *ini)
{
  std::string backendSpecification;
  if (ini->extract("occa", "backend", backendSpecification)) {
    const std::vector<std::string> validBackends = {
        {"serial"},
        {"cpu"},
        {"cuda"},
        {"hip"},
        {"dpcpp"},
        {"opencl"},
    };
    const std::vector<std::string> validArchitectures = {
        {"arch"}, // include the arch= specifier here
        {"x86"},
    };

    std::vector<std::string> validValues = validBackends;
    validValues.insert(validValues.end(), validArchitectures.begin(), validArchitectures.end());

    const std::vector<std::string> list = serializeString(backendSpecification, '+');
    for (const std::string entry : list) {
      const std::vector<std::string> arguments = serializeString(entry, '=');
      for (const std::string argument : arguments) {
        checkValidity(rank, validValues, argument);
      }
    }

    std::string threadModel = "";
    std::string architecture = "";
    for (const std::string entry : list) {
      const std::vector<std::string> arguments = serializeString(entry, '=');
      if (arguments.size() == 1) {
        for (const std::string backend : validBackends) {
          if (backend == arguments.at(0)) {
            threadModel = backend;
          }
        }
      } else if (arguments.size() == 2) {
        for (const std::string arch : validArchitectures) {
          if (arch == arguments.at(1)) {
            architecture = arch;
          }
        }
      } else {
        std::ostringstream error;
        error << "Could not parse string \"" << entry << "\" while parsing OCCA:backend.\n";
        append_error(error.str());
      }
    }

    if (threadModel.empty()) {
      std::ostringstream error;
      error << "Could not parse valid backend from \"" << backendSpecification
            << "\" while parsing OCCA:backend.\n";
      append_error(error.str());
    }

    options.setArgs("THREAD MODEL", upperCase(threadModel));

    if (!architecture.empty()) {
      options.setArgs("ARCHITECTURE", upperCase(architecture));
    }
  }

  std::string deviceNumber;
  if (ini->extract("occa", "devicenumber", deviceNumber)) {
    options.setArgs("DEVICE NUMBER", upperCase(deviceNumber));
  }

  std::string platformNumber;
  if (ini->extract("occa", "platformnumber", platformNumber)) {
    options.setArgs("PLATFORM NUMBER", upperCase(platformNumber));
  }
}

void parseGeneralSection(const int rank, setupAide &options, inipp::Ini *ini)
{
  // GENERAL
  bool verbose = false;
  if (ini->extract("general", "verbose", verbose)) {
    if (verbose) {
      options.setArgs("VERBOSE", "TRUE");
    }
  }

  std::string startFrom;
  if (ini->extract("general", "startfrom", startFrom)) {
    options.setArgs("RESTART FILE NAME", startFrom);
  }

  int N;
  if (ini->extract("general", "polynomialorder", N)) {
    options.setArgs("POLYNOMIAL DEGREE", std::to_string(N));
    if (N > 10) {
      append_error("polynomialOrder > 10 is currently not supported");
    }
  } else {
    append_error("cannot find mandatory parameter GENERAL::polynomialOrder");
  }

  // udf file
  {
    std::string udfFile;
    if (ini->extract("general", "udf", udfFile)) {
      options.setArgs("UDF FILE", udfFile);
    }
  }

  // usr file
  {
    std::string usrFile;
    if (ini->extract("general", "usr", usrFile)) {
      options.setArgs("NEK USR FILE", usrFile);
    }
  }

  // oudf file
  {
    std::string oudfFile;
    if (ini->extract("general", "oudf", oudfFile)) {
      options.setArgs("UDF OKL FILE", oudfFile);
    }
  }

  {
    options.setArgs("SUBCYCLING STEPS", "0");
    int NSubCycles = 0;
    if (ini->extract("general", "advectionsubcyclingsteps", NSubCycles)) {
      options.setArgs("SUBCYCLING STEPS", std::to_string(NSubCycles));
    }
  }

  std::string dtString;
  if (ini->extract("general", "dt", dtString)) {
    const std::vector<std::string> validValues = {
        {"targetcfl"},
        {"max"},
        {"initial"},
    };

    bool useVariableDt = false;
    for (auto &&variableDtEntry : validValues) {
      if (dtString.find(variableDtEntry) != std::string::npos) {
        useVariableDt = true;
      }
    }

    if (useVariableDt) {
      bool userSuppliesInitialDt = false;
      bool userSuppliesTargetCFL = false;
      options.setArgs("VARIABLE DT", "TRUE");
      options.setArgs("TARGET CFL", "0.5");
      std::vector<std::string> entries = serializeString(dtString, '+');
      for (std::string entry : entries) {
        checkValidity(rank, validValues, entry);

        const auto maxStr = parseValueForKey(entry, "max");
        if (!maxStr.empty()) {
          options.setArgs("MAX DT", maxStr);
        }

        const auto initialStr = parseValueForKey(entry, "initial");
        if (!initialStr.empty()) {
          options.setArgs("DT", initialStr);
        }

        const auto cflStr = parseValueForKey(entry, "targetcfl");
        if (!cflStr.empty()) {
          options.setArgs("TARGET CFL", cflStr);
          const double targetCFL = std::stod(cflStr);
          int NSubCycles = std::ceil(targetCFL / 2.0);
          if (targetCFL <= 0.51) {
            NSubCycles = 0;
          }

          int NSubCyclesSpecified = 0;
          if (ini->extract("general", "advectionsubcyclingsteps", NSubCyclesSpecified)) {
            options.setArgs("SUBCYCLING STEPS", std::to_string(NSubCyclesSpecified));
          } else {
            options.setArgs("SUBCYCLING STEPS", std::to_string(NSubCycles));
          }

          userSuppliesTargetCFL = true;
        }
      }

      // if targetCFL is not set, try to infer from subcyclingSteps
      if (!userSuppliesTargetCFL) {
        std::string subCyclingString;
        if (ini->extract("general", "advectionsubcyclingsteps", subCyclingString)) {
          if (subCyclingString.find("auto") != std::string::npos) {
            append_error("subCyclingSteps = auto requires the targetCFL to be set");
            options.setArgs("SUBCYCLING STEPS", "0"); // dummy
          }
        }

        int NSubCycles = 0;
        double targetCFL = 0.5;
        options.getArgs("SUBCYCLING STEPS", NSubCycles);
        if (NSubCycles == 0) {
          targetCFL = 0.5;
        } else {
          targetCFL = 2 * NSubCycles;
        }
        options.setArgs("TARGET CFL", to_string_f(targetCFL));
      }

      // guard against using a higher initial dt than the max
      if (userSuppliesInitialDt) {
        double initialDt = 0.0;
        double maxDt = 0.0;
        options.getArgs("DT", initialDt);
        options.getArgs("MAX DT", maxDt);
        if (maxDt > 0 && initialDt > maxDt) {
          std::ostringstream error;
          error << "initial dt " << initialDt << " is larger than max dt " << maxDt << "\n";
          append_error(error.str());
        }
      }
    } else {
      const double dt = std::stod(dtString);
      options.setArgs("DT", to_string_f(fabs(dt)));
    }
  }

  // check if dt is provided if numSteps or endTime > 0
  {
    int numSteps = 0;
    options.getArgs("NUMBER TIMESTEPS", numSteps);

    double endTime = 0;
    options.getArgs("END TIME", endTime);

    if (numSteps > 0 || endTime > 0) {
      if (options.compareArgs("VARIABLE DT", "FALSE")) {
        const std::string dtString = options.getArgs("DT");
        if (dtString.empty()) {
          append_error("dt not defined!\n");
        }
      }
    }
  }

  options.setArgs("BDF ORDER", "2");
  std::string timeStepper;
  if (ini->extract("general", "timestepper", timeStepper)) {
    if (timeStepper == "bdf3" || timeStepper == "tombo3") {
      options.setArgs("BDF ORDER", "3");
    } else if (timeStepper == "bdf2" || timeStepper == "tombo2") {
      options.setArgs("BDF ORDER", "2");
    } else if (timeStepper == "bdf1" || timeStepper == "tombo1") {
      options.setArgs("BDF ORDER", "1");
    } else {
      std::ostringstream error;
      error << "Could not parse general::timeStepper = " << timeStepper;
      append_error(error.str());
    }
  }

  options.setArgs("EXT ORDER", "3");
  {
    int NSubCycles = 0;
    options.getArgs("SUBCYCLING STEPS", NSubCycles);
    if (NSubCycles) {
      int bdfOrder;
      options.getArgs("BDF ORDER", bdfOrder);
      options.setArgs("EXT ORDER", std::to_string(bdfOrder));
    }
  }

  parseConstFlowRate(rank, options, ini);

  double endTime;
  std::string stopAt = "numsteps";
  ini->extract("general", "stopat", stopAt);
  if (stopAt == "numsteps") {
    int numSteps = 0;
    if (ini->extract("general", "numsteps", numSteps)) {
      options.setArgs("NUMBER TIMESTEPS", std::to_string(numSteps));
      endTime = -1;
    } else {
      append_error("cannot find mandatory parameter GENERAL::numSteps");
    }
    options.setArgs("NUMBER TIMESTEPS", std::to_string(numSteps));
  } else if (stopAt == "endtime") {
    if (!ini->extract("general", "endtime", endTime)) {
      append_error("cannot find mandatory parameter GENERAL::endTime");
    }
    options.setArgs("END TIME", to_string_f(endTime));
  } else if (stopAt == "elapsedtime") {
    double elapsedTime;
    if (!ini->extract("general", "elapsedtime", elapsedTime)) {
      append_error("cannot find mandatory parameter GENERAL::elapsedTime");
    }
    options.setArgs("STOP AT ELAPSED TIME", to_string_f(elapsedTime));
  } else {
    std::ostringstream error;
    error << "Could not parse general::stopAt = " << stopAt;
    append_error(error.str());
  }

  options.setArgs("CHECKPOINT ENGINE", "NEK");
  std::string checkpointEngine;
  if (ini->extract("general", "checkpointengine", checkpointEngine)) {
    if (checkpointEngine == "nek") {
      options.setArgs("CHECKPOINT ENGINE", "NEK");
    } else if (checkpointEngine == "adios") {
      options.setArgs("CHECKPOINT ENGINE", "ADIOS");
#ifndef NEKRS_ENABLE_ADIOS
      append_error("ADIOS engine was requested but is not enabled!\n");
#endif
    } else {
      append_error("invalid checkpointEngine");
    }
  }

  int checkpointPrecision = 0;
  if (ini->extract("general", "checkpointprecision", checkpointPrecision)) {
    if (checkpointPrecision == 64) {
      options.setArgs("CHECKPOINT PRECISION", "FP64");
    } else if (checkpointPrecision == 32) {
      options.setArgs("CHECKPOINT PRECISION", "FP32");
    } else {
      append_error("invalid checkpointPrecision");
    }
  }

  double writeInterval = 0;
  if (!ini->extract("general", "writeinterval", writeInterval)) {
    ini->extract("general", "checkpointinterval", writeInterval);
  }
  options.setArgs("CHECKPOINT INTERVAL", std::to_string(writeInterval));

  std::string writeControl;
  if (!ini->extract("general", "writecontrol", writeControl)) {
    ini->extract("general", "checkpointcontrol", writeControl);
  }

  if (writeControl.size()) {
    checkValidity(rank, {"steps", "simulationtime"}, writeControl);

    if (writeControl == "steps") {
      options.setArgs("CHECKPOINT CONTROL", "STEPS");
    } else if (writeControl == "simulationtime") {
      options.setArgs("CHECKPOINT CONTROL", "SIMULATIONTIME");
    } else {
      std::ostringstream error;
      error << "could not parse general::checkpointControl = " << writeControl;
      append_error(error.str());
    }
  }

  bool dealiasing = true;
  options.setArgs("OVERINTEGRATION", "TRUE");
  if (ini->extract("general", "dealiasing", dealiasing)) {
    if (dealiasing) {
      options.setArgs("OVERINTEGRATION", "TRUE");
    } else {
      options.setArgs("OVERINTEGRATION", "FALSE");
    }
  }

  int cubN = round((3. / 2) * (N + 1) - 1) - 1;
  if (!dealiasing) {
    cubN = 0;
  }
  ini->extract("general", "cubaturepolynomialorder", cubN);
  options.setArgs("CUBATURE POLYNOMIAL DEGREE", std::to_string(cubN));

  {
    parseRegularization(rank, options, ini, "general");
  }
}

void parseGeomSection(const int rank, setupAide &options, inipp::Ini *ini)
{
  if (ini->sections.count("geom")) {
    parseLinearSolver(rank, options, ini, "geom");
    if (options.compareArgs("GEOM SOLVER", "USER")) {
      options.setArgs("MOVING MESH", "TRUE");
      options.setArgs("GEOM SOLVER", "NONE");
      options.setArgs("GEOM INTEGRATION ORDER", "3");
    }

    if (!options.compareArgs("GEOM SOLVER", "NONE")) {
      options.setArgs("MOVING MESH", "TRUE");
      options.setArgs("GEOM INTEGRATION ORDER", "3");
      options.setArgs("GEOM HELMHOLTZ TYPE", "POISSON");
      options.setArgs("GEOM ELLIPTIC COEFF FIELD", "TRUE");

      parseInitialGuess(rank, options, ini, "geom");
      parsePreconditioner(rank, options, ini, "geom");
      parseSolverTolerance(rank, options, ini, "geom");

      std::string m_bcMap;
      if (ini->extract("geom", "boundarytypemap", m_bcMap)) {
        options.setArgs("GEOM BOUNDARY TYPE MAP", m_bcMap);
      } else {
        std::string v_bcMap;
        if (ini->extract("fluid velocity", "boundarytypemap", v_bcMap)) {
          options.setArgs("GEOM DERIVED BOUNDARY TYPE MAP", v_bcMap);
        }
      }
    }

    if (options.compareArgs("MOVING MESH", "TRUE")) {
      options.setArgs("CHECKPOINT OUTPUT MESH", "TRUE");
    }
  }
}

void parseMeshSection(const int rank, setupAide &options, inipp::Ini *ini)
{
  if (ini->sections.count("mesh")) {
    std::string meshFile;
    if (ini->extract("mesh", "file", meshFile)) {
      options.setArgs("MESH FILE", meshFile);
    }

    std::string meshPartitioner;
    if (ini->extract("mesh", "partitioner", meshPartitioner)) {
      if (meshPartitioner != "rcb" && meshPartitioner != "rcb+rsb") {
        std::ostringstream error;
        error << "Could not parse mesh::partitioner = " << meshPartitioner;
        append_error(error.str());
      }
      options.setArgs("MESH PARTITIONER", meshPartitioner);
    }

    std::string meshConTol;
    if (ini->extract("mesh", "connectivitytol", meshConTol)) {
      options.setArgs("MESH CONNECTIVITY TOL", meshConTol);
    }

    std::string boundaryIDs;
    if (ini->extract("mesh", "boundaryidmap", boundaryIDs)) {
      options.setArgs("MESH BOUNDARY ID MAP", boundaryIDs);
    }

    if (ini->extract("mesh", "boundaryidmapV", boundaryIDs)) {
      options.setArgs("MESHV BOUNDARY ID MAP", boundaryIDs);
    }
  }
}

void parsePressureSection(const int rank, setupAide &options, inipp::Ini *ini)
{
  const std::string parScope = "fluid pressure";

  options.setArgs("FLUID PRESSURE ELLIPTIC COEFF FIELD", "FALSE");

  options.setArgs("FLUID PRESSURE HELMHOLTZ TYPE", "POISSON");

  parseSolverTolerance(rank, options, ini, parScope);

  parseInitialGuess(rank, options, ini, parScope);

  parsePreconditioner(rank, options, ini, parScope);

  parseLinearSolver(rank, options, ini, parScope);

  parseBoomerAmgSection(rank, options, ini);

  if (ini->sections.count("amgx")) {
    if (!AMGXenabled()) {
      append_error("AMGX was requested but is not compiled!\n");
    }
    std::string configFile;
    if (ini->extract("amgx", "configfile", configFile)) {
      options.setArgs("AMGX CONFIG FILE", configFile);
    }
  }
}

void parseVelocitySection(const int rank, setupAide &options, inipp::Ini *ini)
{
  std::string vsolver;
  std::string sbuf;

  options.setArgs("FLUID VELOCITY ELLIPTIC COEFF FIELD", "TRUE");
  if (options.getArgs("FLUID STRESSFORMULATION").empty()) {
    options.setArgs("FLUID STRESSFORMULATION", "FALSE");
  }

  const std::string parScope = "fluid velocity";

  parseCheckpointing(rank, options, ini, "fluid");

  parseInitialGuess(rank, options, ini, parScope);

  parsePreconditioner(rank, options, ini, parScope);

  parseLinearSolver(rank, options, ini, parScope);

  parseSolverTolerance(rank, options, ini, parScope);

  std::string v_bcMap;
  if (ini->extract(parScope, "boundarytypemap", v_bcMap)) {
    options.setArgs("FLUID VELOCITY BOUNDARY TYPE MAP", v_bcMap);
  }

  double rho;
  if (ini->extract(parScope, "density", rho) || ini->extract(parScope, "rho", rho)) {
    options.setArgs("FLUID DENSITY", to_string_f(rho));
  }

  if (ini->extract(parScope, "viscosity", sbuf) || ini->extract(parScope, "mu", sbuf)) {
    int err = 0;
    double viscosity = parseFormula(sbuf.c_str(), &err);
    if (err) {
      append_error("Invalid expression for viscosity");
    }
    if (viscosity < 0) {
      viscosity = fabs(1 / viscosity);
    }
    options.setArgs("FLUID VISCOSITY", to_string_f(viscosity));
  }

  parseRegularization(rank, options, ini, parScope);
}

void parseProblemTypeSection(const int rank, setupAide &options, inipp::Ini *ini)
{
  {
    bool stressFormulation;
    if (ini->extract("problemtype", "stressformulation", stressFormulation)) {
      if (stressFormulation) {
        options.setArgs("FLUID STRESSFORMULATION", "TRUE");
      }
    }

    options.setArgs("EQUATION TYPE", upperCase("navierstokes"));

    std::string eqn;
    if (ini->extract("problemtype", "equation", eqn)) {
      const std::vector<std::string> validValues = {
          {"stokes"},
          {"navierstokes"},
          {"stress"},
          {"variableviscosity"},
      };
      const std::vector<std::string> list = serializeString(eqn, '+');

      auto eqnType = list[0];
      options.setArgs("EQUATION TYPE", upperCase(eqnType));

      for (std::string entry : list) {
        checkValidity(rank, validValues, entry);
      }

      if (std::strstr(eqn.c_str(), "stress") || std::strstr(eqn.c_str(), "variableviscosity")) {
        options.setArgs("FLUID STRESSFORMULATION", "TRUE");
      }
    }
  }
}

void parseNekNekSection(const int rank, setupAide &options, inipp::Ini *par)
{
  dlong boundaryEXTOrder = 1;
  if (par->extract("neknek", "boundaryextorder", boundaryEXTOrder)) {
    options.setArgs("NEKNEK BOUNDARY EXT ORDER", std::to_string(boundaryEXTOrder));
  }

  std::string multirateStr;
  if (par->extract("neknek", "multiratetimestepping", multirateStr)) {
    const std::vector<std::string> validValues = {
        {"yes"},
        {"true"},
        {"1"},
        {"no"},
        {"false"},
        {"0"},
        {"correctorsteps"},
    };
    const std::vector<std::string> list = serializeString(multirateStr, '+');
    for (std::string entry : list) {
      checkValidity(rank, validValues, entry);
      const auto correctorStepsStr = parseValueForKey(entry, "correctorsteps");
      if (!correctorStepsStr.empty()) {
        const int correctorSteps = std::stoi(correctorStepsStr);
        options.setArgs("NEKNEK MULTIRATE CORRECTOR STEPS", std::to_string(correctorSteps));
      }
    }
    const bool multirate = checkForTrue(list[0]);
    options.setArgs("NEKNEK MULTIRATE TIMESTEPPER", multirate ? "TRUE" : "FALSE");
  }

  const bool multirate = options.compareArgs("NEKNEK MULTIRATE TIMESTEPPER", "TRUE");

  if (multirate) {
    int correctorSteps = 0;
    options.getArgs("NEKNEK MULTIRATE CORRECTOR STEPS", correctorSteps);
    if (boundaryEXTOrder > 1 && correctorSteps == 0) {
      append_error("Multirate timestepper with boundaryEXTOrder > 1 and correctorSteps = 0 is unstable!\n");
    }
    if (options.compareArgs("VARIABLE DT", "TRUE")) {
      append_error("Multirate timestepper with variable timestep is not supported!\n");
    }
  }
}

void parseScalarSections(const int rank, setupAide &options, inipp::Ini *ini)
{
  nscal = scalarMap.size();
  options.setArgs("NUMBER OF SCALARS", std::to_string(nscal));

  const auto sections = ini->sections;

  auto parseScalarSection = [&](const auto &sec) {
    std::istringstream stream(sec.first);
    std::string firstWord, secondWord;
    stream >> firstWord >> secondWord;
    if (firstWord != "scalar") return;

    std::string sid;
    if (secondWord.empty()) {
      sid = " DEFAULT";
    } else { 
      auto it = scalarMap.find(secondWord);
      if (it != scalarMap.end()) {
        sid = scalarDigitStr(scalarMap.at(secondWord));
        options.setArgs("SCALAR" + sid + " NAME", upperCase(secondWord));
      } else {
        append_error("SCALAR " + secondWord + " not found");
      }
    }

    const auto parScope = sec.first;
    parseCheckpointing(rank, options, ini, parScope);
    parseRegularization(rank, options, ini, parScope);

    std::string solver;
    ini->extract(parScope, "solver", solver);

    if (solver == "cvode") {
      cvodeRequested = true;
      options.setArgs("SCALAR" + sid + " SOLVER", "CVODE");
    }

    options.setArgs("SCALAR" + sid + " ELLIPTIC COEFF FIELD", "TRUE");

    parseInitialGuess(rank, options, ini, parScope);

    parsePreconditioner(rank, options, ini, parScope);

    parseLinearSolver(rank, options, ini, parScope);

    parseSolverTolerance(rank, options, ini, parScope);

    std::string sbuf;

    options.setArgs("SCALAR" + sid + " MESH", "FLUID");
    if (ini->extract(parScope, "mesh", sbuf)) {
      auto keys = serializeString(sbuf, '+');
      if (keys.at(0) != "fluid") { 
        append_error("Valid mesh values are fluid or fluid+solid");
      }

      if (keys.size() > 1) {
        if (keys.at(1) == "solid") { 
          options.setArgs("SCALAR" + sid + " MESH", "FLUID+SOLID");
        } else {
          append_error("Valid mesh values are fluid or fluid+solid");
        }
      }
    }

    if (ini->extract(parScope, "diffusionCoeff", sbuf) || ini->extract(parScope, "conductivity", sbuf)) {
      int err = 0;
      double diffusivity = parseFormula(sbuf.c_str(), &err);
      if (err) {
        append_error("Invalid expression for diffusionCoeff");
      }
      if (diffusivity < 0) {
        diffusivity = fabs(1 / diffusivity);
      }
      options.setArgs("SCALAR" + sid + " DIFFUSIONCOEFF", to_string_f(diffusivity));
    }

    if (ini->extract(parScope, "diffusionCoeffSolid", sbuf) || ini->extract(parScope, "conductivitySolid", sbuf)) {
      int err = 0;
      double diffusivity = parseFormula(sbuf.c_str(), &err);
      if (err) {
        append_error("Invalid expression for diffusionCoeffSolid");
      }
      if (diffusivity < 0) {
        diffusivity = fabs(1 / diffusivity);
      }
      options.setArgs("SCALAR" + sid + " DIFFUSIONCOEFF SOLID", to_string_f(diffusivity));
    }

    if (ini->extract(parScope, "transportCoeff", sbuf) || ini->extract(parScope, "rhoCp", sbuf)) {
      int err = 0;
      double rho = parseFormula(sbuf.c_str(), &err);
      if (err) {
        append_error("Invalid expression for transportCoeff");
      }
      options.setArgs("SCALAR" + sid + " TRANSPORTCOEFF", to_string_f(rho));
    }

    if (ini->extract(parScope, "transportCoeffSolid", sbuf) || ini->extract(parScope, "rhoCpSolid", sbuf)) {
      int err = 0;
      double rho = parseFormula(sbuf.c_str(), &err);
      if (err) {
        append_error("Invalid expression for transportCoeffSolid");
      }
      options.setArgs("SCALAR" + sid + " TRANSPORTCOEFF SOLID", to_string_f(rho));
    }

    std::string s_bcMap;
    if (ini->extract(parScope, "boundarytypemap", s_bcMap)) {
      options.setArgs("SCALAR" + sid + " BOUNDARY TYPE MAP", s_bcMap);
    }
  };

  // read default section first.
  if (sections.count("scalar") != 0) {
    parseScalarSection(std::make_pair(std::string("scalar"), sections.at("scalar")));
  }

  // initialize with default settings if available 
  const std::string defaultSettingStr = "SCALAR DEFAULT ";
  for (int is = 0; is < nscal; ++is) {
    std::string sid = scalarDigitStr(is);
    const auto options_ = options;
    for (auto [keyWord, value] : options_) {
      auto delPos = keyWord.find(defaultSettingStr);
      if (delPos != std::string::npos) {
        auto newKey = keyWord;
        newKey.erase(delPos, defaultSettingStr.size());
        options.setArgs("SCALAR" + sid + " " + newKey, value);
      }
    }
  }

  // override default settings if specified explicitly
  for (auto &&sec : sections) {
    parseScalarSection(sec);
  }

  // set boundarytypemap from default if available and not specified explicitly
  if (sections.count("scalar") != 0) {
    std::string s_bcMapDefault;
    ini->extract("scalar", "boundarytypemap", s_bcMapDefault);
    for (int is = 0; is < nscal; ++is) {
      std::string sid = scalarDigitStr(is);
      std::string dummy;
      if (!ini->extract("scalar" + sid, "boundarytypemap", dummy)) {
        if (s_bcMapDefault.size() > 0) {
          options.setArgs("SCALAR" + sid + " BOUNDARY TYPE MAP", s_bcMapDefault);
        }
      }
    }
  }

  {
    int nscal;
    options.getArgs("NUMBER OF SCALARS", nscal);
    if (nscal) {
      std::string dummy;
      if (!options.getArgs("SCALAR00 SOLVER", dummy)) {
        append_error("scalar index needs to start from 0");
      }
    }
  }

}

void cleanupStaleKeys(const int rank, setupAide &options, inipp::Ini *ini)
{
  std::vector<std::string> sections = {"GEOM", "FLUID PRESSURE", "FLUID VELOCITY", "SCALAR DEFAULT"};
  for (int i = 0; i < nscal; i++) {
    sections.push_back("SCALAR" + scalarDigitStr(i));
  }

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
                                              "MAXIMUM ITERATIONS",
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
    if (option.first.find("SCALAR DEFAULT") == 0) {
      staleOptions.push_back(option.first);
    }

    if (options.compareArgs("FLUID VELOCITY SOLVER", "NONE") && option.first.find("FLUID PRESSURE") == 0) {
      staleOptions.push_back(option.first);
    }
  }
  for (auto const &key : staleOptions) {
    options.removeArgs(key);
  }

  options.removeArgs("REGULARIZATION METHOD");
}

Par::Par(MPI_Comm comm_)
{
  ini = new inipp::Ini();
  comm = comm_;
}

void Par::addValidSection(const std::string &name)
{
  validSections.push_back(name);
}

void Par::parse(setupAide &options)
{
  int rank;
  MPI_Comm_rank(comm, &rank);

  const auto userSections = [&]() {
    std::string value;
    ini->extract("", "usersections", value);
    return serializeString(value, ',');
  }();

  for (auto &section : userSections) {
    addValidSection(section);
  }

  {
    std::string names;
    ini->extract("general", "scalars", names);
    auto list = serializeString(names, ',');
    for(int i = 0; i < list.size(); i++) {
      scalarMap[list[i]] =  i;
    }
  }

  if (rank == 0) {
    validate(ini, userSections);
  }

  processError();

  if (rank == 0) {
    printDeprecation(ini->sections);
  }

  parseOccaSection(rank, options, ini);

  parseGeneralSection(rank, options, ini);

  parseNekNekSection(rank, options, ini);

  parseProblemTypeSection(rank, options, ini);

  parseMeshSection(rank, options, ini);

  parseGeomSection(rank, options, ini);

  if (ini->sections.count("fluid velocity")) {
    parsePressureSection(rank, options, ini);
    parseVelocitySection(rank, options, ini);
    options.setArgs("FLUID SOLVER", "TRUE");
  } else {
    options.setArgs("FLUID SOLVER", "FALSE");
  }

  parseScalarSections(rank, options, ini);

  if (ini->sections.count("cvode") || cvodeRequested) {
    options.setArgs("CVODE", "TRUE");
    parseCvodeSolver(rank, options, ini);
  }

  cleanupStaleKeys(rank, options, ini);

  processError();
}
