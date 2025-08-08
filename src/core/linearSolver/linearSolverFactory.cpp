#include "platform.hpp"
#include "linearSolverFactory.hpp"
#include "cg.hpp"
#include "gmres.hpp"

template <typename T>
std::unique_ptr<linearSolver>
linearSolverFactory<T>::create(const std::string &_solver,
                               const std::string &varName,
                               dlong Nlocal,
                               int Nfields,
                               dlong fieldOffset,
                               const occa::memory &o_weight,
                               std::function<void(const occa::memory &o_q, occa::memory &o_Aq)> Ax,
                               std::function<void(const occa::memory &o_r, occa::memory &o_z)> Pc)
{
  nekrsCheck(!Ax, MPI_COMM_SELF, EXIT_FAILURE, "Ax undefined for %s!\n", varName.c_str());

  auto KSP = [&]() -> std::unique_ptr<linearSolver> {
    const auto solver = lowerCase(_solver);
    auto flexible = false;
    if (solver.find("flexible") != std::string::npos) {
      flexible = true;
    }

    if (solver.find("cg") != std::string::npos) {
      auto combined = false;
      if (solver.find("combined") != std::string::npos) {
        combined = true;
      }

      return std::make_unique<cg<T>>(Nlocal, Nfields, fieldOffset, o_weight, flexible, combined, Ax, Pc);
    } else if (solver.find("gmres") != std::string::npos) {
      int nRestartVectors = 15;
      std::regex pattern("nvector=([0-9]+)");
      std::smatch match;
      if (std::regex_search(solver, match, pattern)) {
        nRestartVectors = std::stoi(match[1]);
      }

      return std::make_unique<gmres<T>>(Nlocal,
                                        Nfields,
                                        fieldOffset,
                                        o_weight,
                                        nRestartVectors,
                                        flexible,
                                        Ax,
                                        Pc);
    } else {
      nekrsAbort(platform->comm.mpiComm(), EXIT_FAILURE, "Unknown linear solver %s!\n", solver.c_str());
      return nullptr;
    }
  }();

  KSP->name(varName);
  return KSP;
}

template class linearSolverFactory<float>;
template class linearSolverFactory<double>;
