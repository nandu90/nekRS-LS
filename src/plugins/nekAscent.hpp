#if !defined(nekrs_nekascent_hpp_)
#define nekrs_nekascent_hpp_

#if !defined(NEKRS_ENABLE_ASCENT)

#error "No Ascent installation was found"

#else

#define NEKRS_ASCENT_ENABLED
#include "platform.hpp"
#include "ascent.hpp"
#include "vtkh/vtkh.hpp"

namespace nekAscent
{
using field = std::tuple<std::string, occa::memory, int>;

void setup(mesh_t *mesh, const std::vector<nekAscent::field> &flds, const std::string inputFile);
void run(const double time, const int tstep);
void finalize();

ascent::Ascent mAscent;
} // namespace nekAscent

namespace
{

std::vector<nekAscent::field> userFieldList;

static deviceMemory<dlong> o_connectivity;
conduit::Node mesh_data;

bool setupCalled = false;
bool updateMesh = true;

std::string actionFile;
} // namespace

static void printStat()
{

  const int verbose = platform->options.compareArgs("VERBOSE", "TRUE") ? 1 : 0;
  const int rank = platform->comm.mpiRank;

  if (verbose && rank == 0) {
    conduit::Node ascent_info, opts;
    nekAscent::mAscent.info(ascent_info);
    opts["num_children_threshold"] = -1;
    opts["num_elements_threshold"] = -1;
    opts["depth"] = 1;
    ascent_info.to_summary_string_stream(std::cout, opts);
    std::cout << std::endl;
  }

  std::cout.setf(std::ios::scientific);
  int outPrecisionSave = std::cout.precision();
  std::cout.precision(5);

  const double tElapsedTimeSolve = platform->timer.query("elapsedStepSum", "DEVICE:MAX");
  platform->timer.printStatSetElapsedTimeSolve(tElapsedTimeSolve);
  const double tudf = platform->timer.query("udfExecuteStep", "DEVICE:MAX");
  const double tSetup = platform->timer.query("nekAscentSetup", "DEVICE:MAX");
  if (rank == 0) {
    std::cout << "nekAscent\n";
    if (tSetup > 0) {
      std::cout << "  setup\n";
      std::cout << "    AscentSetup         " << tSetup << "s\n";
    }

    std::cout << "  solve\n";
    std::cout << "    udfExecuteStep \n";
  }
  platform->timer.printStatEntry("      AscentRun         ", "nekAscentRun", "DEVICE:MAX", tudf);

  std::cout.unsetf(std::ios::scientific);
  std::cout.precision(outPrecisionSave);
}

static void initializeAscent()
{

  const int verbose = platform->options.compareArgs("VERBOSE", "TRUE") ? 1 : 0;

  const double tStart = MPI_Wtime();
  if (platform->comm.mpiRank == 0) {
    printf("initialize Ascent ...\n");
    fflush(stdout);
  }

  platform->par->addValidSection("ascent");
  platform->timer.addPrintStatCallback(printStat);

  MPI_Comm comm;
  MPI_Comm_dup(platform->comm.mpiComm, &comm);

  conduit::Node ascent_opts;
  ascent_opts["mpi_comm"] = MPI_Comm_c2f(comm);

  nekAscent::mAscent.open(ascent_opts);

  const double tSetup = MPI_Wtime() - tStart;
  platform->timer.set("nekAscentInitialize", tSetup);
  if (platform->comm.mpiRank == 0) {
    // select ascent::about() (ascent/libs/ascent/ascent.cpp)
    conduit::Node about;
    ascent::about(about);
    about.remove_child("license");
    about.remove_child("annotations"); // caliper annotations
    about.remove_child("git_sha1_abbrev");
    about.remove_child("git_tag");
    about.remove_child("compilers");
    about.remove_child("platform");
    about.remove_child("system");
    about.remove_child("web_client_root");
    about.remove_child("default_runtime");           // this is always ascent
    about["runtimes/ascent"].remove_child("status"); // always enabled
    std::cout << "---------------- Ascent.about() ----------------";
    std::cout << about.to_yaml() << std::endl;

    // print vtkh adapter
    std::cout << vtkh::AboutVTKH() << std::endl;

    printf("done (%gs)\n\n", tSetup);
  }
  fflush(stdout);
}

void nekAscent::setup(mesh_t *mesh, const std::vector<nekAscent::field> &flds, const std::string inputFile)
{
  const int verbose = platform->options.compareArgs("VERBOSE", "TRUE") ? 1 : 0;

  const double tStart = MPI_Wtime();
  if (platform->comm.mpiRank == 0) {
    printf("initializing nekAscent ...\n");
    fflush(stdout);
  }

  if (platform->comm.mpiRank == 0) {
    nekrsCheck(!fs::exists(inputFile), MPI_COMM_SELF, EXIT_FAILURE, "Cannot find %s\n", inputFile.c_str());
  }
  actionFile = inputFile;

  initializeAscent();

  userFieldList = flds;

  const int Nq = mesh->Nq;
  const dlong Nelements = mesh->Nelements;
  const dlong Ncells = Nelements * (Nq - 1) * (Nq - 1) * (Nq - 1);
  const dlong Nvertices = Ncells * std::pow(2, mesh->dim);

  // allocate work arrays
  o_connectivity.resize(Nvertices);

  // calculate connectivity
  {
    std::vector<dlong> a_etov(Nvertices);
    auto it = a_etov.begin();
    for (int e = 0; e < Nelements; ++e) {
      for (int z = 0; z < Nq - 1; ++z) {
        for (int y = 0; y < Nq - 1; ++y) {
          for (int x = 0; x < Nq - 1; ++x) {
            it[0] = ((e * Nq + z) * Nq + y) * Nq + x;
            it[1] = it[0] + 1;
            it[2] = it[0] + Nq + 1;
            it[3] = it[0] + Nq;
            it[4] = it[0] + Nq * Nq;
            it[5] = it[1] + Nq * Nq;
            it[6] = it[2] + Nq * Nq;
            it[7] = it[3] + Nq * Nq;
            it += 8;
          }
        }
      }
    }
    o_connectivity.copyFrom(a_etov.data(), a_etov.size());
  }

  // Setup pointers for mesh
  mesh_data["coordsets/coords/type"] = "explicit";
  mesh_data["coordsets/coords/values/x"].set_external((dfloat *)mesh->o_x.ptr(), mesh->Nlocal);
  mesh_data["coordsets/coords/values/y"].set_external((dfloat *)mesh->o_y.ptr(), mesh->Nlocal);
  mesh_data["coordsets/coords/values/z"].set_external((dfloat *)mesh->o_z.ptr(), mesh->Nlocal);

  mesh_data["topologies/mesh/type"] = "unstructured";
  mesh_data["topologies/mesh/coordset"] = "coords";
  mesh_data["topologies/mesh/elements/shape"] = "hex";
  mesh_data["topologies/mesh/elements/connectivity"].set_external((dlong *)o_connectivity.ptr(), Nvertices);

  // fields pointer
  int ifld = 0;
  if (platform->comm.mpiRank == 0) {
    printf("(availiable fields:");
  }
  for (auto &entry : userFieldList) {
    std::string fieldName = std::get<0>(entry);
    occa::memory o_fld = std::get<1>(entry);
    dlong fieldLength = std::get<2>(entry);
    if (platform->comm.mpiRank == 0) {
      printf(" %s", fieldName.c_str());
    }

    mesh_data["fields/" + fieldName + "/association"] = "vertex";
    mesh_data["fields/" + fieldName + "/topology"] = "mesh";
    mesh_data["fields/" + fieldName + "/values"].set_external((dfloat *)o_fld.ptr(), fieldLength);
    ifld++;
  }
  if (platform->comm.mpiRank == 0) {
    printf(")\n");
  }

  const double tSetup = MPI_Wtime() - tStart;
  platform->timer.set("nekAscentSetup", tSetup);
  if (platform->comm.mpiRank == 0) {
    printf("done (%gs)\n\n", tSetup);
    if (verbose) {
      mesh_data.print();
    }
  }
  fflush(stdout);

  setupCalled = true;
}

void nekAscent::run(const double time, const int tstep)
{

  nekrsCheck(!setupCalled, MPI_COMM_SELF, EXIT_FAILURE, "%s\n", "called prior to nekAscent::setup()!");

  platform->timer.tic("nekAscentRun", 1);

  const int verbose = platform->options.compareArgs("VERBOSE", "TRUE") ? 1 : 0;

  // Copy data
  mesh_data["state/cycle"] = tstep;
  mesh_data["state/time"] = time;

  mAscent.publish(mesh_data);

  // Ascent actions
  conduit::Node actions;

  // add trigger
  conduit::Node triggers;

  // set action file
  std::string casename = platform->options.getArgs("CASENAME");
  {
    std::string buf = "";
    platform->par->extract("ascent", "actionfile", buf);
    if (!buf.empty()) {
      actionFile = buf;
    }
  }

  triggers["t1/params/condition"] = "True"; // control the condition in udf, not ascent
  triggers["t1/params/actions_file"] = actionFile;

  conduit::Node &add_triggers = actions.append();
  add_triggers["action"] = "add_triggers";
  add_triggers["triggers"] = triggers;

  mAscent.execute(actions);
  platform->timer.toc("nekAscentRun");
}

void nekAscent::finalize()
{
  if (setupCalled) {
    o_connectivity.clear();
    mAscent.close();
  }
}
#endif
#endif // hpp
