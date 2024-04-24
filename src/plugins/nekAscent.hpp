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
using field = std::tuple<std::string, occa::memory, mesh_t *>;

void setup(mesh_t *mesh_,
           const std::vector<nekAscent::field> &flds,
           const std::string& inputFile,
           int Nin_ = 0,
           bool uniform_ = false);
void run(const double time, const int tstep);
void finalize();

ascent::Ascent mAscent;
} // namespace nekAscent

namespace
{
std::vector<nekAscent::field> userFieldList;

static occa::memory o_connectivity;
conduit::Node mesh_data;

int Nfields;
std::vector<dlong> fieldOffsetScan;

mesh_t *mesh_in;
mesh_t *mesh_vis;

bool setupCalled = false;
bool updateMesh = true;

bool interpolate = false;
bool uniform = false;

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
  const double tRun = platform->timer.query("nekAscentRun", "DEVICE:MAX");
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
  if (interpolate) {
    platform->timer.printStatEntry("        AscentMtoN      ", "nekAscentInterpoate", "DEVICE:MAX", tRun);
  }

  std::cout.unsetf(std::ios::scientific);
  std::cout.precision(outPrecisionSave);
}

static void initializeAscent()
{
  const int verbose = platform->options.compareArgs("VERBOSE", "TRUE") ? 1 : 0;

  const double tStart = MPI_Wtime();

  platform->par->addValidSection("ascent");
  platform->timer.addPrintStatCallback(printStat);

  MPI_Comm comm;
  MPI_Comm_dup(platform->comm.mpiComm, &comm);

  conduit::Node ascent_opts;
  ascent_opts["mpi_comm"] = MPI_Comm_c2f(comm);

  nekAscent::mAscent.open(ascent_opts);

  platform->timer.set("nekAscentInitialize", tSetup);
  if (platform->comm.mpiRank == 0) {
    conduit::Node about;
    ascent::about(about);
    about.remove_child("license");
    about.remove_child("annotations");
    about.remove_child("git_sha1_abbrev");
    about.remove_child("git_tag");
    about.remove_child("compilers");
    about.remove_child("platform");
    about.remove_child("system");
    about.remove_child("web_client_root");
    about.remove_child("default_runtime");
    about["runtimes/ascent"].remove_child("status");
    std::cout << "---------------- Ascent.about() ----------------";
    std::cout << about.to_yaml() << std::endl;

    std::cout << vtkh::AboutVTKH() << std::endl;
  }
  fflush(stdout);
}

void nekAscent::setup(mesh_t *mesh_,
                      const std::vector<nekAscent::field> &userFieldList_,
                      const std::string& inputFile,
                      int Nin_,
                      bool uniform_)
{
  const auto verbose = platform->options.compareArgs("VERBOSE", "TRUE") ? 1 : 0;
  userFieldList = userFieldList_;
  mesh_in = mesh_;
  uniform = uniform_;
  const int Nin = (Nin_) ? Nin_ : mesh_in->N;

  interpolate = (uniform || (Nin != mesh_in->N));
  Nfields = userFieldList.size();

  const auto tStart = MPI_Wtime();
  if (platform->comm.mpiRank == 0) {
    printf("initializing nekAscent ");
    if (interpolate) {
      printf("(Nviz=%d", Nin);
      if (uniform)  printf(" +uniform"); 
      printf(") ...\n");
      fflush(stdout);
    }
  }

  if (platform->comm.mpiRank == 0) {
    nekrsCheck(!fs::exists(inputFile), MPI_COMM_SELF, EXIT_FAILURE, "Cannot find %s\n", inputFile.c_str());
  }
  actionFile = inputFile;

  initializeAscent();

  mesh_vis = [&]()
  {
    auto mesh = mesh_in;
    if (interpolate) {
      mesh = new mesh_t();
      mesh->Nelements = mesh_in->Nelements;
      mesh->dim = mesh_in->dim;
      mesh->Nverts = mesh_in->Nverts;
      mesh->Nfaces = mesh_in->Nfaces;
      mesh->NfaceVertices = mesh_in->NfaceVertices;
      meshLoadReferenceNodesHex3D(mesh, Nin, 0);
 
      mesh->o_x = platform->device.malloc<dfloat>(mesh->Nlocal);
      mesh->o_y = platform->device.malloc<dfloat>(mesh->Nlocal);
      mesh->o_z = platform->device.malloc<dfloat>(mesh->Nlocal);
    }
    return mesh;
  }();

  o_connectivity = [&]()
  {
    const dlong Nverts = mesh_vis->Nelements * std::pow(mesh_vis->N, mesh_vis->dim) * mesh_vis->Nverts;
    std::vector<dlong> etov(Nverts);
    auto o_etov = platform->device.malloc<dlong>(etov.size());

    auto it = etov.begin();
    for (int e = 0; e < mesh_vis->Nelements; ++e) {
      for (int z = 0; z < mesh_vis->N; ++z) {
        for (int y = 0; y < mesh_vis->N; ++y) {
          for (int x = 0; x < mesh_vis->N; ++x) {
            const dlong Nq = mesh_vis->Nq;
            it[0] = ((e * Nq + z) * Nq + y) * Nq + x;
            it[1] = it[0] + 1;
            it[2] = it[0] + Nq + 1;
            it[3] = it[0] + Nq;
            it[4] = it[0] + Nq * Nq;
            it[5] = it[1] + Nq * Nq;
            it[6] = it[2] + Nq * Nq;
            it[7] = it[3] + Nq * Nq;
            it += mesh_vis->Nverts;
          }
        }
      }
    }
    o_etov.copyFrom(etov.data());
    return o_etov;
  }();

  mesh_data["coordsets/coords/type"] = "explicit";
  mesh_data["coordsets/coords/values/x"].set_external((dfloat *)mesh_vis->o_x.ptr(), mesh_vis->Nlocal);
  mesh_data["coordsets/coords/values/y"].set_external((dfloat *)mesh_vis->o_y.ptr(), mesh_vis->Nlocal);
  mesh_data["coordsets/coords/values/z"].set_external((dfloat *)mesh_vis->o_z.ptr(), mesh_vis->Nlocal);

  mesh_data["topologies/mesh/type"] = "unstructured";
  mesh_data["topologies/mesh/coordset"] = "coords";
  mesh_data["topologies/mesh/elements/shape"] = "hex";
  mesh_data["topologies/mesh/elements/connectivity"].set_external((dlong *)o_connectivity.ptr(), o_connectivity.size());

  fieldOffsetScan.resize(Nfields + 1, 0);
  int ifld = 0;
  fieldOffsetScan[ifld] = 0;
  ifld++;

  if (platform->comm.mpiRank == 0) {
    printf("availiable fields:");
  }
  for (auto &entry : userFieldList) {
    auto fieldName = std::get<0>(entry);
    auto o_fld = std::get<1>(entry);
    auto mesh_fld = std::get<2>(entry);

    fieldOffsetScan[ifld] =
        fieldOffsetScan[ifld - 1] +
        alignStride<dfloat>(mesh_vis->Np * (mesh_fld->Nelements + mesh_fld->totalHaloPairs));

    if (!interpolate) {
      mesh_data["fields/" + fieldName + "/association"] = "vertex";
      mesh_data["fields/" + fieldName + "/topology"] = "mesh";
      mesh_data["fields/" + fieldName + "/values"].set_external((dfloat *)o_fld.ptr(), mesh_fld->Nlocal);
    }

    if (platform->comm.mpiRank == 0) {
      printf(" %s", fieldName.c_str());
    }

    ifld++;
  }

  const auto tSetup = MPI_Wtime() - tStart;
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
  if (platform->comm.mpiRank == 0) {
    printf("running nekAscent actions ...\n");
  }

  const int verbose = platform->options.compareArgs("VERBOSE", "TRUE") ? 1 : 0;
  const bool movingMesh = platform->options.compareArgs("MOVING MESH", "TRUE");

  mesh_data["state/cycle"] = tstep;
  mesh_data["state/time"] = time;

  occa::memory o_fields;

  if (interpolate) {
    platform->timer.tic("nekAscentInterpoate", 1);

    o_fields = platform->o_memPool.reserve<dfloat>(fieldOffsetScan[Nfields]);

    if (updateMesh || movingMesh) {
      if (uniform) {
        mesh_in->map2Uniform(mesh_vis->N, mesh_in->o_x, mesh_vis->o_x);
        mesh_in->map2Uniform(mesh_vis->N, mesh_in->o_y, mesh_vis->o_y);
        mesh_in->map2Uniform(mesh_vis->N, mesh_in->o_z, mesh_vis->o_z);
      } else {
        mesh_in->interpolate(mesh_vis, mesh_in->o_x, mesh_vis->o_x);
        mesh_in->interpolate(mesh_vis, mesh_in->o_y, mesh_vis->o_y);
        mesh_in->interpolate(mesh_vis, mesh_in->o_z, mesh_vis->o_z);
      }
      updateMesh = false;
    }

    int ifld = 0;
    for (auto &entry : userFieldList) {
      auto fieldName = std::get<0>(entry);
      auto o_fldIn = std::get<1>(entry);
      auto mesh_fld = std::get<2>(entry);
      dlong fieldLength = mesh_fld->Nelements * mesh_vis->Np;

      auto o_fldOut = o_fields.slice(fieldOffsetScan[ifld]);
      if (uniform) {
        mesh_fld->map2Uniform(mesh_vis->N, o_fldIn, o_fldOut);
      } else {
        mesh_fld->interpolate(mesh_vis, o_fldIn, o_fldOut);
      }

      mesh_data["fields/" + fieldName + "/association"] = "vertex";
      mesh_data["fields/" + fieldName + "/topology"] = "mesh";
      mesh_data["fields/" + fieldName + "/values"].set_external((dfloat *)o_fldOut.ptr(), fieldLength);

      ifld++;
    }
    platform->timer.toc("nekAscentInterpoate");
  }

  mAscent.publish(mesh_data);

  conduit::Node actions;
  conduit::Node triggers;

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
    o_connectivity.free();
    mAscent.close();
  }
}

#endif
#endif // hpp
