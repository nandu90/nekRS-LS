#if !defined(nekrs_adios_hpp_)
#define nekrs_adios_hpp_

#if !defined(NEKRS_ENABLE_ADIOS)

#error "No Adios2 installation was found"

#else

#define NEKRS_ADIOS_ENABLED
#include "platform.hpp"
#include "adios2.h"

class NekAdios
{
private:
  adios2::ADIOS *adios;
  adios2::IO io;
  adios2::Engine engine;

  using field = std::tuple<std::string, occa::memory, mesh_t *, dlong>;
  std::vector <field> userFieldList;

  uint32_t VTK_CELL_TYPE;

  bool uniform;

  bool movingMesh;

  bool initialized;
  std::string timerPrefix;

  mesh_t *mesh;
  mesh_t *mesh_vis;

  uint32_t NumOfCells;
  uint32_t NumberOfPoints;

  std::string
  vtkSchema(uint32_t NumberOfPoints, uint32_t NumOfCells, const std::vector<field> &userFieldList)
  {
    std::string schema = R"( 
    <VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
       <UnstructuredGrid>
         <Piece NumberOfPoints=")" +
                         std::to_string(NumberOfPoints) + R"(" NumberOfCells=")" + std::to_string(NumOfCells) +
                         R"(">
           <Points>
             <DataArray Name="vertices" />
           </Points>
           <Cells>
             <DataArray Name="connectivity" />
             <DataArray Name="types" />
           </Cells>
           <PointData>
  )";

    for (auto &entry : userFieldList) {
      const auto fieldName = std::get<0>(entry);
      schema += "        <DataArray Name=\"" + fieldName + "\"/>\n";
    }

    schema += R"( <DataArray Name="TIME"> TIME </DataArray> )";

    schema += R"( 
           </PointData>
         </Piece>
       </UnstructuredGrid>
    </VTKFile>
  )";

    return schema;
  }

  template <typename T> adios2::Variable<T> defineVariable(const std::string &name, adios2::Dims dim = {})
  {
    auto var = io.InquireVariable<T>(name);
    if (var) {
      return var;
    }

    if (dim.size()) {
      return io.DefineVariable<T>(name, {}, {}, dim, adios2::ConstantDims);
    } else {
      return io.DefineVariable<T>(name);
    }
  }

public:
  NekAdios(){};

  NekAdios(mesh_t *meshIn,
           const std::string &name,
           const int N,
           const bool uniform_ = false,
           const std::string streamName = "default",
           const std::string configFile = "")
  {
    mesh = meshIn;
    uniform = uniform_;

    mesh_vis = [&]()
    {
      auto meshNew = meshIn;
      if (uniform || (N > 0 && N != mesh->N)) {
        meshNew = new mesh_t();
        meshNew->Nelements = mesh->Nelements;
        meshNew->dim = mesh->dim;
        meshNew->Nverts = mesh->Nverts;
        meshNew->Nfaces = mesh->Nfaces;
        meshNew->NfaceVertices = mesh->NfaceVertices;
        meshLoadReferenceNodesHex3D(meshNew, N, 0);

        meshNew->x = (dfloat*) std::calloc(meshNew->Nlocal, sizeof(dfloat));
        meshNew->y = (dfloat*) std::calloc(meshNew->Nlocal, sizeof(dfloat));
        meshNew->z = (dfloat*) std::calloc(meshNew->Nlocal, sizeof(dfloat));

        meshNew->o_x = platform->device.malloc<dfloat>(meshNew->Nlocal);
        meshNew->o_y = platform->device.malloc<dfloat>(meshNew->Nlocal);
        meshNew->o_z = platform->device.malloc<dfloat>(meshNew->Nlocal);
      }
      return meshNew;
    }();

    movingMesh = platform->options.compareArgs("MOVING MESH", "TRUE");

    NumOfCells = mesh_vis->Nelements * std::pow(mesh_vis->N, mesh_vis->dim);
    NumberOfPoints = mesh_vis->Nlocal;

    timerPrefix = "nekAdios_" + streamName + "::";
    VTK_CELL_TYPE = 12; // VTK_HEXAHEDRON

    platform->timer.addUserStat(timerPrefix);

    if (configFile.size()) {
      adios = new adios2::ADIOS(configFile, platform->comm.mpiComm);
    } else {
      adios = new adios2::ADIOS(platform->comm.mpiComm);
    }
    io = adios->DeclareIO(streamName);
    io.SetEngine("BP5");
    io.DefineAttribute<uint32_t>("dimension", static_cast<uint32_t>(mesh_vis->dim));

    const auto fileName = name + ".bp";
    engine = io.Open(fileName, adios2::Mode::Write);

    initialized = true;
  }

  NekAdios(mesh_t *mesh,
           const std::string &name,
           const std::string streamName = "default",
           const std::string configFile = "") : 
           NekAdios(mesh, name, mesh->N, false, streamName, configFile) {}

  void close()
  {
    engine.Close();
  }

  ~NekAdios()
  {
    if (mesh_vis != mesh) {
      free(mesh_vis->x);
      free(mesh_vis->y);
      free(mesh_vis->z);

      mesh_vis->o_x.free();
      mesh_vis->o_y.free();
      mesh_vis->o_z.free();
    }
    close();
  }

  void addScalarField(const std::string& name, occa::memory o_fld, mesh_t *mesh_fld)
  {
    userFieldList.push_back(std::tuple{name, o_fld.slice(0, mesh_fld->Nlocal), mesh_fld, 0});
  }

  void addVectorField(const std::string& name, occa::memory o_fld, mesh_t *mesh_fld, dlong offset)
  {
    userFieldList.push_back(std::tuple{name, o_fld.slice(0, mesh_fld->dim * offset), mesh_fld, offset});
  }

  void clearData()
  {
    userFieldList.clear();
  }

  template <typename OutputType = float>
  void write(double time, int tstep)
  {
    nekrsCheck(!initialized, MPI_COMM_SELF, EXIT_FAILURE, "%s\n", "called prior to initialization!");

    platform->timer.tic(timerPrefix + "write");

    if (platform->comm.mpiRank == 0) {
      std::cout << timerPrefix << " writing to " << engine.Name() << " ...";
    }
    const double tStart = MPI_Wtime();

    static bool firstTime = true;
    size_t bytes = 0;

    // ensure that the data remains valid until EndStep() is reached
    std::vector<OutputType> vertices;
    std::vector<uint64_t> etov;
    std::vector<std::vector<OutputType>> fldData;

    engine.BeginStep();

    auto var_time = defineVariable<double>("TIME");
    engine.Put(var_time, time);

    if (firstTime) { 
      io.DefineAttribute<std::string>("vtk.xml", vtkSchema(NumberOfPoints, NumOfCells, userFieldList));
    }

    if (firstTime || movingMesh) {
      if (firstTime) {
        auto var_types = defineVariable<uint32_t>("types");
        engine.Put(var_types, VTK_CELL_TYPE);

        auto var_NumOfCells = defineVariable<uint32_t>("NumOfCells");
        engine.Put(var_NumOfCells, NumOfCells);

        auto var_NumberOfPoints = defineVariable<uint32_t>("NumberOfPoints");
        engine.Put(var_NumberOfPoints, NumberOfPoints);

        auto var_connectivity = defineVariable<uint64_t>("connectivity",
            {static_cast<size_t>(NumOfCells), static_cast<size_t>(mesh_vis->Nverts + 1)});

        {
          etov.resize(NumOfCells * (mesh_vis->Nverts + 1));
          const auto Nq = mesh_vis->Nq;
          const auto NqPlane = Nq * Nq;

          auto it = etov.begin();
          for (int e = 0; e < mesh_vis->Nelements; ++e) {
            for (int z = 0; z < mesh_vis->N; ++z) {
              for (int y = 0; y < mesh_vis->N; ++y) {
                for (int x = 0; x < mesh_vis->N; ++x) {
                  it[0] = mesh_vis->Nverts;

                  // VTK_HEXAHEDRON ordering
                  it[1] = ((e * Nq + z) * Nq + y) * Nq + x;
                  it[2] = it[1] + 1;
                  it[3] = it[1] + 1 + Nq;
                  it[4] = it[1] + Nq;

                  it[5] = it[1] + NqPlane;
                  it[6] = it[2] + NqPlane;
                  it[7] = it[3] + NqPlane;
                  it[8] = it[4] + NqPlane;

                  it += (mesh_vis->Nverts + 1);
                }
              }
            }
          }
        }
        engine.Put(var_connectivity, etov.data());
        bytes += etov.size() * sizeof(uint64_t);
      }

      if (uniform) {
        mesh->map2Uniform(mesh_vis->N, mesh->o_x, mesh_vis->o_x);
        mesh->map2Uniform(mesh_vis->N, mesh->o_y, mesh_vis->o_y);
        mesh->map2Uniform(mesh_vis->N, mesh->o_z, mesh_vis->o_z);
      } else if (mesh_vis->N != mesh->N) {
        mesh->interpolate(mesh_vis, mesh->o_x, mesh_vis->o_x);
        mesh->interpolate(mesh_vis, mesh->o_y, mesh_vis->o_y);
        mesh->interpolate(mesh_vis, mesh->o_z, mesh_vis->o_z);
      }
      mesh_vis->o_x.copyTo(mesh_vis->x);
      mesh_vis->o_y.copyTo(mesh_vis->y);
      mesh_vis->o_z.copyTo(mesh_vis->z);

      auto var_vertices =
          defineVariable<OutputType>("vertices", {static_cast<size_t>(NumberOfPoints), static_cast<size_t>(mesh_vis->dim)});
      vertices.resize(NumberOfPoints * mesh_vis->dim);

      // VTK expects AOS
      for (int i = 0; i < NumberOfPoints; ++i) {
        vertices[i * mesh_vis->dim + 0] = mesh_vis->x[i];
        vertices[i * mesh_vis->dim + 1] = mesh_vis->y[i];
        vertices[i * mesh_vis->dim + 2] = mesh_vis->z[i];
      }

      engine.Put(var_vertices, vertices.data());
      bytes += vertices.size() * sizeof(OutputType);

      firstTime = false; 
    }

    for (auto &entry : userFieldList) {
      const auto& fieldName = std::get<0>(entry);
      const auto& o_fld = std::get<1>(entry);
      const auto& mesh_fld = std::get<2>(entry);
      const auto& offset_fld = std::get<3>(entry);

      nekrsCheck(o_fld.dtype().name() != ogsDfloat,
                 MPI_COMM_SELF,
                 EXIT_FAILURE,
                 "%s %s\n",
                 "invalid field data type=",
                 o_fld.dtype().name().c_str());

      const auto dim_fld = (offset_fld) ? static_cast<int>(o_fld.size() / offset_fld) : 1;
      nekrsCheck(!(dim_fld == 1 || dim_fld == 3),
                 MPI_COMM_SELF,
                 EXIT_FAILURE,
                 "%s %d\n",
                 "invalid field dimension=",
                 dim_fld);

      const auto expectedSize = (dim_fld > 1) ? dim_fld * offset_fld : mesh_fld->Nlocal;
      nekrsCheck(o_fld.size() < expectedSize,
                 MPI_COMM_SELF,
                 EXIT_FAILURE,
                 "%s %s\n",
                 "invalid field data type=",
                 o_fld.dtype().name().c_str());

      std::vector<OutputType> fldEntry(dim_fld * NumberOfPoints, 0);
      for (int dim_i = 0; dim_i < dim_fld; dim_i++) {
        auto o_fldEntryOut = [&]()
        {
          const auto o_fldEntry = o_fld.slice(dim_i * offset_fld, mesh_fld->Nlocal);

          if (uniform || (mesh_vis->N != mesh_fld->N)) {
            auto o_fldEntryOut = 
              platform->o_memPool.reserve<dfloat>(mesh_fld->Nelements * mesh_vis->Np);
            if (uniform) {
              mesh_fld->map2Uniform(mesh_vis->N, o_fldEntry, o_fldEntryOut);
            } else {
              mesh_fld->interpolate(mesh_vis, o_fldEntry, o_fldEntryOut);
            }

            return o_fldEntryOut;
          }

          return o_fldEntry;
        }();

        std::vector<dfloat> fldEntryOut(o_fldEntryOut.size());
        o_fldEntryOut.copyTo(fldEntryOut.data());

        // VTK expects AOS
        for (int n = 0; n < fldEntryOut.size(); ++n) {
          fldEntry[n * dim_fld + dim_i] = fldEntryOut[n];
        }
      }
      fldData.push_back(fldEntry);

      const auto count = [&]() {
        if (dim_fld > 1) {
          return adios2::Dims{static_cast<size_t>(NumberOfPoints), static_cast<size_t>(dim_fld)};
        } else {
          return adios2::Dims{static_cast<size_t>(NumberOfPoints)};
        }
      }();

      auto var = defineVariable<OutputType>(fieldName, count);
      engine.Put(var, fldEntry.data());
      bytes += fldEntry.size() * sizeof(OutputType);
    }


    engine.EndStep();

    platform->timer.toc(timerPrefix + "write");
    if (platform->comm.mpiRank == 0) {
      const auto timeWrite = MPI_Wtime() - tStart;
      printf(" done (%gs, %.1gGB/s)\n", timeWrite, bytes/timeWrite/1e9);
    }
    fflush(stdout);

    initialized = true;
  }

};

#endif
#endif
