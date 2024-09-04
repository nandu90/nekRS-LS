#ifdef NEKRS_ENABLE_ADIOS

#include "iofldAdios.hpp"

void iofldAdios::validateUserSingleValues(const std::string &name) {}

void iofldAdios::validateUserFields(const std::string &name) {}

void iofldAdios::openEngine()
{
  if (fs::exists(configFile)) {
    adios = new adios2::ADIOS(configFile, platform->comm.mpiComm);
  } else {
    adios = new adios2::ADIOS(platform->comm.mpiComm);
  }

  streamName = "default";
  adiosIO = adios->DeclareIO(streamName);
  if (platform->verbose) {
    adiosIO.SetParameters({{"verbose", "4"}});
  }

  const std::string ext = ".bp";

  if (engineMode == iofld::mode::write) {
    fileNameBase += ext;
    adiosIO.DefineAttribute<uint32_t>("dimension", static_cast<uint32_t>(mesh_vis->dim));
    adiosEngine = adiosIO.Open(fileNameBase, adios2::Mode::Write);
  } else {
    if (platform->comm.mpiRank == 0) {
      std::cout << "reading checkpoint ..." << std::endl; 
      std::cout << " fileName: " << fileNameBase + ext << std::endl << std::flush;
    }
    adiosEngine = adiosIO.Open(fileNameBase, adios2::Mode::ReadRandomAccess);
    if (step < 0) {
      step = adiosEngine.Steps() - 1; // last
    }
    nekrsCheck(step + 1 > adiosEngine.Steps(),
               platform->comm.mpiComm,
               EXIT_FAILURE,
               "step number %d has to be smaller than %zu!\n",
               step,
               adiosEngine.Steps());
    if (platform->comm.mpiRank == 0) {
      std::cout << " requested step: " << step;
      if (step + 1 == adiosEngine.Steps()) {
        std::cout << " (last) ";
      }
      std::cout << std::endl << std::flush;
    }

    for (const auto &entry : adiosIO.AvailableVariables(true)) {
      _availableVariables.push_back(entry.first);
    }
  }
}

std::string iofldAdios::vtkSchema()
{
  std::string schema = R"(
  <VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
      <UnstructuredGrid>
          <Piece NumberOfPoints="numOfPoints" NumberOfCells="numOfCells">
              <Points>
                  <DataArray Name="vertices" />
              </Points>
              <Cells>
                  <DataArray Name="connectivity" />
                  <DataArray Name="types" />
              </Cells>
              <PointData>
  )";

  for (auto &entry : userFields) {
    schema += " <DataArray Name=\"" + std::get<0>(entry) + "\"/>\n";
  }

  schema += R"( <DataArray Name="TIME"> time </DataArray> )";

  schema += R"( 
              </PointData>
          </Piece>
      </UnstructuredGrid>
  </VTKFile>
)";

  return schema;
}

void iofldAdios::generateConnectivity(occa::memory etov)
{
  const auto Nq = mesh_vis->Nq;
  const auto NqPlane = Nq * Nq;

  auto it = static_cast<uint64_t *>(etov.ptr());
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

template <typename InputType, typename OutputType>
void iofldAdios::putVariableConvert(const std::vector<occa::memory> &o_fld, occa::memory &o_putVariable)
{
#if 0
  nekrsCheck(o_putVariable.mode() != "Serial",
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "%s\n",
             "put variable has an non-host address!");
#endif

  const auto dim_fld = o_fld.size();
  nekrsCheck(dim_fld != 1 && dim_fld != 3,
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "%s %d\n",
             "invalid field dimension=",
             dim_fld);

  auto map = [&](const int dim_i) {
    auto o_map = platform->device.mallocHost<InputType>(mesh_vis->Nlocal);

    if (!uniform && (mesh_vis->N == mesh->N)) {
      o_map.copyFrom(o_fld[dim_i]);
    } else {
      nekrsCheck((!std::is_same_v<InputType, dfloat>),
                 MPI_COMM_SELF,
                 EXIT_FAILURE,
                 "%s\n",
                 "field has be of type dfloat for mapping to a different N or a uniform mesh");

      auto o_tmp = platform->o_memPool.reserve<dfloat>(mesh_vis->Nlocal);
      if (uniform) {
        mesh->map2Uniform(o_fld[dim_i], mesh_vis, o_tmp);
      } else {
        mesh->interpolate(o_fld[dim_i], mesh_vis, o_tmp);
      }

      o_map.copyFrom(o_tmp);
    }
    return o_map;
  };

  auto o_putVariablePtr = static_cast<OutputType *>(o_putVariable.ptr());
  for (int n = 0; n < o_putVariable.size(); ++n) {
    o_putVariablePtr[n] = 0;
  }

  for (int idim = 0; idim < dim_fld; idim++) {
    auto fld = map(idim);
    auto fldPtr = static_cast<InputType *>(fld.ptr());

    // VTK expects AOS
    for (int n = 0; n < fld.size(); ++n) {
      o_putVariablePtr[n * dim_fld + idim] = static_cast<OutputType>(fldPtr[n]);
    }
  }
};

size_t iofldAdios::write()
{
  nekrsCheck(elementMask.size(), MPI_COMM_SELF, EXIT_FAILURE, "%s\n", "element filter is not supported yet!");
  return (precision == 64) ? write_<double>() : write_<float>();
}

template <typename OutputType> size_t iofldAdios::write_()
{
  if (platform->comm.mpiRank == 0) {
    std::cout << " fileName: " << fileNameBase << std::endl << std::flush;
  }

  nekrsCheck(getStepCounter() > 0 && (N > 0 && N != mesh_vis->N),
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "%s\n",
             "write attribute polynomialOrder order cannot be changed after the initial step");

  if (uniform || (N > 0 && N != mesh_vis->N)) {
    mesh_vis = genVisMesh();
  }
  const uint32_t NumOfCells = mesh_vis->Nelements * std::pow(mesh_vis->N, mesh_vis->dim);
  const uint32_t VTK_CELL_TYPE = VTK_HEXAHEDRON;
  const uint32_t NumOfPoints = mesh_vis->Nlocal;

  size_t writtenBytes = 0;
  adiosEngine.BeginStep();

  // connectivity + mesh coordinates
  if (getStepCounter() == 0 || platform->options.compareArgs("MOVING MESH", "TRUE")) {
    auto adiosMode = adios2::Mode::Sync; // avoid dangling pointer due to potential memPool resize

    putVariable<uint32_t>("types", VTK_CELL_TYPE, adiosMode);
    putVariable<uint32_t>("numOfCells", NumOfCells, adiosMode);
    putVariable<uint32_t>("numOfPoints", NumOfPoints, adiosMode);
    putVariable<uint32_t>("polynomialOrder", mesh_vis->N, adiosMode);

    auto o_globalElementsIds = platform->memPool.reserve<uint64_t>(mesh_vis->Nelements);
    auto globalElementsIdsPtr = static_cast<uint64_t *>(o_globalElementsIds.ptr());
    for (int e = 0; e < o_globalElementsIds.size(); e++) {
      globalElementsIdsPtr[e] = nek::localElementIdToGlobal(e);
    }
    putVariable<uint64_t>("globalElementIds", o_globalElementsIds, {o_globalElementsIds.size()}, adiosMode);
    writtenBytes += o_globalElementsIds.size() * sizeof(uint64_t);

    auto o_connectivity =
        platform->memPool.reserve<uint64_t>(static_cast<uint64_t>(NumOfCells) * (mesh_vis->Nverts + 1));
    generateConnectivity(o_connectivity);
    putVariable<uint64_t>("connectivity",
                          o_connectivity,
                          {static_cast<size_t>(NumOfCells), static_cast<size_t>(mesh_vis->Nverts + 1)},
                          adiosMode);
    writtenBytes += o_connectivity.size() * sizeof(uint64_t);

    auto o_coordVertices = platform->memPool.reserve<OutputType>(mesh_vis->dim * mesh_vis->Nlocal);

    std::vector<occa::memory> o_xyz;
    o_xyz.push_back(mesh->o_x);
    o_xyz.push_back(mesh->o_y);
    o_xyz.push_back(mesh->o_z);
    putVariableConvert<dfloat, OutputType>(o_xyz, o_coordVertices);
    putVariable<OutputType>("vertices",
                            o_coordVertices,
                            {static_cast<size_t>(mesh_vis->Nlocal), static_cast<size_t>(mesh_vis->dim)},
                            adiosMode);
    writtenBytes += o_coordVertices.size() * sizeof(OutputType);
  }

  std::vector<occa::memory> o_fldDataScratch;
  for (auto &fld : userFields) {
    size_t fldSize = 0;
    for (const auto &entry : std::get<1>(fld)) {
      fldSize += entry.size();
    }
    o_fldDataScratch.push_back(platform->memPool.reserve<OutputType>(fldSize));
  }

  // after this point no memPool reservations are allowed to ensure pointers
  // in o_fldDataScratch do not change

  auto o_fldDataScratchIt = o_fldDataScratch.begin();
  for (const auto &entry : userFields) {
    const auto &name = entry.first;
    const auto &o_fld = entry.second;
    auto &o_fldConverted = *o_fldDataScratchIt;

    const auto inputType = o_fld.at(0).dtype();
    if (inputType == occa::dtype::get<double>()) {
      putVariableConvert<double, OutputType>(o_fld, o_fldConverted);
    } else {
      putVariableConvert<dfloat, OutputType>(o_fld, o_fldConverted);
    }

    o_fldDataScratchIt++;

    const auto count = [&]() {
      const size_t dim_fld = o_fldConverted.size() / mesh_vis->Nlocal;
      if (dim_fld > 1) {
        return adios2::Dims{static_cast<size_t>(mesh_vis->Nlocal), dim_fld};
      } else {
        return adios2::Dims{static_cast<size_t>(mesh_vis->Nlocal)};
      }
    }();

    putVariable<OutputType>(name, o_fldConverted, count);
    writtenBytes += o_fldConverted.size() * sizeof(OutputType);
  }

  for (auto &entry : userSingleValues) {
    auto name = std::get<0>(entry);
    putVariable(name, std::get<1>(entry));
  }

  if (getStepCounter() == 0) {
    adiosIO.DefineAttribute<std::string>("vtk.xml", vtkSchema());
  }

  adiosEngine.PerformDataWrite();
  adiosEngine.EndStep();

  return writtenBytes;
}

template <typename T>
std::vector<occa::memory> iofldAdios::redistributeField(const std::vector<occa::memory> &o_in)
{
  // all entries of o_in are of the same size
  int o_inEntrySize = (o_in.size()) ? o_in.at(0).size() : 0;

  const auto targetInfo = [&]() {
    std::vector<std::pair<int, int>> targetInfo;
    if (o_inEntrySize == 0) {
      return targetInfo;
    }

    auto &data = variables["globalElementIds"].data;
    const auto gids = static_cast<uint64_t *>(data.ptr());
    std::vector<uint64_t> globalElementIds(gids, gids + data.size());

    for (int e = 0; e < globalElementIds.size(); e++) {
      const auto gid = globalElementIds[e];
      targetInfo.push_back(std::make_pair(nek::globalElementIdToRank(gid), nek::globalElementIdToLocal(gid)));
    }
    return targetInfo;
  }();

  const auto maxRemoteSizes = [&]() {
    std::vector<int> work(2);
    work[0] = o_in.size();
    work[1] = o_inEntrySize;
    MPI_Allreduce(MPI_IN_PLACE,
                  work.data(),
                  work.size(),
                  MPI_INT,
                  MPI_MAX,
                  platform->comm.mpiComm); // ensure win is large enough on all ranks
    return std::make_pair(work[0], work[1]);
  }();

  const size_t winFieldOffset = maxRemoteSizes.second;
#if 0
  auto o_win = platform->o_memPool.reserve<T>(maxRemoteSizes.first * winFieldOffset);
#else
  auto o_win = platform->device.malloc<T>(maxRemoteSizes.first * winFieldOffset);
#endif

  int typeSize;
  MPI_Datatype mpiType;
  if constexpr (std::is_same_v<T, double>) {
    mpiType = MPI_DOUBLE;
  } else if constexpr (std::is_same_v<T, float>) {
    mpiType = MPI_FLOAT;
  } else {
    nekrsAbort(MPI_COMM_SELF, EXIT_FAILURE, "%s\n", "unsupported data type!");
  }
  MPI_Type_size(mpiType, &typeSize);

  MPI_Win win;
  MPI_Win_create(o_win.ptr(), o_win.byte_size(), typeSize, MPI_INFO_NULL, platform->comm.mpiComm, &win);

  MPI_Win_lock_all(MPI_MODE_NOCHECK, win);
  if (o_inEntrySize) {
    const int Np = o_inEntrySize / targetInfo.size();
    for (int e = 0; e < targetInfo.size(); e++) {
      const auto targetRank = targetInfo[e].first;
      const auto localElementIndex = targetInfo[e].second;

      for (int dim = 0; dim < o_in.size(); dim++) {
        const MPI_Aint winOffset = Np * localElementIndex + dim * winFieldOffset;
        const auto o_buf = o_in.at(dim).slice(e * Np, Np);
        MPI_Put(o_buf.ptr(), Np, mpiType, targetRank, winOffset, Np, mpiType, win);
      }
    }
  }
  MPI_Win_unlock_all(win);
  MPI_Win_free(&win); // after this point data is visible remotely

  std::vector<occa::memory> o_out;
  const auto o_outSize = maxRemoteSizes.first;
  for (int dim = 0; dim < o_outSize; dim++) {
    o_out.push_back(o_win.slice(dim * winFieldOffset));
  }

  return o_out;
}

template <typename Tadios, typename Tout>
std::vector<occa::memory> iofldAdios::getDataConvert(const std::string &name)
{
  const auto &var = variables[name];
  auto in = var.data;

  std::vector<occa::memory> o_out;
  if (!in.isInitialized()) {
    return o_out;
  }

#if 0
  nekrsCheck(in.mode() != "Serial",
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "variable %s has an non-host address!\n",
             name.c_str());
#endif

  auto inPtr = static_cast<Tadios *>(in.ptr());
  const auto nDim = var.dim;
  const size_t Nlocal = in.size() / nDim;

  auto out = platform->memPool.reserve<Tout>(in.size());
  auto outPtr = static_cast<Tout *>(out.ptr());

  for (int dim = 0; dim < nDim; dim++) {
    // AOS -> SOA
    for (int i = 0; i < Nlocal; i++) {
      outPtr[i + dim * Nlocal] = static_cast<Tout>(inPtr[dim + i * nDim]);
    }
    o_out.push_back(platform->o_memPool.reserve<Tout>(Nlocal));
    o_out.at(dim).copyFrom(outPtr + dim * Nlocal);
  }

  return o_out;
}

template <typename Tadios>
void iofldAdios::getData(const std::string &name, std::vector<occa::memory> &o_userBuf)
{
  //  convert to target datatype and map vector from AOS to SOA
  std::vector<occa::memory> o_convDistributedData;
  if (o_userBuf.at(0).dtype() == occa::dtype::get<double>()) {
    auto o_convData = getDataConvert<Tadios, double>(name);
    o_convDistributedData = redistributeField<double>(o_convData);
  }
  if (o_userBuf.at(0).dtype() == occa::dtype::get<float>()) {
    auto o_convData = getDataConvert<Tadios, float>(name);
    o_convDistributedData = redistributeField<float>(o_convData);
  }

  for (int dim = 0; dim < o_convDistributedData.size(); dim++) {
    if (mesh_vis != mesh) {
      mesh_vis->interpolate(o_convDistributedData[dim], mesh, o_userBuf[dim]);
    } else {
      o_userBuf[dim].copyFrom(o_convDistributedData[dim]);
    }
  }
}

template <typename Tadios> void iofldAdios::getData(const std::string &name, variantType &variant)
{
  if (std::holds_alternative<std::reference_wrapper<int>>(variant)) {
    auto &value = std::get<std::reference_wrapper<int>>(variant).get();
    if (platform->comm.mpiRank == 0) {
      value = *(variables[name].data.ptr<Tadios>());
    }
    MPI_Bcast(&value, 1, MPI_INT, 0, platform->comm.mpiComm);
  } else if (std::holds_alternative<std::reference_wrapper<long long int>>(variant)) {
    auto &value = std::get<std::reference_wrapper<long long int>>(variant).get();
    if (platform->comm.mpiRank == 0) {
      value = *(variables[name].data.ptr<Tadios>());
    }
    MPI_Bcast(&value, 1, MPI_LONG_LONG_INT, 0, platform->comm.mpiComm);
  } else if (std::holds_alternative<std::reference_wrapper<float>>(variant)) {
    auto &value = std::get<std::reference_wrapper<float>>(variant).get();
    if (platform->comm.mpiRank == 0) {
      value = *(variables[name].data.ptr<Tadios>());
    }
    MPI_Bcast(&value, 1, MPI_FLOAT, 0, platform->comm.mpiComm);
  } else if (std::holds_alternative<std::reference_wrapper<double>>(variant)) {
    auto &value = std::get<std::reference_wrapper<double>>(variant).get();
    if (platform->comm.mpiRank == 0) {
      value = *(variables[name].data.ptr<Tadios>());
    }
    MPI_Bcast(&value, 1, MPI_DOUBLE, 0, platform->comm.mpiComm);
  }
}

template <class T> void iofldAdios::getVariable(bool allocateOnly, const std::string &name, int _step)
{
  auto adiosVariable = adiosIO.InquireVariable<T>(name);
  nekrsCheck(!static_cast<bool>(adiosVariable),
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "variable %s not found!\n",
             name.c_str());

  if (!allocateOnly) {
    nekrsCheck(variables.count(name) == 0,
               MPI_COMM_SELF,
               EXIT_FAILURE,
               "variable %s was not allocated yet!\n",
               name.c_str());

    auto var = variables[name];
    if (!var.data.isInitialized()) {
      return;
    }

    adiosVariable.SetStepSelection(adios2::Box<std::size_t>(var.step, 1)); // single step only

    const auto getMode = adios2::Mode::Deferred;

    if (var.dim > 0) {
      size_t offset = 0;
      for (size_t b = 0; b < var.blocks.size(); b++) {
        adiosVariable.SetBlockSelection(var.blocks.at(b).first);
        adiosEngine.Get(name, var.data.ptr<T>() + offset, getMode);
        offset += var.blocks.at(b).second;
      }
    } else {
      if (platform->comm.mpiRank == 0) {
        adiosEngine.Get(name, var.data.ptr<T>(), getMode);
      }
    }
    return;
  }

  const size_t varStep = (_step > -1) ? _step : step;
  adiosVariable.SetStepSelection(adios2::Box<std::size_t>(varStep, 1));

  const auto &globalBlocks = adiosEngine.BlocksInfo(adiosVariable, varStep);
  nekrsCheck(globalBlocks.size() == 0,
             MPI_COMM_SELF,
             EXIT_FAILURE,
             "required step %d for variable %s not found!\n",
             varStep,
             name.c_str());

  variable var;
  var.type = adios2::GetType<T>();
  var.step = varStep;
  var.blocks = std::vector<std::pair<size_t, size_t>>();

  for (auto &block : globalBlocks) {
    if ((block.BlockID % static_cast<size_t>(platform->comm.mpiCommSize)) ==
        static_cast<size_t>(platform->comm.mpiRank)) {
      size_t NlocalBlock = 0;
      if (block.Count.size() == 0) { // scalar
        var.dim = 0;
        NlocalBlock = 1;
      } else if (block.Count.size() == 1) {
        var.dim = 1;
        NlocalBlock = block.Count[0];
      } else if (block.Count.size() == 2) {
        var.dim = block.Count[1];
        NlocalBlock = block.Count[0] * block.Count[1];
      }
      var.blocks.push_back(std::make_pair(block.BlockID, NlocalBlock));
    }
  }

  if (var.blocks.size() == 0) {
    var.data = o_NULL; // no data on this proc
  } else if (var.dim == 0) {
    var.data = platform->memPool.reserve<T>(1);
  } else {
    const auto Nlocal = [&]() {
      size_t sum = 0;
      for (const auto &entry : var.blocks) {
        sum += entry.second;
      }
      return sum;
    }();
    var.data = platform->memPool.reserve<T>(Nlocal);
  }

  if (platform->verbose && var.blocks.size()) {
    std::cout << " " << name << " on rank " << platform->comm.mpiRank << " is of type " << var.type;

    if (var.dim) {
      std::cout << " has " << var.blocks.size() << " out of " << globalBlocks.size() << " blocks in step "
                << varStep << " with total entries " << var.data.size() << " and dim " << var.dim
                << std::endl;
    } else {
      std::cout << std::endl;
    }
  }

  variables.insert_or_assign(name, var);
}

size_t iofldAdios::read()
{
  const auto userVariables = [&]() {
    std::vector<std::string> variables;
    for (const auto &entry : userFields) {
      variables.push_back(entry.first);
    }
    for (const auto &entry : userSingleValues) {
      variables.push_back(entry.first);
    }
    return variables;
  }();

  // first allocate then get variable to ensure deferred pointer to memPool remains valid
  for (int pass = 0; pass < 2; pass++) {
    const auto allocateOnly = (pass == 0) ? true : false;
    getVariable<uint32_t>(allocateOnly, "polynomialOrder", 0);
    getVariable<uint64_t>(allocateOnly, "globalElementIds", 0);

    for (const auto &name : userVariables) {
      if (std::find(_availableVariables.begin(), _availableVariables.end(), name) ==
          _availableVariables.end()) {
        continue; // skip if not available!
      }

      const auto &type = adiosIO.VariableType(name);

      if (type == adios2::GetType<double>()) {
        getVariable<double>(allocateOnly, name);
      } else if (type == adios2::GetType<float>()) {
        getVariable<float>(allocateOnly, name);
      } else if (type == adios2::GetType<int>()) {
        getVariable<int>(allocateOnly, name);
      } else if (type == adios2::GetType<long long int>()) {
        getVariable<long long int>(allocateOnly, name);
      } else {
        nekrsAbort(MPI_COMM_SELF, EXIT_FAILURE, "ADIOS variable type %s is not supported!\n", type.c_str());
      }
    }
  }

  adiosEngine.PerformGets();
  MPI_Barrier(platform->comm.mpiComm);

  for (auto &entry : userSingleValues) {
    const auto &name = entry.first;
    auto &variant = entry.second;

    if (std::find(_availableVariables.begin(), _availableVariables.end(), name) ==
        _availableVariables.end()) {
      continue;
    }

    const auto &adiosType = variables[name].type;

    if (adiosType == adios2::GetType<double>()) {
      getData<double>(name, variant);
    } else if (adiosType == adios2::GetType<float>()) {
      getData<float>(name, variant);
    } else if (adiosType == adios2::GetType<int>()) {
      getData<int>(name, variant);
    } else if (adiosType == adios2::GetType<long long int>()) {
      getData<long long int>(name, variant);
    }
  }

  mesh_vis = [&]() {
    variantType v = std::ref(N);
    getData<uint32_t>("polynomialOrder", v);
    if (N != mesh->N) {
      return genVisMesh();
    } else {
      return mesh;
    }
  }();

  for (auto &o_entry : userFields) {
    const auto &name = o_entry.first;
    auto &o_userBuf = o_entry.second;

    if (std::find(_availableVariables.begin(), _availableVariables.end(), name) ==
        _availableVariables.end()) {
      continue;
    }

    const auto &adiosType = variables[name].type;

    if (adiosType == adios2::GetType<double>()) {
      getData<double>(name, o_userBuf);
    } else if (adiosType == adios2::GetType<float>()) {
      getData<float>(name, o_userBuf);
    }
  }

  return 0;
}

void iofldAdios::close()
{
  if (static_cast<bool>(adiosEngine)) {
    adiosEngine.Close();
  }
}

#endif
