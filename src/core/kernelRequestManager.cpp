#include "nekrsSys.hpp"
#include "kernelRequestManager.hpp"
#include "platform.hpp"
#include "fileUtils.hpp"
#include <regex>

kernelRequestManager_t::kernelRequestManager_t(const platform_t &m_platform)
    : kernelsProcessed(false), platformRef(m_platform)
{
}

void kernelRequestManager_t::add(const std::string& requestName, occa::kernel kernel)
{
  add(requestName, kernel.sourceFilename(), kernel.properties(), ""); 
  requestToKernelMap[requestName] = kernel;
}

void kernelRequestManager_t::add(const std::string &m_requestName,
                                 const std::string &m_fileName,
                                 const occa::properties &m_props,
                                 std::string m_suffix,
                                 bool checkUnique)
{
  auto adjustedProps = platformRef.device.adjustKernelProps(m_fileName, m_props);
  this->add(kernelRequest_t{m_requestName, m_fileName, adjustedProps, m_suffix}, checkUnique);
}

void kernelRequestManager_t::add(kernelRequest_t request, bool checkUnique)
{
  if (!platform->options.compareArgs("REGISTER ONLY", "TRUE")) return; 

  auto iterAndBoolPair = kernels.insert(request);

  // typically checkUnique is false as some requests (source file + props)
  // might map to different kernels 
  if (checkUnique) {
    int unique = (iterAndBoolPair.second) ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &unique, 1, MPI_INT, MPI_MIN, platformRef.comm.mpiComm);

    nekrsCheck(!unique,
               platformRef.comm.mpiComm,
               EXIT_FAILURE,
               "request already exists:\n%s",
               request.to_string().c_str());
  }

  requestNameToRequestMap.insert({request.requestName, request});
  requestToKernelMap.insert({request.requestName, occa::kernel()}); // not loaded yet
}

occa::kernel kernelRequestManager_t::load(const std::string& requestName, const std::string& kernelName, bool checkValid)
{
  if (checkValid) {
    const bool issueError = !processed() || (requestNameToRequestMap.find(requestName) == requestNameToRequestMap.end());
    int errorFlag = issueError ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &errorFlag, 1, MPI_INT, MPI_MAX, platformRef.comm.mpiComm);

    auto errTxt = [&]() {
      std::stringstream txt;
      txt << "\n";
      txt << "Cannot find request " << "<" << requestName << ">" << "!\n";
      txt << "Available:\n";
      for (auto &keyAndValue : requestNameToRequestMap) {
        txt << "\t" << "<" << keyAndValue.second.requestName << ">" << "\n";
      }

      txt << "===========================================================\n";
      return txt.str();
    };

    nekrsCheck(errorFlag, platformRef.comm.mpiComm, EXIT_FAILURE, errTxt().c_str(), "");
  }

  auto kernel = [&]() 
  {
    bool knlIsInitialized = false;
    if (requestToKernelMap.find(requestName) != requestToKernelMap.end()) {
      if (requestToKernelMap.at(requestName).isInitialized()) {
        knlIsInitialized = true;
      }
    }

    occa::kernel knl;
    if (knlIsInitialized) {
      knl = requestToKernelMap.at(requestName);
    } else {
      const auto& req = requestNameToRequestMap.find(requestName)->second;
 
      knl = platformRef.device.buildKernel(req.fileName, kernelName, req.props, MPI_COMM_SELF);
      requestToKernelMap.at(requestName) = knl;
    }
    nekrsCheck(!knl.isInitialized(),
               MPI_COMM_SELF,
               EXIT_FAILURE,
               "kernel %s for request %s could not be initialized!\n",
               kernelName.c_str(),
               requestName.c_str());
    return knl;
  };

  return kernel();
}

occa::kernel kernelRequestManager_t::load(const std::string &requestName, bool checkValid)
{
  const auto& req = requestNameToRequestMap.find(requestName)->second;
  auto fileName = req.fileName;

  auto extractKernelName = [](const std::string& fullPath)
  {
  std::regex kernelNameRegex(R"((.+)\/(.+)\.)");
  std::smatch kernelNameMatch;
  const auto foundKernelName = std::regex_search(fullPath, kernelNameMatch, kernelNameRegex);

  // capture group
  // 0:   /path/to/install/nekrs/kernels/cds/advectMeshVelocityHex3D.okl
  // 1:   /path/to/install/nekrs/kernels/cds
  // 2:   advectMeshVelocityHex3D.okl

  return (foundKernelName && kernelNameMatch.size() == 3) ? kernelNameMatch[2].str() : "";
  };

  return load(requestName, extractKernelName(fileName), checkValid);
}

void kernelRequestManager_t::compile()
{
  if (kernelsProcessed) {
    return;
  } else {
    kernelsProcessed = true;
  }

  const auto &device = platformRef.device;

  constexpr int maxCompilingRanks{32}; // large enough to speed things up, small enough to control pressure on filesystem
  const int rank = platform->cacheLocal ? platformRef.comm.localRank : platformRef.comm.mpiRank;
  const int ranksCompiling = std::min(
                               maxCompilingRanks,
                               platform->cacheLocal ? platformRef.comm.mpiCommLocalSize : platformRef.comm.mpiCommSize
                             );

  if (platformRef.comm.mpiRank == 0 && platform->verbose) {
    std::cout << "requests.size(): " << kernels.size() << std::endl;
  }

  // compile requests (assumed to have a unique occa hash) on build ranks
  if (rank < ranksCompiling) { 
    for (auto&& req: kernels) {
      const int reqId = distance(kernels.begin(), kernels.find(req));
      if (reqId % ranksCompiling == rank) {
        if (!requestToKernelMap[req.requestName].isInitialized()) {
          if (platform->verbose) {
            std::cerr << "Compiling request <" << req.requestName << "> on rank " << rank << std::endl;
          }
          device.compileKernel(req.fileName, req.props, req.suffix, MPI_COMM_SELF);
        }
      }
    }
  }

  MPI_Barrier(platform->comm.mpiComm); // finish compilation

  if (platform->cacheBcast) {
    const auto srcPath = fs::path(getenv("OCCA_CACHE_DIR"));
    const auto dstPath = platform->tmpDir / fs::path("occa/") / "..";
    fileBcast(srcPath, dstPath, platform->comm.mpiComm, platform->verbose);
  }
}
