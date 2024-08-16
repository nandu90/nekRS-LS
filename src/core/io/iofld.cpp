#include "platform.hpp"
#include "nekInterfaceAdapter.hpp"
#include "iofld.hpp"

static std::map<std::string, int > fileCounter;

iofld::iofld(int mode_, const std::string& prefix_, const std::string& engine_)
{
  prefix = prefix_;
  mode = mode_;
  engine = engine_;

  nekrsCheck(engine != "nek", MPI_COMM_SELF, EXIT_FAILURE, "%s\n", "Unsupported engine!");
  nekrsCheck(mode != iofld::read && mode != iofld::write, MPI_COMM_SELF, EXIT_FAILURE, "%s\n", "Invalid mode!");

  tStart = MPI_Wtime();

  platform->options.getArgs("POLYNOMIAL DEGREE", N); 

  if (mode == iofld::write) {
    platform->timer.tic("checkpointing");

    auto counter = [&]() 
    {
      fileCounter[prefix]++;
      std::ostringstream oss;
      oss << std::setw(5) << std::setfill('0') << fileCounter[prefix];
      return oss.str(); 
    };

    std::string casename; 
    platform->options.getArgs("CASENAME", casename);
    fileName = prefix + casename + "0.f" + counter();

     if (platform->comm.mpiRank == 0) { 
      std::cout << "writing checkpoint ..." << std::endl 
                << " fileName: " << fileName << std::endl << std::flush;
    }
  }

  if (mode == iofld::read) {
    fileName = prefix;

    if (platform->comm.mpiRank == 0) { 
      std::cout << "reading checkpoint ..." << std::endl
                << " fileName: " << fileName << std::endl;
    }

    if (engine == "nek") {
      double time;
      double p0th;
      nek::openfld(fileName, time, p0th);
 
      defineVariable<double>("time", time);
      defineVariable<double>("p0th", p0th);
    }
  }
}

void iofld::writeElementFilter(const std::vector<int>& elementMask_)
{
  elementMask = elementMask_;
}

void iofld::writeAttribute(const std::string& key_, const std::string& val)
{
  std::string key = key_;
  lowerCase(key); 

  if (key == "polynomialorder")  
    N = std::stoi(val);
  else if (key == "precision") 
    precision = stoi(val);
  else if (key == "uniform") 
    uniform = (val == "true") ? true : false;
  else
    nekrsAbort(MPI_COMM_SELF, EXIT_FAILURE, "Invalid attribute %s\n", key_.c_str());
}

void iofld::close()
{
  if (mode == iofld::read) {
    std::vector<occa::memory> o_x;
    std::vector<occa::memory> o_u;
    occa::memory o_pp;
    occa::memory o_tt;
    std::vector<occa::memory> o_s;

    nek::readfld(o_x, o_u, o_pp, o_tt, o_s);

    std::vector<occa::memory> o_p{o_pp};
    std::vector<occa::memory> o_t{o_tt};

    // put read buffer to user buffer for requested variables 
    if(o_x.size()) defineVariable<decltype(o_x)>("mesh", std::move(o_x));
    if(o_u.size()) defineVariable<decltype(o_u)>("velocity", std::move(o_u));
    if(o_p.size()) defineVariable<decltype(o_p)>("pressure", std::move(o_p));
    if(o_t.size()) defineVariable<decltype(o_t)>("temperature", std::move(o_t));
    if(o_s.size()) defineVariable<decltype(o_s)>("scalars", std::move(o_s));
  }

  if (mode == iofld::write) {
     if (platform->comm.mpiRank == 0) { 
      std::cout << " settings: N=" << N << "  precision=" << precision << "  uniform=" << uniform << std::endl << std::flush;
    }

    if (engine == "nek") {
      auto getFld = [&](const std::string& name)
      {
         if (fields.count(name) == 0) return std::vector<occa::memory>{};
         return fields[name];
      };

      auto o_x = getFld("mesh");
      auto o_u = getFld("velocity");
      auto o_p = (getFld("pressure").size()) ? getFld("pressure")[0] : o_NULL;
      auto o_t = (getFld("temperature").size()) ? getFld("temperature")[0] : o_NULL;
      auto o_s = getFld("scalars");
 
      const auto time = inquireVariable<double>("time");
      const auto p0th = (variables.count("p0th")) ? inquireVariable<double>("p0th") : 0.0;
 
      const auto FP64 = (precision == 64 ? true : false);

      nek::outfld(fileName, time, /* step */ 0, p0th, FP64, o_x, o_u, o_p, o_t, o_s, elementMask, N, uniform);
    }

    if (platform->comm.mpiRank == 0) {
      std::string casename; 
      platform->options.getArgs("CASENAME", casename);
 
      std::ofstream outFile(prefix + casename + ".nek5000");
      outFile << "filetemplate: " << prefix + casename + R"(%01d.f%05d)" << std::endl 
              << "firsttimestep: 1" << std::endl 
              << "numtimesteps: " <<  fileCounter[prefix] << std::endl; 
      outFile.close();
    } 

    platform->timer.toc("checkpointing");
  }

  MPI_Barrier(platform->comm.mpiComm);
  if (platform->comm.mpiRank == 0) {
    const auto elapsed = MPI_Wtime() - tStart; 
    std::cout << " elapsed time: " << elapsed << "s" 
              << "  (" << fs::file_size(fileName)/elapsed/1e9 << " GB/s)" << std::endl;
  }
}
