/*---------------------------------------------------------------------------*\
   Copyright (c) 2019-2024, UCHICAGO ARGONNE, LLC.

   The UChicago Argonne, LLC as Operator of Argonne National
   Laboratory holds copyright in the Software. The copyright holder
   reserves all rights except those expressly granted to licensees,
   and U.S. Government license rights.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the disclaimer below.

   2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the disclaimer (as noted below)
   in the documentation and/or other materials provided with the
   distribution.

   3. Neither the name of ANL nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
   FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
   UCHICAGO ARGONNE, LLC, THE U.S. DEPARTMENT OF
   ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
   TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

   Additional BSD Notice
   ---------------------
   1. This notice is required to be provided under our contract with
   the U.S. Department of Energy (DOE). This work was produced at
   Argonne National Laboratory under Contract
   No. DE-AC02-06CH11357 with the DOE.

   2. Neither the United States Government nor UCHICAGO ARGONNE,
   LLC nor any of their employees, makes any warranty,
   express or implied, or assumes any liability or responsibility for the
   accuracy, completeness, or usefulness of any information, apparatus,
   product, or process disclosed, or represents that its use would not
   infringe privately-owned rights.

   3. Also, reference herein to any specific commercial products, process,
   or services by trade name, trademark, manufacturer or otherwise does
   not necessarily constitute or imply its endorsement, recommendation,
   or favoring by the United States Government or UCHICAGO ARGONNE LLC.
   The views and opinions of authors expressed
   herein do not necessarily state or reflect those of the United States
   Government or UCHICAGO ARGONNE, LLC, and shall
   not be used for advertising or product endorsement purposes.

\*---------------------------------------------------------------------------*/

#include "main.hpp"
#include "nekrs.hpp"

int main(int argc, char** argv)
{

  const auto timeStart = std::chrono::high_resolution_clock::now();
  {
    int request = MPI_THREAD_SINGLE;
    const char* env_val = std::getenv ("NEKRS_MPI_THREAD_MULTIPLE");
    if (env_val)
      if (std::stoi(env_val)) request = MPI_THREAD_MULTIPLE;

    int provided;
    int retval =  MPI_Init_thread(&argc, &argv, request, &provided);
    if (retval != MPI_SUCCESS) {
      std::cout << "FATAL ERROR: Cannot initialize MPI!" << "\n";
      exit(EXIT_FAILURE);
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
  }

  std::signal(SIGUSR2, signalHandlerUpdateFile);  
  {
    const char* env_val = std::getenv("NEKRS_SIGNUM_BACKTRACE");
    if (env_val)
      std::signal(std::atoi(env_val), signalHandlerBacktrace);  
  }

  MPI_Barrier(MPI_COMM_WORLD);
  const double time0 = MPI_Wtime(); 

  MPI_Comm commGlobal;
  MPI_Comm_dup(MPI_COMM_WORLD, &commGlobal);

  {
    if(!getenv("NEKRS_HOME")) {
      std::cout << "FATAL ERROR: Cannot find env variable NEKRS_HOME!" << "\n";
      fflush(stdout);
      MPI_Abort(commGlobal, EXIT_FAILURE);
    }

    std::string bin(getenv("NEKRS_HOME"));
    bin += "/bin/nekrs";
    const char* ptr = realpath(bin.c_str(), NULL);
    if(!ptr) {
      std::cout << "FATAL ERROR: Cannot find " << bin << "!\n";
      fflush(stdout);
      MPI_Abort(commGlobal, EXIT_FAILURE);
    }
  }

  cmdOptions* cmdOpt = processCmdLineOptions(argc, argv, commGlobal);

  MPI_Comm comm = setupSession(cmdOpt, commGlobal);

  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  if (rank == 0) {
     time_t now = time(0);
     tm *gmtm = gmtime(&now);
     char *dt = asctime(gmtm);
     std::cout << "UTC time: " << dt << std::endl;
  }
  MPI_Barrier(comm);

  if (cmdOpt->attach) {
    fprintf(stderr, "rank %d: pid<%d>\n", rank, getpid());
    MPI_Barrier(comm);
    if (rank == 0) {
      fprintf(stderr, "Attach debugger, then press enter to continue\n");
      std::cin.get();
    }
  }

  auto abort = [&]()
  {
    if (cmdOpt->debug) {
      throw;
    } else {
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
  };

  try {
    { // change working dir
      const size_t last_slash = cmdOpt->setupFile.rfind('/') + 1;
      const std::string casepath = cmdOpt->setupFile.substr(0,last_slash);
      chdir(casepath.c_str());
      const std::string casename = cmdOpt->setupFile.substr(last_slash, cmdOpt->setupFile.length() - last_slash);
      if(casepath.length() > 0) chdir(casepath.c_str());
      cmdOpt->setupFile.assign(casename);
    }

    auto parKeyValuePairs = readPar(cmdOpt->setupFile, comm);

    nekrs::setup(commGlobal, 
                 comm, 
      	         cmdOpt->buildOnly, 
                 cmdOpt->sizeTarget,
                 cmdOpt->ciMode, 
                 parKeyValuePairs,
                 cmdOpt->setupFile,
                 cmdOpt->backend, 
                 cmdOpt->deviceID,
                 cmdOpt->nSessions, 
                 cmdOpt->sessionID,
                 cmdOpt->debug);
 
    if (cmdOpt->buildOnly) {
      nekrs::finalize();
      MPI_Finalize();
      return EXIT_SUCCESS;
    }
 
    double time = nekrs::startTime();
 
    double elapsedTime = 0;
    {
      MPI_Barrier(comm);
      const auto timeStop = std::chrono::high_resolution_clock::now();
      elapsedTime += std::chrono::duration<double, std::milli>(timeStop - timeStart).count() / 1e3;
      MPI_Allreduce(MPI_IN_PLACE, &elapsedTime, 1, MPI_DOUBLE, MPI_MAX, comm);
      nekrs::updateTimer("setup", elapsedTime);
      if (rank == 0)
        std::cout << "initialization took " << elapsedTime << " s" << std::endl;
    }
 
    int tStep = 0;
    int isLastStep = 0;
    nekrs::lastStep(isLastStep);
 
    nekrs::udfExecuteStep(time, tStep, /* outputStep */ 0);
    nekrs::resetTimer("udfExecuteStep");
 
    double elapsedStepSum = 0;
 
    double tSolveStepMin = std::numeric_limits<double>::max();
    double tSolveStepMax = std::numeric_limits<double>::min();
 
    if (nekrs::endTime() > nekrs::startTime()) {
      if (rank == 0) std::cout << "\ntimestepping to time " << nekrs::endTime() << " ...\n";
    } else if (nekrs::numSteps() > tStep) {
      if (rank == 0) std::cout << "\ntimestepping for " << nekrs::numSteps() << " steps ...\n";
    } else {
      isLastStep = 1;
      if (rank == 0) std::cout << "endTime or numSteps reached already -> skip timestepping\n"; 
    }

    fflush(stdout);
    MPI_Pcontrol(1);
    while (!isLastStep) {
      MPI_Barrier(comm);
      const double timeStartStep = MPI_Wtime();
 
      ++tStep;
      double dt = nekrs::dt(tStep);
      isLastStep = nekrs::lastStep(time + dt, tStep, elapsedTime);
      if (isLastStep && nekrs::endTime() > 0) dt = nekrs::endTime() - time;

      int outputStep = nekrs::outputStep(time + dt, tStep);
      if (nekrs::writeInterval() == 0) outputStep = 0;
      if (isLastStep) outputStep = 1;
      if (nekrs::writeInterval() < 0) outputStep = 0;
      nekrs::outputStep(outputStep);
 
      nekrs::initStep(time, dt, tStep);
      
      int corrector = 1;
      bool converged = false;
      do {
        converged = nekrs::runStep(corrector++);
      } while (!converged);
  
      time = nekrs::finishStep();
 
      {
        int read = sig_processUpdFile;
        MPI_Bcast(&read, 1, MPI_INT, 0, comm);
        if(read) {
          nekrs::processUpdFile();
          sig_processUpdFile = 0;
        }
      }
 
      if (nekrs::printInfoFreq()) {
        if (tStep % nekrs::printInfoFreq() == 0)
          nekrs::printInfo(time, tStep, false, true);
      }
 
      if (outputStep) nekrs::outfld(time, tStep);
 
      MPI_Barrier(comm);
      const double elapsedStep = MPI_Wtime() - timeStartStep;
      tSolveStepMin = std::min(elapsedStep, tSolveStepMin);
      tSolveStepMax = std::max(elapsedStep, tSolveStepMax);
      nekrs::updateTimer("minSolveStep", tSolveStepMin);
      nekrs::updateTimer("maxSolveStep", tSolveStepMax);
 
      elapsedStepSum += elapsedStep;
      elapsedTime += elapsedStep;
      nekrs::updateTimer("elapsedStep", elapsedStep);
      nekrs::updateTimer("elapsedStepSum", elapsedStepSum);
      nekrs::updateTimer("elapsed", elapsedTime);
 
      if (nekrs::printInfoFreq()) {
        if (tStep % nekrs::printInfoFreq() == 0)
          nekrs::printInfo(time, tStep, true, false);
      }
 
      if(nekrs::runTimeStatFreq()) {
        if (tStep % nekrs::runTimeStatFreq() == 0 || isLastStep)
          nekrs::printRuntimeStatistics(tStep);
      }
 
      if (tStep % 100 == 0) fflush(stdout);
    }
    MPI_Pcontrol(0);
 
    delete cmdOpt;
 
    const int exitValue = nekrs::finalize();
 
    MPI_Barrier(commGlobal);
    MPI_Finalize();
 
    if(exitValue)
      return EXIT_FAILURE;
    else
      return EXIT_SUCCESS;

  }
  catch (const std::overflow_error& e)
  {
    std::cerr << e.what() << std::endl;
    abort();
  }
  catch (const std::underflow_error& e)
  {
    std::cerr << e.what() << std::endl;
    abort();
  }
  catch (const std::runtime_error& e)
  {
    std::cerr << e.what() << std::endl;
    abort();
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    abort();
  }
  catch (...)
  {
    std::cerr << "unknown exception thrown!" << std::endl;
    abort();
  }

}
