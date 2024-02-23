#include <iostream>
#include <string>
#include <map>
#include <algorithm>
#include <tuple>

#include "platform.hpp"
#include "ogs.hpp"
#include "orderedMap.hpp"

namespace timer
{
namespace
{
struct tagData {
  long long int count;
  double hostElapsed;
  double deviceElapsed;
  double startTime;
  occa::streamTag startTag;
};

orderedMap<std::string, tagData> m_;

const int NEKRS_TIMER_INVALID_KEY = -1;
const int NEKRS_TIMER_INVALID_METRIC = -2;

int ifSync_;

inline int ifSync()
{
  return ifSync_;
}

int enable_sync_;

int enabled;

occa::device device_;
MPI_Comm comm_;

inline void sync()
{
  if (enable_sync_) {
    MPI_Barrier(comm_);
  }
}

double tElapsedTimeSolve = 0;

} // namespace

void timer_t::printStatSetElapsedTimeSolve(double time)
{
  tElapsedTimeSolve = time;
}

std::tuple<double, long long int> timer_t::sumAllMatchingTags(std::function<bool(std::string)> predicate, const std::string metric)
{
  long long int count = 0;
  double elapsed = 0;
  const auto timerTags = platform->timer.tags();

  // filter out tags that do contain tag in the name
  std::vector<std::string> filteredTags;
  std::copy_if(timerTags.begin(),
               timerTags.end(),
               std::back_inserter(filteredTags),
               [&](const std::string &tag) { return predicate(tag); });

  for (auto &&t : filteredTags) {
    count += platform->timer.count(t);
    elapsed += platform->timer.query(t, metric);
  }

  return std::make_tuple(elapsed, count);
}


timer_t::timer_t(MPI_Comm comm, occa::device device, int ifSyncDefault, int enableSync)
{
  init(comm, device, ifSyncDefault, enableSync);
}

void timer_t::init(MPI_Comm comm, occa::device device, int ifSyncDefault, int enableSync)
{
  device_ = device;
  ifSync_ = ifSyncDefault;
  comm_ = comm;
  enable_sync_ = enableSync;
  enabled = 1;
}

void timer_t::set(const std::string tag, double time, long long int count)
{
  m_[tag].startTime = time;
  auto it = m_.find(tag);

  nekrsCheck(it == m_.end(), MPI_COMM_SELF, EXIT_FAILURE, "Invalid tag name %s\n", tag.c_str());

  it->second.hostElapsed = time;
  it->second.deviceElapsed = it->second.hostElapsed;
  it->second.count = count;
}

void timer_t::enable()
{
  enabled = 1;
}

void timer_t::disable()
{
  enabled = 0;
}

void timer_t::reset()
{
  for (auto &it : m_) {
    it.second.startTime = 0;
    it.second.hostElapsed = 0;
    it.second.deviceElapsed = 0;
    it.second.count = 0;
  }
  ogsResetTime();
}

void timer_t::clear()
{
  m_.clear();
  ogsResetTime();
}

void timer_t::enableSync()
{
  enable_sync_ = 1;
}

void timer_t::disableSync()
{
  enable_sync_ = 0;
}

void timer_t::reset(const std::string tag)
{
  std::map<std::string, tagData>::iterator it = m_.find(tag);
  if (it != m_.end()) {
    it->second.startTime = 0;
    it->second.hostElapsed = 0;
    it->second.deviceElapsed = 0;
    it->second.count = 0;
  }
}

void timer_t::finalize()
{
  reset();
}

void timer_t::deviceTic(const std::string tag, int ifSync)
{
  if (!enabled) {
    return;
  }
  if (ifSync) {
    sync();
  }
  m_[tag].startTag = device_.tagStream();
}

void timer_t::deviceTic(const std::string tag)
{
  if (!enabled) {
    return;
  }
  if (ifSync()) {
    sync();
  }
  m_[tag].startTag = device_.tagStream();
}

void timer_t::deviceToc(const std::string tag)
{
  if (!enabled) {
    return;
  }
  occa::streamTag stopTag = device_.tagStream();

  std::map<std::string, tagData>::iterator it = m_.find(tag);
  nekrsCheck(it == m_.end(), MPI_COMM_SELF, EXIT_FAILURE, "Invalid tag name %s\n", tag.c_str());

  it->second.deviceElapsed += device_.timeBetween(it->second.startTag, stopTag);
  it->second.count++;
}

void timer_t::hostTic(const std::string tag, int ifSync)
{
  if (!enabled) {
    return;
  }
  if (ifSync) {
    sync();
  }
  m_[tag].startTime = MPI_Wtime();
}

void timer_t::hostTic(const std::string tag)
{
  if (!enabled) {
    return;
  }
  if (ifSync()) {
    sync();
  }
  m_[tag].startTime = MPI_Wtime();
}

void timer_t::hostToc(const std::string tag)
{
  if (!enabled) {
    return;
  }
  double stopTime = MPI_Wtime();

  auto it = m_.find(tag);
  nekrsCheck(it == m_.end(), MPI_COMM_SELF, EXIT_FAILURE, "Invalid tag name %s\n", tag.c_str());

  it->second.hostElapsed += (stopTime - it->second.startTime);
  it->second.count++;
}

void timer_t::tic(const std::string tag, int ifSync)
{
  if (!enabled) {
    return;
  }
  if (ifSync) {
    sync();
  }
  m_[tag].startTime = MPI_Wtime();
  m_[tag].startTag = device_.tagStream();
}

void timer_t::tic(const std::string tag)
{
  if (!enabled) {
    return;
  }
  if (ifSync()) {
    sync();
  }
  m_[tag].startTime = MPI_Wtime();
  m_[tag].startTag = device_.tagStream();
}

void timer_t::toc(const std::string tag)
{
  if (!enabled) {
    return;
  }
  auto stopTime = MPI_Wtime();
  auto stopTag = device_.tagStream();

  auto it = m_.find(tag);
  nekrsCheck(it == m_.end(), MPI_COMM_SELF, EXIT_FAILURE, "Invalid tag name %s\n", tag.c_str());

  it->second.hostElapsed += (stopTime - it->second.startTime);
  it->second.deviceElapsed += device_.timeBetween(it->second.startTag, stopTag);
  it->second.count++;
}

double timer_t::hostElapsed(const std::string tag)
{
  auto it = m_.find(tag);
  if (it == m_.end()) {
    return NEKRS_TIMER_INVALID_KEY;
  }
  return it->second.hostElapsed;
}

double timer_t::deviceElapsed(const std::string tag)
{
  auto it = m_.find(tag);
  if (it == m_.end()) {
    return NEKRS_TIMER_INVALID_KEY;
  }
  return it->second.deviceElapsed;
}

long long int timer_t::count(const std::string tag)
{
  auto it = m_.find(tag);
  if (it == m_.end()) {
    return NEKRS_TIMER_INVALID_KEY;
  }
  return it->second.count;
}

double timer_t::query(const std::string tag, const std::string metric)
{
  int size;
  MPI_Comm_size(comm_, &size);

  auto it = m_.find(tag);
  if (it == m_.end()) {
    return NEKRS_TIMER_INVALID_KEY;
  }
  auto hostElapsed = it->second.hostElapsed;
  auto deviceElapsed = it->second.deviceElapsed;
  auto count = it->second.count;

  double retVal;

  std::string upperMetric = metric;
  std::transform(upperMetric.begin(), upperMetric.end(), upperMetric.begin(), ::toupper);

  if (upperMetric.compare("HOST:MIN") == 0) {
    MPI_Allreduce(&hostElapsed, &retVal, 1, MPI_DOUBLE, MPI_MIN, comm_);
    return retVal;
  }
  if (upperMetric.compare("HOST:MAX") == 0) {
    MPI_Allreduce(&hostElapsed, &retVal, 1, MPI_DOUBLE, MPI_MAX, comm_);
    return retVal;
  }
  if (upperMetric.compare("HOST:SUM") == 0) {
    MPI_Allreduce(&hostElapsed, &retVal, 1, MPI_DOUBLE, MPI_SUM, comm_);
    return retVal;
  }
  if (upperMetric.compare("HOST:AVG") == 0) {
    MPI_Allreduce(&hostElapsed, &retVal, 1, MPI_DOUBLE, MPI_SUM, comm_);
    return retVal / (size * count);
  }
  if (upperMetric.compare("DEVICE:MIN") == 0) {
    MPI_Allreduce(&deviceElapsed, &retVal, 1, MPI_DOUBLE, MPI_MIN, comm_);
    return retVal;
  }
  if (upperMetric.compare("DEVICE:MAX") == 0) {
    MPI_Allreduce(&deviceElapsed, &retVal, 1, MPI_DOUBLE, MPI_MAX, comm_);
    return retVal;
  }
  if (upperMetric.compare("DEVICE:SUM") == 0) {
    MPI_Allreduce(&deviceElapsed, &retVal, 1, MPI_DOUBLE, MPI_SUM, comm_);
    return retVal;
  }
  if (upperMetric.compare("DEVICE:AVG") == 0) {
    MPI_Allreduce(&deviceElapsed, &retVal, 1, MPI_DOUBLE, MPI_SUM, comm_);
    return retVal / (size * count);
  }
  return NEKRS_TIMER_INVALID_METRIC;
}

std::string printPercentage(double num, double dom)
{
  char buf[4096];
  double frac = num / dom;
  snprintf(buf, sizeof(buf), "%4.1f", 100 * frac);
  return std::string(buf);
}

void timer_t::printStatEntry(std::string name, double tTag, long long int nCalls, double tNorm)
{
  int rank;
  MPI_Comm_rank(comm_, &rank);
  const bool child = (tNorm != tElapsedTimeSolve);
  if (tTag > 0) {
    if (rank == 0) {
      std::cout << name << tTag << "s"
                << "  " << printPercentage(tTag, tElapsedTimeSolve);
      if (child) {
        std::cout << "  " << printPercentage(tTag, tNorm);
      } else {
        std::cout << "      ";
      }
      std::cout << "  " << nCalls << "\n";
    }
  }
}

void timer_t::printStatEntry(std::string name, std::string tag, std::string type, double tNorm)
{
  int rank;
  MPI_Comm_rank(comm_, &rank);
  const long long int nCalls = count(tag);
  const double tTag = query(tag, type);
  const bool child = (tNorm != tElapsedTimeSolve);
  if (tTag > 0) {
    if (rank == 0) {
      std::cout << name << tTag << "s"
                << "  " << printPercentage(tTag, tElapsedTimeSolve);
      if (child) {
        std::cout << "  " << printPercentage(tTag, tNorm);
      } else {
        std::cout << "      ";
      }
      std::cout << "  " << nCalls << "\n";
    }
  }
}

void timer_t::printStatEntry(std::string name, double time, double tNorm)
{
  int rank;
  MPI_Comm_rank(comm_, &rank);
  const bool child = (tNorm != tElapsedTimeSolve);
  if (time > 0) {
    if (rank == 0) {
      std::cout << name << time << "s"
                << "  " << printPercentage(time, tElapsedTimeSolve);
      if (child) {
        std::cout << "  " << printPercentage(time, tNorm);
      } else {
        std::cout << "      ";
      }
      std::cout << "\n";
    }
  }
}


void timer_t::printAll()
{
  if (platform->comm.mpiRank != 0) {
    return;
  }
  std::cout << "Device timers: {\n";
  for (auto &&[name, data] : m_) {
    std::cout << "\t" << name << " " << data.deviceElapsed << ",\n";
  }
  std::cout << "}\n";

  std::cout << "Host timers: {\n";
  for (auto &&[name, data] : m_) {
    std::cout << "\t" << name << " " << data.hostElapsed << ",\n";
  }
  std::cout << "}\n";
}

std::vector<std::string> timer_t::tags()
{
  std::vector<std::string> entries;
  auto &keys = m_.keys();
  std::copy(keys.begin(), keys.end(), std::back_inserter(entries));
  return entries;
}

} // namespace timer
