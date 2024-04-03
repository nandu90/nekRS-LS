#if !defined(nekrs_par_hpp_)
#define nekrs_par_hpp_

#include "nekrsSys.hpp"
#include "inipp.hpp"

class Par
{

public:
  Par(MPI_Comm comm);
  void addValidSection(const std::string& name);
  void parse(setupAide& options);
  bool extract(const std::string& section, const std::string& key, std::string& val) { return ini->extract(section, key, val); };

  template <typename T>
  bool extract(const std::string& section, const std::string& key, T& val) { return ini->extract(section, key, val); };

  inipp::Ini *ini;

private:
  setupAide options;
  MPI_Comm comm;


};

#endif
