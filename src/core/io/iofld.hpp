#if !defined(nekrs_iofld_hpp_)
#define nekrs_iofld_hpp_

#include "platform.hpp"
#include <variant>

#include <variant>
#include <type_traits>

namespace {
// Helper to check if a type is in a variant
template <typename T, typename Variant>
struct is_in_variant;

template <typename T, typename... Types>
struct is_in_variant<T, std::variant<Types...>> : std::disjunction<std::is_same<T, Types>...> {};

// Helper to check if a type is NOT in a variant
template <typename T, typename Variant>
constexpr bool is_in_variant_v = is_in_variant<T, Variant>::value;
}

class iofld
{
public:
  using variableType = std::variant< dlong, hlong, double, float, std::string >; 

  static constexpr int read = 1;
  static constexpr int write = 2;

  iofld(int mode_, const std::string& prefix_ = std::string(), const std::string& engine_ = std::string("nek"));

  void writeAttribute(const std::string& key, const std::string& val);
  void writeElementFilter(const std::vector<int>& elementMask);
  void close();

  template<typename T = std::vector<occa::memory>>
  void defineVariable(const std::string& name, T u)
  {
    if (engine == "nek") {
      std::vector<std::string> validNames = 
        {"time", "p0th", "mesh", "velocity", "pressure", "temperature", "scalars"}; 
      nekrsCheck(std::find(validNames.begin(), validNames.end(), name) == validNames.end(), 
                 MPI_COMM_SELF, EXIT_FAILURE, "%s\n", "Unsupport variable name!");
    }

    if constexpr (std::is_same_v<T, std::vector<occa::memory>>) {
      for(const auto& v : u) {
        nekrsCheck(v.dtype() != occa::dtype::get<dfloat>(), MPI_COMM_SELF, EXIT_FAILURE, "%s!\n", "Invalid typed occa::memory");
      }
      fields.insert({name, u});  
    } else {
      variables.insert({name, u});  
    }
    // add error hanling for unspported data types!
  };

  template<typename T = std::vector<occa::memory>>
  const T inquireVariable(const std::string& name)
  {
     nekrsCheck(fields.count(name) == 0 && variables.count(name) == 0, 
                MPI_COMM_SELF, EXIT_FAILURE, "Invalid variable name %s!\n", name.c_str());

    if constexpr (std::is_same_v<T, std::vector<occa::memory>>) {
      return fields.at(name);
    } else {
      static_assert(is_in_variant_v<T, variableType>, "Unsupported variable type");

      auto visitor = [](const auto& value) -> variableType 
      {
        if constexpr (std::is_same_v<decltype(value), dlong>) {
          return value;
        } else if constexpr (std::is_same_v<decltype(value), hlong>) {
          return value;
        } else if constexpr (std::is_same_v<decltype(value), float>) {
          return value;
        } else if constexpr (std::is_same_v<decltype(value), double>) {
          return value;
        } else {
          return value;
        }
      };

      return std::get<T>(std::visit(visitor, variables.at(name)));
    }
  };

  std::vector<std::string> variableNames()
  {
    std::vector<std::string> list;
    for(const auto& var : variables) list.push_back(var.first);
    for(const auto& var : fields) list.push_back(var.first);
    return list;
  };


private:
  int mode;
  std::string engine;

  std::string fileName;
  std::string prefix;

  double tStart = 0;

  int N;
  bool uniform = false;
  int precision = 32;

  std::map<std::string, std::vector<occa::memory>> fields;
  std::map<std::string, variableType> variables; 

  std::vector<int> elementMask;


};

#endif
