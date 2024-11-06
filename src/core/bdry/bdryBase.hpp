#if !defined(nekrs_bcmap_hpp_)
#define nekrs_bcmap_hpp_

#include "nekrsSys.hpp"
#include "mesh.h"
#include "bcType.h"
#include <set>

class bdryBase
{

public:
  static constexpr int bcType_interpolation = p_bcType_interpolation;
  static constexpr int bcType_zeroDirichlet = p_bcType_zeroDirichlet;
  static constexpr int bcType_udfDirichlet = p_bcType_udfDirichlet;
  static constexpr int bcType_zeroDirichletX_zeroNeumann = p_bcType_zeroDirichletX_zeroNeumann;
  static constexpr int bcType_zeroDirichletY_zeroNeumann = p_bcType_zeroDirichletY_zeroNeumann;
  static constexpr int bcType_zeroDirichletZ_zeroNeumann = p_bcType_zeroDirichletZ_zeroNeumann;
  static constexpr int bcType_zeroDirichletN_zeroNeumann = p_bcType_zeroDirichletN_zeroNeumann;
  static constexpr int bcType_zeroDirichletX_udfNeumann = p_bcType_zeroDirichletX_udfNeumann;
  static constexpr int bcType_zeroDirichletY_udfNeumann = p_bcType_zeroDirichletY_udfNeumann;
  static constexpr int bcType_zeroDirichletZ_udfNeumann = p_bcType_zeroDirichletZ_udfNeumann;
  static constexpr int bcType_zeroDirichletN_udfNeumann = p_bcType_zeroDirichletN_udfNeumann;
  static constexpr int bcType_zeroDirichletYZ_zeroNeumann = p_bcType_zeroDirichletYZ_zeroNeumann;
  static constexpr int bcType_zeroDirichletXZ_zeroNeumann = p_bcType_zeroDirichletXZ_zeroNeumann;
  static constexpr int bcType_zeroDirichletXY_zeroNeumann = p_bcType_zeroDirichletXY_zeroNeumann;
  static constexpr int bcType_zeroDirichletT_zeroNeumann = p_bcType_zeroDirichletT_zeroNeumann;
  static constexpr int bcType_zeroNeumann = p_bcType_zeroNeumann;
  static constexpr int bcType_udfNeumann = p_bcType_udfNeumann;
  static constexpr int bcType_none = p_bcType_none;

  virtual ~bdryBase() = default;

  virtual void setup() = 0;

  bool useNek() const
  {
    return importFromNek;
  };

  int typeId(int bid, std::string field) const
  {
    if (bid < 1) {
      return bcType_none;
    }

    try {
      return bToBc.at({field, bid - 1});
    } catch (const std::out_of_range &oor) {
      nekrsAbort(MPI_COMM_SELF, EXIT_FAILURE, "lookup of bid %d field %s failed!\n", bid, field.c_str());
    }

    return -1;
  };

  virtual int typeElliptic(int bid, std::string field) const = 0;
  virtual std::string typeText(int bid, std::string field) const = 0;

  int size(const std::string &_field) const
  {
    std::string field = _field;
    lowerCase(field);

    int cnt = 0;
    for (auto &entry : bToBc) {
      if (entry.first.first == field) {
        cnt++;
      }
    }
    return cnt;
  };

  std::map<std::pair<std::string, int>, int> bIdToTypeId() const
  {
    return bToBc;
  };

  void setBcMap(std::string field, int *map, int nIDs)
  {
    fields.insert(field);
    for (int i = 0; i < nIDs; i++) {
      bToBc[make_pair(field, i)] = map[i];
    }
  };

  virtual void checkAlignment(mesh_t *mesh) const = 0;

  virtual bool unalignedMixedBoundary(std::string field) const = 0;

  virtual void addKernelConstants(occa::properties &kernelInfo)
  {
    const std::string installDir = getenv("NEKRS_HOME");
    kernelInfo["includes"].asArray();
    kernelInfo["includes"] += installDir + "/include/core/bdry/bcType.h";
  };

protected:
  static std::set<std::string> fields;
  static std::map<std::pair<std::string, int>, int> bToBc;
  static bool importFromNek;
};

#endif
