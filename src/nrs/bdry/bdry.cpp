#include <fstream>
#include <set>
#include <utility>

#include "platform.hpp"
#include "udf.hpp"

#include "elliptic.hpp"
#include "alignment.hpp"
#include "nrs.hpp"

nrs_t::bdry::bdry()
{
  importFromNek = true;
}

static boundaryAlignment_t computeAlignment(mesh_t *mesh, dlong element, dlong face)
{
  const dfloat alignmentTol = 1e-3;
  dfloat nxDiff = 0.0;
  dfloat nyDiff = 0.0;
  dfloat nzDiff = 0.0;

  std::vector<dfloat> sgeo;
  sgeo.reserve(mesh->o_sgeo.length());
  mesh->o_sgeo.copyTo(sgeo.data());

  for (int fp = 0; fp < mesh->Nfp; ++fp) {
    const dlong sid = mesh->Nsgeo * (mesh->Nfaces * mesh->Nfp * element + mesh->Nfp * face + fp);
    const dfloat nx = sgeo[sid + NXID];
    const dfloat ny = sgeo[sid + NYID];
    const dfloat nz = sgeo[sid + NZID];
    nxDiff += std::abs(std::abs(nx) - 1.0);
    nyDiff += std::abs(std::abs(ny) - 1.0);
    nzDiff += std::abs(std::abs(nz) - 1.0);
  }

  nxDiff /= mesh->Nfp;
  nyDiff /= mesh->Nfp;
  nzDiff /= mesh->Nfp;

  if (nxDiff < alignmentTol) {
    return boundaryAlignment_t::X;
  }
  if (nyDiff < alignmentTol) {
    return boundaryAlignment_t::Y;
  }
  if (nzDiff < alignmentTol) {
    return boundaryAlignment_t::Z;
  }

  return boundaryAlignment_t::UNALIGNED;
}

void nrs_t::bdry::velocitySetup(std::string field, std::vector<std::string> slist)
{
  int foundAligned = 0;
  int foundUnaligned = 0;

  for (int bid = 0; bid < slist.size(); bid++) {
    std::string key = slist[bid];

    if (key.compare("p") == 0) {
      key = "periodic";
    }

    if (key.compare("w") == 0) {
      key = "zerodirichlet";
    }
    if (key.compare("wall") == 0) {
      key = "zerodirichlet";
    }

    if (key.compare("int") == 0) {
      key = "interpolation";
    }
    if (key.compare("interpolation") == 0) {
      key = "interpolation";
    }

    if (key.compare("inlet") == 0) {
      key = "udfdirichlet";
    }
    if (key.compare("v") == 0) {
      key = "udfdirichlet";
    }

    if (key.compare("mv") == 0) {
      key = "udfdirichlet";
    }

    if (key.compare("udfdirichlet+moving") == 0) {
      key = "udfdirichlet";
    }

    if (key.compare("slipx") == 0 || key.compare("symx") == 0) {
      key = "zeroxvalue/zeroneumann";
      foundAligned++;
    }
    if (key.compare("slipy") == 0 || key.compare("symy") == 0) {
      key = "zerodirichlety/zeroneumann";
      foundAligned++;
    }
    if (key.compare("slipz") == 0 || key.compare("symz") == 0) {
      key = "zerodirichletz/zeroneumann";
      foundAligned++;
    }
    if (key.compare("slip") == 0 || key.compare("sym") == 0) {
      key = "zerodirichletn/zeroneumann";
      foundUnaligned++;
    }

    if (key.compare("tractionx") == 0 || key.compare("shlx") == 0) {
      key = "zerodirichletx/udfneumann";
      foundAligned++;
    }
    if (key.compare("tractiony") == 0 || key.compare("shly") == 0) {
      key = "zerodirichlety/udfneumann";
      foundAligned++;
    }
    if (key.compare("tractionz") == 0 || key.compare("shlz") == 0) {
      key = "zerodirichletz/udfneumann";
      foundAligned++;
    }
    if (key.compare("traction") == 0 || key.compare("shl") == 0) {
      key = "zerodirichletn/udfneumann";
      foundUnaligned++;
    }

    if (key.compare("outlet") == 0) {
      key = "zeroneumann";
    }
    if (key.compare("outflow") == 0) {
      key = "zeroneumann";
    }
    if (key.compare("o") == 0) {
      key = "zeroneumann";
    }

    if (key.compare("onx") == 0) {
      key = "zerodirichletyz/zeroneumann";
      foundAligned++;
    }
    if (key.compare("ony") == 0) {
      key = "zerodirichletxz/zeroneumann";
      foundAligned++;
    }
    if (key.compare("onz") == 0) {
      key = "zerodirichletxy/zeroneumann";
      foundAligned++;
    }
// not supported yet
#if 0
    if (key.compare("on") == 0) {
      key = "zerodirichlett/zeroneumann";
      foundUnaligned++;
    }
#endif
    lowerCase(key);

    nekrsCheck(vBcTextToID.find(key) == vBcTextToID.end(),
               platform->comm.mpiComm,
               EXIT_FAILURE,
               "Invalid velocity bcType (%s)\n",
               key.c_str());

    bToBc[make_pair(field, bid)] = vBcTextToID.at(key);

    nekrsCheck(foundAligned && foundUnaligned,
               platform->comm.mpiComm,
               EXIT_FAILURE,
               "%s\n",
               "Aligned together with unaligned mixed boundary types are not supported!");
  }
}

void nrs_t::bdry::scalarSetup(std::string field, std::vector<std::string> slist)
{
  for (int bid = 0; bid < slist.size(); bid++) {
    std::string key = slist[bid];
    if (key.compare("p") == 0) {
      key = "periodic";
    }

    if (key.compare("int") == 0) {
      key = "interpolation";
    }
    if (key.compare("interpolation") == 0) {
      key = "interpolation";
    }

    if (key.compare("t") == 0) {
      key = "udfdirichlet";
    }
    if (key.compare("inlet") == 0) {
      key = "udfdirichlet";
    }

    if (key.compare("flux") == 0) {
      key = "udfneumann";
    }
    if (key.compare("f") == 0) {
      key = "udfneumann";
    }

    if (key.compare("zeroflux") == 0) {
      key = "zeroneumann";
    }
    if (key.compare("i") == 0) {
      key = "zeroneumann";
    }
    if (key.compare("insulated") == 0) {
      key = "zeroneumann";
    }

    if (key.compare("outflow") == 0) {
      key = "zeroneumann";
    }
    if (key.compare("outlet") == 0) {
      key = "zeroneumann";
    }
    if (key.compare("o") == 0) {
      key = "zeroneumann";
    }

    lowerCase(key);

    nekrsCheck(sBcTextToID.find(key) == sBcTextToID.end(),
               platform->comm.mpiComm,
               EXIT_FAILURE,
               "Invalid scalar bcType (%s)\n",
               key.c_str());

    bToBc[make_pair(field, bid)] = sBcTextToID.at(key);
  }
}

void nrs_t::bdry::setupField(std::vector<std::string> slist, std::string field)
{
  if (slist.size() == 0) {
    return;
  }

  if (slist.size()) {
    importFromNek = false;

    if (slist.size() == 1 && slist[0] == "none") {
      return;
    }
  }

  lowerCase(field);
  fields.insert(field);

  if (field.compare("velocity") == 0) {
    velocitySetup(field, slist);
  } else if (field.compare("mesh") == 0) {
    velocitySetup(field, slist);
  } else if (field.compare(0, 6, "scalar") == 0) {
    scalarSetup(field, slist);
  } else {
    nekrsAbort(platform->comm.mpiComm, EXIT_FAILURE, "unknown field %s\n", field.c_str());
  }
}

void nrs_t::bdry::setup()
{
  int nscal = 0;
  platform->options.getArgs("NUMBER OF SCALARS", nscal);

  std::vector<std::string> sections = {"MESH", "VELOCITY"};
  for (int i = 0; i < nscal; i++) {
    sections.push_back("SCALAR" + scalarDigitStr(i));
  }

  int count = 0;
  int expectedCount = 0;

  auto process = [&](const std::string &section) {
    std::vector<std::string> staleOptions;
    for (auto const &option : platform->options) {
      if (option.first.find(section) == 0) {

        if (option.first.compare(section + " SOLVER") == 0 &&
            option.second.find("NONE") == std::string::npos) {
          expectedCount++;
        }

        if (option.first.find("BOUNDARY TYPE MAP") != std::string::npos) {
          count++;

          if (section == "MESH" && option.first.find("DERIVED") != std::string::npos) {
            deriveMeshBoundaryConditions(serializeString(option.second, ','));
          } else {
            setupField(serializeString(option.second, ','), section);
          }

          staleOptions.push_back(option.first);
        }
      }
    }
    for (auto const &key : staleOptions) {
      platform->options.removeArgs(key);
    }
  };

  for (const auto &section : sections) {
    process(section);
  }

  nekrsCheck(count > 0 && count != expectedCount,
             platform->comm.mpiComm,
             EXIT_FAILURE,
             "%s\n",
             "boundaryTypeMap specfied for some but not all fields!");

  addKernelConstants(platform->kernelInfo);
}

void nrs_t::bdry::deriveMeshBoundaryConditions(std::vector<std::string> velocityBCs)
{
  if (velocityBCs.size() == 0) {
    return;
  }

  meshConditionsDerived = true;

  const std::string field = "mesh";

  fields.insert(field);

  for (int bid = 0; bid < velocityBCs.size(); bid++) {
    const std::string keyIn = velocityBCs[bid];

    std::string key = "zerodirichletn/zeroneumann"; // default

    if (keyIn.compare("none") == 0) {
      key = "none";
    }

    if (keyIn.compare("zerodirichlet") == 0) {
      key = "zerodirichlet";
    }
    if (keyIn.compare("udfdirichlet") == 0) {
      key = "udfdirichlet";
    }

    if (keyIn.compare("p") == 0) {
      key = "periodic";
    }

    if (keyIn.compare("w") == 0) {
      key = "zerodirichlet";
    }
    if (keyIn.compare("wall") == 0) {
      key = "zerodirichlet";
    }
    if (keyIn.compare("inlet") == 0) {
      key = "zerodirichlet";
    }
    if (keyIn.compare("v") == 0) {
      key = "zerodirichlet";
    }

    if (key.compare("int") == 0) {
      key = "zerodirichlet";
    }
    if (key.compare("interpolation") == 0) {
      key = "zerodirichlet";
    }

    if (keyIn.compare("mv") == 0) {
      key = "udfdirichlet";
    }
    if (keyIn.compare("udfdirichlet+moving") == 0) {
      key = "udfdirichlet";
    }

    lowerCase(key);

    nekrsCheck(vBcTextToID.find(key) == vBcTextToID.end(),
               platform->comm.mpiComm,
               EXIT_FAILURE,
               "Invalid bcType (%s)\n",
               key.c_str());

    bToBc[make_pair(field, bid)] = vBcTextToID.at(key);
  }
}

int nrs_t::bdry::typeElliptic(int bid, std::string field) const
{
  if (bid < 1) {
    return ellipticBcType::NO_OP;
  }

  try {
    int bcType = -1;
    if (field.compare("x-velocity") == 0 || field.compare("x-mesh") == 0) {
      const std::string fld = (field.compare("x-velocity") == 0) ? "velocity" : "mesh";
      const int bcID = bToBc.at({fld, bid - 1});

      bcType = ellipticBcType::DIRICHLET;
      if (bcID == bdryBase::bcType_zeroNeumann) {
        bcType = ellipticBcType::NEUMANN;
      }
      if (bcID == bdryBase::bcType_zeroDirichletY_zeroNeumann ||
          bcID == bdryBase::bcType_zeroDirichletY_udfNeumann) {
        bcType = ellipticBcType::NEUMANN;
      }
      if (bcID == bdryBase::bcType_zeroDirichletZ_zeroNeumann ||
          bcID == bdryBase::bcType_zeroDirichletZ_udfNeumann) {
        bcType = ellipticBcType::NEUMANN;
      }
      if (bcID == bdryBase::bcType_zeroDirichletN_zeroNeumann ||
          bcID == bdryBase::bcType_zeroDirichletN_udfNeumann) {
        bcType = ellipticBcType::ZERO_NORMAL;
      }
      if (bcID == bdryBase::bcType_zeroDirichletYZ_zeroNeumann) {
        bcType = ellipticBcType::NEUMANN;
      }
      if (bcID == bdryBase::bcType_zeroDirichletT_zeroNeumann) {
        bcType = ellipticBcType::ZERO_TANGENTIAL;
      }
      if (bcID == bdryBase::bcType_none) {
        bcType = ellipticBcType::NO_OP;
      }
    } else if (field.compare("y-velocity") == 0 || field.compare("y-mesh") == 0) {
      const std::string fld = (field.compare("y-velocity") == 0) ? "velocity" : "mesh";
      const int bcID = bToBc.at({fld, bid - 1});

      bcType = ellipticBcType::DIRICHLET;
      if (bcID == bdryBase::bcType_zeroNeumann) {
        bcType = ellipticBcType::NEUMANN;
      }
      if (bcID == bdryBase::bcType_zeroDirichletX_zeroNeumann ||
          bcID == bdryBase::bcType_zeroDirichletX_udfNeumann) {
        bcType = ellipticBcType::NEUMANN;
      }
      if (bcID == bdryBase::bcType_zeroDirichletZ_zeroNeumann ||
          bcID == bdryBase::bcType_zeroDirichletZ_udfNeumann) {
        bcType = ellipticBcType::NEUMANN;
      }
      if (bcID == bdryBase::bcType_zeroDirichletN_zeroNeumann ||
          bcID == bdryBase::bcType_zeroDirichletN_udfNeumann) {
        bcType = ellipticBcType::ZERO_NORMAL;
      }
      if (bcID == bdryBase::bcType_zeroDirichletXZ_zeroNeumann) {
        bcType = ellipticBcType::NEUMANN;
      }
      if (bcID == bdryBase::bcType_zeroDirichletT_zeroNeumann) {
        bcType = ellipticBcType::ZERO_TANGENTIAL;
      }
      if (bcID == bdryBase::bcType_none) {
        bcType = ellipticBcType::NO_OP;
      }
    } else if (field.compare("z-velocity") == 0 || field.compare("z-mesh") == 0) {
      const std::string fld = (field.compare("z-velocity") == 0) ? "velocity" : "mesh";
      const int bcID = bToBc.at({fld, bid - 1});

      bcType = ellipticBcType::DIRICHLET;
      if (bcID == bdryBase::bcType_zeroNeumann) {
        bcType = ellipticBcType::NEUMANN;
      }
      if (bcID == bdryBase::bcType_zeroDirichletX_zeroNeumann ||
          bcID == bdryBase::bcType_zeroDirichletX_udfNeumann) {
        bcType = ellipticBcType::NEUMANN;
      }
      if (bcID == bdryBase::bcType_zeroDirichletY_zeroNeumann ||
          bcID == bdryBase::bcType_zeroDirichletY_udfNeumann) {
        bcType = ellipticBcType::NEUMANN;
      }
      if (bcID == bdryBase::bcType_zeroDirichletN_zeroNeumann ||
          bcID == bdryBase::bcType_zeroDirichletN_udfNeumann) {
        bcType = ellipticBcType::ZERO_NORMAL;
      }
      if (bcID == bdryBase::bcType_zeroDirichletXY_zeroNeumann) {
        bcType = ellipticBcType::NEUMANN;
      }
      if (bcID == bdryBase::bcType_zeroDirichletT_zeroNeumann) {
        bcType = ellipticBcType::ZERO_TANGENTIAL;
      }
      if (bcID == bdryBase::bcType_none) {
        bcType = ellipticBcType::NO_OP;
      }
    } else if (field.compare("pressure") == 0) {
      const int bcID = bToBc.at({"velocity", bid - 1});
      bcType = ellipticBcType::NEUMANN;
      if (bcID == bdryBase::bcType_zeroNeumann || bcID == bdryBase::bcType_zeroDirichletYZ_zeroNeumann ||
          bcID == bdryBase::bcType_zeroDirichletXZ_zeroNeumann ||
          bcID == bdryBase::bcType_zeroDirichletXY_zeroNeumann ||
          bcID == bdryBase::bcType_zeroDirichletT_zeroNeumann) {
        bcType = ellipticBcType::DIRICHLET;
      }
      if (bcID == bdryBase::bcType_none) {
        bcType = ellipticBcType::NO_OP;
      }
    } else if (field.compare(0, 6, "scalar") == 0) {
      const int bcID = bToBc.at({field, bid - 1});

      bcType = ellipticBcType::NEUMANN;
      if (bcID == bdryBase::bcType_udfDirichlet || bcID == bdryBase::bcType_interpolation) {
        bcType = ellipticBcType::DIRICHLET;
      }
      if (bcID == bdryBase::bcType_none) {
        bcType = ellipticBcType::NO_OP;
      }
    }

    nekrsCheck(bcType == -1,
               MPI_COMM_SELF,
               EXIT_FAILURE,
               "ellipticType lookup of bid %d field %s failed!\n",
               bid,
               field.c_str());

    return bcType;
  } catch (const std::out_of_range &oor) {
    nekrsAbort(MPI_COMM_SELF,
               EXIT_FAILURE,
               "ellipticType lookup of bid %d field %s failed!\n",
               bid,
               field.c_str());
  }

  return 0;
}

std::string nrs_t::bdry::typeText(int bid, std::string field) const
{
  if (bid < 1) {
    return std::string();
  }

  const int bcID = bToBc.at({field, bid - 1});

  if (bcID == bdryBase::bcType_none) {
    return std::string("");
  }

  if (field.compare("velocity") == 0 || field.compare("mesh") == 0) {
    return vBcIDToText.at(bcID);
  } else if (field.compare(0, 6, "scalar") == 0) {
    return sBcIDToText.at(bcID);
  }

  nekrsAbort(MPI_COMM_SELF, EXIT_FAILURE, "%s\n", "Unexpected error occured!");

  return 0;
}

bool nrs_t::bdry::useDerivedMeshBoundaryConditions()
{
  if (importFromNek) {
    return true;
  } else {
    return meshConditionsDerived;
  }
}

void nrs_t::bdry::checkAlignment(mesh_t *mesh) const
{
  bool bail = false;
  for (auto &&field : fields) {
    if (field != std::string("velocity") && field != std::string("mesh")) {
      continue;
    }

    const int nid = size(field);

    std::map<int, boundaryAlignment_t> expectedAlignmentInvalidBIDs;
    std::map<int, std::set<boundaryAlignment_t>> actualAlignmentsInvalidBIDs;

    for (int e = 0; e < mesh->Nelements; e++) {
      for (int f = 0; f < mesh->Nfaces; f++) {
        int bid = mesh->EToB[e * mesh->Nfaces + f];
        int bc = typeId(bid, field);
        if (bc == bdryBase::bcType_zeroDirichletX_zeroNeumann ||
            bc == bdryBase::bcType_zeroDirichletY_zeroNeumann ||
            bc == bdryBase::bcType_zeroDirichletZ_zeroNeumann ||
            bc == bdryBase::bcType_zeroDirichletX_udfNeumann ||
            bc == bdryBase::bcType_zeroDirichletY_udfNeumann ||
            bc == bdryBase::bcType_zeroDirichletZ_udfNeumann ||
            bc == bdryBase::bcType_zeroDirichletYZ_zeroNeumann ||
            bc == bdryBase::bcType_zeroDirichletXZ_zeroNeumann ||
            bc == bdryBase::bcType_zeroDirichletXY_zeroNeumann) {
          auto expectedAlignment = boundaryAlignment_t::UNALIGNED;
          switch (bc) {
          case bdryBase::bcType_zeroDirichletX_zeroNeumann:
            expectedAlignment = boundaryAlignment_t::X;
            break;
          case bdryBase::bcType_zeroDirichletX_udfNeumann:
            expectedAlignment = boundaryAlignment_t::X;
            break;
          case bdryBase::bcType_zeroDirichletYZ_zeroNeumann:
            expectedAlignment = boundaryAlignment_t::X;
            break;
          case bdryBase::bcType_zeroDirichletY_zeroNeumann:
            expectedAlignment = boundaryAlignment_t::Y;
            break;
          case bdryBase::bcType_zeroDirichletY_udfNeumann:
            expectedAlignment = boundaryAlignment_t::Y;
            break;
          case bdryBase::bcType_zeroDirichletXZ_zeroNeumann:
            expectedAlignment = boundaryAlignment_t::Y;
            break;
          case bdryBase::bcType_zeroDirichletZ_zeroNeumann:
            expectedAlignment = boundaryAlignment_t::Z;
            break;
          case bdryBase::bcType_zeroDirichletZ_udfNeumann:
            expectedAlignment = boundaryAlignment_t::Z;
            break;
          case bdryBase::bcType_zeroDirichletXY_zeroNeumann:
            expectedAlignment = boundaryAlignment_t::Z;
            break;
          }

          auto alignment = computeAlignment(mesh, e, f);
          if (alignment != expectedAlignment) {
            expectedAlignmentInvalidBIDs[bid] = expectedAlignment;
            actualAlignmentsInvalidBIDs[bid].insert(alignment);
          }
        }
      }
    }

    int err = expectedAlignmentInvalidBIDs.size();
    MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_INT, MPI_MAX, platform->comm.mpiComm);
    if (err > 0) {
      bail = true;

      std::vector<int> valid(nid, 1);
      for (int bid = 1; bid <= nid; bid++) {
        valid[bid - 1] = expectedAlignmentInvalidBIDs.count(bid) == 0;
      }

      constexpr int invalidAlignment = -1;
      constexpr int nAlignments = 4;
      std::vector<int> expectedAlignments(nid, invalidAlignment);
      std::vector<int> encounteredAlignments(nid * nAlignments, invalidAlignment);
      for (auto &&bidAndAlignments : actualAlignmentsInvalidBIDs) {
        const auto bid = bidAndAlignments.first;
        const auto &alignments = bidAndAlignments.second;
        encounteredAlignments[(bid - 1) * nAlignments + 0] = (alignments.count(boundaryAlignment_t::X));
        encounteredAlignments[(bid - 1) * nAlignments + 1] = (alignments.count(boundaryAlignment_t::Y));
        encounteredAlignments[(bid - 1) * nAlignments + 2] = (alignments.count(boundaryAlignment_t::Z));
        encounteredAlignments[(bid - 1) * nAlignments + 3] =
            (alignments.count(boundaryAlignment_t::UNALIGNED));
        expectedAlignments[(bid - 1)] = static_cast<int>(expectedAlignmentInvalidBIDs[bid]);
      }
      MPI_Allreduce(MPI_IN_PLACE, valid.data(), nid, MPI_INT, MPI_MIN, platform->comm.mpiComm);
      MPI_Allreduce(MPI_IN_PLACE,
                    encounteredAlignments.data(),
                    nid * nAlignments,
                    MPI_INT,
                    MPI_MAX,
                    platform->comm.mpiComm);
      MPI_Allreduce(MPI_IN_PLACE, expectedAlignments.data(), nid, MPI_INT, MPI_MAX, platform->comm.mpiComm);

      if (platform->comm.mpiRank == 0) {
        std::cout << "Encountered incorrectly aligned boundaries in field \"" << field << "\":\n";
        for (int bid = 1; bid <= nid; bid++) {
          if (valid[bid - 1] == 0) {
            std::cout << "\tBoundary ID " << bid << ":\n";
            std::cout << "\t\texpected alignment : "
                      << to_string(static_cast<boundaryAlignment_t>(expectedAlignments[bid - 1])) << "\n";
            std::cout << "\t\tencountered alignments:\n";
            if (encounteredAlignments[(bid - 1) * nAlignments + 0]) {
              std::cout << "\t\t\tX\n";
            }
            if (encounteredAlignments[(bid - 1) * nAlignments + 1]) {
              std::cout << "\t\t\tY\n";
            }
            if (encounteredAlignments[(bid - 1) * nAlignments + 2]) {
              std::cout << "\t\t\tZ\n";
            }
            if (encounteredAlignments[(bid - 1) * nAlignments + 3]) {
              std::cout << "\t\t\tUNALIGNED\n";
            }
          }
        }
      }

      fflush(stdout);
      MPI_Barrier(platform->comm.mpiComm);
    }
  }

  nekrsCheck(bail, platform->comm.mpiComm, EXIT_FAILURE, "%s\n", "");
}

bool nrs_t::bdry::unalignedMixedBoundary(std::string field) const
{
  const auto nid = size(field);

  for (int bid = 1; bid <= nid; bid++) {
    const auto bcType = typeId(bid, field);
    if (bcType == bdryBase::bcType_zeroDirichletN_zeroNeumann) {
      return true;
    }
    if (bcType == bdryBase::bcType_zeroDirichletN_udfNeumann) {
      return true;
    }
    if (bcType == bdryBase::bcType_zeroDirichletT_zeroNeumann) {
      return true;
    }
  }

  return false;
}
