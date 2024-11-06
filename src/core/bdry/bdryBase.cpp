#include "bdryBase.hpp"

std::set<std::string> bdryBase::fields;
std::map<std::pair<std::string, int>, int> bdryBase::bToBc;
bool bdryBase::importFromNek = false;
