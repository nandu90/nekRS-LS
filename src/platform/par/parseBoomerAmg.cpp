void parseBoomerAmgSection(const int rank, setupAide &options, inipp::Ini *ini)
{
  if (ini->sections.count("boomeramg")) {
    int coarsenType;
    if (ini->extract("boomeramg", "coarsentype", coarsenType)) {
      options.setArgs("BOOMERAMG COARSEN TYPE", std::to_string(coarsenType));
    }
    int interpolationType;
    if (ini->extract("boomeramg", "interpolationtype", interpolationType)) {
      options.setArgs("BOOMERAMG INTERPOLATION TYPE", std::to_string(interpolationType));
    }
    int smootherType;
    if (ini->extract("boomeramg", "smoothertype", smootherType)) {
      options.setArgs("BOOMERAMG SMOOTHER TYPE", std::to_string(smootherType));
    }
    int coarseSmootherType;
    if (ini->extract("boomeramg", "coarsesmoothertype", coarseSmootherType)) {
      options.setArgs("BOOMERAMG COARSE SMOOTHER TYPE", std::to_string(coarseSmootherType));
    }
    int numCycles;
    if (ini->extract("boomeramg", "iterations", numCycles)) {
      options.setArgs("BOOMERAMG ITERATIONS", std::to_string(numCycles));
    }
    double strongThres;
    if (ini->extract("boomeramg", "strongthreshold", strongThres)) {
      options.setArgs("BOOMERAMG STRONG THRESHOLD", to_string_f(strongThres));
    }
    double nonGalerkinTol;
    if (ini->extract("boomeramg", "nongalerkintol", nonGalerkinTol)) {
      options.setArgs("BOOMERAMG NONGALERKIN TOLERANCE", to_string_f(nonGalerkinTol));
    }
    int aggLevels;
    if (ini->extract("boomeramg", "aggressivecoarseninglevels", aggLevels)) {
      options.setArgs("BOOMERAMG AGGRESSIVE COARSENING LEVELS", std::to_string(aggLevels));
    }
    int chebyRelaxOrder;
    if (ini->extract("boomeramg", "chebyshevrelaxorder", chebyRelaxOrder)) {
      options.setArgs("BOOMERAMG CHEBYSHEV RELAX ORDER", std::to_string(chebyRelaxOrder));
    }
    double chebyFraction;
    if (ini->extract("boomeramg", "chebyshevfraction", chebyFraction)) {
      options.setArgs("BOOMERAMG CHEBYSHEV FRACTION", std::to_string(chebyFraction));
    }
  }
}

