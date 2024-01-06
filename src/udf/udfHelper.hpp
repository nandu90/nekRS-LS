#if !defined(nekrs_udfhelp_hpp_)
#define nekrs_udfhelp_hpp_

#define CIPASS                                                                                               \
{                                                                                                            \
if (platform->comm.mpiRank == 0)                                                                             \
printf("TESTS passed \n");                                                                                   \
platform->exitValue = 0;                                                                                     \
}
#define CIFAIL                                                                                               \
{                                                                                                            \
if (platform->comm.mpiRank == 0)                                                                             \
printf("TESTS failed!\n");                                                                                   \
platform->exitValue += 1;                                                                                    \
}

static auto linAlg = platform->linAlg;
static auto comm = platform->comm;

#endif
