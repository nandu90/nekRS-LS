/*

The MIT License (MIT)

Copyright (c) 2017 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#include "ogstypes.h"
#include "ogs.hpp"
#include "ogsKernels.hpp"

namespace ogs {

int Nrefs = 0;
const int gatherNodesPerBlock =
    std::min(BLOCKSIZE, 1024); // should be a multiple of blockSize for good unrolling

void *hostBuf;
size_t hostBufSize = 0;

void *haloBuf;
occa::memory o_haloBuf;
occa::memory h_haloBuf;

occa::stream defaultStream;
occa::stream dataStream;

occa::properties kernelInfo;

occa::kernel gatherScatterNewKernel_floatAdd;
occa::kernel gatherScatterNewKernel_doubleAdd;
occa::kernel gatherScatterNewKernel_doubleMin;
occa::kernel gatherScatterNewKernel_doubleMax;

occa::kernel gatherScatterKernel_floatAdd;
occa::kernel gatherScatterKernel_floatMul;
occa::kernel gatherScatterKernel_floatMin;
occa::kernel gatherScatterKernel_floatMax;
occa::kernel gatherScatterKernel_doubleAdd;
occa::kernel gatherScatterKernel_doubleMul;
occa::kernel gatherScatterKernel_doubleMin;
occa::kernel gatherScatterKernel_doubleMax;
occa::kernel gatherScatterKernel_intAdd;
occa::kernel gatherScatterKernel_intMul;
occa::kernel gatherScatterKernel_intMin;
occa::kernel gatherScatterKernel_intMax;
occa::kernel gatherScatterKernel_longAdd;
occa::kernel gatherScatterKernel_longMul;
occa::kernel gatherScatterKernel_longMin;
occa::kernel gatherScatterKernel_longMax;
occa::kernel gatherScatterVecKernel_floatAdd;
occa::kernel gatherScatterVecKernel_floatMul;
occa::kernel gatherScatterVecKernel_floatMin;
occa::kernel gatherScatterVecKernel_floatMax;
occa::kernel gatherScatterVecKernel_doubleAdd;
occa::kernel gatherScatterVecKernel_doubleMul;
occa::kernel gatherScatterVecKernel_doubleMin;
occa::kernel gatherScatterVecKernel_doubleMax;
occa::kernel gatherScatterVecKernel_intAdd;
occa::kernel gatherScatterVecKernel_intMul;
occa::kernel gatherScatterVecKernel_intMin;
occa::kernel gatherScatterVecKernel_intMax;
occa::kernel gatherScatterVecKernel_longAdd;
occa::kernel gatherScatterVecKernel_longMul;
occa::kernel gatherScatterVecKernel_longMin;
occa::kernel gatherScatterVecKernel_longMax;
occa::kernel gatherScatterManyKernel_floatAdd;
occa::kernel gatherScatterManyKernel_floatMul;
occa::kernel gatherScatterManyKernel_floatMin;
occa::kernel gatherScatterManyKernel_floatMax;
occa::kernel gatherScatterManyKernel_doubleAdd;
occa::kernel gatherScatterManyKernel_doubleMul;
occa::kernel gatherScatterManyKernel_doubleMin;
occa::kernel gatherScatterManyKernel_doubleMax;
occa::kernel gatherScatterManyKernel_intAdd;
occa::kernel gatherScatterManyKernel_intMul;
occa::kernel gatherScatterManyKernel_intMin;
occa::kernel gatherScatterManyKernel_intMax;
occa::kernel gatherScatterManyKernel_longAdd;
occa::kernel gatherScatterManyKernel_longMul;
occa::kernel gatherScatterManyKernel_longMin;
occa::kernel gatherScatterManyKernel_longMax;

occa::kernel gatherKernel_floatAdd;
occa::kernel gatherKernel_floatMul;
occa::kernel gatherKernel_floatMin;
occa::kernel gatherKernel_floatMax;
occa::kernel gatherKernel_doubleAdd;
occa::kernel gatherKernel_doubleMul;
occa::kernel gatherKernel_doubleMin;
occa::kernel gatherKernel_doubleMax;
occa::kernel gatherKernel_intAdd;
occa::kernel gatherKernel_intMul;
occa::kernel gatherKernel_intMin;
occa::kernel gatherKernel_intMax;
occa::kernel gatherKernel_longAdd;
occa::kernel gatherKernel_longMul;
occa::kernel gatherKernel_longMin;
occa::kernel gatherKernel_longMax;
occa::kernel gatherVecKernel_floatAdd;
occa::kernel gatherVecKernel_floatMul;
occa::kernel gatherVecKernel_floatMin;
occa::kernel gatherVecKernel_floatMax;
occa::kernel gatherVecKernel_doubleAdd;
occa::kernel gatherVecKernel_doubleMul;
occa::kernel gatherVecKernel_doubleMin;
occa::kernel gatherVecKernel_doubleMax;
occa::kernel gatherVecKernel_intAdd;
occa::kernel gatherVecKernel_intMul;
occa::kernel gatherVecKernel_intMin;
occa::kernel gatherVecKernel_intMax;
occa::kernel gatherVecKernel_longAdd;
occa::kernel gatherVecKernel_longMul;
occa::kernel gatherVecKernel_longMin;
occa::kernel gatherVecKernel_longMax;
occa::kernel gatherManyKernel_floatAdd;
occa::kernel gatherManyKernel_floatMul;
occa::kernel gatherManyKernel_floatMin;
occa::kernel gatherManyKernel_floatMax;
occa::kernel gatherManyKernel_doubleAdd;
occa::kernel gatherManyKernel_doubleMul;
occa::kernel gatherManyKernel_doubleMin;
occa::kernel gatherManyKernel_doubleMax;
occa::kernel gatherManyKernel_intAdd;
occa::kernel gatherManyKernel_intMul;
occa::kernel gatherManyKernel_intMin;
occa::kernel gatherManyKernel_intMax;
occa::kernel gatherManyKernel_longAdd;
occa::kernel gatherManyKernel_longMul;
occa::kernel gatherManyKernel_longMin;
occa::kernel gatherManyKernel_longMax;

occa::kernel scatterKernel_float;
occa::kernel scatterKernel_double;
occa::kernel scatterKernel_int;
occa::kernel scatterKernel_long;
occa::kernel scatterVecKernel_float;
occa::kernel scatterVecKernel_double;
occa::kernel scatterVecKernel_int;
occa::kernel scatterVecKernel_long;
occa::kernel scatterManyKernel_float;
occa::kernel scatterManyKernel_double;
occa::kernel scatterManyKernel_int;
occa::kernel scatterManyKernel_long;
} // namespace ogs

void ogs::initKernels(MPI_Comm comm, occa::device device, ogsBuildKernel_t buildKernel, bool verbose)
{

  ogs::kernelInfo["defines/ "
                  "p_blockSize"] = BLOCKSIZE;
  ogs::kernelInfo["defines/ "
                  "dlong"] = dlongString;
  ogs::kernelInfo["defines/ "
                  "hlong"] = hlongString;

  if ("OpenCL" == device.mode())
    ogs::kernelInfo["defines/"
                    "hlong"] = "long";

  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  const std::string oklpath = std::string(getenv("NEKRS_KERNEL_DIR")) + "/core/ogs/";

  ogs::defaultStream = device.getStream();
  ogs::dataStream = device.createStream();

  ogs::kernelInfo["defines"].asObject();
  ogs::kernelInfo["includes"].asArray();
  ogs::kernelInfo["header"].asArray();
  ogs::kernelInfo["flags"].asObject();

  occa::properties& props = ogs::kernelInfo;

  props["includes"] += oklpath + "/ogsDefs.h";

  props["defines/init_"
        "float"
        "_add"] = (float)0;
  props["defines/init_"
        "float"
        "_mul"] = (float)1;
  props["defines/init_"
        "float"
        "_min"] = (float)std::numeric_limits<float>::max();
  props["defines/init_"
        "float"
        "_max"] = (float)std::numeric_limits<float>::lowest();

  props["defines/init_"
        "double"
        "_add"] = (double)0;
  props["defines/init_"
        "double"
        "_mul"] = (double)1;
  props["defines/init_"
        "double"
        "_min"] = (double)std::numeric_limits<double>::max();
  props["defines/init_"
         "double"
         "_max"] = (double)std::numeric_limits<double>::lowest();

  props["defines/init_"
        "int"
        "_add"] = (int)0;
  props["defines/init_"
        "int"
        "_mul"] = (int)1;
  props["defines/init_"
        "int"
        "_min"] = (int)std::numeric_limits<int>::max();
  props["defines/init_"
         "int"
         "_max"] = (int)std::numeric_limits<int>::lowest();

  props["defines/init_"
        "long_long"
        "_add"] = (long long int)0;
  props["defines/init_"
        "long_long"
        "_mul"] = (long long int)1;
  props["defines/init_"
        "long_long"
        "_min"] = (long long int)std::numeric_limits<long long int>::max();
  props["defines/init_"
         "long_long"
         "_max"] = (long long int)std::numeric_limits<long long int>::lowest();


  // gatherScatter
  props["defines/p_gatherNodesPerBlock"] = gatherNodesPerBlock;

  ogs::gatherScatterKernel_floatAdd =
      buildKernel(oklpath + "gatherScatter.okl", "gatherScatter_float_add", props);
  ogs::gatherScatterNewKernel_floatAdd =
      buildKernel(oklpath + "gatherScatterNew.okl", "gatherScatter_float_add", props);
  ogs::gatherScatterKernel_floatMul =
      buildKernel(oklpath + "gatherScatter.okl", "gatherScatter_float_mul", props);
  ogs::gatherScatterKernel_floatMin =
      buildKernel(oklpath + "gatherScatter.okl", "gatherScatter_float_min", props);
  ogs::gatherScatterKernel_floatMax =
      buildKernel(oklpath + "gatherScatter.okl", "gatherScatter_float_max", props);

  ogs::gatherScatterKernel_doubleAdd =
      buildKernel(oklpath + "gatherScatter.okl", "gatherScatter_double_add", props);
  ogs::gatherScatterNewKernel_doubleAdd =
      buildKernel(oklpath + "gatherScatterNew.okl", "gatherScatter_double_add", props);
  ogs::gatherScatterKernel_doubleMul =
      buildKernel(oklpath + "gatherScatter.okl", "gatherScatter_double_mul", props);
  ogs::gatherScatterKernel_doubleMin =
      buildKernel(oklpath + "gatherScatter.okl", "gatherScatter_double_min", props);
  ogs::gatherScatterNewKernel_doubleMin =
      buildKernel(oklpath + "gatherScatterNew.okl", "gatherScatter_double_min", props);
  ogs::gatherScatterKernel_doubleMax =
      buildKernel(oklpath + "gatherScatter.okl", "gatherScatter_double_max", props);
  ogs::gatherScatterNewKernel_doubleMax =
      buildKernel(oklpath + "gatherScatterNew.okl", "gatherScatter_double_max", props);

  ogs::gatherScatterKernel_intAdd = buildKernel(oklpath + "gatherScatter.okl", "gatherScatter_int_add", props);
  ogs::gatherScatterKernel_intMul = buildKernel(oklpath + "gatherScatter.okl", "gatherScatter_int_mul", props);
  ogs::gatherScatterKernel_intMin = buildKernel(oklpath + "gatherScatter.okl", "gatherScatter_int_min", props);
  ogs::gatherScatterKernel_intMax = buildKernel(oklpath + "gatherScatter.okl", "gatherScatter_int_max", props);

  ogs::gatherScatterKernel_longAdd =
      buildKernel(oklpath + "gatherScatter.okl", "gatherScatter_long_long_add", props);
  ogs::gatherScatterKernel_longMul =
      buildKernel(oklpath + "gatherScatter.okl", "gatherScatter_long_long_mul", props);
  ogs::gatherScatterKernel_longMin =
      buildKernel(oklpath + "gatherScatter.okl", "gatherScatter_long_long_min", props);
  ogs::gatherScatterKernel_longMax =
      buildKernel(oklpath + "gatherScatter.okl", "gatherScatter_long_long_max", props);

  // gatherScatterMany

  ogs::gatherScatterManyKernel_floatAdd =
      buildKernel(oklpath + "gatherScatterMany.okl", "gatherScatterMany_float_add", props);
  ogs::gatherScatterManyKernel_doubleAdd =
      buildKernel(oklpath + "gatherScatterMany.okl", "gatherScatterMany_double_add", props);
  ogs::gatherScatterManyKernel_intAdd =
      buildKernel(oklpath + "gatherScatterMany.okl", "gatherScatterMany_int_add", props);
  ogs::gatherScatterManyKernel_longAdd =
      buildKernel(oklpath + "gatherScatterMany.okl", "gatherScatterMany_long_long_add", props);


  ogs::gatherScatterManyKernel_floatMul =
      buildKernel(oklpath + "gatherScatterMany.okl", "gatherScatterMany_float_mul", props);
  ogs::gatherScatterManyKernel_floatMin =
      buildKernel(oklpath + "gatherScatterMany.okl", "gatherScatterMany_float_min", props);
  ogs::gatherScatterManyKernel_floatMax =
      buildKernel(oklpath + "gatherScatterMany.okl", "gatherScatterMany_float_max", props);

  ogs::gatherScatterManyKernel_doubleMul =
      buildKernel(oklpath + "gatherScatterMany.okl", "gatherScatterMany_double_mul", props);
  ogs::gatherScatterManyKernel_doubleMin =
      buildKernel(oklpath + "gatherScatterMany.okl", "gatherScatterMany_double_min", props);
  ogs::gatherScatterManyKernel_doubleMax =
      buildKernel(oklpath + "gatherScatterMany.okl", "gatherScatterMany_double_max", props);

  ogs::gatherScatterManyKernel_intMul =
      buildKernel(oklpath + "gatherScatterMany.okl", "gatherScatterMany_int_mul", props);
  ogs::gatherScatterManyKernel_intMin =
      buildKernel(oklpath + "gatherScatterMany.okl", "gatherScatterMany_int_min", props);
  ogs::gatherScatterManyKernel_intMax =
      buildKernel(oklpath + "gatherScatterMany.okl", "gatherScatterMany_int_max", props);

  ogs::gatherScatterManyKernel_longMul =
      buildKernel(oklpath + "gatherScatterMany.okl", "gatherScatterMany_long_long_mul", props);
  ogs::gatherScatterManyKernel_longMin =
      buildKernel(oklpath + "gatherScatterMany.okl", "gatherScatterMany_long_long_min", props);
  ogs::gatherScatterManyKernel_longMax =
      buildKernel(oklpath + "gatherScatterMany.okl", "gatherScatterMany_long_long_max", props);

  // gather

  ogs::gatherKernel_floatAdd = buildKernel(oklpath + "gather.okl", "gather_float_add", props);
  ogs::gatherKernel_doubleAdd = buildKernel(oklpath + "gather.okl", "gather_double_add", props);
  ogs::gatherKernel_intAdd = buildKernel(oklpath + "gather.okl", "gather_int_add", props);
  ogs::gatherKernel_longAdd = buildKernel(oklpath + "gather.okl", "gather_long_long_add", props);

  ogs::gatherKernel_floatMul = buildKernel(oklpath + "gather.okl", "gather_float_mul", props);
  ogs::gatherKernel_floatMin = buildKernel(oklpath + "gather.okl", "gather_float_min", props);
  ogs::gatherKernel_floatMax = buildKernel(oklpath + "gather.okl", "gather_float_max", props);

  ogs::gatherKernel_doubleMul = buildKernel(oklpath + "gather.okl", "gather_double_mul", props);
  ogs::gatherKernel_doubleMin = buildKernel(oklpath + "gather.okl", "gather_double_min", props);
  ogs::gatherKernel_doubleMax = buildKernel(oklpath + "gather.okl", "gather_double_max", props);

  ogs::gatherKernel_intMul = buildKernel(oklpath + "gather.okl", "gather_int_mul", props);
  ogs::gatherKernel_intMin = buildKernel(oklpath + "gather.okl", "gather_int_min", props);
  ogs::gatherKernel_intMax = buildKernel(oklpath + "gather.okl", "gather_int_max", props);

  ogs::gatherKernel_longMul = buildKernel(oklpath + "gather.okl", "gather_long_long_mul", props);
  ogs::gatherKernel_longMin = buildKernel(oklpath + "gather.okl", "gather_long_long_min", props);
  ogs::gatherKernel_longMax = buildKernel(oklpath + "gather.okl", "gather_long_long_max", props);

  // gatherMany

  ogs::gatherManyKernel_floatAdd = buildKernel(oklpath + "gatherMany.okl", "gatherMany_float_add", props);
  ogs::gatherManyKernel_doubleAdd = buildKernel(oklpath + "gatherMany.okl", "gatherMany_double_add", props);
  ogs::gatherManyKernel_intAdd = buildKernel(oklpath + "gatherMany.okl", "gatherMany_int_add", props);
  ogs::gatherManyKernel_longAdd = buildKernel(oklpath + "gatherMany.okl", "gatherMany_long_long_add", props);

  ogs::gatherManyKernel_floatMul = buildKernel(oklpath + "gatherMany.okl", "gatherMany_float_mul", props);
  ogs::gatherManyKernel_floatMin = buildKernel(oklpath + "gatherMany.okl", "gatherMany_float_min", props);
  ogs::gatherManyKernel_floatMax = buildKernel(oklpath + "gatherMany.okl", "gatherMany_float_max", props);

  ogs::gatherManyKernel_doubleMul = buildKernel(oklpath + "gatherMany.okl", "gatherMany_double_mul", props);
  ogs::gatherManyKernel_doubleMin = buildKernel(oklpath + "gatherMany.okl", "gatherMany_double_min", props);
  ogs::gatherManyKernel_doubleMax = buildKernel(oklpath + "gatherMany.okl", "gatherMany_double_max", props);

  ogs::gatherManyKernel_intMul = buildKernel(oklpath + "gatherMany.okl", "gatherMany_int_mul", props);
  ogs::gatherManyKernel_intMin = buildKernel(oklpath + "gatherMany.okl", "gatherMany_int_min", props);
  ogs::gatherManyKernel_intMax = buildKernel(oklpath + "gatherMany.okl", "gatherMany_int_max", props);

  ogs::gatherManyKernel_longMul = buildKernel(oklpath + "gatherMany.okl", "gatherMany_long_long_mul", props);
  ogs::gatherManyKernel_longMin = buildKernel(oklpath + "gatherMany.okl", "gatherMany_long_long_min", props);
  ogs::gatherManyKernel_longMax = buildKernel(oklpath + "gatherMany.okl", "gatherMany_long_long_max", props);

  // scatter

  ogs::scatterKernel_float = buildKernel(oklpath + "scatter.okl", "scatter_float", props);
  ogs::scatterKernel_double = buildKernel(oklpath + "scatter.okl", "scatter_double", props);
  ogs::scatterKernel_int = buildKernel(oklpath + "scatter.okl", "scatter_int", props);
  ogs::scatterKernel_long = buildKernel(oklpath + "scatter.okl", "scatter_long_long", props);

  // scatterMany

  ogs::scatterManyKernel_float = buildKernel(oklpath + "scatterMany.okl", "scatterMany_float", props);
  ogs::scatterManyKernel_double = buildKernel(oklpath + "scatterMany.okl", "scatterMany_double", props);
  ogs::scatterManyKernel_int = buildKernel(oklpath + "scatterMany.okl", "scatterMany_int", props);
  ogs::scatterManyKernel_long = buildKernel(oklpath + "scatterMany.okl", "scatterMany_long_long", props);
}

void ogs::freeKernels()
{

  ogs::gatherScatterKernel_floatAdd.free();
  ogs::gatherScatterKernel_floatMul.free();
  ogs::gatherScatterKernel_floatMin.free();
  ogs::gatherScatterKernel_floatMax.free();
  ogs::gatherScatterKernel_doubleAdd.free();
  ogs::gatherScatterKernel_doubleMul.free();
  ogs::gatherScatterKernel_doubleMin.free();
  ogs::gatherScatterKernel_doubleMax.free();
  ogs::gatherScatterKernel_intAdd.free();
  ogs::gatherScatterKernel_intMul.free();
  ogs::gatherScatterKernel_intMin.free();
  ogs::gatherScatterKernel_intMax.free();
  ogs::gatherScatterKernel_longAdd.free();
  ogs::gatherScatterKernel_longMul.free();
  ogs::gatherScatterKernel_longMin.free();
  ogs::gatherScatterKernel_longMax.free();
  ogs::gatherScatterVecKernel_floatAdd.free();
  ogs::gatherScatterVecKernel_floatMul.free();
  ogs::gatherScatterVecKernel_floatMin.free();
  ogs::gatherScatterVecKernel_floatMax.free();
  ogs::gatherScatterVecKernel_doubleAdd.free();
  ogs::gatherScatterVecKernel_doubleMul.free();
  ogs::gatherScatterVecKernel_doubleMin.free();
  ogs::gatherScatterVecKernel_doubleMax.free();
  ogs::gatherScatterVecKernel_intAdd.free();
  ogs::gatherScatterVecKernel_intMul.free();
  ogs::gatherScatterVecKernel_intMin.free();
  ogs::gatherScatterVecKernel_intMax.free();
  ogs::gatherScatterVecKernel_longAdd.free();
  ogs::gatherScatterVecKernel_longMul.free();
  ogs::gatherScatterVecKernel_longMin.free();
  ogs::gatherScatterVecKernel_longMax.free();
  ogs::gatherScatterManyKernel_floatAdd.free();
  ogs::gatherScatterManyKernel_floatMul.free();
  ogs::gatherScatterManyKernel_floatMin.free();
  ogs::gatherScatterManyKernel_floatMax.free();
  ogs::gatherScatterManyKernel_doubleAdd.free();
  ogs::gatherScatterManyKernel_doubleMul.free();
  ogs::gatherScatterManyKernel_doubleMin.free();
  ogs::gatherScatterManyKernel_doubleMax.free();
  ogs::gatherScatterManyKernel_intAdd.free();
  ogs::gatherScatterManyKernel_intMul.free();
  ogs::gatherScatterManyKernel_intMin.free();
  ogs::gatherScatterManyKernel_intMax.free();
  ogs::gatherScatterManyKernel_longAdd.free();
  ogs::gatherScatterManyKernel_longMul.free();
  ogs::gatherScatterManyKernel_longMin.free();
  ogs::gatherScatterManyKernel_longMax.free();

  ogs::gatherKernel_floatAdd.free();
  ogs::gatherKernel_floatMul.free();
  ogs::gatherKernel_floatMin.free();
  ogs::gatherKernel_floatMax.free();
  ogs::gatherKernel_doubleAdd.free();
  ogs::gatherKernel_doubleMul.free();
  ogs::gatherKernel_doubleMin.free();
  ogs::gatherKernel_doubleMax.free();
  ogs::gatherKernel_intAdd.free();
  ogs::gatherKernel_intMul.free();
  ogs::gatherKernel_intMin.free();
  ogs::gatherKernel_intMax.free();
  ogs::gatherKernel_longAdd.free();
  ogs::gatherKernel_longMul.free();
  ogs::gatherKernel_longMin.free();
  ogs::gatherKernel_longMax.free();
  ogs::gatherVecKernel_floatAdd.free();
  ogs::gatherVecKernel_floatMul.free();
  ogs::gatherVecKernel_floatMin.free();
  ogs::gatherVecKernel_floatMax.free();
  ogs::gatherVecKernel_doubleAdd.free();
  ogs::gatherVecKernel_doubleMul.free();
  ogs::gatherVecKernel_doubleMin.free();
  ogs::gatherVecKernel_doubleMax.free();
  ogs::gatherVecKernel_intAdd.free();
  ogs::gatherVecKernel_intMul.free();
  ogs::gatherVecKernel_intMin.free();
  ogs::gatherVecKernel_intMax.free();
  ogs::gatherVecKernel_longAdd.free();
  ogs::gatherVecKernel_longMul.free();
  ogs::gatherVecKernel_longMin.free();
  ogs::gatherVecKernel_longMax.free();
  ogs::gatherManyKernel_floatAdd.free();
  ogs::gatherManyKernel_floatMul.free();
  ogs::gatherManyKernel_floatMin.free();
  ogs::gatherManyKernel_floatMax.free();
  ogs::gatherManyKernel_doubleAdd.free();
  ogs::gatherManyKernel_doubleMul.free();
  ogs::gatherManyKernel_doubleMin.free();
  ogs::gatherManyKernel_doubleMax.free();
  ogs::gatherManyKernel_intAdd.free();
  ogs::gatherManyKernel_intMul.free();
  ogs::gatherManyKernel_intMin.free();
  ogs::gatherManyKernel_intMax.free();
  ogs::gatherManyKernel_longAdd.free();
  ogs::gatherManyKernel_longMul.free();
  ogs::gatherManyKernel_longMin.free();
  ogs::gatherManyKernel_longMax.free();

  ogs::scatterKernel_float.free();
  ogs::scatterKernel_double.free();
  ogs::scatterKernel_int.free();
  ogs::scatterKernel_long.free();
  ogs::scatterVecKernel_float.free();
  ogs::scatterVecKernel_double.free();
  ogs::scatterVecKernel_int.free();
  ogs::scatterVecKernel_long.free();
  ogs::scatterManyKernel_float.free();
  ogs::scatterManyKernel_double.free();
  ogs::scatterManyKernel_int.free();
  ogs::scatterManyKernel_long.free();

  ogs::o_haloBuf.free();
  ogs::haloBuf = NULL;
}
