#include "platform.hpp"
#include "mesh.h"

static const std::string section = "core-";
static const std::string suffix = "Hex3D";

namespace opSEM
{

occa::memory grad(mesh_t *mesh, const occa::memory &o_in, dlong offset)
{
  occa::memory o_out = platform->o_memPool.reserve<dfloat>(mesh->dim * offset);
  static occa::kernel kernel;
  if (!kernel.isInitialized()) {
    kernel = platform->kernelRequests.load(section + "wgradientVolume" + suffix);
  }
  kernel(mesh->Nelements, mesh->o_vgeo, mesh->o_D, offset, o_in, o_out);

  return o_out;
}

occa::memory strongGrad(mesh_t *mesh, const occa::memory &o_in, dlong offset)
{
  occa::memory o_out = platform->o_memPool.reserve<dfloat>(mesh->dim * offset);
  static occa::kernel kernel;
  if (!kernel.isInitialized()) {
    kernel = platform->kernelRequests.load(section + "gradientVolume" + suffix);
  }
  kernel(mesh->Nelements, mesh->o_vgeo, mesh->o_D, o_in, o_out);

  return o_out;
}

occa::memory divergence(mesh_t *mesh, const occa::memory &o_in, dlong offset)
{
  occa::memory o_out = platform->o_memPool.reserve<dfloat>(offset);
  static occa::kernel kernel;
  if (!kernel.isInitialized()) {
    kernel = platform->kernelRequests.load(section + "wDivergenceVolume" + suffix);
  }
  kernel(mesh->Nelements, mesh->o_vgeo, mesh->o_D, offset, o_in, o_out);

  return o_out;
}

occa::memory strongDivergence(mesh_t *mesh, const occa::memory &o_in, dlong offset)
{
  occa::memory o_out = platform->o_memPool.reserve<dfloat>(offset);
  static occa::kernel kernel;
  if (!kernel.isInitialized()) {
    kernel = platform->kernelRequests.load(section + "divergenceVolume" + suffix);
  }
  kernel(mesh->Nelements, mesh->o_vgeo, mesh->o_D, offset, o_in, o_out);

  return o_out;
}

occa::memory laplacian(mesh_t *mesh, const occa::memory &o_lambda, const occa::memory &o_in, dlong offset)
{
  occa::memory o_out = platform->o_memPool.reserve<dfloat>(offset);
  static occa::kernel kernel;
  if (!kernel.isInitialized()) {
    kernel = platform->kernelRequests.load(section + "weakLaplacian" + suffix);
  }
  kernel(mesh->Nelements, 1, 0, mesh->o_ggeo, mesh->o_D, o_lambda, o_in, o_out);

  return o_out;
}

occa::memory
strongLaplacian(mesh_t *mesh, const occa::memory &o_lambda, const occa::memory &o_in, dlong offset)
{
  auto o_tmp = strongGrad(mesh, o_in, offset);
  oogs::startFinish(o_tmp, mesh->dim, offset, ogsDfloat, ogsAdd, mesh->oogs);
  return strongDivergence(mesh, o_tmp, offset);
}

} // namespace opSEM
