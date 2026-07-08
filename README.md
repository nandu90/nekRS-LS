```
███    ██ ███████ ██   ██ ██████  ███████       ██      ███████
████   ██ ██      ██  ██  ██   ██ ██            ██      ██
██ ██  ██ █████   █████   ██████  ███████ █████ ██      ███████
██  ██ ██ ██      ██  ██  ██   ██      ██       ██           ██
██   ████ ███████ ██   ██ ██   ██ ███████       ███████ ███████
(c) 2019-2026 UCHICAGO ARGONNE, LLC
```

[![Build Status](https://travis-ci.com/Nek5000/nekRS.svg?branch=master)](https://travis-ci.com/Nek5000/nekRS)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-orange.svg)](https://opensource.org/licenses/BSD-3-Clause)

**NekRS-LS** is a fork of the offical [NekRS](https://github.com/Nek5000/nekRS) repo based on [v26](https://github.com/Nek5000/nekRS/releases/tag/v26.0). This repository contains the two-phase solver plugin to NekRS based on the conservative level-set method.

The two-phase examples are hosted [here](https://github.com/nandu90/nrsLS_Examples).
Example simulation videos are hosted [here](https://nandu90.github.io/).

This fork is under active development and subject to changes.
We try hard not to break userland but the code is evolving quickly so things might change from one version to another without being backward compatible. Please consult `RELEASE.md` *before* using the code.  

Capabilities:

* Incompressible two-phase solver
* Level-set method
* Incompressible and low Mach-number Navier-Stokes + scalar transport 
* High-order curvilinear conformal Hex spectral elements in space 
* Variable time step 2nd/3rd order semi-implicit time integration
* MPI + [OCCA](https://github.com/libocca/occa) supporting CUDA, HIP, DPC++, SERIAL (C++)
* LES and RANS turbulence models
* Arbitrary-Lagrangian-Eulerian moving mesh
* Lagrangian phase model
* Overlapping overset grids
* Conjugate fluid-solid heat transfer
* Various boundary conditions
* VisIt & Paraview for data analysis and visualization including in-situ support through Ascent
* Legacy interface

## Build Instructions

Requirements:
* Linux, Mac OS X (Microsoft WSL and Windows is not supported) 
* GNU/oneAPI/NVHPC/ROCm compilers (C++17/C99 compatible)
* MPI-3.1 or later
* CMake version 3.21 or later 

Clone our GitHub repository:

```sh
https://github.com/nandu90/nekRS-LS.git
```
The [master](https://github.com/nandu90/nekRS-LS/tree/master) branch always points to the latest stable release while [nekLS](https://github.com/nandu90/nekRS-LS/tree/nekLS) 
provides an early preview of the next upcoming release (do not use in a production environment).

#
If you're on an HPC system, ensure you log in to a compute node. You can find installation instructions and job submission scripts for common HPC systems [here](https://github.com/Nek5000/nekRS_HPCsupport).
Now, just run:

```sh
CC=mpicc CXX=mpic++ FC=mpif77 ./build.sh [-DCMAKE_INSTALL_PREFIX=$HOME/.local/nekrs] [<options>]
```
Adjust the compilers as necessary. Make sure to remove the previous build and installation directory if updating.

## Setting the Environment

Assuming you run `bash` and your install directory is $HOME/.local/nekrs, 
add the following line to your $HOME/.bash_profile:

```sh
export NEKRS_HOME=$HOME/.local/nekrs
export PATH=$NEKRS_HOME/bin:$PATH
```
then type `source $HOME/.bash_profile` in the current terminal window. 

## Documentation 
For documentation, see [readthedocs page](https://nekrs.readthedocs.io/en/latest/). 
The manual pages for the `par` file and environment variables `env` can be accessed through `nrsman`
The manual page for two-phase specific parameters in the `par` file can be accessed through `nlsman par`

## License
nekRS is released under the BSD 3-clause license (see `LICENSE` file). 
All new contributions must be made under the BSD 3-clause license.

## Citing
The two-phase level-set implementation in NekRS is detailed in [A high order continuous Galerkin spectrally stabilized level-set approach for incompressible two-phase flows](https://doi.org/10.1016/j.jcp.2026.114961).
Please cite our work if you find it useful.

## Collaboration
Feel free to reach out for any collaborative multiphase research opportunities at: nsaini.ne@gmail.com.
New capabilites are being actively added to the coded, guided by research needs.
Ideas on enhancing the implementation or code performance are encouraged :)

## Acknowledgment
This research was supported by the Exascale Computing Project (17-SC-20-SC), 
a joint project of the U.S. Department of Energy's Office of Science and National Nuclear Security 
Administration, responsible for delivering a capable exascale ecosystem, including software, 
applications, and hardware technology, to support the nation's exascale computing imperative.
