#ifndef AERO_FORCE_HPP_
#define AERO_FORCE_HPP_ 

#include "platform.hpp"

class AeroForce
{
public:
  AeroForce() {};
  AeroForce(std::tuple< std::array<dfloat, 3>, std::array<dfloat, 3> > f) 
  {
    forceV = std::get<0>(f);
    forceP = std::get<1>(f); 
  };

  std::array<dfloat, 3> viscous() const { return forceV; };
  std::array<dfloat, 3> viscousNormal() const { return forceVn; };
  std::array<dfloat, 3> viscousTangential() const { return forceVt; };
  std::array<dfloat, 3> viscousFriction() const { return forceVt; };
  void setViscousForce(std::array<dfloat, 3> f) { forceV = f; }
  void setViscousForceNormal(std::array<dfloat, 3> f) { forceVn = f; }
  void setViscousForceTangential(std::array<dfloat, 3> f) { forceVt = f; }

  std::array<dfloat, 3> pressure() const { return forceP; };
  void setPressureForce(std::array<dfloat, 3> f) { forceP = f; }

  std::array<dfloat, 3> total() const
  { 
     return {forceV[0] + forceP[0], forceV[1] + forceP[1], forceV[2] + forceP[2]}; 
  };

  void p(const occa::memory& o_in) { o_P = o_in; }
  occa::memory p() { return o_P; }

private:
  occa::memory o_P;

  std::array<dfloat, 3> forceVn;
  std::array<dfloat, 3> forceVt;
  std::array<dfloat, 3> forceV;
  std::array<dfloat, 3> forceP;
};

#endif
