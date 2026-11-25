#pragma once
#include "appearance.hpp"
using Jac=Eigen::Matrix<double,2,6>;
std::vector<Jac> jacobian(const Pair&,const M4&);
struct Evaluation {double value;Vec7 gradient;double geometry,appearance;double supplemental=0;};
Evaluation objective(const std::vector<Pair>&,const M4&,double,bool coarse,bool score);
double audit_pixels(const std::vector<Pair>&,const M4&,double);
