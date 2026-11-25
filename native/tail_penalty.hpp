#pragma once
#include <cmath>
#include <utility>
// Second entry is the derivative multiplier relative to log(1+e^2).
inline std::pair<double,double> tail_penalty(double e){double s=e*e;if(s<=1)return {std::log1p(s),1};return {std::log(2.)+(s-1)/(s+1),2/(s+1)};}
