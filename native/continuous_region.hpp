#pragma once
#include "contour.hpp"
// d is measured in pixels; the transition scale is fixed at one pixel.
inline std::pair<double,V2> continuous_region_gate(const Pair& p,V2 u){
 bool inside=u.x()>=-.5&&u.x()<p.mask.cols-.5&&u.y()>=-.5&&u.y()<p.mask.rows-.5&&p.mask.at<uint8_t>(int(std::floor(u.y()+.5)),int(std::floor(u.x()+.5)));
 if(inside)return {0,V2::Zero()};auto c=contour_cost(p,u);double inverse=std::exp(-c.first);
 return {-std::expm1(-c.first),inverse*c.second};
}
