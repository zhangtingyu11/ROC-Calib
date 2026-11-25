#pragma once
#include "core.hpp"
#include "rank.hpp"
struct Stencil {double p=0,dx=0,dy=0;std::array<int,3> levels{};std::array<double,8> K{};std::array<std::array<double,3>,8> dl{},du{};};
struct Appearance {std::vector<std::array<double,8>> ratios;std::vector<Stencil> stencils;std::array<std::array<long double,256>,3> q{};std::array<long double,3> mass{};};
Appearance appearance(const Pair&,const std::vector<V2>&);
std::vector<V2> appearance_pullback(const Appearance&,const std::vector<std::array<double,8>>&);
struct Mixture {double value,gap;std::array<double,9> lambda;std::vector<std::vector<std::array<double,8>>> gradients;};
Mixture mixture(const std::vector<Appearance>&);
std::pair<double,std::vector<V2>> coarse_appearance(const Pair&,const std::vector<V2>&);
