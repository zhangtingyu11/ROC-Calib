#pragma once
#include "vendor/json.hpp"
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include <filesystem>
#include <fstream>
#include <array>
#include <map>
#include <set>
#include <vector>
#include <cmath>
#include <functional>
#include <chrono>
#include <stdexcept>
using json=nlohmann::json;namespace fs=std::filesystem;
using V2=Eigen::Vector2d;using V3=Eigen::Vector3d;using M3=Eigen::Matrix3d;using M4=Eigen::Matrix4d;
using Vec7=Eigen::Matrix<double,7,1>;
inline constexpr double units[7]={.01,.01,.01,.01,.01,.01,.1};
inline json read_json(const fs::path& p){std::ifstream f(p);if(!f)throw std::runtime_error("Cannot read "+p.string());json j;f>>j;return j;}
inline M4 matrix(const json& j){M4 m;if(j.size()==4&&j[0].is_array()){for(int r=0;r<4;++r)for(int c=0;c<4;++c)m(r,c)=j[r][c];}else{for(int r=0;r<4;++r)for(int c=0;c<4;++c)m(r,c)=j[r*4+c];}return m;}
inline json as_json(const M4& m){json j=json::array();for(int r=0;r<4;++r){json row=json::array();for(int c=0;c<4;++c)row.push_back(m(r,c));j.push_back(row);}return j;}
inline M3 skew(V3 v){M3 m;m<<0,-v.z(),v.y(),v.z(),0,-v.x(),-v.y(),v.x(),0;return m;}
inline M4 delta(const Vec7& d){M4 m=M4::Identity();m.block<3,3>(0,0)=(Eigen::AngleAxisd(d[2],V3::UnitZ())*Eigen::AngleAxisd(d[1],V3::UnitY())*Eigen::AngleAxisd(d[0],V3::UnitX())).toRotationMatrix();m.block<3,1>(0,3)=d.segment<3>(3);return m;}
inline double bounded(const std::function<double(double)>& f,double a,double b,double tol=1e-10){
 const double gold=.5*(3-std::sqrt(5.)),seps=std::sqrt(2.2e-16);double xf=a+gold*(b-a),nf=xf,ff=xf,fx=f(xf),fn=fx,fff=fx,e=0,rat=0;int calls=1;
 while(calls<500){double xm=.5*(a+b),tol1=seps*std::abs(xf)+tol/3,tol2=2*tol1;if(std::abs(xf-xm)<=tol2-.5*(b-a))break;bool golden=true;
 if(std::abs(e)>tol1){double r=(xf-nf)*(fx-fff),q=(xf-ff)*(fx-fn),p=(xf-ff)*q-(xf-nf)*r;q=2*(q-r);if(q>0)p=-p;q=std::abs(q);r=e;e=rat;if(std::abs(p)<std::abs(.5*q*r)&&p>q*(a-xf)&&p<q*(b-xf)){rat=p/q;golden=false;double x=xf+rat;if(x-a<tol2||b-x<tol2)rat=std::copysign(tol1,xm-xf);}}
 if(golden){e=xf>=xm?a-xf:b-xf;rat=gold*e;}double x=xf+std::copysign(std::max(std::abs(rat),tol1),rat==0?1:rat),v=f(x);++calls;
 if(v<=fx){if(x>=xf)a=xf;else b=xf;ff=nf;fff=fn;nf=xf;fn=fx;xf=x;fx=v;}else{if(x<xf)a=x;else b=x;if(v<=fn||nf==xf){ff=nf;fff=fn;nf=x;fn=v;}else if(v<=fff||ff==xf||ff==nf){ff=x;fff=v;}}
 }return xf;
}
struct Pair {
 mutable std::vector<size_t> supported_hull;
 mutable std::vector<double> allowances;
 mutable std::vector<std::array<size_t,4>> spacing_chains;
 mutable std::vector<V2> spacing_normals;
 std::vector<std::array<double,4>> contour;
 std::vector<std::array<double,4>> region_edges;
 std::string id;std::vector<V3> points;std::vector<double> intensity,c,w;std::vector<int> a,b;cv::Mat mask,color,gray;std::array<double,256> gray_rank{};M3 K;std::array<double,8> D{};std::string distortion;json temporal;
 std::vector<double> xs,ys;std::vector<long> ix,iy;
};
struct Dataset {std::vector<Pair> pairs;json provenance;};
Dataset load_data(const fs::path& root,int group);
M4 initialize(const std::vector<Pair>& pairs);
std::vector<V2> project(const Pair&,const M4&);
void verify_sources(const json&);
