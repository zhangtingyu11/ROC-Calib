#include <unistd.h>
#include <iostream>
extern "C" {
#define restrict __restrict
#include "vendor/__nnls.h"
#undef restrict
int cauchy_scores_native(const double*,long,const double*,long,const long*,const long*,long,const double*,long,double,double*);
}
bool inside(const Pair& p,V2 u){return u.x()>=-.5&&u.y()>=-.5&&u.x()<p.mask.cols-.5&&u.y()<p.mask.rows-.5&&p.mask.at<uint8_t>(int(std::floor(u.y()+.5)),int(std::floor(u.x()+.5)));}
struct Edge{V2 a,b,n;};
std::vector<Edge> edges(const Pair& p){
 std::vector<Edge> e;
 auto occupied=[&](int x,int y){return x>=0&&y>=0&&x<p.mask.cols&&y<p.mask.rows&&p.mask.at<uint8_t>(y,x);};
 for(int y=0;y<p.mask.rows;++y)for(int x=0;x<p.mask.cols;++x)if(occupied(x,y)){
  if(!occupied(x-1,y))e.push_back({V2(x-.5,y-.5),V2(x-.5,y+.5),V2(1,0)});
  if(!occupied(x+1,y))e.push_back({V2(x+.5,y-.5),V2(x+.5,y+.5),V2(-1,0)});
  if(!occupied(x,y-1))e.push_back({V2(x-.5,y-.5),V2(x+.5,y-.5),V2(0,1)});
  if(!occupied(x,y+1))e.push_back({V2(x-.5,y+.5),V2(x+.5,y+.5),V2(0,-1)});
 }return e;
}
Vec7 cone(const std::vector<Vec7>& rows,Vec7 g){
 if(rows.empty())return -g;
 int n=rows.size();Eigen::Matrix<double,7,Eigen::Dynamic> A(7,n);
 for(int i=0;i<n;++i)A.col(i)=rows[i];
 auto original=A;Vec7 rhs=g;std::vector<double>x(n),w(n),zz(7);std::vector<int> ids(n);double norm;int info;
 __nnls(7,n,A.data(),rhs.data(),x.data(),w.data(),zz.data(),ids.data(),10000,&norm,&info);
 if(info!=0&&info!=1)throw std::runtime_error("Cone NNLS failed");
 Vec7 d=-g;for(int i=0;i<n;++i)d+=x[i]*original.col(i);
 for(int i=0;i<n;++i)if(rows[i].dot(d)<-1e-9||x[i]<0||std::abs(x[i]*rows[i].dot(d))>1e-9)throw std::runtime_error("Cone KKT failed");
 return d;
}
struct Constraint {size_t pair,point,edge;};
bool project_affine(const std::vector<Vec7>& rows,const std::vector<double>& bounds,Vec7& x){
 x.setZero();std::vector<double> lambda(rows.size(),0);
 for(int sweep=0;sweep<10000;++sweep){
  for(size_t i=0;i<rows.size();++i){
   double next=std::max(0.,lambda[i]+bounds[i]-rows[i].dot(x));
   x+=(next-lambda[i])*rows[i];lambda[i]=next;
  }
  bool valid=true;
  for(size_t i=0;i<rows.size();++i){
   double slack=rows[i].dot(x)-bounds[i];
   if(slack < -1e-11||std::abs(lambda[i]*slack)>1e-11)valid=false;
  }
  if(valid)return x.allFinite();
 }return false;
}
