#pragma once
#include <vector>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
extern "C" int rank_narrow(int,int64_t,const int64_t*,const int64_t*,const double*,const double*,int,int,const double*,const double*,const double*,double*,int64_t*);
extern "C" int rank_wide_parallel_window(int,long,const long long*,const long long*,const double*,const double*,double*);
struct RankQuery {int64_t a,b;double c,q;};
struct RankKernel {
 int n,j,stride;std::vector<double> co,fac,den;
 explicit RankKernel(int N):n(N){
  const int m=n-1;const double eps=8*std::numeric_limits<double>::epsilon(),s=double(m)/n;
  auto factorial=[](int k){double v=1;for(int i=2;i<=k;++i)v*=i;return v;};
  for(j=0;j<=m/2;++j){if(2*j+2>m)break;int k=j+1;double ratio=s*s/((2*k+2.)*(2*k+3.)),d=factorial(2*k+1);double bv=.5*std::pow(s,2*k)/d/(1-ratio),bc=m*std::pow(s,2*k)/d/(1-ratio),bq=k*m*std::pow(s,2*k-1)/d/(1-2*ratio);if(bv*n<=eps&&(.5*bc+bq)*n<=eps)break;}
  stride=2*j+2;co.assign(stride*stride,0);fac.assign(stride,1);
  for(int order=1;order<std::min(stride,m+1);++order){long double f=1;for(int k=m-order+1;k<=m;++k)f*=k;fac[order]=double(f);long long binom=1;for(int k=0;k<order;++k){co[order*stride+k]=((order-1-k)%2?-1:1)*binom;binom=binom*(order-1-k)/(k+1);}}
  for(int k=0;k<j+2;++k)den.push_back(factorial(2*k+1));
 }
 std::vector<std::array<double,3>> evaluate(const std::vector<RankQuery>& qs)const{
  std::vector<std::array<double,3>> answer(qs.size());
  for(int narrow=0;narrow<2;++narrow){std::vector<int64_t> a,b,ids;std::vector<double> c,q;
   for(size_t i=0;i<qs.size();++i)if((n*qs[i].q<=1)==bool(narrow)){a.push_back(qs[i].a);b.push_back(qs[i].b);c.push_back(qs[i].c);q.push_back(qs[i].q);ids.push_back(i);}
   if(ids.empty())continue;std::vector<double> out(3*ids.size());std::vector<int64_t> terms(ids.size());
   int status=narrow?rank_narrow(n,ids.size(),a.data(),b.data(),c.data(),q.data(),j,stride,co.data(),fac.data(),den.data(),out.data(),terms.data()):rank_wide_parallel_window(n,ids.size(),reinterpret_cast<const long long*>(a.data()),reinterpret_cast<const long long*>(b.data()),c.data(),q.data(),out.data());
   if(status)throw std::runtime_error("native rank kernel status "+std::to_string(status));
   for(size_t i=0;i<ids.size();++i)for(int k=0;k<3;++k)answer[ids[i]][k]=out[3*i+k];
  }return answer;
 }
};
