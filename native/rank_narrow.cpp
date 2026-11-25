#include <cstdint>
#include <cmath>
#include <algorithm>
#include <limits>
#include <vector>
#include <boost/math/special_functions/beta.hpp>
#include <boost/math/distributions/binomial.hpp>
struct Kernels {};
static double cdf(const Kernels& f,int m,int k,double t,bool complement){
 if(k<0)return complement?1:0;if(k>=m)return complement?0:1;
 return complement?boost::math::ibeta(double(k+1),double(m-k),t):boost::math::ibetac(double(k+1),double(m-k),t);
}
static double probability(const Kernels& f,int n,int a,int b,double t){
 double lo=cdf(f,n-1,a-1,t,false),hi=cdf(f,n-1,b-1,t,false);
 double slo=cdf(f,n-1,a-1,t,true),shi=cdf(f,n-1,b-1,t,true);
 return lo+hi>1?slo-shi:hi-lo;
}
static double pmf(const Kernels& f,int m,int k,double t){
 if(k<0||k>m)return 0;return std::clamp(boost::math::pdf(boost::math::binomial_distribution<double>(m,t),k),0.,1.);
}
static double derivative(const Kernels& f,int n,int a,int b,double t,int order,int stride,const double* coefficients,const double* factors){
 if(order==0)return probability(f,n,a,b,t);int m=n-1;if(order>m)return 0;
 double v=0;for(int j=0;j<order;++j){double d=pmf(f,m-order,a-1-j,t)-pmf(f,m-order,b-1-j,t);v+=coefficients[order*stride+j]*d;}
 return v*factors[order];
}
extern "C" int rank_narrow(int n,int64_t count,const int64_t* a,const int64_t* b,const double* c,const double* q,int maxJ,int stride,const double* coefficients,const double* factors,const double* denoms,double* out,int64_t* terms){
 if(n<1||count<0||maxJ<0||stride<2*maxJ+2)return 1;
 Kernels f;int m=n-1;const double eps=8*std::numeric_limits<double>::epsilon();
 std::vector<double> rowA(stride),rowB(stride),derivatives(stride);
 for(int64_t i=0;i<count;++i){
  if(a[i]<0||b[i]>n||a[i]>=b[i]||q[i]<=0||n*q[i]>1||c[i]<0||c[i]>1)return 2;
  double w=double(b[i]-a[i])/n;int localJ=0;
  for(;localJ<=maxJ;++localJ){
   if(2*localJ+2>m)break;
   int k=localJ+1;double ss=m*q[i],ratio=ss*ss/((2*k+2.)*(2*k+3.)),den=denoms[k];
   double bv=.5*std::pow(ss,2*k)/den/(1-ratio),bc=m*std::pow(ss,2*k)/den/(1-ratio),bq=k*double(m)*std::pow(ss,2*k-1)/den/(1-2*ratio);
   if(bv/w<=eps&&(.5*bc+bq)/w<=eps)break;
  }
  if(localJ>maxJ)return 5;
  std::fill(derivatives.begin(),derivatives.end(),0.);int top=std::min(m,2*localJ+1);
  for(int h=0;h<top;++h)rowA[h]=pmf(f,m-top,a[i]-1-h,c[i]);
  int span=b[i]-a[i];
  for(int h=0;h<top;++h)rowB[h]=h>=span?rowA[h-span]:pmf(f,m-top,b[i]-1-h,c[i]);
  for(int order=top;order>=1;--order){
   double sum=0;for(int h=0;h<order;++h)sum+=coefficients[order*stride+h]*(rowA[h]-rowB[h]);
   derivatives[order]=sum*factors[order];
   for(int h=0;h<order-1;++h){rowA[h]=(1-c[i])*rowA[h]+c[i]*rowA[h+1];rowB[h]=(1-c[i])*rowB[h]+c[i]*rowB[h+1];}
  }
  double v=probability(f,n,a[i],b[i],c[i]);double dc0=derivatives[1],dq=0;bool done=false;
  for(int j=0;j<=localJ;++j){
   if(j){double even=derivatives[2*j];double weight=std::pow(q[i],2*j)/(std::pow(2.,2*j)*denoms[j]);v+=even*weight;dc0+=derivatives[2*j+1]*weight;dq+=even*(2*j)*weight/q[i];}
   terms[i]=j+1;
   if(2*j+2>m){done=true;break;}
   int k=j+1;double s=m*q[i],ratio=s*s/((2*k+2.)*(2*k+3.));double power=std::pow(s,2*k),den=denoms[k];
   double bv=.5*power/den/(1-ratio),bc=m*power/den/(1-ratio),bq=k*double(m)*std::pow(s,2*k-1)/den/(1-2*ratio);
   double l=.5*dc0-dq,u=.5*dc0+dq;
   if(bv/w<=eps*std::max(1.,std::abs(v/w))&&(.5*bc+bq)/w<=eps*std::max(1.,std::max(std::abs(l/w),std::abs(u/w)))){done=true;break;}
  }
  if(!done)return 3;out[3*i]=v;out[3*i+1]=.5*dc0-dq;out[3*i+2]=.5*dc0+dq;
  if(!std::isfinite(v)||!std::isfinite(out[3*i+1])||!std::isfinite(out[3*i+2]))return 4;
 }
 return 0;
}
