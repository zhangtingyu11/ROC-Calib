#include "appearance.hpp"
extern "C" {
#define restrict __restrict
#include "vendor/__slsqp.h"
#undef restrict
}
Mixture mixture(const std::vector<Appearance>& pairs){
 std::vector<std::array<double,9>> R;std::vector<double> weights;for(auto& p:pairs)for(auto r:p.ratios){std::array<double,9> row{};row[0]=1;std::copy(r.begin(),r.end(),row.begin()+1);R.push_back(row);weights.push_back(1./(pairs.size()*p.ratios.size()));}
 std::array<double,9> lam{},gradient{};lam[0]=1;double value=0;
 auto evaluate=[&](){value=0;gradient.fill(0);for(size_t i=0;i<R.size();++i){double r=0;for(int j=0;j<9;++j)r+=R[i][j]*lam[j];if(r<=0){value=INFINITY;return;}value-=weights[i]*std::log(r);for(int j=0;j<9;++j)gradient[j]-=weights[i]*R[i][j]/r;}};
 SLSQP_vars state{};state.acc=1e-14;state.tol=1e-13;state.itermax=300;state.m=1;state.meq=1;state.n=9;
 std::array<double,9> C,lower{},upper;C.fill(1);upper.fill(1);double constraint=0;std::vector<double> mult(21),buffer(10000);std::vector<int> indices(100);evaluate();
 for(int calls=0;calls<3000;++calls){__slsqp_body(&state,&value,gradient.data(),C.data(),&constraint,lam.data(),mult.data(),lower.data(),upper.data(),buffer.data(),indices.data());if(state.mode!=1&&state.mode!=-1)break;evaluate();constraint=-1;for(auto v:lam)constraint+=v;}
 double sum=0;for(auto& v:lam){v=std::max(0.,v);sum+=v;}for(auto& v:lam)v/=sum;evaluate();if(!std::isfinite(value)||value>0){lam.fill(0);lam[0]=1;}
 double gap=INFINITY;int iter=0;for(;iter<10000;++iter){evaluate();double dot=0;int enter=0,leave=-1;for(int j=0;j<9;++j){dot+=lam[j]*gradient[j];if(gradient[j]<gradient[enter])enter=j;if(lam[j]>0&&(leave<0||gradient[j]>gradient[leave]))leave=j;}gap=dot-gradient[enter];if(gap<=1e-10)break;if(enter==leave)throw std::runtime_error("Mixture stalled");
 std::vector<double> r(R.size()),dir(R.size());for(size_t i=0;i<R.size();++i){for(int j=0;j<9;++j)r[i]+=R[i][j]*lam[j];dir[i]=R[i][enter]-R[i][leave];}auto derivative=[&](double step)->double{double v=0;for(size_t i=0;i<R.size();++i){double trial=r[i]+step*dir[i];if(trial<=0)return INFINITY;v-=weights[i]*dir[i]/trial;}return v;};double step=lam[leave];if(derivative(step)>0){double lo=0,hi=step;for(int k=0;k<100&&hi-lo>5e-16;++k){double mid=(lo+hi)/2;if(derivative(mid)<=0)lo=mid;else hi=mid;}step=(lo+hi)/2;}if(step<=0)throw std::runtime_error("Mixture zero step");lam[enter]+=step;lam[leave]-=step;}
 if(iter==10000||!std::isfinite(value))throw std::runtime_error("Mixture not certified");Mixture answer;answer.value=value;answer.gap=gap;answer.lambda=lam;size_t at=0;for(auto& p:pairs){std::vector<std::array<double,8>> gradients(p.ratios.size());for(size_t i=0;i<p.ratios.size();++i,++at){double r=0;for(int j=0;j<9;++j)r+=R[at][j]*lam[j];for(int j=0;j<8;++j)gradients[i][j]=-weights[at]*lam[j+1]/r;}answer.gradients.push_back(std::move(gradients));}return answer;
}
