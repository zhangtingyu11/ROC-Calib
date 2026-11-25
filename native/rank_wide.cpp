#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
static std::vector<long double> pmf(int n,long double t){
 std::vector<long double> p(n+1,0.L);
 if(t==0){p[0]=1;return p;}if(t==1){p[n]=1;return p;}
 int mode=std::min(n,(int)std::floor((n+1)*t));p[mode]=1;
 for(int k=mode;k<n;++k)p[k+1]=p[k]*((long double)(n-k)/(k+1))*(t/(1-t));
 for(int k=mode;k>0;--k)p[k-1]=p[k]*((long double)k/(n-k+1))*((1-t)/t);
 long double sum=0;for(auto v:p)sum+=v;const long double reciprocal=1.L/sum;for(auto &v:p)v*=reciprocal;return p;
}
#include <map>
#include <utility>
#include <cstdint>
#include <cstring>
struct Window {std::vector<long double> interval,left_density,right_density;};
#include <atomic>
#include <omp.h>
extern "C" int rank_window_workers(){int count=0;
 #pragma omp parallel num_threads(4)
 {
  #pragma omp single
  count=omp_get_num_threads();
 }
 return count;
}
extern "C" int rank_wide_parallel_window(int n,long count,const long long* a,const long long* b,const double* c,const double* q,double* out){
 try{
  if(std::numeric_limits<long double>::digits<=53)return 2;
  std::map<std::pair<uint64_t,uint64_t>,size_t> cache;
  std::vector<size_t> indexes(count,std::numeric_limits<size_t>::max());
  std::vector<long double> left,right;
  for(long i=0;i<count;++i){
   long double C=c[i],Q=q[i],L=C-Q/2,U=C+Q/2;
   const long double eps=8*std::numeric_limits<double>::epsilon();
   if(L < -eps||U > 1+eps||Q<=0||a[i]<0||b[i]>n||a[i]>=b[i])return 3;
   L=std::max(0.L,L);U=std::min(1.L,U);
   if(a[i]==0&&b[i]==n)continue;
   uint64_t ck,qk;std::memcpy(&ck,c+i,sizeof(ck));std::memcpy(&qk,q+i,sizeof(qk));auto key=std::make_pair(ck,qk);
   auto found=cache.find(key);
   if(found==cache.end()){
    size_t id=left.size();left.push_back(L);right.push_back(U);found=cache.emplace(key,id).first;
   }
   indexes[i]=found->second;
  }
  std::vector<Window> windows(left.size());std::atomic<int> failure(0);
  #pragma omp parallel for num_threads(4) schedule(static)
  for(long id=0;id<(long)left.size();++id){
   try{
    long double L=left[id],U=right[id];
  auto pl=pmf(n,L),pu=pmf(n,U);
  Window w;w.interval.resize(n+1);w.left_density=pmf(n-1,L);w.right_density=pmf(n-1,U);
  std::vector<long double> cl(n+1),cu(n+1);cl[0]=pl[0];cu[0]=pu[0];
  for(int k=1;k<=n;++k){cl[k]=cl[k-1]+pl[k];cu[k]=cu[k-1]+pu[k];}
  long double tl=0,tu=0;
  for(int k=n;k>0;--k){tl+=pl[k];tu+=pu[k];w.interval[k]=(tu+tl>1 ? cl[k-1]-cu[k-1] : tu-tl);}

    windows[id]=std::move(w);
   }catch(...){failure.store(1);}
  }
  if(failure.load())return 1;
  for(long i=0;i<count;++i){
   if(indexes[i]==std::numeric_limits<size_t>::max()){out[3*i]=1;out[3*i+1]=out[3*i+2]=0;;continue;}
   const auto &w=windows[indexes[i]];long double Q=q[i];long double mass=0,pleft=0,pright=0;
   for(int k=b[i];k>a[i];--k)mass+=w.interval[k];
   for(int k=a[i];k<b[i];++k){pleft+=w.left_density[k];pright+=w.right_density[k];}
   long double v=mass/n/Q;
   out[3*i]=(double)v;out[3*i+1]=(double)((v-pleft)/Q);out[3*i+2]=(double)((pright-v)/Q);

  }
  return 0;
 }catch(...){return 1;}
}
