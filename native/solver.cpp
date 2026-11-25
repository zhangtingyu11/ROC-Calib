#include "objective.hpp"
#include "contour.hpp"
#include "boundary_refine.hpp"
#include <iostream>
#include <cstring>
extern "C" {
#include "vendor/__lbfgsb.h"
#define restrict __restrict
#include "vendor/__nnls.h"
#undef restrict
}
using Clock=std::chrono::steady_clock;
static Vec7 pullback(const Vec7& local,const Vec7& d){Vec7 raw,answer;for(int i=0;i<7;++i)raw[i]=local[i]/units[i];double cy=std::cos(d[1]),sy=std::sin(d[1]),cz=std::cos(d[2]),sz=std::sin(d[2]);M3 J;J<<cz*cy,-sz,0,sz*cy,cz,0,-sy,0,1;answer=raw;answer.head<3>()=J.transpose()*(raw.head<3>()-d.segment<3>(3).cross(raw.segment<3>(3)));for(int i=0;i<7;++i)answer[i]*=units[i];return answer;}
static std::pair<double,double> brent(const std::function<double(double)>& f){
 const double gold=1.618034;double a=0,b=1,fa=f(a),fb=f(b);if(fa<fb){std::swap(a,b);std::swap(fa,fb);}double c=b+gold*(b-a),fc=f(c);
 for(int it=0;fc<fb&&it<1000;++it){double t1=(b-a)*(fb-fc),t2=(b-c)*(fb-fa),v=t2-t1,den=2*(std::abs(v)<1e-21?1e-21:v),w=b-((b-c)*t2-(b-a)*t1)/den,limit=b+110*(c-b),fw;
 if((w-c)*(b-w)>0){fw=f(w);if(fw<fc){a=b;fa=fb;b=w;fb=fw;break;}if(fw>fb){c=w;fc=fw;break;}w=c+gold*(c-b);fw=f(w);}else if((w-limit)*(limit-c)>=0){w=limit;fw=f(w);}else if((w-limit)*(c-w)>0){fw=f(w);if(fw<fc){b=c;fb=fc;c=w;fc=fw;w=c+gold*(c-b);fw=f(w);}}else{w=c+gold*(c-b);fw=f(w);}a=b;fa=fb;b=c;fb=fc;c=w;fc=fw;}
 if(!((fb<fc&&fb<=fa)||(fb<fa&&fb<=fc))){if(fa<fb&&fa<=fc)return {a,fa};if(fc<fb)return {c,fc};return {b,fb};}
 double x=b,w=b,v=b,fx=fb,fw=fb,fv=fb,lo=std::min(a,c),hi=std::max(a,c),dx=0,rat=0;
 for(int it=0;it<500;++it){double tol1=1e-6*std::abs(x)+1e-11,tol2=2*tol1,mid=.5*(lo+hi);if(std::abs(x-mid)<tol2-.5*(hi-lo))break;if(std::abs(dx)<=tol1){dx=x>=mid?lo-x:hi-x;rat=.3819660*dx;}else{double t1=(x-w)*(fx-fv),t2=(x-v)*(fx-fw),p=(x-v)*t2-(x-w)*t1;t2=2*(t2-t1);if(t2>0)p=-p;t2=std::abs(t2);double prev=dx;dx=rat;if(p>t2*(lo-x)&&p<t2*(hi-x)&&std::abs(p)<std::abs(.5*t2*prev)){rat=p/t2;double u=x+rat;if(u-lo<tol2||hi-u<tol2)rat=mid-x>=0?tol1:-tol1;}else{dx=x>=mid?lo-x:hi-x;rat=.3819660*dx;}}
 double u=x+(std::abs(rat)<tol1?(rat>=0?tol1:-tol1):rat),fu=f(u);if(fu>fx){if(u<x)lo=u;else hi=u;if(fu<=fw||w==x){v=w;fv=fw;w=u;fw=fu;}else if(fu<=fv||v==x||v==w){v=u;fv=fu;}}else{if(u>=x)lo=x;else hi=x;v=w;fv=fw;w=x;fw=fx;x=u;fx=fu;}}
 return {x,fx};
}
struct Stage {M4 T;double b,value;Vec7 parameters;bool success;};
json solve_native(const std::vector<Pair>& pairs,const M4& seed,const fs::path& output,Clock::time_point started,int group){
 double finebest=INFINITY;M4 fineT=seed;double fineb=1;size_t calls=0;json stages=json::array();
 auto save=[&](const M4& T,double b,double value,bool coarse){if(!coarse&&value<finebest){finebest=value;fineT=T;fineb=b;fs::path checkpoint=output.string()+".checkpoint.json",temp=checkpoint.string()+".tmp";fs::create_directories(checkpoint.parent_path());std::ofstream(temp)<<json{{"matrix",as_json(T)},{"scalePixels",b},{"loss",value},{"evaluation",calls},{"finalAuditCompleted",false},{"pythonRuntime",false}}.dump(2);fs::rename(temp,checkpoint);}};
 auto stage=[&](M4 center,double b0,bool coarse){
 Stage best{center,b0,INFINITY,Vec7::Zero(),false};std::map<std::array<double,7>,std::pair<double,Vec7>> cache;std::map<std::array<double,7>,double> scalarcache;
 auto state=[&](const Vec7& x){Vec7 d;for(int i=0;i<7;++i)d[i]=x[i]*units[i];M4 T=delta(d)*center;return std::tuple<Vec7,M4,double>(d,T,b0*std::exp(d[6]));};
 auto remember=[&](const Vec7& x,const M4& T,double b,double f){if(f<best.value){best={T,b,f,x,false};save(T,b,f,coarse);}};
 auto score=[&](const Vec7& x){if(!x.allFinite())throw std::runtime_error("Nonfinite solver coordinate");std::array<double,7> key;std::copy(x.data(),x.data()+7,key.begin());if(cache.count(key))return cache.at(key);double f=INFINITY;Vec7 grad=Vec7::Zero();auto [d,T,b]=state(x);try{auto e=objective(pairs,T,b,coarse,true);f=e.value;grad=pullback(e.gradient,d);++calls;remember(x,T,b,f);}catch(const std::runtime_error&){}return cache[key]=std::make_pair(f,grad);};
 auto value=[&](const Vec7& x){if(!x.allFinite())throw std::runtime_error("Nonfinite solver coordinate");std::array<double,7> key;std::copy(x.data(),x.data()+7,key.begin());if(scalarcache.count(key))return scalarcache.at(key);auto [d,T,b]=state(x);double f=INFINITY;try{f=objective(pairs,T,b,coarse,false).value;++calls;remember(x,T,b,f);}catch(const std::runtime_error&){}return scalarcache[key]=f;};
 Vec7 x=Vec7::Zero();auto initial=score(x);if(!std::isfinite(initial.first))throw std::runtime_error("Nonfinite native initial objective");double f=initial.first;Vec7 g=initial.second;std::vector<double> wa(2*10*7+11*100+5*7+8*10),dsave(29),lower(7),upper(7);std::vector<int> iwa(21),nbd(7),lsave(4),isave(44);int task[2]={0,0},ln[2]={0,0},iterations=0;
 while(true){setulb(7,10,x.data(),lower.data(),upper.data(),nbd.data(),&f,g.data(),1e-14/std::numeric_limits<double>::epsilon(),1e-8,wa.data(),iwa.data(),task,lsave.data(),isave.data(),dsave.data(),40,ln);if(task[0]==3){auto fg=score(x);f=fg.first;g=fg.second;}else if(task[0]==1){if(++iterations>=260)break;}else break;}
 std::cout<<json{{"stage",coarse?"coarse_lbfgs":"fine_lbfgs"},{"loss",best.value},{"calls",calls}}.dump()<<std::endl;
 x=best.parameters;f=value(x);Vec7 x1=x;Eigen::Matrix<double,7,7> directions=Eigen::Matrix<double,7,7>::Identity();int pcalls=0;bool converged=false;
 struct MaxCalls{};auto pv=[&](const Vec7& a){if(pcalls++>=3000)throw MaxCalls();return value(a);};auto line=[&](Vec7& point,Vec7& direction){if(direction.isZero(0))return pv(point);auto r=brent([&](double alpha){return pv(point+alpha*direction);});direction*=r.first;point+=direction;return r.second;};
 try{for(int it=0;it<100;++it){double before=f,decrease=0;int ibig=0;for(int j=0;j<7;++j){Vec7 dir=directions.col(j);double prev=f;f=line(x,dir);if(prev-f>decrease){decrease=prev-f;ibig=j;}}if(2*(before-f)<=1e-12*(std::abs(before)+std::abs(f))+1e-20){converged=true;break;}Vec7 dir=x-x1;x1=x;double ext=pv(x+dir);if(before>ext){double t=2*(before+ext-2*f)*std::pow(before-f-decrease,2)-decrease*std::pow(before-ext,2);if(t<0){f=line(x,dir);if(!dir.isZero(0)){directions.col(ibig)=directions.col(6);directions.col(6)=dir;}}}}}catch(MaxCalls&){}
 std::cout<<json{{"stage",coarse?"coarse_powell":"fine_powell"},{"loss",best.value},{"calls",calls}}.dump()<<std::endl;
 // Ablation: omit sampled-gradient continuation in both stages.
 best.success=converged;return best;
 };
 auto initial=objective(pairs,seed,1,false,true);save(seed,1,initial.value,false);bool completed=false;std::string status;{Stage coarse=stage(seed,1,true);stages.push_back({{"stage","coarse"},{"matrix",as_json(coarse.T)},{"scalePixels",coarse.b},{"loss",coarse.value},{"optimizerSuccess",coarse.success}});prepare_support(pairs,coarse.T);finebest=INFINITY;initial=objective(pairs,seed,1,false,true);save(seed,1,initial.value,false);json supports=json::array();for(const auto& p:pairs){json allowances=json::array();for(double a:p.allowances)allowances.push_back(std::isfinite(a)?json(a):json());supports.push_back({{"id",p.id},{"indices",p.supported_hull},{"allowances",allowances}});}std::ofstream(output.string()+".support.json")<<supports.dump(2);Stage fine=stage(coarse.T,coarse.b,false);stages.push_back({{"stage","fine"},{"matrix",as_json(fine.T)},{"scalePixels",fine.b},{"loss",fine.value},{"optimizerSuccess",fine.success}});completed=true;status="finished";}
 if(completed)stages.push_back(boundary_refine(pairs,fineT,fineb,save,calls));
 auto verified=objective(pairs,fineT,fineb,false,true);double scalar=objective(pairs,fineT,fineb,false,false).value;if(std::abs(scalar-verified.value)>1e-12)throw std::runtime_error("Final scalar/score mismatch");double pixel_error=std::max(audit_pixels(pairs,fineT,fineb),audit_pixels(pairs,seed,1.));if(!stages.empty()){M4 coarse_matrix=matrix(stages[0].at("matrix"));double coarse_scale=stages[0].at("scalePixels");pixel_error=std::max(pixel_error,audit_pixels(pairs,coarse_matrix,coarse_scale));}if(std::abs(verified.value-finebest)>1e-10)throw std::runtime_error("Native final objective mismatch");return json{{"matrix",as_json(fineT)},{"scalePixels",fineb},{"loss",verified.value},{"initialLoss",initial.value},{"completed",completed},{"finalAuditCompleted",true},{"maxOriginalPixelLogDifference",pixel_error},{"status",status},{"stages",stages},{"objectiveCalls",calls},{"referenceRead",false},{"pythonRuntime",false}};
}
