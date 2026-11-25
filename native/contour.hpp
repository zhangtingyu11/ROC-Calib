#pragma once
#include "objective.hpp"
#include "tail_penalty.hpp"
inline void prepare_contour(Pair& p){
 std::map<std::pair<int,int>,std::vector<int>> lines;
 auto has=[&](int x,int y){return x>=0&&y>=0&&x<p.mask.cols&&y<p.mask.rows&&p.mask.at<uint8_t>(y,x)>0;};
 for(int y=0;y<p.mask.rows;++y)for(int x=0;x<p.mask.cols;++x)if(has(x,y)){
  if(!has(x,y-1))lines[{0,2*y-1}].push_back(x);if(!has(x,y+1))lines[{0,2*y+1}].push_back(x);
  if(!has(x-1,y))lines[{1,2*x-1}].push_back(y);if(!has(x+1,y))lines[{1,2*x+1}].push_back(y);
 }
 p.contour.clear();for(auto& kv:lines){auto& a=kv.second;std::sort(a.begin(),a.end());
  for(size_t i=0;i<a.size();){int start=a[i],end=start;while(++i<a.size()&&a[i]==end+1)end=a[i];double c=.5*kv.first.second;
   if(kv.first.first==0)p.contour.push_back({start-.5,c,end+.5,c});else p.contour.push_back({c,start-.5,c,end+.5});
  }
 }
 if(p.contour.empty())throw std::runtime_error("Missing original contour");
}
inline std::vector<size_t> support_hull(const std::vector<V2>& uv){
 std::vector<size_t> ids(uv.size());for(size_t i=0;i<ids.size();++i)ids[i]=i;
 std::sort(ids.begin(),ids.end(),[&](size_t a,size_t b){return uv[a].x()!=uv[b].x()?uv[a].x()<uv[b].x():uv[a].y()!=uv[b].y()?uv[a].y()<uv[b].y():a<b;});
 ids.erase(std::unique(ids.begin(),ids.end(),[&](size_t a,size_t b){return uv[a]==uv[b];}),ids.end());if(ids.size()<3)return ids;
 auto cross=[&](size_t a,size_t b,size_t c){V2 u=uv[b]-uv[a],v=uv[c]-uv[a];return u.x()*v.y()-u.y()*v.x();};
 std::vector<size_t> h;
 for(auto i:ids){while(h.size()>1&&cross(h[h.size()-2],h.back(),i)<=0)h.pop_back();h.push_back(i);}
 size_t lower=h.size();for(size_t k=ids.size()-1;k-->0;){auto i=ids[k];while(h.size()>lower&&cross(h[h.size()-2],h.back(),i)<=0)h.pop_back();h.push_back(i);}
 h.pop_back();return h;
}
inline std::pair<double,V2> contour_cost(const Pair& p,V2 u,bool useInterpolated=false){
 double best=INFINITY;V2 residual=V2::Zero();
 for(auto s:(useInterpolated?p.region_edges:p.contour)){V2 a(s[0],s[1]),v(s[2]-s[0],s[3]-s[1]);double t=std::clamp((u-a).dot(v)/v.squaredNorm(),0.,1.);V2 e=u-a-t*v;
  if(e.squaredNorm()<best){best=e.squaredNorm();residual=e;}}
 return {std::log1p(best),2*residual/(1+best)};
}
inline double chain_allowance(const std::vector<V2>& uv,const std::vector<V3>& xyz,size_t start,V2 inward){
 if(inward.norm()<1e-12)return NAN;inward.normalize();size_t at=start;double ds[3],ns[3],xs[3];
 for(int k=0;k<3;++k){double best=INFINITY;size_t next=uv.size();for(size_t j=0;j<uv.size();++j){V2 v=uv[j]-uv[at];double n=v.norm();if(n<1e-9||v.dot(inward)<std::sqrt(3.)*.5*n)continue;if(n<best){best=n;next=j;}}if(next==uv.size())return NAN;ds[k]=best;ns[k]=(uv[next]-uv[at]).dot(inward);xs[k]=(xyz[next]-xyz[at]).norm();at=next;}
 auto consistent=[](double* v){auto mm=std::minmax_element(v,v+3);return *mm.first>1e-12&&*mm.second<=2*(*mm.first);};
 if(!consistent(ds)||!consistent(xs))return NAN;std::sort(ns,ns+3);return ns[1];
}
inline void prepare_spacing(const std::vector<Pair>&,const M4&);
inline void prepare_support(const std::vector<Pair>& pairs,const M4& T){for(const auto& p:pairs){auto uv=project(p,T);p.supported_hull=support_hull(uv);p.allowances.clear();V2 center=V2::Zero();for(auto u:uv)center+=u/double(uv.size());for(auto i:p.supported_hull){double best=INFINITY;V2 direction=V2::Zero();for(auto s:p.contour){V2 a(s[0],s[1]),v(s[2]-s[0],s[3]-s[1]);V2 e=uv[i]-a-std::clamp((uv[i]-a).dot(v)/v.squaredNorm(),0.,1.)*v;if(e.squaredNorm()<best){best=e.squaredNorm();direction=V2(-v.y(),v.x()).normalized();}}if(direction.dot(center-uv[i])<0)direction=-direction;p.allowances.push_back(chain_allowance(uv,p.points,i,direction));}}prepare_spacing(pairs,T);}
inline std::pair<double,V2> supported_cost(const Pair& p,V2 u,double allowance){
 if(!std::isfinite(allowance))return {0,V2::Zero()};auto c=contour_cost(p,u);double d=std::sqrt(std::expm1(c.first));if(d<=allowance||d==0)return {0,V2::Zero()};double e=d-allowance;return {std::log1p(e*e),c.second*(e/d)*(1+d*d)/(1+e*e)};
}
inline Evaluation frozen_contour_term(const Pair& p,const std::vector<V2>& uv,const std::vector<Jac>& J,bool score){Evaluation out{0,Vec7::Zero(),0,0};for(size_t k=0;k<p.supported_hull.size();++k){size_t i=p.supported_hull[k];auto c=supported_cost(p,uv[i],p.allowances[k]);out.value+=c.first/p.supported_hull.size();if(score)out.gradient.head<6>()+=J[i].transpose()*c.second/p.supported_hull.size();}return out;}
inline void prepare_spacing(const std::vector<Pair>& pairs,const M4& T){for(const auto&p:pairs){auto uv=project(p,T);V2 center=V2::Zero();for(auto u:uv)center+=u/double(uv.size());p.spacing_chains.clear();p.spacing_normals.clear();for(size_t k=0;k<p.supported_hull.size();++k){size_t i=p.supported_hull[k];std::array<size_t,4> chain{i,i,i,i};V2 n=V2::Zero();if(std::isfinite(p.allowances[k])){double best=INFINITY;for(auto s:p.contour){V2 a(s[0],s[1]),v(s[2]-s[0],s[3]-s[1]);V2 e=uv[i]-a-std::clamp((uv[i]-a).dot(v)/v.squaredNorm(),0.,1.)*v;if(e.squaredNorm()<best){best=e.squaredNorm();n=V2(-v.y(),v.x()).normalized();}}if(n.dot(center-uv[i])<0)n=-n;for(int j=0;j<3;++j){double nearest=INFINITY;size_t next=uv.size();for(size_t z=0;z<uv.size();++z){V2 v=uv[z]-uv[chain[j]];double length=v.norm();if(length<1e-9||v.dot(n)<std::sqrt(3.)*.5*length)continue;if(length<nearest){nearest=length;next=z;}}if(next==uv.size())throw std::runtime_error("frozen chain lost");chain[j+1]=next;}std::array<double,3> gaps;for(int j=0;j<3;++j)gaps[j]=std::abs(n.dot(uv[chain[j+1]]-uv[chain[j]]));std::sort(gaps.begin(),gaps.end());if(std::abs(gaps[1]-p.allowances[k])>1e-12)throw std::runtime_error("coarse allowance mismatch");}p.spacing_chains.push_back(chain);p.spacing_normals.push_back(n);}}}
inline std::pair<double,Eigen::Matrix<double,6,1>> projected_allowance(const Pair&p,size_t k,const std::vector<V2>&uv,const std::vector<Jac>&J,bool score){auto chain=p.spacing_chains[k];V2 n=p.spacing_normals[k];std::array<std::pair<double,int>,3> gaps;for(int j=0;j<3;++j)gaps[j]={std::abs(n.dot(uv[chain[j+1]]-uv[chain[j]])),j};std::sort(gaps.begin(),gaps.end());int j=gaps[1].second;Eigen::Matrix<double,6,1> g=Eigen::Matrix<double,6,1>::Zero();if(score){double s=n.dot(uv[chain[j+1]]-uv[chain[j]]);g=(J[chain[j+1]]-J[chain[j]]).transpose()*n*(s>0?1:s<0?-1:0);}return {gaps[1].first,g};}
inline Evaluation contour_term(const Pair&p,const std::vector<V2>&uv,const std::vector<Jac>&J,bool score){Evaluation out{0,Vec7::Zero(),0,0};for(size_t k=0;k<p.supported_hull.size();++k){if(!std::isfinite(p.allowances[k]))continue;size_t i=p.supported_hull[k];auto a=projected_allowance(p,k,uv,J,score);auto c=contour_cost(p,uv[i],true);double d=std::sqrt(std::expm1(c.first));if(d<=a.first||d==0)continue;double e=d-a.first;auto rho=tail_penalty(e);out.value+=rho.first/p.supported_hull.size();if(score)out.gradient.head<6>()+=(J[i].transpose()*c.second*(e/d)*(1+d*d)-2*e*a.second)*rho.second/((1+e*e)*p.supported_hull.size());}return out;}
inline Evaluation pixel_contour_term(const Pair&p,const std::vector<V2>&uv,const std::vector<Jac>&J,bool score){Evaluation out{0,Vec7::Zero(),0,0};for(size_t k=0;k<p.supported_hull.size();++k){if(!std::isfinite(p.allowances[k]))continue;size_t i=p.supported_hull[k];auto a=projected_allowance(p,k,uv,J,score);auto c=contour_cost(p,uv[i]);double d=std::sqrt(std::expm1(c.first));if(d<=a.first||d==0)continue;double e=d-a.first;out.value+=std::log1p(e*e)/p.supported_hull.size();if(score)out.gradient.head<6>()+=(J[i].transpose()*c.second*(e/d)*(1+d*d)-2*e*a.second)/((1+e*e)*p.supported_hull.size());}return out;}
