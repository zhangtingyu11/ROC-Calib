#include "boundary_helpers.hpp"
json boundary_refine(const std::vector<Pair>& pairs,M4 T,double b,
 const std::function<void(const M4&,double,double,bool)>& save,
 size_t& calls){
 json history=json::array();std::string status="iteration_limit";
 std::vector<std::vector<Edge>> boundaries;for(const auto& p:pairs)boundaries.push_back(edges(p));
 for(int outer=0;outer<100;++outer){
  auto initial=objective(pairs,T,b,false,true);++calls;
  std::vector<std::vector<bool>> membership;std::vector<std::vector<V2>> baseUV;std::vector<Vec7> initialRows;
  for(size_t k=0;k<pairs.size();++k){const auto& p=pairs[k];auto uv=project(p,T);auto J=jacobian(p,T);std::vector<bool> flags;
   for(size_t i=0;i<uv.size();++i){flags.push_back(inside(p,uv[i]));if(!flags.back())continue;
    for(auto e:boundaries[k]){V2 v=e.b-e.a;double t=std::clamp((uv[i]-e.a).dot(v)/v.squaredNorm(),0.,1.);
     if((uv[i]-e.a-t*v).squaredNorm()>1e-12)continue;
     Vec7 a=Vec7::Zero();a.head<6>()=J[i].transpose()*e.n;if(a.norm()>0)initialRows.push_back(a.normalized());
    }
   }membership.push_back(std::move(flags));baseUV.push_back(std::move(uv));
  }
  Vec7 direction=cone(initialRows,initial.gradient);if(direction.norm()==0){status="zero_cone";break;}direction.normalize();
  double best=initial.value,bestScale=b,bestStep=0;M4 bestT=T;size_t acceptedEntries=0;
  for(double step:std::vector<double>{1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8}){
   Vec7 physical;for(int j=0;j<7;++j)physical[j]=units[j]*step*direction[j];
   M4 trial=delta(physical)*T;double scale=b*std::exp(physical[6]);
   std::vector<Constraint> constraints;bool success=false,qpFailed=false;size_t entries=0;
   for(int iteration=0;iteration<=10;++iteration){
     std::vector<std::vector<V2>> uv;std::vector<std::vector<Jac>> J;size_t exits=0;entries=0;
    for(size_t k=0;k<pairs.size();++k){uv.push_back(project(pairs[k],trial));
     for(size_t i=0;i<uv.back().size();++i){
      bool now=inside(pairs[k],uv[k][i]);if(!membership[k][i]&&now)++entries;
      if(!membership[k][i]||now)continue;
      ++exits;size_t nearest=size_t(-1);double distance=1e300;
      for(size_t e=0;e<boundaries[k].size();++e){auto edge=boundaries[k][e];auto u=baseUV[k][i];
       if(edge.n.dot(u-edge.a)<0)continue;V2 v=edge.b-edge.a;
       double t=std::clamp((u-edge.a).dot(v)/v.squaredNorm(),0.,1.);double d=(u-edge.a-t*v).squaredNorm();
       if(d<distance){distance=d;nearest=e;}
      }
      if(nearest==size_t(-1))throw std::runtime_error("No inside boundary");
      bool present=false;for(auto c:constraints)if(c.pair==k&&c.point==i&&c.edge==nearest)present=true;
      if(!present)constraints.push_back({k,i,nearest});
     }
    }
    if(!exits){success=true;break;}if(iteration==10)break;
    for(const auto& p:pairs)J.push_back(jacobian(p,trial));
    std::vector<Vec7> rows;std::vector<double> bounds;
    for(auto c:constraints){auto e=boundaries[c.pair][c.edge];Vec7 a=Vec7::Zero();a.head<6>()=J[c.pair][c.point].transpose()*e.n;
     double norm=a.norm();if(norm==0){qpFailed=true;break;}rows.push_back(a/norm);
     bounds.push_back((1e-9-e.n.dot(uv[c.pair][c.point]-e.a))/norm);
    }
    Vec7 correction;if(qpFailed||!project_affine(rows,bounds,correction))break;
    for(int j=0;j<7;++j)physical[j]=units[j]*correction[j];trial=delta(physical)*trial;
   }
   if(success){double value=objective(pairs,trial,scale,false,false).value;++calls;
    if(value<best-1e-14){best=value;bestT=trial;bestScale=scale;bestStep=step;acceptedEntries=entries;}
   }
  }
  if(best<initial.value-1e-14){T=bestT;b=bestScale;save(T,b,best,false);
   history.push_back({{"iteration",outer},{"before",initial.value},{"loss",best},{"step",bestStep},{"entries",acceptedEntries},{"exits",0}});
  }else{status="no_descent";break;}
 }
 return {{"stage","boundary_continuation"},{"status",status},{"history",history},{"referenceRead",false},{"intensityRead",false}};
}
