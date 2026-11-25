#include "objective.hpp"
#include <iostream>
#include <iomanip>
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
#include <thread>
static volatile sig_atomic_t supervised_child=0;
static void forward_signal(int signal){if(supervised_child>0)kill(-supervised_child,signal);}
json solve_native(const std::vector<Pair>&,const M4&,const fs::path&,std::chrono::steady_clock::time_point,int);
int main(int argc,char** argv){try{fs::path root=".",state,output;int group=0;bool prepare=false,audit=false;for(int i=1;i<argc;++i){std::string arg=argv[i];if(arg=="--root"&&i+1<argc)root=argv[++i];else if(arg=="--group"&&i+1<argc)group=std::stoi(argv[++i]);else if(arg=="--state"&&i+1<argc)state=argv[++i];else if(arg=="--output"&&i+1<argc)output=argv[++i];else if(arg=="--prepare-only")prepare=true;else if(arg=="--audit")audit=true;else if(arg=="--help"){std::cout<<"autocalib_native --root REPO --group N [--state CHECKPOINT_JSON | --prepare-only] --output NEW_JSON\n";return 0;}else throw std::runtime_error("Unknown/incomplete argument "+arg);}if(group<0||output.empty()||fs::exists(output))throw std::runtime_error("Group and new output file required");auto start=std::chrono::steady_clock::now();
 if(!output.parent_path().empty())fs::create_directories(output.parent_path());
 pid_t child=fork();if(child<0)throw std::runtime_error("Native supervisor fork failed");
 if(child>0){supervised_child=child;signal(SIGTERM,forward_signal);signal(SIGINT,forward_signal);setpgid(child,child);int status=0;while(waitpid(child,&status,WNOHANG)==0){std::this_thread::sleep_for(std::chrono::milliseconds(20));}int exitcode=WIFEXITED(status)?WEXITSTATUS(status):128+WTERMSIG(status);std::ofstream(output.string()+".execution.json")<<json{{"hardBudgetSeconds",nullptr},{"budgetExceeded",false},{"exitCode",exitcode},{"processSeconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()},{"pythonRuntime",false},{"supervisor","native C++ POSIX process group"}}.dump(2)<<"\n";return exitcode;}
 setpgid(0,0);
 auto d=load_data(root,group);size_t points=0;for(auto& p:d.pairs)points+=p.points.size();json result={{"group",group},{"pairs",d.pairs.size()},{"points",points},{"pythonRuntime",false},{"appearanceEnabled",false},{"intensityRead",false},{"sourceSHA256",d.provenance}};
 if(!state.empty()){auto s=read_json(state);M4 T=matrix(s.at("matrix"));double b=s.at("scalePixels");auto tick=std::chrono::steady_clock::now();auto e=objective(d.pairs,T,b,false,true);result["loss"]=e.value;result["geometry"]=e.geometry;result["appearance"]=e.appearance;result["gradient"]=std::vector<double>(e.gradient.data(),e.gradient.data()+7);result["callbackSeconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-tick).count();
 if(audit){result["maxOriginalPixelLogDifference"]=audit_pixels(d.pairs,T,b);const auto& native_pairs=d.pairs;result["auditStates"]=json::array();for(int k=0;k<3;++k){Vec7 d=Vec7::Zero();if(k==1){d[0]=1e-4;d[4]=1e-3;}if(k==2){d[2]=-2e-4;d[3]=-1e-3;}M4 pose=delta(d)*T;double scale=b*(k==2?1.1:1.);json row={{"state",k},{"matrix",as_json(pose)},{"scalePixels",scale}};for(bool coarse:{false,true}){auto scored=objective(native_pairs,pose,scale,coarse,true);auto scalar=objective(native_pairs,pose,scale,coarse,false);row[coarse?"coarse":"fine"]={{"loss",scored.value},{"scalarLoss",scalar.value},{"gradient",std::vector<double>(scored.gradient.data(),scored.gradient.data()+7)}};}result["auditStates"].push_back(row);}}
}
 else{M4 seed=initialize(d.pairs);result["initialMatrix"]=as_json(seed);if(!prepare)result["solve"]=solve_native(d.pairs,seed,output,start,group);}
 verify_sources(d.provenance);
 result["endToEndSeconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();if(!output.parent_path().empty())fs::create_directories(output.parent_path());std::ofstream(output)<<result.dump(2)<<"\n";std::cout<<json{{"group",group},{"points",points},{"seconds",result["endToEndSeconds"]}}.dump()<<std::endl;return 0;
 }catch(std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
