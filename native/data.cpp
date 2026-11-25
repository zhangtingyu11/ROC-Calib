#include "core.hpp"
#include "contour.hpp"
#include "interpolated_region.hpp"
#include <cstring>
#include <openssl/sha.h>
#include <iomanip>
#include <sstream>
static std::vector<char> bytes(const fs::path& p){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error("Missing "+p.string());size_t n=f.tellg();std::vector<char> b(n);f.seekg(0);if(!f.read(b.data(),n))throw std::runtime_error("Short read "+p.string());return b;}
static std::string sha(const fs::path& p){std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("Missing hash input "+p.string());SHA256_CTX ctx;SHA256_Init(&ctx);std::array<char,1048576> b;while(f){f.read(b.data(),b.size());SHA256_Update(&ctx,b.data(),f.gcount());}unsigned char hash[32];SHA256_Final(hash,&ctx);std::ostringstream s;for(auto v:hash)s<<std::hex<<std::setw(2)<<std::setfill('0')<<int(v);return s.str();}
static fs::path resolve(const fs::path& root,std::string s){if(s.rfind("/data/cache/",0)==0)return root/s.substr(12);if(s.rfind("/workspace/",0)==0)return root/s.substr(11);return fs::path(s).is_absolute()?fs::path(s):root/s;}
static std::string label(const json& x){return x.is_string()?x.get<std::string>():x.dump();}
static const json& find(const json& ar,const char* key,const json& val){for(const auto& x:ar)if(label(x.at(key))==label(val))return x;throw std::runtime_error(std::string("Missing record ")+key+"="+label(val));}
static void rectangles(Pair& p){std::map<std::pair<int,int>,int> active;std::vector<std::array<double,4>> rect;
 for(int y=0;y<=p.mask.rows;++y){std::set<std::pair<int,int>> runs;if(y<p.mask.rows)for(int x=0;x<p.mask.cols;){if(!p.mask.at<uint8_t>(y,x)){++x;continue;}int first=x;while(x<p.mask.cols&&p.mask.at<uint8_t>(y,x))++x;runs.emplace(first,x);}
 for(auto it=active.begin();it!=active.end();){if(!runs.count(it->first)){rect.push_back({it->first.first-.5,it->first.second-.5,it->second-.5,y-.5});it=active.erase(it);}else ++it;}
 for(auto interval:runs)if(!active.count(interval))active[interval]=y;}
 std::set<std::array<double,2>> X,Y;for(auto r:rect){X.insert({r[0],r[1]});Y.insert({r[2],r[3]});}std::map<std::array<double,2>,long> xi,yi;for(auto x:X){xi[x]=xi.size();p.xs.insert(p.xs.end(),x.begin(),x.end());}for(auto y:Y){yi[y]=yi.size();p.ys.insert(p.ys.end(),y.begin(),y.end());}for(auto r:rect){p.ix.push_back(xi[{r[0],r[1]}]);p.iy.push_back(yi[{r[2],r[3]}]);}
}
Dataset load_data(const fs::path& root,int group){
 auto base=root;auto req=read_json(base/"request.json");Dataset out;
 const auto mode=req.at("initialization_mode").get<std::string>();
 if(mode=="existing"){
   const auto& values=req.at("base_matrix");
   if(!values.is_array()||values.size()!=16)throw std::runtime_error("Expected a row-major 4x4 initial transform");
   M4 seed=matrix(values);
   M3 R=seed.block<3,3>(0,0);
   if(!seed.allFinite()||!seed.row(3).isApprox(Eigen::RowVector4d(0,0,0,1),1e-6)||
      !(R.transpose()*R).isApprox(M3::Identity(),1e-5)||std::abs(R.determinant()-1)>1e-5)
     throw std::runtime_error("Initial transform must be a finite rigid camera-from-LiDAR matrix");
   out.supplied_seed=seed;
 }else if(mode!="forward")throw std::runtime_error("Unknown initialization mode");
 json expected=fs::exists(root/"checksums.json")?read_json(root/"checksums.json"):json::object();std::map<std::string,std::string> hashed;
 auto check=[&](const fs::path& p){auto key=p.string();if(!hashed.count(key))hashed[key]=sha(p);auto relative=fs::relative(p,root).generic_string();if(expected.contains(relative)&&hashed[key]!=expected.at(relative).get<std::string>())throw std::runtime_error("Source hash mismatch "+key);};check(base/"request.json");
 for(const auto& q:req.at("pairs")){
 Pair p;p.id=q.at("annotation_id");auto cloud=resolve(root,q.at("cloud_url"));check(cloud);auto blob=bytes(cloud);if(blob.size()<16||std::memcmp(blob.data(),"ACP1",4))throw std::runtime_error("ACP1 expected");uint32_t count;std::memcpy(&count,blob.data()+4,4);if(blob.size()!=16+16*size_t(count))throw std::runtime_error("ACP1 size mismatch");std::vector<V3> xyz(count);for(size_t i=0;i<count;++i){float f[3];std::memcpy(f,blob.data()+16+16*i,12);xyz[i]=V3(f[0],f[1],f[2]);}

 for(auto& idx:q.at("point_indices")){size_t i=idx.get<size_t>();if(i>=count||uint8_t(blob[16+16*i+12])==2||!xyz[i].allFinite())throw std::runtime_error("Invalid selected record");p.points.push_back(xyz[i]);}
 p.mask=cv::Mat::zeros(q.at("mask_height").get<int>(),q.at("mask_width").get<int>(),CV_8U);size_t at=0;bool value=false;for(auto& run:q.at("mask_rle")){size_t end=std::min(p.mask.total(),at+size_t(std::max(0,run.get<int>())));if(value)std::fill(p.mask.data+at,p.mask.data+end,1);at=end;value=!value;}if(!cv::countNonZero(p.mask))throw std::runtime_error("Empty mask");rectangles(p);prepare_contour(p);prepare_interpolated_region(p);
 for(int r=0;r<3;++r)for(int c=0;c<3;++c)p.K(r,c)=q.at("intrinsic")[3*r+c];for(size_t i=0;i<std::min(size_t(8),q.at("distortion").size());++i)p.D[i]=q.at("distortion")[i];p.distortion=q.at("distortion_model");p.temporal=q.value("temporal_projection",json());
 out.pairs.push_back(std::move(p));
 }
 out.provenance=hashed;return out;
}
void verify_sources(const json& hashes){for(auto it=hashes.begin();it!=hashes.end();++it)if(sha(it.key())!=it.value().get<std::string>())throw std::runtime_error("Input changed during native run: "+it.key());}
