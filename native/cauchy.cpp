#include <cmath>
#include <algorithm>
#include <vector>
static constexpr double PI=3.141592653589793238462643383279502884;
struct Score { double lp,du,db; };
static Score interval(double left,double right,double u,double b,double logb,double logw){
    double a=left-u,c=right-u,w=right-left;
    double h=std::max(std::max(std::abs(a),std::abs(c)),b);
    double aa=a/h,cc=c/h,ss=b/h;
    double den=aa*cc+ss*ss;
    double lognum=logb+logw-2*std::log(h);
    double ratio=den>0 ? lognum-std::log(den) : 0.;
    double lp=den>0&&ratio<-18. ? ratio-std::log(PI) : std::log(std::atan2(std::exp(lognum),den))-std::log(PI);
    double da=aa*aa+ss*ss,dc=cc*cc+ss*ss;
    double factor=std::exp(lognum-std::log(da)-std::log(dc)-lp-std::log(PI));
    double du=factor*(aa+cc)/h,db=factor*(aa*cc-ss*ss);
    if(!(da>0)||!(dc>0)||!std::isfinite(factor)||!std::isfinite(du)||!std::isfinite(db)){
        long double al=(long double)left-u,cl=(long double)right-u,bl=b;
        long double hl=std::max(std::max(std::abs(al),std::abs(cl)),bl);
        long double ax=al/hl,cx=cl/hl,bx=bl/hl;
        long double lf=std::log(bl)+std::log((long double)right-left)-2*std::log(hl)
            -std::log(ax*ax+bx*bx)-std::log(cx*cx+bx*bx)-(long double)lp-std::log((long double)PI);
        long double f=std::exp(lf);
        du=(double)(f*(ax+cx)/hl);db=(double)(f*(ax*cx-bx*bx));
    }
    return {lp,du,db};
}
extern "C" int cauchy_scores_native(const double* xs,long nx,const double* ys,long ny,
        const long* ix,const long* iy,long nr,const double* uv,long np,double b,double* out){
    if(!(b>0)||!std::isfinite(b)||nx<=0||ny<=0||nr<=0)return 1;
    for(long k=0;k<nx;k++)if(!(xs[2*k+1]>xs[2*k])||!std::isfinite(xs[2*k])||!std::isfinite(xs[2*k+1]))return 2;
    for(long k=0;k<ny;k++)if(!(ys[2*k+1]>ys[2*k])||!std::isfinite(ys[2*k])||!std::isfinite(ys[2*k+1]))return 2;
    for(long j=0;j<nr;j++)if(ix[j]<0||ix[j]>=nx||iy[j]<0||iy[j]>=ny)return 2;
    double logb=std::log(b);
    std::vector<double> logx(nx),logy(ny);
    for(long k=0;k<nx;k++)logx[k]=std::log(xs[2*k+1]-xs[2*k]);
    for(long k=0;k<ny;k++)logy[k]=std::log(ys[2*k+1]-ys[2*k]);
    std::vector<Score> x(nx),y(ny);
    for(long i=0;i<np;i++){
        if(!std::isfinite(uv[2*i])||!std::isfinite(uv[2*i+1]))return 3;
        for(long k=0;k<nx;k++)x[k]=interval(xs[2*k],xs[2*k+1],uv[2*i],b,logb,logx[k]);
        for(long k=0;k<ny;k++)y[k]=interval(ys[2*k],ys[2*k+1],uv[2*i+1],b,logb,logy[k]);
        double m=-INFINITY;
        for(long j=0;j<nr;j++)m=std::max(m,x[ix[j]].lp+y[iy[j]].lp);
        double mass=0,du=0,dv=0,db=0;
        for(long j=0;j<nr;j++){
            const Score& a=x[ix[j]];const Score& c=y[iy[j]];
            double w=std::exp(a.lp+c.lp-m);
            mass+=w;du+=w*a.du;dv+=w*c.du;db+=w*(a.db+c.db);
        }
        double lp=m+std::log(mass);du/=mass;dv/=mass;db/=mass;
        if(!std::isfinite(lp)||!std::isfinite(du)||!std::isfinite(dv)||!std::isfinite(db)||lp>1e-12)return 4;
        out[4*i]=std::min(lp,0.);out[4*i+1]=du;out[4*i+2]=dv;out[4*i+3]=db;
    }
    return 0;
}
