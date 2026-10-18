// FP64 support/clearance predicates. No fast-math.
struct V { double x,y,z; };
__device__ V vec(const double* p) { return {p[0],p[1],p[2]}; }
__device__ V add(V a,V b) { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
__device__ V sub(V a,V b) { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
__device__ V mul(V a,double s) { return {a.x*s,a.y*s,a.z*s}; }
__device__ double dot(V a,V b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
__device__ double seg(V p,V a,V d) {
    V offset=sub(p,a); double length=dot(d,d);
    double t=length>0 ? fmin(1.,fmax(0.,dot(offset,d)/length)) : 0.;
    V residual=sub(offset,mul(d,t)); return dot(residual,residual);
}
__device__ double sweep(V p,V a,V b,V ref,double height) {
    V delta=sub(b,a),up=mul(ref,height);
    double result=fmin(fmin(seg(p,a,delta),seg(p,add(a,up),delta)),
                       fmin(seg(p,a,up),seg(p,b,up)));
    double vertical=dot(delta,ref); V horizontal=sub(delta,mul(ref,vertical));
    double length=dot(horizontal,horizontal);
    if(length>0) {
        V offset=sub(p,a); double t=dot(offset,horizontal)/length;
        double u=dot(offset,ref)-t*vertical;
        if(t>=0 && t<=1 && u>=0 && u<=height) {
            V residual=sub(sub(offset,mul(delta,t)),mul(ref,u));
            result=fmin(result,dot(residual,residual));
        }
    }
    return result;
}
__device__ long long lower_bound(const long long* keys,long long n,long long key) {
    long long lo=0,hi=n;
    while(lo<hi) { long long mid=(lo+hi)/2; if(keys[mid]<key) lo=mid+1; else hi=mid; }
    return lo;
}
__device__ long long upper_bound(const long long* keys,long long n,long long key) {
    long long lo=0,hi=n;
    while(lo<hi) { long long mid=(lo+hi)/2; if(keys[mid]<=key) lo=mid+1; else hi=mid; }
    return lo;
}
__device__ bool bounds(V lower,V upper,const double* origin,const long long* dims,
                        double cell,long long* lo,long long* hi) {
    double a[3]={lower.x,lower.y,lower.z},b[3]={upper.x,upper.y,upper.z};
    for(int j=0;j<3;j++) {
        lo[j]=max(0LL,(long long)floor((a[j]-origin[j])/cell)-1);
        hi[j]=min(dims[j]-1,(long long)floor((b[j]-origin[j])/cell)+1);
        if(lo[j]>hi[j]) return false;
    }
    return true;
}
extern "C" __global__ void support_check(const double* pts,const unsigned char* labels,
    const long long* keys,long long n,const double* origin,const long long* dims,double cell,
    const double* queries,const double* radii,int nq,int required,unsigned char* result) {
    int q=blockIdx.x*blockDim.x+threadIdx.x; if(q>=nq) return;
    V p=vec(queries+3*q); double radius=radii[q]; V pad={radius,radius,radius};
    long long lo[3],hi[3]; result[q]=0;
    if(!bounds(sub(p,pad),add(p,pad),origin,dims,cell,lo,hi)) return;
    V distinct[6]; int count=0;
    for(long long x=lo[0];x<=hi[0];x++) for(long long y=lo[1];y<=hi[1];y++) for(long long z=lo[2];z<=hi[2];z++) {
        long long key=(x*dims[1]+y)*dims[2]+z;
        long long begin=lower_bound(keys,n,key),end=upper_bound(keys,n,key);
        for(long long i=begin;i<end;i++) if(labels[i]&1) {
            V point=vec(pts+3*i),delta=sub(point,p);
            if(sqrt(dot(delta,delta))>radius) continue;
            if(required==1) { result[q]=1; return; }
            bool duplicate=false;
            for(int k=0;k<count;k++) if(point.x==distinct[k].x && point.y==distinct[k].y && point.z==distinct[k].z) duplicate=true;
            if(!duplicate) { distinct[count++]=point; if(count>=required) { result[q]=1; return; } }
        }
    }
}
extern "C" __global__ void obstacle_check(const double* pts,const unsigned char* labels,
    const long long* keys,long long n,const double* origin,const long long* dims,double cell,
    const double* route,int edges,const double* normal,double height,double margin,
    unsigned char* result) {
    int e=blockIdx.x*blockDim.x+threadIdx.x; if(e>=edges) return;
    V a=vec(route+3*e),b=vec(route+3*(e+1)),ref=vec(normal),up=mul(ref,height);
    V au=add(a,up),bu=add(b,up);
    V lower={fmin(fmin(a.x,b.x),fmin(au.x,bu.x))-margin,
             fmin(fmin(a.y,b.y),fmin(au.y,bu.y))-margin,
             fmin(fmin(a.z,b.z),fmin(au.z,bu.z))-margin};
    V upper={fmax(fmax(a.x,b.x),fmax(au.x,bu.x))+margin,
             fmax(fmax(a.y,b.y),fmax(au.y,bu.y))+margin,
             fmax(fmax(a.z,b.z),fmax(au.z,bu.z))+margin};
    long long lo[3],hi[3]; result[e]=1;
    if(!bounds(lower,upper,origin,dims,cell,lo,hi)) return;
    for(long long x=lo[0];x<=hi[0];x++) for(long long y=lo[1];y<=hi[1];y++) for(long long z=lo[2];z<=hi[2];z++) {
        long long key=(x*dims[1]+y)*dims[2]+z;
        long long begin=lower_bound(keys,n,key),end=upper_bound(keys,n,key);
        for(long long i=begin;i<end;i++) if(labels[i]&2) {
            V point=vec(pts+3*i);
            if(sweep(point,a,b,ref,height)<=margin*margin) { result[e]=0; return; }
        }
    }
}
