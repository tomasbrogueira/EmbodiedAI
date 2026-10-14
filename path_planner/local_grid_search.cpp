// Fixed-capacity local terrain search with a monotonic deadline.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <utility>

static bool observed_line(const double* costs,int width,int x,int y,int dx,int dy) {
    const int tx=x+dx,ty=y+dy,sx=(dx>0)-(dx<0),sy=(dy>0)-(dy<0);
    const double inf=std::numeric_limits<double>::infinity();
    const double stepx=dx?1./std::abs(dx):inf,stepy=dy?1./std::abs(dy):inf;
    double nextx=dx?.5/std::abs(dx):inf,nexty=dy?.5/std::abs(dy):inf;
    while(x!=tx || y!=ty) {
        if(std::abs(nextx-nexty)<1e-12) {
            if(!std::isfinite(costs[y*width+x+sx]) || !std::isfinite(costs[(y+sy)*width+x])) return false;
            x+=sx;y+=sy;nextx+=stepx;nexty+=stepy;
        } else if(nextx<nexty) {x+=sx;nextx+=stepx;}
        else {y+=sy;nexty+=stepy;}
        if(!std::isfinite(costs[y*width+x])) return false;
    }
    return true;
}

extern "C" int grid_search(const double* costs, const double* heights, int width,
    int source, double cell, double max_slope, int enforce, double budget_ms,int max_stride,
    double* distance, int32_t* previous, uint8_t* settled, int* expanded) {
    const auto started=std::chrono::steady_clock::now();
    const int n=width*width;
    if(source<0 || source>=n || !std::isfinite(costs[source])) return -1;
    for(int i=0;i<n;++i) {
        distance[i]=std::numeric_limits<double>::infinity();
        previous[i]=-1; settled[i]=0;
    }
    using Item=std::pair<double,int>;
    std::priority_queue<Item,std::vector<Item>,std::greater<Item>> queue;
    distance[source]=0.; queue.push({0.,source}); *expanded=0;
    const double grade=std::tan(max_slope);
    max_stride=std::clamp(max_stride,1,4);
    while(!queue.empty()) {
        auto [value,first]=queue.top(); queue.pop();
        if(value!=distance[first] || settled[first]) continue;
        if((*expanded % 64)==0 && std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-started).count()>=budget_ms) return 1;
        settled[first]=1; ++*expanded;
        const int y=first/width,x=first%width;
        for(int dy=-max_stride;dy<=max_stride;++dy) for(int dx=-max_stride;dx<=max_stride;++dx) {
            if((!dx&&!dy) || x+dx<0 || x+dx>=width || y+dy<0 || y+dy>=width) continue;
            if(max_stride>1 && dx*dx+dy*dy>max_stride*max_stride) continue;
            const int other=(y+dy)*width+x+dx;
            if(!std::isfinite(costs[other])) continue;
            if(!observed_line(costs,width,x,y,dx,dy)) continue;
            const double horizontal=cell*std::sqrt(double(dx*dx+dy*dy));
            const double dh=heights[other]-heights[first];
            if(enforce && std::abs(dh)>=horizontal*grade) continue;
            const double candidate=value+std::hypot(horizontal,dh)*(costs[first]+costs[other])/2;
            if(candidate<distance[other]) {
                distance[other]=candidate; previous[other]=first;
                queue.push({candidate,other});
            }
        }
    }
    return 0;
}
