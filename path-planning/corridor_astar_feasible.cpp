// External relaxed search with conservative complete-map feasibility checks.
// Build: g++ -O3 -std=c++17 -ffp-contract=off -shared -fPIC corridor_astar_feasible.cpp -o corridor_astar_feasible.so
// IDs here are local indices; the caller retains their original-row mapping.
// Returns 1 on success, 0 when no route fits U, -1 for invalid input,
// -2 for allocation failure, -3 for another exception. On success g[goal]
// is the ORIGINAL edge-weight cost; prev follows local predecessor IDs.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <new>
#include <numeric>
#include <unordered_set>
#include <vector>

namespace {
using Id = std::int64_t;

struct Pair {
    Id a, b;
    bool operator==(const Pair& other) const { return a == other.a && b == other.b; }
};
Pair pair_key(Id a, Id b) { return {std::min(a, b), std::max(a, b)}; }
struct PairHash {
    std::size_t operator()(const Pair& p) const {
        const auto a = static_cast<std::uint64_t>(p.a);
        const auto b = static_cast<std::uint64_t>(p.b);
        return static_cast<std::size_t>(a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2)));
    }
};

struct KDTree {
    struct Node {
        Id begin, end;
        double lo[3], hi[3];
        int left = -1, right = -1;
    };
    const double* xyz;
    std::vector<Id> order;
    std::vector<Node> nodes;

    KDTree(const double* data, Id n, const Id* rows = nullptr) : xyz(data), order(static_cast<std::size_t>(n)) {
        if (rows) std::copy(rows, rows+n, order.begin());
        else std::iota(order.begin(), order.end(), Id(0));
        nodes.reserve(static_cast<std::size_t>(n / 16 + 1));
        if (n) build(0, n);
    }
    int build(Id begin, Id end) {
        Node node;
        node.begin = begin;
        node.end = end;
        for (int axis = 0; axis < 3; ++axis) {
            node.lo[axis] = node.hi[axis] = xyz[3 * order[begin] + axis];
            for (Id i = begin + 1; i < end; ++i) {
                const double value = xyz[3 * order[i] + axis];
                node.lo[axis] = std::min(node.lo[axis], value);
                node.hi[axis] = std::max(node.hi[axis], value);
            }
        }
        const int index = static_cast<int>(nodes.size());
        nodes.push_back(node);
        if (end - begin > 64) {
            int axis = 0;
            for (int k = 1; k < 3; ++k)
                if (node.hi[k] - node.lo[k] > node.hi[axis] - node.lo[axis]) axis = k;
            const Id middle = begin + (end - begin) / 2;
            std::nth_element(order.begin() + begin, order.begin() + middle, order.begin() + end,
                             [&](Id a, Id b) {
                const double x = xyz[3 * a + axis], y = xyz[3 * b + axis];
                return x < y || (x == y && a < b);
            });
            const int left = build(begin, middle), right = build(middle, end);
            nodes[index].left = left;
            nodes[index].right = right;
        }
        return index;
    }
    template<class Visitor>
    void query(int index, const double* position, double expanded_radius2, Visitor& visit) const {
        const Node& node = nodes[index];
        double lower2 = 0;
        for (int k = 0; k < 3; ++k) {
            const double delta = std::max(std::max(node.lo[k] - position[k],
                                                 position[k] - node.hi[k]), 0.0);
            lower2 += delta * delta;
        }
        if (lower2 > expanded_radius2) return;
        if (node.left < 0) {
            for (Id i = node.begin; i < node.end; ++i) visit(order[i]);
        } else {
            query(node.left, position, expanded_radius2, visit);
            query(node.right, position, expanded_radius2, visit);
        }
    }
    template<class Predicate>
    bool any(int index, const double* position, double radius2, Predicate& accept) const {
        const Node& node = nodes[index];
        double lower2 = 0;
        for (int k=0; k<3; ++k) {
            const double delta = std::max(std::max(node.lo[k]-position[k], position[k]-node.hi[k]), 0.0);
            lower2 += delta*delta;
        }
        if (lower2 > std::nextafter(radius2*(1+32*std::numeric_limits<double>::epsilon()),
                                    std::numeric_limits<double>::infinity())) return false;
        if (node.left < 0) {
            for (Id i=node.begin; i<node.end; ++i) if (accept(order[i])) return true;
            return false;
        }
        return any(node.left, position, radius2, accept) || any(node.right, position, radius2, accept);
    }
    template<class Predicate>
    bool any_box(int index, const double* position, const double* halfwidth, Predicate& accept) const {
        const Node& node = nodes[index];
        for (int k=0; k<3; ++k)
            if (node.lo[k]-position[k] > halfwidth[k] || position[k]-node.hi[k] > halfwidth[k])
                return false;
        if (node.left < 0) {
            for (Id i=node.begin; i<node.end; ++i) if (accept(order[i])) return true;
            return false;
        }
        return any_box(node.left, position, halfwidth, accept)
            || any_box(node.right, position, halfwidth, accept);
    }
};

struct Entry { double f, g; Id row; };
struct Later {
    bool operator()(const Entry& a, const Entry& b) const {
        if (a.f != b.f) return a.f > b.f;
        if (a.g != b.g) return a.g > b.g;
        return a.row > b.row;
    }
};

// One live entry per row. Popping clears its position, so a later strict
// improvement can reopen the row using the same insertion operation.
struct IndexedHeap {
    std::vector<Entry> entries;
    std::vector<int> position;
    explicit IndexedHeap(Id n) : position(static_cast<std::size_t>(n), -1) {}
    bool empty() const { return entries.empty(); }
    std::size_t size() const { return entries.size(); }
    void swap_entries(int a, int b) {
        std::swap(entries[a], entries[b]);
        position[entries[a].row] = a;
        position[entries[b].row] = b;
    }
    void push(Entry entry) {
        int index = position[entry.row];
        if (index < 0) {
            index = static_cast<int>(entries.size());
            entries.push_back(entry);
            position[entry.row] = index;
        } else {
            entries[index] = entry; // Caller supplies only strict g decreases.
        }
        while (index > 0) {
            const int parent = (index - 1) / 2;
            if (!Later{}(entries[parent], entries[index])) break;
            swap_entries(parent, index);
            index = parent;
        }
    }
    Entry pop() {
        const Entry result = entries.front();
        position[result.row] = -1;
        if (entries.size() == 1) {
            entries.pop_back();
            return result;
        }
        entries.front() = entries.back();
        entries.pop_back();
        position[entries.front().row] = 0;
        int index = 0;
        while (static_cast<std::size_t>(index) * 2 + 1 < entries.size()) {
            const int left = static_cast<int>(static_cast<std::size_t>(index) * 2 + 1);
            int child = left;
            if (static_cast<std::size_t>(left) + 1 < entries.size()
                    && Later{}(entries[left], entries[left + 1])) child = left + 1;
            if (!Later{}(entries[index], entries[child])) break;
            swap_entries(index, child);
            index = child;
        }
        return result;
    }
};

// These trees MUST contain ALL original finite-cost support rows and ALL
// original finite non-floor rows, even when the search graph is ellipse-pruned.
// Coordinates are borrowed; only row indices and bounding boxes are allocated.
// Rejection is conservative: enlarged support balls, reduced obstacle margins,
// and every plausible ceil probe count retain floating-boundary ambiguities.
// A production-valid edge therefore remains in this relaxation; Python still
// certifies every returned edge with the unchanged original connection rule.
double swept_distance2(const double* offset, const double* delta, const double* up, double height) {
    double minimum = std::numeric_limits<double>::infinity();
    auto segment = [&](const double* origin, const double* direction) {
        double length2=0, projection=0;
        for (int k=0; k<3; ++k) {
            length2 += direction[k]*direction[k];
            projection += (offset[k]-origin[k])*direction[k];
        }
        const double t=length2 > 0 ? std::clamp(projection/length2, 0.0, 1.0) : 0;
        double distance2=0;
        for (int k=0; k<3; ++k) {
            const double d=offset[k]-origin[k]-t*direction[k];
            distance2 += d*d;
        }
        minimum=std::min(minimum, distance2);
    };
    const double zero[3]={0,0,0};
    double vertical[3], horizontal[3], dh=0, qh=0, horizontal2=0, along=0;
    for (int k=0; k<3; ++k) {
        vertical[k]=height*up[k];
        dh += delta[k]*up[k];
        qh += offset[k]*up[k];
    }
    segment(zero, delta);
    if (height == 0) return minimum;
    segment(vertical, delta);
    segment(zero, vertical);
    segment(delta, vertical);
    for (int k=0; k<3; ++k) {
        horizontal[k]=delta[k]-dh*up[k];
        horizontal2 += horizontal[k]*horizontal[k];
        along += offset[k]*horizontal[k];
    }
    if (horizontal2 > 0) {
        const double t=along/horizontal2, h=qh-t*dh;
        if (t >= 0 && t <= 1 && h >= 0 && h <= height) {
            double distance2=0;
            for (int k=0; k<3; ++k) {
                const double d=offset[k]-t*delta[k]-h*up[k];
                distance2 += d*d;
            }
            minimum=std::min(minimum, distance2);
        }
    }
    return minimum;
}

bool impossible(const double* a, const double* b, double length, double radius,
                const KDTree& support, const KDTree& obstacles, double gap, double margin,
                const double* up, double clearance_height) {
    const double delta[3] = {b[0]-a[0], b[1]-a[1], b[2]-a[2]};
    double scale = std::max({1.0, length, radius, clearance_height});
    for (int k=0; k<3; ++k) scale = std::max({scale, std::abs(a[k]), std::abs(b[k])});
    const double tolerance = 1e-10*scale, lower_length = std::max(0.0, length-tolerance);
    const double first = std::max(1.0, std::ceil(lower_length/(gap/4)));
    const double last = std::max(1.0, std::ceil((length+tolerance)/(gap/4)));
    // Defaults require at most nine probes. Numerically unresolved or extreme
    // configurations remain relaxed rather than risking a false exclusion.
    if (!support.nodes.empty() && std::isfinite(last) && last <= 65536 && last-first <= 2) {
        bool possible_grid = false;
        for (Id steps=static_cast<Id>(first); steps<=static_cast<Id>(last); ++steps) {
            const double reach = gap/2-lower_length/(2*steps)+tolerance, reach2=reach*reach;
            bool gap_found = false;
            // Endpoint probes are support rows themselves and padding <= gap/8.
            for (Id j=1; j<steps; ++j) {
                double p[3];
                for (int k=0; k<3; ++k) p[k]=a[k]+(double(j)/steps)*delta[k];
                auto present = [&](Id row) {
                    double distance2=0;
                    for (int k=0; k<3; ++k) {
                        const double d=support.xyz[3*row+k]-p[k];
                        distance2 += d*d;
                    }
                    return distance2 <= reach2;
                };
                if (!support.any(0, p, reach2, present)) { gap_found=true; break; }
            }
            if (!gap_found) { possible_grid=true; break; }
        }
        if (!possible_grid) return true;
    }
    if (!obstacles.nodes.empty() && margin > tolerance) {
        double middle[3], halfwidth[3];
        for (int k=0; k<3; ++k) {
            middle[k]=(a[k]+b[k])/2+clearance_height*up[k]/2;
            halfwidth[k]=std::abs(delta[k])/2+clearance_height*std::abs(up[k])/2+margin+tolerance;
        }
        const double inner2=(margin-tolerance)*(margin-tolerance);
        auto blocked = [&](Id row) {
            double offset[3], local_scale=scale;
            for (int k=0; k<3; ++k) {
                offset[k]=obstacles.xyz[3*row+k]-a[k];
                local_scale=std::max(local_scale, std::abs(obstacles.xyz[3*row+k]));
            }
            const double distance2=swept_distance2(offset, delta, up, clearance_height);
            const double conservative_margin=margin-1e-10*local_scale;
            return conservative_margin > 0 && distance2 < std::min(inner2, conservative_margin*conservative_margin);
        };
        if (obstacles.any_box(0, middle, halfwidth, blocked)) return true;
    }
    return false;
}
}

static int search_policy(const double* xyz, const double* cost, const double* reference,
                               Id n, double radius, double tan_slope, double cmin, double upper,
                               Id start, Id goal, const Id* banned_pairs, Id banned_count,
                               double* g, Id* previous,
                               const double* original_xyz, Id original_count,
                               const Id* support_rows, Id support_count,
                               const Id* obstacle_rows, Id obstacle_count,
                               double max_gap, double obstacle_margin, double clearance_height, int slope_enabled) {
    try {
        if (!xyz || !cost || !reference || !g || !previous || n <= 0
                || n > std::numeric_limits<int>::max() || start < 0 || start >= n
                || goal < 0 || goal >= n || banned_count < 0 || (banned_count && !banned_pairs)
                || !std::isfinite(radius) || radius < 0 || !std::isfinite(radius * radius)
                || (slope_enabled != 0 && slope_enabled != 1)
                || !std::isfinite(tan_slope) || tan_slope <= 0
                || !std::isfinite(cmin) || cmin <= 0 || !std::isfinite(upper) || upper < 0
                || original_count < 0 || original_count > std::numeric_limits<int>::max()
                || support_count < 0 || support_count > original_count
                || obstacle_count < 0 || obstacle_count > original_count
                || ((support_count || obstacle_count) && !original_xyz)
                || (support_count && !support_rows) || (obstacle_count && !obstacle_rows)
                || !std::isfinite(max_gap) || max_gap <= 0
                || !std::isfinite(obstacle_margin) || obstacle_margin < 0
                || !std::isfinite(clearance_height) || clearance_height < 0)
            return -1;
        for (int kind=0; kind<2; ++kind) {
            const Id* rows=kind == 0 ? support_rows : obstacle_rows;
            const Id count=kind == 0 ? support_count : obstacle_count;
            for (Id i=0; i<count; ++i) {
                if (rows[i] < 0 || rows[i] >= original_count) return -1;
                for (int k=0; k<3; ++k) if (!std::isfinite(original_xyz[3*rows[i]+k])) return -1;
            }
        }
        double reference2 = 0;
        for (int k = 0; k < 3; ++k) {
            if (!std::isfinite(reference[k])) return -1;
            reference2 += reference[k] * reference[k];
        }
        if (std::abs(reference2 - 1.0) > 1e-8) return -1;
        std::vector<double> heuristic(static_cast<std::size_t>(n));
        const double infinity = std::numeric_limits<double>::infinity();
        const double slack = 1e-10 * std::max(1.0, upper);
        const double limit = upper + slack;
        for (Id row = 0; row < n; ++row) {
            if (!std::isfinite(cost[row]) || cost[row] < cmin) return -1;
            double distance2 = 0;
            for (int k = 0; k < 3; ++k) {
                if (!std::isfinite(xyz[3 * row + k])) return -1;
                const double delta = xyz[3 * row + k] - xyz[3 * goal + k];
                distance2 += delta * delta;
            }
            // A tiny downward allowance protects the mathematical lower bound
            // against floating-point evaluation differences from NumPy.
            heuristic[row] = std::max(0.0, cmin * std::sqrt(distance2) - slack);
            g[row] = infinity;
            previous[row] = -1;
        }
        std::unordered_set<Pair, PairHash> banned;
        banned.reserve(static_cast<std::size_t>(banned_count));
        for (Id i = 0; i < banned_count; ++i) {
            const Id a = banned_pairs[2 * i], b = banned_pairs[2 * i + 1];
            if (a < 0 || a >= n || b < 0 || b >= n) return -1;
            banned.insert(pair_key(a, b));
        }
        g[start] = 0;
        if (start == goal) return 1;
        if (heuristic[start] > limit) return 0;
        KDTree tree(xyz, n);
        KDTree support(original_xyz, support_count, support_rows), obstacles(original_xyz, obstacle_count, obstacle_rows);
        const double radius2 = radius * radius;
        const double expanded_radius2 = std::nextafter(
            radius2 * (1 + 32 * std::numeric_limits<double>::epsilon()), infinity);
        IndexedHeap queue(n);
        queue.push({heuristic[start], 0.0, start});
        std::uint64_t expanded = 0;
        while (!queue.empty()) {
            const Entry current = queue.pop();
            if (current.g != g[current.row]) continue;
            if (current.f > limit) break;
            if (current.row == goal) {
                std::fprintf(stderr, "A* found: expanded=%llu cost=%.17g\n",
                             static_cast<unsigned long long>(expanded), current.g);
                std::fflush(stderr);
                return 1;
            }
            if (++expanded % 50000 == 0) {
                std::fprintf(stderr, "A* expanded=%llu frontier=%zu bound=%.12g\n",
                             static_cast<unsigned long long>(expanded), queue.size(), current.f);
                std::fflush(stderr);
            }
            const double* a = xyz + 3 * current.row;
            auto visit = [&](Id other) {
                if (other == current.row) return;
                const double* b = xyz + 3 * other;
                const double dx = b[0] - a[0], dy = b[1] - a[1], dz = b[2] - a[2];
                const double length2 = dx * dx + dy * dy + dz * dz;
                if (length2 > expanded_radius2) return; // Caller validates the original radius rule.
                if (!banned.empty() && banned.count(pair_key(current.row, other))) return;
                const double length = std::sqrt(length2);
                const double candidate = current.g + length * (cost[current.row] / 2 + cost[other] / 2);
                if (!(candidate < g[other]) || candidate + heuristic[other] > limit) return;
                if (length > 0) {
                    const double height = dx * reference[0] + dy * reference[1] + dz * reference[2];
                    const double x = dx - height * reference[0];
                    const double y = dy - height * reference[1];
                    const double z = dz - height * reference[2];
                    if (slope_enabled && std::abs(height) > std::sqrt(x * x + y * y + z * z) * tan_slope + 1e-12)
                        return; // Relaxation only; final edges use original feasibility.
                    if ((support_count || obstacle_count)
                            && impossible(current.row < other ? a : b, current.row < other ? b : a,
                                          length, radius, support, obstacles, max_gap, obstacle_margin,
                                          reference, clearance_height)) return;
                }
                g[other] = candidate;
                previous[other] = current.row;
                queue.push({candidate + heuristic[other], candidate, other});
            };
            tree.query(0, a, expanded_radius2, visit);
        }
        return 0;
    } catch (const std::bad_alloc&) {
        return -2;
    } catch (...) {
        return -3;
    }
}

// Versioned policy export: slope_enabled is an explicit switch, not a fake angle.
extern "C" int corridor_astar_policy(const double* xyz, const double* cost, const double* reference,
                               Id n, double radius, double tan_slope, double cmin, double upper,
                               Id start, Id goal, const Id* banned_pairs, Id banned_count,
                               double* g, Id* previous,
                               const double* original_xyz, Id original_count,
                               const Id* support_rows, Id support_count,
                               const Id* obstacle_rows, Id obstacle_count,
                               double max_gap, double obstacle_margin, double clearance_height, int slope_enabled) {
    return search_policy(xyz, cost, reference, n, radius, tan_slope, cmin, upper,
        start, goal, banned_pairs, banned_count, g, previous, original_xyz, original_count,
        support_rows, support_count, obstacle_rows, obstacle_count, max_gap, obstacle_margin,
        clearance_height, slope_enabled);
}

extern "C" int corridor_astar_clearance(const double* xyz, const double* cost, const double* reference,
                               Id n, double radius, double tan_slope, double cmin, double upper,
                               Id start, Id goal, const Id* banned_pairs, Id banned_count,
                               double* g, Id* previous,
                               const double* original_xyz, Id original_count,
                               const Id* support_rows, Id support_count,
                               const Id* obstacle_rows, Id obstacle_count,
                               double max_gap, double obstacle_margin, double clearance_height) {
    return search_policy(xyz, cost, reference, n, radius, tan_slope, cmin, upper,
        start, goal, banned_pairs, banned_count, g, previous, original_xyz, original_count,
        support_rows, support_count, obstacle_rows, obstacle_count, max_gap, obstacle_margin,
        clearance_height, 1);
}

// Keep the previous ABI usable for baseline parity checks and older callers.
extern "C" int corridor_astar_feasible(const double* xyz, const double* cost, const double* reference,
                               Id n, double radius, double tan_slope, double cmin, double upper,
                               Id start, Id goal, const Id* banned_pairs, Id banned_count,
                               double* g, Id* previous,
                               const double* original_xyz, Id original_count,
                               const Id* support_rows, Id support_count,
                               const Id* obstacle_rows, Id obstacle_count,
                               double max_gap, double obstacle_margin) {
    return corridor_astar_clearance(xyz, cost, reference, n, radius, tan_slope, cmin, upper,
        start, goal, banned_pairs, banned_count, g, previous, original_xyz, original_count,
        support_rows, support_count, obstacle_rows, obstacle_count, max_gap, obstacle_margin, 0);
}
