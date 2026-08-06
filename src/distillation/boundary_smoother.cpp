#include "distillation/boundary_smoother.hpp"

#include <Geom2dAPI_PointsToBSpline.hxx>
#include <GeomAPI_PointsToBSpline.hxx>
#include <TColgp_Array1OfPnt2d.hxx>
#include <TColgp_Array1OfPnt.hxx>

#include <map>
#include <set>
#include <queue>
#include <deque>
#include <algorithm>
#include <cmath>
#include <utility>
#include <numeric>

namespace distillation {

static inline std::pair<int,int> mkEdge(int a, int b) {
    return (a < b) ? std::make_pair(a, b) : std::make_pair(b, a);
}

static inline double triArea2D(const Vec2& a, const Vec2& b, const Vec2& c) {
    return 0.5 * ((b.x() - a.x()) * (c.y() - a.y()) - (b.y() - a.y()) * (c.x() - a.x()));
}

// ═══════════════════════════════════════════════════════════════════
// Step 0: Vertex labels -> Face labels
// ═══════════════════════════════════════════════════════════════════

IntArr convertVertexLabelsToFaceLabels(
    const IntVecSet& vertexPartitions,
    const FaceArr& faces,
    int numVertices)
{
    IntArr vertexLabels(numVertices, -1);
    for (int pid = 0; pid < static_cast<int>(vertexPartitions.size()); ++pid) {
        for (int v : vertexPartitions[pid]) {
            if (v >= 0 && v < numVertices) {
                vertexLabels[v] = (vertexLabels[v] < 0) ? pid : -2;
            }
        }
    }
    for (int i = 0; i < numVertices; ++i)
        if (vertexLabels[i] == -2) vertexLabels[i] = -1;

    int nf = static_cast<int>(faces.size());
    IntArr faceLabels(nf, -1);

    for (int fi = 0; fi < nf; ++fi) {
        int l0 = vertexLabels[faces[fi].v0];
        int l1 = vertexLabels[faces[fi].v1];
        int l2 = vertexLabels[faces[fi].v2];
        if      (l0 >= 0 && l0 == l1 && l1 == l2) faceLabels[fi] = l0;
        else if (l0 >= 0 && l0 == l1)              faceLabels[fi] = l0;
        else if (l1 >= 0 && l1 == l2)              faceLabels[fi] = l1;
        else if (l0 >= 0 && l0 == l2)              faceLabels[fi] = l0;
    }

    std::vector<IntArr> faceAdj(nf);
    {
        std::map<std::pair<int,int>, IntArr> edgeFaces;
        for (int fi = 0; fi < nf; ++fi) {
            const auto& f = faces[fi];
            for (int i = 0; i < 3; ++i)
                edgeFaces[mkEdge(f[i], f[(i + 1) % 3])].push_back(fi);
        }
        for (auto& kv : edgeFaces) {
            const auto& fl = kv.second;
            for (size_t i = 0; i < fl.size(); ++i)
                for (size_t j = i + 1; j < fl.size(); ++j) {
                    faceAdj[fl[i]].push_back(fl[j]);
                    faceAdj[fl[j]].push_back(fl[i]);
                }
        }
    }

    std::queue<int> q;
    for (int fi = 0; fi < nf; ++fi)
        if (faceLabels[fi] >= 0) q.push(fi);

    while (!q.empty()) {
        int fi = q.front(); q.pop();
        int lbl = faceLabels[fi];
        for (int nb : faceAdj[fi]) {
            if (faceLabels[nb] < 0) {
                faceLabels[nb] = lbl;
                q.push(nb);
            }
        }
    }

    return faceLabels;
}

// ═══════════════════════════════════════════════════════════════════
// Merge tiny partition regions
// ═══════════════════════════════════════════════════════════════════

void mergeTinyRegions(
    IntArr& faceLabels,
    const FaceArr& faces,
    int minRegionFaces)
{
    int nf = static_cast<int>(faces.size());
    if (nf == 0) return;

    // Build face adjacency from shared edges
    std::vector<IntArr> faceAdj(nf);
    {
        std::map<std::pair<int,int>, IntArr> edgeFaces;
        for (int fi = 0; fi < nf; ++fi) {
            const auto& f = faces[fi];
            for (int i = 0; i < 3; ++i) {
                int a = f[i], b = f[(i + 1) % 3];
                edgeFaces[(a < b) ? std::make_pair(a, b) : std::make_pair(b, a)]
                    .push_back(fi);
            }
        }
        for (auto& kv : edgeFaces) {
            const auto& fl = kv.second;
            for (size_t i = 0; i < fl.size(); ++i)
                for (size_t j = i + 1; j < fl.size(); ++j) {
                    faceAdj[fl[i]].push_back(fl[j]);
                    faceAdj[fl[j]].push_back(fl[i]);
                }
        }
    }

    int totalMerged = 0;
    int nPass = 0;
    const int MAX_PASSES = 10;
    int dynThreshold = (minRegionFaces > 0) ? minRegionFaces : 0;

    while (nPass < MAX_PASSES) {
        ++nPass;

        // Find connected components per label
        std::vector<bool> visited(nf, false);
        using Component = std::vector<int>;
        std::vector<Component> components;
        std::vector<int> componentLabel;

        for (int fi = 0; fi < nf; ++fi) {
            if (visited[fi]) continue;
            int lbl = faceLabels[fi];
            if (lbl < 0) continue;

            std::vector<int> comp;
            std::queue<int> q;
            q.push(fi); visited[fi] = true;
            while (!q.empty()) {
                int cur = q.front(); q.pop();
                comp.push_back(cur);
                for (int nb : faceAdj[cur]) {
                    if (!visited[nb] && faceLabels[nb] == lbl) {
                        visited[nb] = true;
                        q.push(nb);
                    }
                }
            }
            components.push_back(std::move(comp));
            componentLabel.push_back(lbl);
        }

        // Dynamic threshold: elbow in component size distribution
        if (dynThreshold <= 0 && !components.empty()) {
            std::vector<int> sizes;
            for (const auto& c : components)
                sizes.push_back(static_cast<int>(c.size()));
            std::sort(sizes.begin(), sizes.end());

            // Find first elbow: ratio > 3x between consecutive sizes (after min size 3)
            int elbow = sizes.back();  // default: merge nothing if no clear elbow
            for (size_t i = 0; i + 1 < sizes.size(); ++i) {
                if (sizes[i] < 3) continue;
                if (sizes[i + 1] >= sizes[i] * 3) {
                    elbow = sizes[i];
                    break;
                }
            }
            // Floor: must merge at least very tiny islands
            dynThreshold = std::max(5, std::min(elbow, nf / 50));
        } else if (dynThreshold <= 0) {
            dynThreshold = 5;
        }

        // Merge components below threshold
        int nMerged = 0;
        for (size_t ci = 0; ci < components.size(); ++ci) {
            if (static_cast<int>(components[ci].size()) >= dynThreshold) continue;

            std::map<int, int> adjCompCount;
            std::set<int> compSet(components[ci].begin(), components[ci].end());

            for (int fi : components[ci]) {
                for (int nb : faceAdj[fi]) {
                    if (compSet.count(nb)) continue;
                    if (faceLabels[nb] < 0) continue;
                    // Find target component: largest adjacent component
                    for (size_t cj = 0; cj < components.size(); ++cj) {
                        if (cj == ci) continue;
                        if (static_cast<int>(components[cj].size()) < dynThreshold) continue;
                        if (faceLabels[nb] == componentLabel[cj]) {
                            for (int fcj : components[cj]) {
                                if (fcj == nb) {
                                    adjCompCount[static_cast<int>(cj)]++;
                                    break;
                                }
                            }
                            break;
                        }
                    }
                }
            }

            if (adjCompCount.empty()) {
                // No large neighbor — merge to the largest adjacent component regardless
                for (int fi : components[ci]) {
                    for (int nb : faceAdj[fi]) {
                        if (compSet.count(nb)) continue;
                        if (faceLabels[nb] < 0) continue;
                        for (size_t cj = 0; cj < components.size(); ++cj) {
                            if (cj == ci && faceLabels[nb] == componentLabel[cj]) {
                                adjCompCount[static_cast<int>(cj)]++;
                                break;
                            }
                        }
                    }
                }
            }

            if (adjCompCount.empty()) continue;

            int bestComp = -1, bestCount = 0;
            for (auto& ac : adjCompCount) {
                if (ac.second > bestCount) {
                    bestCount = ac.second;
                    bestComp = ac.first;
                }
            }

            if (bestComp >= 0) {
                int targetLabel = componentLabel[bestComp];
                for (int fi : components[ci])
                    faceLabels[fi] = targetLabel;
                ++nMerged;
            }
        }

        totalMerged += nMerged;
        if (nMerged == 0) break;
    }

    if (totalMerged > 0)
        std::cout << "  [MergeTiny] " << totalMerged << " regions in "
                  << nPass << " pass(es), threshold=" << dynThreshold << "\n";
}

// ═══════════════════════════════════════════════════════════════════
// Step 1: Extract boundary network
// ═══════════════════════════════════════════════════════════════════

BoundaryNetwork extractBoundaryNetwork(
    const Vec2Arr& uvs,
    const FaceArr& faces,
    const IntArr& faceLabels,
    double uMin, double uMax,
    double vMin, double vMax)
{
    BoundaryNetwork net;
    int nv = static_cast<int>(uvs.size());
    int nf = static_cast<int>(faces.size());

    std::map<std::pair<int,int>, IntArr> edgeFaces;
    for (int fi = 0; fi < nf; ++fi) {
        const auto& f = faces[fi];
        for (int i = 0; i < 3; ++i)
            edgeFaces[mkEdge(f[i], f[(i + 1) % 3])].push_back(fi);
    }

    std::vector<std::pair<int,int>> boundaryEdges;
    std::vector<int> boundaryEdgeFace0, boundaryEdgeFace1;
    for (const auto& kv : edgeFaces) {
        const auto& fl = kv.second;
        if (fl.size() == 1) {
            boundaryEdges.push_back(kv.first);
            boundaryEdgeFace0.push_back(fl[0]);
            boundaryEdgeFace1.push_back(-1);
        } else if (fl.size() == 2 && faceLabels[fl[0]] != faceLabels[fl[1]]) {
            boundaryEdges.push_back(kv.first);
            boundaryEdgeFace0.push_back(fl[0]);
            boundaryEdgeFace1.push_back(fl[1]);
        }
    }

    std::set<int> bvSet;
    for (const auto& e : boundaryEdges) {
        bvSet.insert(e.first);
        bvSet.insert(e.second);
    }
    int nBv = static_cast<int>(bvSet.size());

    net.globalToLocal.assign(nv, -1);
    net.localToGlobal.resize(nBv);
    int idx = 0;
    for (int v : bvSet) {
        net.globalToLocal[v] = idx;
        net.localToGlobal[idx] = v;
        ++idx;
    }

    net.smoothedUVs.resize(nBv);
    for (int i = 0; i < nBv; ++i)
        net.smoothedUVs[i] = uvs[net.localToGlobal[i]];

    net.edges.reserve(boundaryEdges.size());
    net.neighbors.resize(nBv);
    for (size_t ei = 0; ei < boundaryEdges.size(); ++ei) {
        const auto& e = boundaryEdges[ei];
        int l0 = net.globalToLocal[e.first];
        int l1 = net.globalToLocal[e.second];
        if (l0 < 0 || l1 < 0) continue;
        net.edges.push_back({e.first, e.second,
                             boundaryEdgeFace0[ei], boundaryEdgeFace1[ei]});
        net.neighbors[l0].push_back(l1);
        net.neighbors[l1].push_back(l0);
    }

    double sumLen = 0.0;
    for (const auto& e : boundaryEdges)
        sumLen += (uvs[e.first] - uvs[e.second]).norm();
    net.avgEdgeLength = boundaryEdges.empty() ? 0.0 : sumLen / boundaryEdges.size();

    double nearU = (uMax - uMin) * 0.001;
    double nearV = (vMax - vMin) * 0.001;
    net.isExternal.assign(nBv, false);
    for (int i = 0; i < nBv; ++i) {
        const Vec2& uv = net.smoothedUVs[i];
        if (std::abs(uv.x() - uMin) < nearU || std::abs(uv.x() - uMax) < nearU ||
            std::abs(uv.y() - vMin) < nearV || std::abs(uv.y() - vMax) < nearV)
            net.isExternal[i] = true;
    }

    return net;
}

// ═══════════════════════════════════════════════════════════════════
// Domain boundary projection
// ═══════════════════════════════════════════════════════════════════

Vec2 projectToDomainBoundary(
    const Vec2& pt,
    double uMin, double uMax,
    double vMin, double vMax)
{
    struct Segment { Vec2 s, e; };
    Segment segs[4] = {
        {Vec2(uMin, vMin), Vec2(uMax, vMin)},
        {Vec2(uMax, vMin), Vec2(uMax, vMax)},
        {Vec2(uMax, vMax), Vec2(uMin, vMax)},
        {Vec2(uMin, vMax), Vec2(uMin, vMin)},
    };
    double bestDist = std::numeric_limits<double>::max();
    Vec2 bestPt = segs[0].s;
    for (int k = 0; k < 4; ++k) {
        Vec2 seg = segs[k].e - segs[k].s;
        double lenSq = seg.squaredNorm();
        double t = (lenSq < EPS) ? 0.0 : clamp((pt - segs[k].s).dot(seg) / lenSq, 0.0, 1.0);
        Vec2 proj = segs[k].s + t * seg;
        double d = (proj - pt).squaredNorm();
        if (d < bestDist) { bestDist = d; bestPt = proj; }
    }
    return bestPt;
}

// ═══════════════════════════════════════════════════════════════════
// Step 2: Graph Laplacian smoothing (with displacement clamping)
// ═══════════════════════════════════════════════════════════════════

std::vector<SmoothIterationEntry> laplacianSmoothBoundary(
    BoundaryNetwork& net,
    double uMin, double uMax,
    double vMin, double vMax,
    double sigmaTarget,
    double tolerance)
{
    std::vector<SmoothIterationEntry> history;
    int nBv = static_cast<int>(net.smoothedUVs.size());
    if (nBv == 0 || net.avgEdgeLength < EPS) return history;

    int K = static_cast<int>(std::ceil(
        std::pow(2.0 * sigmaTarget / net.avgEdgeLength, 2.0)));
    K = std::max(1, std::min(K, 200));

    double maxStep = net.avgEdgeLength * 0.3;

    Vec2Arr newUVs(nBv);

    for (int iter = 0; iter < K; ++iter) {
        double maxDisp = 0.0;
        double sumDisp = 0.0;

        for (int i = 0; i < nBv; ++i) {
            const auto& nb = net.neighbors[i];
            if (nb.empty()) {
                newUVs[i] = net.smoothedUVs[i];
                continue;
            }

            Vec2 avg(0, 0);
            for (int nv : nb) avg += net.smoothedUVs[nv];
            avg /= static_cast<double>(nb.size());

            Vec2 target;
            if (net.isExternal[i])
                target = projectToDomainBoundary(avg, uMin, uMax, vMin, vMax);
            else
                target = avg;

            Vec2 disp = target - net.smoothedUVs[i];
            double dispNorm = disp.norm();
            if (dispNorm > maxStep && dispNorm > EPS)
                disp *= maxStep / dispNorm;

            newUVs[i] = net.smoothedUVs[i] + disp;
            double d = (newUVs[i] - net.smoothedUVs[i]).norm();
            maxDisp = std::max(maxDisp, d);
            sumDisp += d;
        }

        net.smoothedUVs = newUVs;

        SmoothIterationEntry entry;
        entry.iteration = iter + 1;
        entry.maxDisplacement = maxDisp;
        entry.avgDisplacement = sumDisp / nBv;
        entry.nBoundaryVertices = nBv;
        history.push_back(entry);

        if (iter % std::max(1, K / 10) == 0 || maxDisp < tolerance)
            std::cout << "    iter " << (iter + 1) << "/" << K
                      << " maxDisp=" << maxDisp
                      << " avgDisp=" << entry.avgDisplacement << "\n";

        if (maxDisp < tolerance) break;
    }

    return history;
}

SmoothIterationEntry laplacianSmoothSingle(
    BoundaryNetwork& net,
    double uMin, double uMax,
    double vMin, double vMax)
{
    int nBv = static_cast<int>(net.smoothedUVs.size());
    SmoothIterationEntry entry{0, 0.0, 0.0, nBv};
    if (nBv == 0) return entry;

    double maxStep = net.avgEdgeLength * 0.3;
    Vec2Arr newUVs(nBv);
    double maxDisp = 0.0, sumDisp = 0.0;

    for (int i = 0; i < nBv; ++i) {
        const auto& nb = net.neighbors[i];
        if (nb.empty()) {
            newUVs[i] = net.smoothedUVs[i];
            continue;
        }

        Vec2 avg(0, 0);
        for (int nv : nb) avg += net.smoothedUVs[nv];
        avg /= static_cast<double>(nb.size());

        Vec2 target = net.isExternal[i]
            ? projectToDomainBoundary(avg, uMin, uMax, vMin, vMax)
            : avg;

        Vec2 disp = target - net.smoothedUVs[i];
        double dispNorm = disp.norm();
        if (dispNorm > maxStep && dispNorm > EPS)
            disp *= maxStep / dispNorm;

        newUVs[i] = net.smoothedUVs[i] + disp;
        double d = (newUVs[i] - net.smoothedUVs[i]).norm();
        maxDisp = std::max(maxDisp, d);
        sumDisp += d;
    }

    net.smoothedUVs = std::move(newUVs);
    entry.iteration = 1;
    entry.maxDisplacement = maxDisp;
    entry.avgDisplacement = sumDisp / nBv;
    return entry;
}

// ═══════════════════════════════════════════════════════════════════
// Polylines from boundary network — SEGMENTED at branch points
// ═══════════════════════════════════════════════════════════════════

std::vector<std::vector<Vec2>> extractPolylinesFromNetwork(
    const BoundaryNetwork& net)
{
    int nBv = static_cast<int>(net.smoothedUVs.size());
    std::map<int, IntArr> graph;
    for (const auto& e : net.edges) {
        if (e.face0 < 0 || e.face1 < 0) continue;
        int l0 = net.globalToLocal[e.v0];
        int l1 = net.globalToLocal[e.v1];
        if (l0 < 0 || l1 < 0) continue;
        graph[l0].push_back(l1);
        graph[l1].push_back(l0);
    }

    std::set<int> corners;
    for (auto& kv : graph) {
        if (kv.second.size() != 2)
            corners.insert(kv.first);
    }

    auto removeEdge = [&](int a, int b) {
        auto& na = graph[a];
        na.erase(std::remove(na.begin(), na.end(), b), na.end());
        auto& nb = graph[b];
        nb.erase(std::remove(nb.begin(), nb.end(), a), nb.end());
    };

    std::vector<std::vector<Vec2>> polylines;

    for (int start : corners) {
        auto& nb = graph[start];
        while (!nb.empty()) {
            int next = nb.back(); nb.pop_back();
            removeEdge(start, next);

            std::deque<int> chain;
            chain.push_back(start);
            chain.push_back(next);
            int prev = start, cur = next;

            while (corners.find(cur) == corners.end()) {
                auto& cnb = graph[cur];
                auto it = std::find(cnb.begin(), cnb.end(), prev);
                if (it != cnb.end()) cnb.erase(it);
                if (cnb.empty()) break;
                int n = cnb.back(); cnb.pop_back();
                chain.push_back(n);
                removeEdge(cur, n);
                prev = cur; cur = n;
            }

            std::vector<Vec2> poly;
            for (int vid : chain) poly.push_back(net.smoothedUVs[vid]);
            if (poly.size() >= 2) polylines.push_back(poly);
        }
    }

    for (auto& kv : graph) {
        int start = kv.first;
        if (graph[start].size() != 2) continue;
        std::deque<int> chain;
        chain.push_back(start);
        int cur = start, prev = -1;
        while (true) {
            auto& nb = graph[cur];
            int next = -1;
            for (int n : nb) {
                if (n != prev) { next = n; break; }
            }
            if (next < 0) break;
            if (next == start) { chain.push_back(next); break; }
            chain.push_back(next);
            prev = cur; cur = next;
        }
        for (size_t i = 0; i + 1 < chain.size(); ++i)
            removeEdge(chain[i], chain[i + 1]);

        std::vector<Vec2> poly;
        for (int vid : chain) poly.push_back(net.smoothedUVs[vid]);
        if (poly.size() >= 2) polylines.push_back(poly);
    }

    return polylines;
}

// ═══════════════════════════════════════════════════════════════════
// Polylines from mesh — SEGMENTED at branch points (degree≠2)
// ═══════════════════════════════════════════════════════════════════

std::vector<std::vector<Vec2>> extractBoundaryPolylines(
    const Vec2Arr& uvs,
    const FaceArr& faces,
    const IntArr& faceLabels)
{
    int nf = static_cast<int>(faces.size());
    std::map<std::pair<int,int>, IntArr> edgeFaces;
    for (int fi = 0; fi < nf; ++fi) {
        const auto& f = faces[fi];
        for (int i = 0; i < 3; ++i)
            edgeFaces[mkEdge(f[i], f[(i + 1) % 3])].push_back(fi);
    }

    struct PE { int v0, v1; };
    std::vector<PE> polyEdges;
    for (const auto& kv : edgeFaces) {
        const auto& fl = kv.second;
        if (fl.size() == 1 || (fl.size() == 2 && faceLabels[fl[0]] != faceLabels[fl[1]]))
            polyEdges.push_back({kv.first.first, kv.first.second});
    }

    std::map<int, IntArr> graph;
    for (const auto& e : polyEdges) {
        graph[e.v0].push_back(e.v1);
        graph[e.v1].push_back(e.v0);
    }

    std::set<int> corners;
    for (auto& kv : graph) {
        if (kv.second.size() != 2)
            corners.insert(kv.first);
    }

    auto removeEdge = [&](int a, int b) {
        auto& na = graph[a];
        na.erase(std::remove(na.begin(), na.end(), b), na.end());
        auto& nb = graph[b];
        nb.erase(std::remove(nb.begin(), nb.end(), a), nb.end());
    };

    std::vector<std::vector<Vec2>> polylines;

    for (int start : corners) {
        auto& nb = graph[start];
        while (!nb.empty()) {
            int next = nb.back(); nb.pop_back();
            removeEdge(start, next);

            std::deque<int> chain;
            chain.push_back(start);
            chain.push_back(next);
            int prev = start, cur = next;

            while (corners.find(cur) == corners.end()) {
                auto& cnb = graph[cur];
                auto it = std::find(cnb.begin(), cnb.end(), prev);
                if (it != cnb.end()) cnb.erase(it);
                if (cnb.empty()) break;
                int n = cnb.back(); cnb.pop_back();
                chain.push_back(n);
                removeEdge(cur, n);
                prev = cur; cur = n;
            }

            std::vector<Vec2> poly;
            for (int vid : chain) poly.push_back(uvs[vid]);
            if (poly.size() >= 2) polylines.push_back(poly);
        }
    }

    for (auto& kv : graph) {
        int start = kv.first;
        if (graph[start].size() != 2) continue;
        std::deque<int> chain;
        chain.push_back(start);
        int cur = start, prev = -1;
        while (true) {
            auto& nb = graph[cur];
            int next = -1;
            for (int n : nb) { if (n != prev) { next = n; break; } }
            if (next < 0) break;
            if (next == start) { chain.push_back(next); break; }
            chain.push_back(next);
            prev = cur; cur = next;
        }
        for (size_t i = 0; i + 1 < chain.size(); ++i)
            removeEdge(chain[i], chain[i + 1]);

        std::vector<Vec2> poly;
        for (int vid : chain) poly.push_back(uvs[vid]);
        if (poly.size() >= 2) polylines.push_back(poly);
    }

    return polylines;
}

// ═══════════════════════════════════════════════════════════════════
// Step 3a: Harmonic mesh update (Dirichlet: Δv=0, boundary fixed)
// ═══════════════════════════════════════════════════════════════════

void harmonicMeshUpdate(
    Vec2Arr& uvs,
    const FaceArr& faces,
    const BoundaryNetwork& net)
{
    int nv = static_cast<int>(uvs.size());
    std::vector<bool> isFixed(nv, false);
    for (int i = 0; i < static_cast<int>(net.localToGlobal.size()); ++i)
        isFixed[net.localToGlobal[i]] = true;

    // Build adjacency
    std::vector<IntArr> neighbors(nv);
    for (const auto& f : faces) {
        for (int i = 0; i < 3; ++i) {
            int a = f[i], b = f[(i+1)%3];
            neighbors[a].push_back(b);
            neighbors[b].push_back(a);
        }
    }

    int maxIters = 2000;
    double tol = 1e-8;
    Vec2Arr newUVs = uvs;

    for (int iter = 0; iter < maxIters; ++iter) {
        double maxChange = 0.0;
        for (int v = 0; v < nv; ++v) {
            if (isFixed[v]) continue;
            Vec2 sum(0, 0);
            for (int nb : neighbors[v]) sum += uvs[nb];
            newUVs[v] = sum / static_cast<double>(neighbors[v].size());
            maxChange = std::max(maxChange, (newUVs[v] - uvs[v]).norm());
        }
        uvs.swap(newUVs);
        if (maxChange < tol) break;
    }

    int nFixed = 0;
    for (int i = 0; i < nv; ++i) if (isFixed[i]) ++nFixed;
    std::cout << "  [Harmonic] " << nFixed << " boundary verts fixed, "
              << (nv - nFixed) << " interior solved via Jacobi\n";
}

// ═══════════════════════════════════════════════════════════════════
// Step 3: Mesh repair (edge flips)
// ═══════════════════════════════════════════════════════════════════

bool repairMeshAfterSmoothing(
    Vec2Arr& uvs,
    FaceArr& faces,
    IntArr& faceLabels,
    const BoundaryNetwork& net)
{
    int nf = static_cast<int>(faces.size());
    Vec2Arr candidateUVs = uvs;

    for (int i = 0; i < static_cast<int>(net.localToGlobal.size()); ++i) {
        int gidx = net.localToGlobal[i];
        if (gidx >= 0 && gidx < static_cast<int>(candidateUVs.size()))
            candidateUVs[gidx] = net.smoothedUVs[i];
    }

    std::vector<bool> flipped(nf, false);
    int nFlipped = 0;
    for (int fi = 0; fi < nf; ++fi) {
        const auto& f = faces[fi];
        double origArea = triArea2D(uvs[f.v0], uvs[f.v1], uvs[f.v2]);
        double newArea = triArea2D(candidateUVs[f.v0], candidateUVs[f.v1], candidateUVs[f.v2]);
        bool signFlipped = (origArea > EPS && newArea < -EPS)
                        || (origArea < -EPS && newArea > EPS);
        bool degenerated = std::abs(newArea) < EPS && std::abs(origArea) > EPS;
        if (signFlipped || degenerated) {
            flipped[fi] = true;
            ++nFlipped;
        }
    }

    if (nFlipped == 0) {
        uvs = candidateUVs;
        return true;
    }

    if (nFlipped > nf / 4) {
        std::cout << "  [MeshRepair] " << nFlipped << "/" << nf
                  << " flipped — too many, skip repair\n";
        return false;
    }

    std::cout << "  [MeshRepair] fixing " << nFlipped << " flipped triangles...\n";

    std::map<std::pair<int,int>, IntArr> edgeFaces;
    for (int fi = 0; fi < nf; ++fi) {
        const auto& f = faces[fi];
        for (int i = 0; i < 3; ++i)
            edgeFaces[mkEdge(f[i], f[(i + 1) % 3])].push_back(fi);
    }

    for (int fi = 0; fi < nf; ++fi) {
        if (!flipped[fi]) continue;
        const auto& f = faces[fi];
        for (int k = 0; k < 3; ++k) {
            auto key = mkEdge(f[k], f[(k + 1) % 3]);
            auto& fl = edgeFaces[key];
            if (fl.size() != 2) continue;
            int other = (fl[0] == fi) ? fl[1] : fl[0];
            if (flipped[other]) continue;

            int a = f[k], b = f[(k + 1) % 3], c = f[(k + 2) % 3];
            const auto& fo = faces[other];
            int d = -1;
            if (fo.v0 != a && fo.v0 != b) d = fo.v0;
            else if (fo.v1 != a && fo.v1 != b) d = fo.v1;
            else if (fo.v2 != a && fo.v2 != b) d = fo.v2;
            if (d < 0) continue;

            double area1 = triArea2D(candidateUVs[a], candidateUVs[d], candidateUVs[c]);
            double area2 = triArea2D(candidateUVs[d], candidateUVs[b], candidateUVs[c]);

            if (std::abs(area1) > EPS && std::abs(area2) > EPS) {
                faces[fi] = {a, d, c};
                faces[other] = {d, b, c};
                flipped[fi] = false;
            }
        }
    }

    int nBad = 0;
    for (int fi = 0; fi < nf; ++fi) {
        const auto& f = faces[fi];
        double origArea = triArea2D(uvs[f.v0], uvs[f.v1], uvs[f.v2]);
        double newArea = triArea2D(candidateUVs[f.v0], candidateUVs[f.v1], candidateUVs[f.v2]);
        if ((origArea > EPS && newArea < -EPS) || (origArea < -EPS && newArea > EPS)) ++nBad;
    }

    if (nBad > 0) {
        std::cout << "  [MeshRepair] " << nBad << " flipped remain — skip\n";
        return false;
    }

    uvs = candidateUVs;
    std::cout << "  [MeshRepair] success\n";
    return true;
}

// ═══════════════════════════════════════════════════════════════════
// Step 5: Fit NURBS curves — with fallback
// ═══════════════════════════════════════════════════════════════════

static bool validateCurve2D(const Handle(Geom2d_BSplineCurve)& curve,
                            double margin = 0.05)
{
    if (curve.IsNull()) return false;
    double u1 = curve->FirstParameter();
    double u2 = curve->LastParameter();
    for (int k = 0; k <= 20; ++k) {
        double t = u1 + (u2 - u1) * k / 20.0;
        gp_Pnt2d p = curve->Value(t);
        if (p.X() < -margin || p.X() > 1.0 + margin ||
            p.Y() < -margin || p.Y() > 1.0 + margin)
            return false;
    }
    return true;
}

static Handle(Geom2d_BSplineCurve) fitOneCurve(
    const std::vector<Vec2>& poly,
    int degree, int nCtrl, double tolerance)
{
    int n = static_cast<int>(poly.size());
    TColgp_Array1OfPnt2d arr(1, n);
    for (int i = 0; i < n; ++i)
        arr(i + 1) = gp_Pnt2d(poly[i].x(), poly[i].y());
    try {
        Geom2dAPI_PointsToBSpline fitter(arr, degree, nCtrl,
                                          GeomAbs_C1, tolerance);
        if (fitter.IsDone()) return fitter.Curve();
    } catch (...) {}
    return nullptr;
}

std::vector<Handle(Geom2d_BSplineCurve)> fitBoundaryCurves(
    const std::vector<std::vector<Vec2>>& polylines,
    int degree, int minCtrlPts, int maxCtrlPts, double tolerance)
{
    std::vector<Handle(Geom2d_BSplineCurve)> result;

    for (const auto& poly : polylines) {
        int n = static_cast<int>(poly.size());
        if (n < 4) { result.push_back(nullptr); continue; }

        int nCtrl1 = std::max(degree + 1,
            std::min(n - 1, std::min(maxCtrlPts, std::max(minCtrlPts, n / 3))));

        // Strategy 1: standard
        auto c = fitOneCurve(poly, degree, nCtrl1, tolerance);
        if (!c.IsNull() && validateCurve2D(c)) {
            result.push_back(c);
            continue;
        }

        // Strategy 2: fewer control points
        int nCtrl2 = std::max(degree + 1, nCtrl1 / 2);
        if (nCtrl2 < nCtrl1) {
            c = fitOneCurve(poly, degree, nCtrl2, tolerance * 2.0);
            if (!c.IsNull() && validateCurve2D(c)) {
                result.push_back(c);
                continue;
            }
        }

        // Strategy 3: lower degree
        if (degree >= 3 && n >= 4) {
            c = fitOneCurve(poly, 2,
                std::max(3, std::min(n - 1, nCtrl1 / 2)),
                tolerance * 5.0);
            if (!c.IsNull() && validateCurve2D(c)) {
                result.push_back(c);
                continue;
            }
        }

        // Fallback: return nullptr (visualization will use raw polyline)
        result.push_back(nullptr);
    }

    return result;
}

// ═══════════════════════════════════════════════════════════════════
// Step 6: Lift to 3D
// ═══════════════════════════════════════════════════════════════════

std::vector<Vec3Arr> liftCurvesTo3DPolylines(
    const std::vector<Handle(Geom2d_BSplineCurve)>& curves2d,
    const NurbsSurfaceWrapper& nurbs,
    int samplesPerCurve)
{
    std::vector<Vec3Arr> result;

    for (const auto& curve : curves2d) {
        Vec3Arr poly;
        if (curve.IsNull()) continue;

        double u1 = curve->FirstParameter();
        double u2 = curve->LastParameter();
        poly.reserve(samplesPerCurve);
        for (int i = 0; i < samplesPerCurve; ++i) {
            double t = u1 + (u2 - u1) * static_cast<double>(i) / (samplesPerCurve - 1);
            gp_Pnt2d p2d = curve->Value(t);
            poly.push_back(nurbs.evaluate(p2d.X(), p2d.Y()));
        }
        result.push_back(poly);
    }

    return result;
}

Vec3Arr liftMeshTo3D(const Vec2Arr& uvs, const NurbsSurfaceWrapper& nurbs)
{
    Vec3Arr result;
    result.reserve(uvs.size());
    for (const auto& uv : uvs)
        result.push_back(nurbs.evaluate(uv.x(), uv.y()));
    return result;
}

} // namespace distillation
