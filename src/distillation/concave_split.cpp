#include "distillation/concave_split.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <queue>
#include <set>
#include <stack>

namespace distillation {

// ═════════════════════════════════════════════════════════════════
// Union-Find
// ═════════════════════════════════════════════════════════════════
UnionFind::UnionFind(int n) : parent(n), rank(n, 0) {
    for (int i = 0; i < n; ++i) parent[i] = i;
}
int UnionFind::find(int x) {
    if (parent[x] != x) parent[x] = find(parent[x]);
    return parent[x];
}
void UnionFind::unite(int x, int y) {
    x = find(x); y = find(y);
    if (x == y) return;
    if (rank[x] < rank[y]) parent[x] = y;
    else if (rank[x] > rank[y]) parent[y] = x;
    else { parent[y] = x; rank[x]++; }
}

// ═════════════════════════════════════════════════════════════════
// 2D convex hull (Andrew's monotone chain)
// ═════════════════════════════════════════════════════════════════
static double cross2D(const Vec2& o, const Vec2& a, const Vec2& b) {
    return (a.x() - o.x()) * (b.y() - o.y()) - (a.y() - o.y()) * (b.x() - o.x());
}

std::vector<int> convexHull2D(const Vec2Arr& pts) {
    int n = (int)pts.size();
    if (n < 3) { std::vector<int> all(n); for (int i = 0; i < n; ++i) all[i] = i; return all; }

    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        if (pts[a].x() != pts[b].x()) return pts[a].x() < pts[b].x();
        return pts[a].y() < pts[b].y();
    });

    std::vector<int> hull;
    for (int i = 0; i < n; ++i) {
        while (hull.size() >= 2 && cross2D(pts[hull[hull.size()-2]], pts[hull.back()], pts[idx[i]]) <= 0)
            hull.pop_back();
        hull.push_back(idx[i]);
    }
    int lowerSize = (int)hull.size();
    for (int i = n - 2; i >= 0; --i) {
        while ((int)hull.size() > lowerSize && cross2D(pts[hull[hull.size()-2]], pts[hull.back()], pts[idx[i]]) <= 0)
            hull.pop_back();
        hull.push_back(idx[i]);
    }
    if (hull.size() > 1) hull.pop_back(); // last == first
    return hull;
}

// ═════════════════════════════════════════════════════════════════
// Pocket detection
// ═════════════════════════════════════════════════════════════════
static double pointToLineDist(const Vec2& p, const Vec2& a, const Vec2& b) {
    Vec2 ab = b - a;
    double len2 = ab.x() * ab.x() + ab.y() * ab.y();
    if (len2 < 1e-12) return (p - a).norm();
    double t = ((p.x() - a.x()) * ab.x() + (p.y() - a.y()) * ab.y()) / len2;
    t = std::max(0.0, std::min(1.0, t));
    Vec2 proj(a.x() + t * ab.x(), a.y() + t * ab.y());
    return (p - proj).norm();
}

std::vector<PocketInfo> detectPockets(const Vec2Arr& poly) {
    std::vector<PocketInfo> pockets;
    int n = (int)poly.size();
    if (n < 4) return pockets;

    auto hull = convexHull2D(poly);
    int h = (int)hull.size();
    if (h < 3) return pockets;

    // Build hull vertex -> polygon index map
    std::map<int, int> hullPos; // polygon index -> hull position
    for (int i = 0; i < h; ++i) hullPos[hull[i]] = i;

    // For each hull edge, trace the polygon arc between the two hull vertices
    for (int hi = 0; hi < h; ++hi) {
        int hnext = (hi + 1) % h;
        int pA = hull[hi], pB = hull[hnext];

        // Find arc from pA to pB along polygon (forward direction)
        std::vector<int> arc;
        int cur = (pA + 1) % n;
        while (cur != pB) {
            arc.push_back(cur);
            cur = (cur + 1) % n;
            if (arc.size() > (size_t)n) break; // safety
        }

        if (arc.empty()) {
            // pA and pB are consecutive on polygon → part of hull edge, no pocket
            continue;
        }

        // Measure pocket depth
        double maxDist = 0.0;
        int deepestIdx = -1;
        for (int vi : arc) {
            double d = pointToLineDist(poly[vi], poly[pA], poly[pB]);
            if (d > maxDist) { maxDist = d; deepestIdx = vi; }
        }

        double width = (poly[pA] - poly[pB]).norm();
        if (width < 1e-8) continue;

        PocketInfo pkt;
        pkt.startIdx = pA;
        pkt.endIdx = pB;
        pkt.arc = std::move(arc);
        pkt.maxDepth = maxDist;
        pkt.width = width;
        pockets.push_back(pkt);
    }

    return pockets;
}

// ═════════════════════════════════════════════════════════════════
// Split tip (切尖): find neck in pocket arc, cut across
// ═════════════════════════════════════════════════════════════════
bool splitTip(const Vec2Arr& poly, const PocketInfo& pocket, Vec2& p0, Vec2& p1) {
    const auto& arc = pocket.arc;
    int m = (int)arc.size();
    if (m < 4) return false;

    // Evaluate all chord pairs (arc[i], arc[j]) and find the neck:
    // minimum chord_length / max_arc_depth
    double bestRatio = 1e9;
    int bestI = -1, bestJ = -1;

    for (int i = 0; i < m; ++i) {
        for (int j = i + 3; j < m; ++j) { // at least 3 vertices apart
            Vec2 chord = poly[arc[j]] - poly[arc[i]];
            double chordLen = chord.norm();
            if (chordLen < 1e-8) continue;

            // Max depth of the arc segment between i and j relative to chord
            double maxD = 0.0;
            for (int k = i + 1; k < j; ++k) {
                double d = pointToLineDist(poly[arc[k]], poly[arc[i]], poly[arc[j]]);
                if (d > maxD) maxD = d;
            }

            if (maxD > 0) {
                double ratio = chordLen / maxD;
                if (ratio < bestRatio) {
                    bestRatio = ratio;
                    bestI = i; bestJ = j;
                }
            }
        }
    }

    if (bestI < 0 || bestJ < 0) return false;
    p0 = poly[arc[bestI]];
    p1 = poly[arc[bestJ]];
    return true;
}

// ═════════════════════════════════════════════════════════════════
// Split corner (切角): angle bisector from deepest point
// ═════════════════════════════════════════════════════════════════
bool splitCorner(const Vec2Arr& poly, const PocketInfo& pocket, Vec2& p0, Vec2& p1) {
    const auto& arc = pocket.arc;
    if (arc.empty()) return false;

    // Find deepest point on the arc
    double maxD = -1;
    int deepestIdx = -1;
    for (int vi : arc) {
        double d = pointToLineDist(poly[vi], poly[pocket.startIdx], poly[pocket.endIdx]);
        if (d > maxD) { maxD = d; deepestIdx = vi; }
    }
    if (deepestIdx < 0) return false;

    int n = (int)poly.size();
    Vec2 vDeep = poly[deepestIdx];

    // Tangent direction at deepest point: average of incoming and outgoing edges
    Vec2 vPrev = poly[(deepestIdx + n - 1) % n];
    Vec2 vNext = poly[(deepestIdx + 1) % n];
    Vec2 inDir = (vDeep - vPrev); inDir = inDir / (inDir.norm() + 1e-12);
    Vec2 outDir = (vNext - vDeep); outDir = outDir / (outDir.norm() + 1e-12);

    // Angle bisector pointing INTO the polygon (internal)
    Vec2 bisector = inDir + outDir;
    double len = bisector.norm();
    if (len < 1e-12) return false;
    bisector = bisector / len;

    // Ensure bisector points into the polygon (not outward)
    // The pocket is on one side of the hull edge. Check: cross(hullEdge, bisector) sign
    Vec2 hullEdge = poly[pocket.endIdx] - poly[pocket.startIdx];
    if (bisector.x() * hullEdge.y() - bisector.y() * hullEdge.x() < 0)
        bisector = Vec2(-bisector.x(), -bisector.y());

    // Ray-cast: find intersection with opposite boundary
    double maxRay = (poly[pocket.startIdx] - poly[pocket.endIdx]).norm() * 2.0;
    Vec2 rayEnd(vDeep.x() + bisector.x() * maxRay, vDeep.y() + bisector.y() * maxRay);

    double bestT = 1e9;
    Vec2 bestHit;
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        // Skip edges near the deepest point
        if (i == deepestIdx || j == deepestIdx) continue;
        if (i == (deepestIdx + n - 1) % n || j == (deepestIdx + 1) % n) continue;

        // Line segment intersection
        Vec2 s1 = poly[i], e1 = poly[j];
        Vec2 d1 = e1 - s1;
        Vec2 d2 = rayEnd - vDeep;

        double cross = d1.x() * d2.y() - d1.y() * d2.x();
        if (std::abs(cross) < 1e-12) continue;

        double t = ((vDeep.x() - s1.x()) * d2.y() - (vDeep.y() - s1.y()) * d2.x()) / cross;
        double u = ((vDeep.x() - s1.x()) * d1.y() - (vDeep.y() - s1.y()) * d1.x()) / cross;

        if (t >= 0.0 && t <= 1.0 && u > 1e-3) {
            if (u < bestT) {
                bestT = u;
                bestHit = Vec2(vDeep.x() + u * d2.x(), vDeep.y() + u * d2.y());
            }
        }
    }

    if (bestT > 1e8) return false;
    p0 = vDeep;
    p1 = bestHit;
    return true;
}

// ═════════════════════════════════════════════════════════════════
// Classify pocket and generate split line
// ═════════════════════════════════════════════════════════════════
ConcavityInfo classifyPocket(const Vec2Arr& poly, const PocketInfo& pocket, int pid) {
    ConcavityInfo info;
    info.pid = pid;
    info.isConcave = false;
    info.splitType = 0;

    double ratio = pocket.width > 1e-8 ? pocket.maxDepth / pocket.width : 0;
    if (ratio < 0.3) return info;

    // Determine type: check winding of the arc relative to polygon
    // For a pocket, the arc is "inside" relative to the hull edge
    // If arc wrapping is outward → tip (切尖)
    // If arc wrapping is inward → corner (切角)

    // Compute area sign of the pocket polygon (arc + hull edge)
    double area2 = 0;
    Vec2 pA = poly[pocket.startIdx], pB = poly[pocket.endIdx];
    area2 += pA.x() * pB.y() - pA.y() * pB.x();
    for (int vi : pocket.arc) {
        const Vec2& p = poly[vi];
        area2 += pB.x() * p.y() - pB.y() * p.x();
        pB = p;
    }
    // positively oriented = tip pocket (convex pocket outward)
    // negatively oriented = corner pocket (concave pocket inward)

    if (area2 > 0) {
        // Tip: narrow neck
        if (splitTip(poly, pocket, info.splitP0, info.splitP1)) {
            info.splitType = 1;
            info.isConcave = true;
        }
    } else {
        // Corner: angle bisector
        if (splitCorner(poly, pocket, info.splitP0, info.splitP1)) {
            info.splitType = 2;
            info.isConcave = true;
        }
    }

    return info;
}

// ═════════════════════════════════════════════════════════════════
// Apply split to face labels
// ═════════════════════════════════════════════════════════════════
static int lineSide(const Vec2& p, const Vec2& a, const Vec2& b) {
    double cross = (b.x() - a.x()) * (p.y() - a.y()) - (b.y() - a.y()) * (p.x() - a.x());
    if (cross > 1e-12) return 1;
    if (cross < -1e-12) return -1;
    return 0;
}

int applySplit(IntArr& faceLabels, const FaceArr& faces,
               const Vec2Arr& uvs, const Vec2& p0, const Vec2& p1,
               int srcPid, int& nParts, int minFaces) {
    int nFaces = (int)faces.size();
    int newPid = nParts;

    // Count faces on each side and assign
    std::vector<int> side(nFaces, 0);
    int countLeft = 0, countRight = 0;

    for (int fi = 0; fi < nFaces; ++fi) {
        if (faceLabels[fi] != srcPid) continue;
        const Face& f = faces[fi];
        Vec2 c((uvs[f.v0].x() + uvs[f.v1].x() + uvs[f.v2].x()) / 3.0,
               (uvs[f.v0].y() + uvs[f.v1].y() + uvs[f.v2].y()) / 3.0);
        int s = lineSide(c, p0, p1);
        side[fi] = s;
        if (s > 0) countRight++;
        else countLeft++;
    }

    if (countLeft < minFaces || countRight < minFaces) return -1;

    // Apply: left side keeps srcPid, right side gets newPid
    for (int fi = 0; fi < nFaces; ++fi) {
        if (faceLabels[fi] != srcPid) continue;
        if (side[fi] > 0) faceLabels[fi] = newPid;
    }

    nParts++;
    return countRight;
}

// ═════════════════════════════════════════════════════════════════
// Polygon area
// ═════════════════════════════════════════════════════════════════
static double polygonArea(const Vec2Arr& poly) {
    int n = (int)poly.size();
    double a = 0.0;
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        a += poly[i].x() * poly[j].y() - poly[j].x() * poly[i].y();
    }
    return std::abs(a) / 2.0;
}


// ═════════════════════════════════════════════════════════════════
// Main split function
// ═════════════════════════════════════════════════════════════════
int splitConcavePartitions(IntArr& faceLabels, const FaceArr& faces,
                           const Vec2Arr& uvs,
                           const std::vector<Vec2Arr>& polylines,
                           int& nParts, double depthRatioThreshold,
                           int minFaces) {
    int nPoly = (int)polylines.size();
    int totalSplits = 0;

    std::cout << "  [Concave] scanning " << std::min(nParts, nPoly) << " partitions (threshold="
              << depthRatioThreshold << "):\n";
    int nConcave = 0, nSkipped = 0;

    for (int pid = 0; pid < nParts && pid < nPoly; ++pid) {
        const auto& poly = polylines[pid];
        if (poly.size() < 4) { nSkipped++; continue; }

        // Macro-concavity: area deficit = 1 - area(poly)/area(hull)
        double pa = polygonArea(poly);
        auto hullIdx = convexHull2D(poly);
        Vec2Arr hullPts; for (int hi : hullIdx) hullPts.push_back(poly[hi]);
        double ha = polygonArea(hullPts);
        double deficit = (ha > 1e-12) ? (1.0 - pa / ha) : 0.0;

        if (deficit < depthRatioThreshold) continue;

        // Detect pockets for split line generation
        auto pockets = detectPockets(poly);
        if (pockets.empty()) continue;

        // Find deepest pocket with bounding-box normalization
        double xmin = poly[0].x(), xmax = poly[0].x();
        double ymin = poly[0].y(), ymax = poly[0].y();
        for (size_t i = 1; i < poly.size(); ++i) {
            if (poly[i].x() < xmin) xmin = poly[i].x();
            if (poly[i].x() > xmax) xmax = poly[i].x();
            if (poly[i].y() < ymin) ymin = poly[i].y();
            if (poly[i].y() > ymax) ymax = poly[i].y();
        }
        double bbDiag = std::sqrt((xmax-xmin)*(xmax-xmin) + (ymax-ymin)*(ymax-ymin));
        int bestPkt = -1;
        double bestDepth = 0.0;
        for (int pi = 0; pi < (int)pockets.size(); ++pi) {
            if (pockets[pi].maxDepth > bestDepth) { bestDepth = pockets[pi].maxDepth; bestPkt = pi; }
        }
        if (bestPkt < 0) continue;

        std::cout << "    part " << pid << ": deficit=" << deficit
                  << " (depth/bb=" << bestDepth/bbDiag << ") -> macro-concave\n";

        // Classify and get split line
        ConcavityInfo info = classifyPocket(poly, pockets[bestPkt], pid);
        if (!info.isConcave) continue;

        nConcave++;

        // Apply split
        int nAssigned = applySplit(faceLabels, faces, uvs,
                                   info.splitP0, info.splitP1,
                                   pid, nParts, minFaces);
        if (nAssigned > 0) {
            totalSplits++;
            std::cout << "    -> Split partition " << pid << " (type="
                      << (info.splitType == 1 ? "tip" : "corner")
                      << "), " << nAssigned << " faces -> partition " << (nParts - 1)
                      << "\n";
        }
    }

    if (nSkipped > 0)
        std::cout << "  [Concave] " << nSkipped << " partitions skipped (too small)\n";
    if (totalSplits == 0 && nConcave == 0)
        std::cout << "  [Concave] no macro-concave partitions need splitting\n";
    else if (totalSplits == 0)
        std::cout << "  [Concave] " << nConcave << " macro-concave found but split failed\n";

    return totalSplits;
}

} // namespace distillation
