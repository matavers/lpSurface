#pragma once

#include "common.hpp"

namespace distillation {

/// Union-Find for connected components
class UnionFind {
public:
    UnionFind(int n);
    int find(int x);
    void unite(int x, int y);
private:
    std::vector<int> parent, rank;
};

/// Result of concavity analysis on a single partition
struct ConcavityInfo {
    int pid;                 // partition ID
    bool isConcave;          // macro-concave?
    int splitType;           // 0=none, 1=tip (切尖), 2=corner (切角)
    Vec2 splitP0;            // split line endpoint (UV)
    Vec2 splitP1;            // split line endpoint (UV)
};

/// Concavity analysis on smoothed boundary polylines
struct PocketInfo {
    int startIdx, endIdx;    // indices on convex hull into polyline
    std::vector<int> arc;    // boundary vertex indices forming the pocket
    double maxDepth;         // max perpendicular distance to hull edge
    double width;            // hull edge length
};

/// Detect macro-concave pockets on a 2D polygon
std::vector<PocketInfo> detectPockets(const Vec2Arr& poly);

/// Compute convex hull of a 2D polygon, returns hull vertex indices (CCW)
std::vector<int> convexHull2D(const Vec2Arr& pts);

/// For a pocket, determine split type and split line
ConcavityInfo classifyPocket(const Vec2Arr& poly, const PocketInfo& pocket, int pid);

/// Split tip: find neck points and return cut line
bool splitTip(const Vec2Arr& poly, const PocketInfo& pocket, Vec2& p0, Vec2& p1);

/// Split corner: ray from deepest point along angle bisector to opposite side
bool splitCorner(const Vec2Arr& poly, const PocketInfo& pocket, Vec2& p0, Vec2& p1);

/// Apply split to face labels: faces on each side of UV split line get different labels
/// Returns number of faces assigned to the new partition (-1 if split invalid)
int applySplit(IntArr& faceLabels, const FaceArr& faces,
               const Vec2Arr& uvs, const Vec2& p0, const Vec2& p1,
               int srcPid, int& nParts, int minFaces);

/// Main function: inspect polylines for macro-concave pockets, split as needed
/// Returns number of partitions split (0 = nothing to split)
int splitConcavePartitions(IntArr& faceLabels, const FaceArr& faces,
                           const Vec2Arr& uvs,
                           const std::vector<Vec2Arr>& polylines,
                           int& nParts, double depthRatioThreshold = 0.3,
                           int minFaces = 4);

} // namespace distillation
