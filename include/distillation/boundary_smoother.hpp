#pragma once

#include "common.hpp"
#include "nurbs_surface_wrapper.hpp"

#include <Geom2d_BSplineCurve.hxx>
#include <Geom_BSplineCurve.hxx>

namespace distillation {

struct SmoothIterationEntry {
    int iteration;
    double maxDisplacement;
    double avgDisplacement;
    int nBoundaryVertices;
};

struct BoundaryEdge {
    int v0, v1;
    int face0, face1;
};

struct BoundaryNetwork {
    Vec2Arr smoothedUVs;
    std::vector<BoundaryEdge> edges;
    std::vector<IntArr> neighbors;
    IntArr globalToLocal;
    IntArr localToGlobal;
    std::vector<bool> isExternal;
    double avgEdgeLength = 0.0;
};

IntArr convertVertexLabelsToFaceLabels(
    const IntVecSet& vertexPartitions,
    const FaceArr& faces,
    int numVertices);

void mergeTinyRegions(
    IntArr& faceLabels,
    const FaceArr& faces,
    int minRegionFaces = 0);

BoundaryNetwork extractBoundaryNetwork(
    const Vec2Arr& uvs,
    const FaceArr& faces,
    const IntArr& faceLabels,
    double uMin, double uMax,
    double vMin, double vMax);

Vec2 projectToDomainBoundary(
    const Vec2& pt,
    double uMin, double uMax,
    double vMin, double vMax);

std::vector<SmoothIterationEntry> laplacianSmoothBoundary(
    BoundaryNetwork& net,
    double uMin, double uMax,
    double vMin, double vMax,
    double sigmaTarget,
    double tolerance = 1e-6);

SmoothIterationEntry laplacianSmoothSingle(
    BoundaryNetwork& net,
    double uMin, double uMax,
    double vMin, double vMax);

std::vector<std::vector<Vec2>> extractPolylinesFromNetwork(
    const BoundaryNetwork& net);

std::vector<std::vector<Vec2>> extractBoundaryPolylines(
    const Vec2Arr& uvs,
    const FaceArr& faces,
    const IntArr& faceLabels);

bool repairMeshAfterSmoothing(
    Vec2Arr& uvs,
    FaceArr& faces,
    IntArr& faceLabels,
    const BoundaryNetwork& net);

void harmonicMeshUpdate(
    Vec2Arr& uvs,
    const FaceArr& faces,
    const BoundaryNetwork& net);

std::vector<Handle(Geom2d_BSplineCurve)> fitBoundaryCurves(
    const std::vector<std::vector<Vec2>>& polylines,
    int degree = 3, int minCtrlPts = 6,
    int maxCtrlPts = 20, double tolerance = 0.001);

std::vector<Vec3Arr> liftCurvesTo3DPolylines(
    const std::vector<Handle(Geom2d_BSplineCurve)>& curves2d,
    const NurbsSurfaceWrapper& nurbs,
    int samplesPerCurve = 200);

Vec3Arr liftMeshTo3D(
    const Vec2Arr& uvs,
    const NurbsSurfaceWrapper& nurbs);

} // namespace distillation
