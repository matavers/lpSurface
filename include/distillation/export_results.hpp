#pragma once

#include "common.hpp"
#include "boundary_smoother.hpp"
#include "tolerance.hpp"

#include <fstream>
#include <string>
#include <vector>

namespace distillation {

bool exportOBJ(const std::string& path, const Vec3Arr& vertices, const FaceArr& faces);

bool exportMeshUV(const std::string& path, const Vec2Arr& uvs);

bool exportPartitionLabels(const std::string& path, int numVertices, const IntVecSet& partitions);

bool exportPartitionLabels(const std::string& path, int numVertices, const IntArr& labels);

bool exportFaceLabels(const std::string& path, const IntArr& labels);

bool exportBoundaries(const std::string& path,
                      const std::map<int, std::vector<Vec3Arr>>& boundaries);

bool exportCorners(const std::string& path, const Vec3Arr& corners);

bool exportBoundaryCurves3D(const std::string& path,
                            const std::vector<Vec3Arr>& curves3d);

bool exportBoundaryCurves2D(const std::string& path,
                            const std::vector<std::vector<Vec2>>& curves2d);

bool exportSmoothHistory(const std::string& path,
                         const std::vector<SmoothIterationEntry>& history);

bool exportBoundaryPolylinesIter(const std::string& path,
                                 const std::vector<std::vector<Vec2>>& polylines2D,
                                 const NurbsSurfaceWrapper& nurbs,
                                 int iter);

bool exportToleranceData(const std::string& path,
                         const std::vector<TolResult>& tolerances);

bool exportRuledSurface(const std::string& path,
                        const Handle(Geom2d_BSplineCurve)& C0,
                        const Handle(Geom2d_BSplineCurve)& C1,
                        const NurbsSurfaceWrapper& nurbs,
                        double bestAngleDeg,
                        int samplesU = 30, int samplesV = 12);

} // namespace distillation
