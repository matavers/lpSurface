#pragma once

#include "common.hpp"
#include "nurbs_mesh_processor.hpp"

#include <AIS_InteractiveContext.hxx>
#include <V3d_Viewer.hxx>
#include <AIS_Shape.hxx>

namespace distillation {

class PartitionViewer {
public:
    PartitionViewer(const Vec3Arr& vertices, const FaceArr& faces,
                    const std::vector<IterationEntry>& iterHistory,
                    const IntVecSet& partitionsOrdered,
                    const std::vector<Vec3Arr>& boundaryCurves = {},
                    const Vec3Arr& cornerPoints = {});
    void show();

private:
    Vec3Arr m_vertices;
    FaceArr m_faces;
    std::vector<IterationEntry> m_iterHistory;
    IntVecSet m_partitions;
    std::vector<Vec3Arr> m_boundaryCurves;
    Vec3Arr m_cornerPoints;

    int m_nIters, m_nParts;
    Handle(V3d_Viewer) m_viewer;
    Handle(AIS_InteractiveContext) m_context;
};

} // namespace distillation
