#pragma once

#include "common.hpp"
#include "nurbs_surface_wrapper.hpp"

#include <memory>

namespace distillation {

class NurbsMeshProcessor {
public:
    NurbsMeshProcessor(const Vec3Arr& vertices, const FaceArr& faces,
                       const NurbsSurfaceWrapper& nurbs,
                       const Vec2Arr& uvs = {});

    int numVertices() const { return static_cast<int>(m_vertices.size()); }
    const Vec3Arr& vertices() const { return m_vertices; }
    const FaceArr& faces() const { return m_faces; }
    const Vec2Arr& uvs() const { return m_uvs; }

    const Vec3Arr& vertexNormals() const { return m_vertexNormals; }
    const Vec3Arr& principalDirections1() const { return m_principalDirs1; }
    const Vec3Arr& principalDirections2() const { return m_principalDirs2; }
    const MatX& principalCurvatures() const { return m_principalCurvatures; }
    const ArrX& gaussianCurvatures() const { return m_gaussianCurvatures; }
    const ArrX& meanCurvatures() const { return m_meanCurvatures; }
    const ArrX& faceAreas() const { return m_faceAreas; }
    double avgEdgeLength() const { return m_avgEdgeLength; }

    const std::vector<IntArr>& adjacency() const { return m_adjacency; }
    const std::vector<std::pair<int, int>>& edgeVertices() const { return m_edgeVertices; }

private:
    void buildAdjacency();
    void computeAnalyticalGeometry();
    void computeFaceAreas();

    Vec3Arr m_vertices;
    FaceArr m_faces;
    Vec2Arr m_uvs;
    const NurbsSurfaceWrapper& m_nurbs;

    Vec3Arr m_vertexNormals;
    Vec3Arr m_principalDirs1;
    Vec3Arr m_principalDirs2;
    MatX m_principalCurvatures;
    ArrX m_gaussianCurvatures;
    ArrX m_meanCurvatures;
    ArrX m_faceAreas;
    double m_avgEdgeLength = 0.0;

    std::vector<IntArr> m_adjacency;
    std::vector<std::pair<int, int>> m_edgeVertices;
};

} // namespace distillation
