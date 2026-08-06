#include "distillation/nurbs_mesh_processor.hpp"

namespace distillation {

NurbsMeshProcessor::NurbsMeshProcessor(const Vec3Arr& vertices, const FaceArr& faces,
                                       const NurbsSurfaceWrapper& nurbs,
                                       const Vec2Arr& uvs)
    : m_vertices(vertices), m_faces(faces), m_nurbs(nurbs)
{
    int nV = numVertices();

    if (!uvs.empty()) {
        m_uvs = uvs;
    } else {
        m_uvs.resize(nV);
        for (int i = 0; i < nV; ++i) {
            auto proj = m_nurbs.projectToUV(m_vertices[i]);
            m_uvs[i] = Vec2(proj.u, proj.v);
        }
    }

    buildAdjacency();
    computeAnalyticalGeometry();
    computeFaceAreas();
}

void NurbsMeshProcessor::buildAdjacency() {
    int nV = numVertices();
    m_adjacency.assign(nV, IntArr{});

    std::unordered_set<uint64_t> edgeSet;
    for (const auto& f : m_faces) {
        for (int i = 0; i < 3; ++i) {
            int a = f[i], b = f[(i + 1) % 3];
            if (a > b) std::swap(a, b);
            uint64_t key = (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
            if (edgeSet.insert(key).second) {
                m_adjacency[a].push_back(b);
                m_adjacency[b].push_back(a);
            }
        }
    }

    m_edgeVertices.clear();
    for (uint64_t key : edgeSet) {
        int a = static_cast<int>(key >> 32);
        int b = static_cast<int>(key & 0xFFFFFFFF);
        m_edgeVertices.push_back({a, b});
    }

    double total = 0.0;
    for (const auto& e : m_edgeVertices) {
        total += (m_vertices[e.first] - m_vertices[e.second]).norm();
    }
    m_avgEdgeLength = m_edgeVertices.empty() ? 1.0 : total / m_edgeVertices.size();
}

void NurbsMeshProcessor::computeAnalyticalGeometry() {
    int nV = numVertices();
    m_vertexNormals.resize(nV);
    m_principalDirs1.resize(nV);
    m_principalDirs2.resize(nV);
    m_principalCurvatures.resize(nV, 2);
    m_gaussianCurvatures.resize(nV);
    m_meanCurvatures.resize(nV);

    for (int i = 0; i < nV; ++i) {
        double u = m_uvs[i].x(), v = m_uvs[i].y();
        try {
            m_vertexNormals[i] = m_nurbs.normal(u, v);
            auto c = m_nurbs.principalCurvatures(u, v);
            m_principalCurvatures(i, 0) = c.k1;
            m_principalCurvatures(i, 1) = c.k2;
            m_principalDirs1[i] = c.d1;
            m_principalDirs2[i] = c.d2;
            m_gaussianCurvatures[i] = c.k1 * c.k2;
            m_meanCurvatures[i] = (c.k1 + c.k2) * 0.5;
        } catch (...) {
            m_vertexNormals[i] = Vec3(0, 0, 1);
            m_principalCurvatures(i, 0) = m_principalCurvatures(i, 1) = 0.0;
            m_principalDirs1[i] = Vec3(1, 0, 0);
            m_principalDirs2[i] = Vec3(0, 1, 0);
            m_gaussianCurvatures[i] = 0.0;
            m_meanCurvatures[i] = 0.0;
        }
    }
}

void NurbsMeshProcessor::computeFaceAreas() {
    int nF = static_cast<int>(m_faces.size());
    m_faceAreas.resize(nF);
    for (int i = 0; i < nF; ++i) {
        const auto& f = m_faces[i];
        const Vec3& v0 = m_vertices[f.v0];
        const Vec3& v1 = m_vertices[f.v1];
        const Vec3& v2 = m_vertices[f.v2];
        m_faceAreas[i] = 0.5 * (v1 - v0).cross(v2 - v0).norm();
    }
}

} // namespace distillation
