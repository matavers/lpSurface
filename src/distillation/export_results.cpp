#include "distillation/export_results.hpp"
#include <filesystem>
#include <array>

namespace distillation {

static bool ensureDir(const std::string& path) {
    std::filesystem::path p(path);
    auto dir = p.parent_path();
    if (!dir.empty() && !std::filesystem::exists(dir)) {
        std::filesystem::create_directories(dir);
    }
    return true;
}

bool exportOBJ(const std::string& path, const Vec3Arr& vertices, const FaceArr& faces) {
    ensureDir(path);
    std::ofstream out(path);
    if (!out) { std::cerr << "Cannot write " << path << std::endl; return false; }
    out.precision(10);
    for (const auto& v : vertices)
        out << "v " << v.x() << " " << v.y() << " " << v.z() << "\n";
    for (const auto& f : faces)
        out << "f " << (f.v0 + 1) << " " << (f.v1 + 1) << " " << (f.v2 + 1) << "\n";
    std::cout << "  wrote " << path << " (" << vertices.size() << " verts, "
              << faces.size() << " faces)\n";
    return true;
}

bool exportPartitionLabels(const std::string& path, int numVertices, const IntVecSet& partitions) {
    std::ofstream out(path);
    if (!out) { std::cerr << "Cannot write " << path << std::endl; return false; }

    VecXi labels = VecXi::Constant(numVertices, -1);
    for (int pid = 0; pid < static_cast<int>(partitions.size()); ++pid) {
        for (int v : partitions[pid]) {
            if (v >= 0 && v < numVertices)
                labels[v] = pid;
        }
    }

    for (int i = 0; i < numVertices; ++i)
        out << labels[i] << "\n";

    int assigned = (labels.array() >= 0).count();
    std::cout << "  wrote " << path << " (" << numVertices << " verts, "
              << assigned << " assigned)\n";
    return true;
}

bool exportPartitionLabels(const std::string& path, int numVertices, const IntArr& labels) {
    std::ofstream out(path);
    if (!out) { std::cerr << "Cannot write " << path << std::endl; return false; }
    for (int i = 0; i < numVertices; ++i)
        out << (i < static_cast<int>(labels.size()) ? labels[i] : -1) << "\n";
    std::cout << "  wrote " << path << " (" << numVertices << " verts)\n";
    return true;
}

bool exportFaceLabels(const std::string& path, const IntArr& labels) {
    std::ofstream out(path);
    if (!out) { std::cerr << "Cannot write " << path << std::endl; return false; }
    for (int lbl : labels) out << lbl << "\n";
    std::cout << "  wrote " << path << " (" << labels.size() << " faces)\n";
    return true;
}

bool exportBoundaries(const std::string& path,
                      const std::map<int, std::vector<Vec3Arr>>& boundaries) {
    std::ofstream out(path);
    if (!out) { std::cerr << "Cannot write " << path << std::endl; return false; }
    out.precision(10);
    for (const auto& [pid, segments] : boundaries) {
        for (const auto& pts : segments) {
            out << pts.size() << " " << pid << "\n";
            for (const auto& p : pts)
                out << p.x() << " " << p.y() << " " << p.z() << "\n";
        }
    }
    std::cout << "  wrote " << path << std::endl;
    return true;
}

bool exportCorners(const std::string& path, const Vec3Arr& corners) {
    std::ofstream out(path);
    if (!out) { std::cerr << "Cannot write " << path << std::endl; return false; }
    out.precision(10);
    for (const auto& c : corners)
        out << c.x() << " " << c.y() << " " << c.z() << "\n";
    std::cout << "  wrote " << path << " (" << corners.size() << " corners)\n";
    return true;
}

bool exportBoundaryCurves3D(const std::string& path,
                            const std::vector<Vec3Arr>& curves3d) {
    ensureDir(path);
    std::ofstream out(path);
    if (!out) { std::cerr << "Cannot write " << path << std::endl; return false; }
    out.precision(10);
    for (size_t ci = 0; ci < curves3d.size(); ++ci) {
        const auto& pts = curves3d[ci];
        out << pts.size() << " " << ci << "\n";
        for (const auto& p : pts)
            out << p.x() << " " << p.y() << " " << p.z() << "\n";
    }
    std::cout << "  wrote " << path << " (" << curves3d.size() << " curves)\n";
    return true;
}

bool exportBoundaryCurves2D(const std::string& path,
                            const std::vector<std::vector<Vec2>>& curves2d) {
    ensureDir(path);
    std::ofstream out(path);
    if (!out) { std::cerr << "Cannot write " << path << std::endl; return false; }
    out.precision(10);
    for (size_t ci = 0; ci < curves2d.size(); ++ci) {
        const auto& pts = curves2d[ci];
        out << pts.size() << " " << ci << "\n";
        for (const auto& p : pts)
            out << p.x() << " " << p.y() << "\n";
    }
    std::cout << "  wrote " << path << " (" << curves2d.size() << " curves)\n";
    return true;
}

bool exportSmoothHistory(const std::string& path,
                         const std::vector<SmoothIterationEntry>& history) {
    ensureDir(path);
    std::ofstream out(path);
    if (!out) { std::cerr << "Cannot write " << path << std::endl; return false; }
    out.precision(10);
    out << "iter max_displacement avg_displacement n_boundary_vertices\n";
    for (const auto& h : history) {
        out << h.iteration << " "
            << h.maxDisplacement << " "
            << h.avgDisplacement << " "
            << h.nBoundaryVertices << "\n";
    }
    if (!history.empty())
        std::cout << "  wrote " << path << " (" << history.size() << " iterations)\n";
    return true;
}

bool exportBoundaryPolylinesIter(const std::string& path,
                                 const std::vector<std::vector<Vec2>>& polylines2D,
                                 const NurbsSurfaceWrapper& nurbs,
                                 int iter)
{
    ensureDir(path);
    std::ofstream out(path);
    if (!out) { std::cerr << "Cannot write " << path << std::endl; return false; }
    out.precision(10);

    for (size_t ci = 0; ci < polylines2D.size(); ++ci) {
        const auto& poly = polylines2D[ci];
        if (poly.size() < 2) continue;
        out << poly.size() << " " << ci << "\n";
        for (const auto& uv : poly) {
            Vec3 p3d = nurbs.evaluate(uv.x(), uv.y());
            out << p3d.x() << " " << p3d.y() << " " << p3d.z() << "\n";
        }
    }
    return true;
}

bool exportToleranceData(const std::string& path,
                         const std::vector<TolResult>& tolerances)
{
    ensureDir(path);
    std::ofstream out(path);
    if (!out) { std::cerr << "Cannot write " << path << std::endl; return false; }
    out.precision(6);
    out << "pair_id maxDist rmsDist bestAngleDeg\n";
    for (size_t i = 0; i < tolerances.size(); ++i) {
        out << i << " "
            << tolerances[i].maxDist << " "
            << tolerances[i].rmsDist << " "
            << tolerances[i].bestAngleDeg << "\n";
    }
    std::cout << "  wrote " << path << " (" << tolerances.size() << " pairs)\n";
    return true;
}

bool exportRuledSurface(const std::string& path,
                        const Handle(Geom2d_BSplineCurve)& C0,
                        const Handle(Geom2d_BSplineCurve)& C1,
                        const NurbsSurfaceWrapper& nurbs,
                        double bestAngleDeg,
                        int samplesU, int samplesV)
{
    ensureDir(path);
    if (C0.IsNull() || C1.IsNull()) return false;

    double ua = C0->FirstParameter(), ub = C0->LastParameter();
    double va = C1->FirstParameter(), vb = C1->LastParameter();
    double rad = bestAngleDeg * M_PI / 180.0;
    double dx = cos(rad), dy = sin(rad);

    // Build vertex grid: (samplesU+1) x (samplesV+1)
    std::vector<Vec3> verts;
    std::vector<std::array<int, 4>> quads;

    for (int i = 0; i <= samplesU; ++i) {
        double t0 = ua + (ub - ua) * double(i) / samplesU;
        gp_Pnt2d p0 = C0->Value(t0);

        // Find C1 point along direction
        double bestCos = -2.0, bestT = va;
        for (int k = 0; k <= 200; ++k) {
            double t = va + (vb - va) * double(k) / 200;
            gp_Pnt2d p = C1->Value(t);
            double tox = p.X() - p0.X(), toy = p.Y() - p0.Y();
            double len = std::sqrt(tox * tox + toy * toy);
            if (len < 1e-12) continue;
            double cosA = (tox * dx + toy * dy) / len;
            if (cosA > bestCos) { bestCos = cosA; bestT = t; }
        }
        gp_Pnt2d p1 = C1->Value(bestT);

        for (int j = 0; j <= samplesV; ++j) {
            double lamb = double(j) / samplesV;
            double u = p0.X() + lamb * (p1.X() - p0.X());
            double v = p0.Y() + lamb * (p1.Y() - p0.Y());
            verts.push_back(nurbs.evaluate(u, v));
        }
    }

    int cols = samplesV + 1;
    for (int i = 0; i < samplesU; ++i)
        for (int j = 0; j < samplesV; ++j) {
            int a = i * cols + j;
            int b = a + 1;
            int c = (i + 1) * cols + j;
            int d = c + 1;
            quads.push_back({a, b, d, c});
        }

    std::ofstream out(path);
    if (!out) return false;
    out.precision(8);
    for (const auto& v : verts)
        out << "v " << v.x() << " " << v.y() << " " << v.z() << "\n";
    for (const auto& q : quads)
        out << "f " << (q[0] + 1) << " " << (q[1] + 1) << " "
            << (q[2] + 1) << " " << (q[3] + 1) << "\n";
    return true;
}

} // namespace distillation
