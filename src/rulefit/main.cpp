#include "rulefit/surface.hpp"
#include "rulefit/direction_field.hpp"
#include "rulefit/developable_fit.hpp"
#include "rulefit/partition.hpp"
#include "rulefit/evaluate.hpp"

#include <fstream>
#include <filesystem>

using namespace distillation;

static void ensureDir(const std::string& path) {
    std::filesystem::create_directories(path);
}

static void exportMeshOBJ(const std::string& path, const Vec3Arr& verts, const FaceArr& faces) {
    std::ofstream out(path);
    if (!out) return;
    out.precision(10);
    for (const auto& v : verts) out << "v " << v.x() << " " << v.y() << " " << v.z() << "\n";
    for (const auto& f : faces) out << "f " << (f.v0 + 1) << " " << (f.v1 + 1) << " " << (f.v2 + 1) << "\n";
}

static void exportFaceLabels(const std::string& path, const IntArr& labels) {
    std::ofstream out(path);
    for (int l : labels) out << l << "\n";
}

static void exportDirectionField(const std::string& path, const Vec3Arr& verts,
                                 const rulefit::DirectionField& field) {
    std::ofstream out(path);
    if (!out) return;
    out.precision(8);
    out << "x y z ax1 ay1 az1 ax2 ay2 az2 hyperbolic k1 k2\n";
    for (size_t i = 0; i < verts.size(); ++i) {
        const Vec3& p = verts[i];
        const Vec3& a1 = field.asym1[i];
        const Vec3& a2 = field.asym2[i];
        out << p.x() << " " << p.y() << " " << p.z() << " "
            << a1.x() << " " << a1.y() << " " << a1.z() << " "
            << a2.x() << " " << a2.y() << " " << a2.z() << " "
            << (int)field.hyperbolic[i] << " " << field.k1[i] << " " << field.k2[i] << "\n";
    }
}

static void exportPatches(const std::string& dir, const std::vector<rulefit::DevelopablePatch>& patches) {
    ensureDir(dir);
    for (size_t k = 0; k < patches.size(); ++k) {
        const auto& p = patches[k];
        if (p.numPoints < 6) continue;
        std::ofstream out(dir + "/patch_" + std::to_string(k) + ".txt");
        if (!out) continue;
        out.precision(8);
        out << "gamma_x gamma_y gamma_z ruling_x ruling_y ruling_z length\n";
        for (size_t s = 0; s < p.gamma.size(); ++s) {
            out << p.gamma[s].x() << " " << p.gamma[s].y() << " " << p.gamma[s].z() << " "
                << p.ruling[s].x() << " " << p.ruling[s].y() << " " << p.ruling[s].z() << " "
                << p.length[s] << "\n";
        }
    }
}

int main(int argc, char* argv[]) {
    std::string surface = "wavy";
    std::string outDir = "./out";
    int res = 40;
    int K = 16;
    int maxIter = 20;
    double lambdaDev = 1.0;
    double lambdaCenter = 0.3;
    double lambdaSmooth = 0.05;
    int seed = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--surface=", 0) == 0) surface = a.substr(10);
        else if (a.rfind("--out=", 0) == 0) outDir = a.substr(6);
        else if (a.rfind("--res=", 0) == 0) res = std::stoi(a.substr(6));
        else if (a.rfind("--K=", 0) == 0) K = std::stoi(a.substr(4));
        else if (a.rfind("--max-iter=", 0) == 0) maxIter = std::stoi(a.substr(11));
        else if (a.rfind("--lambda-dev=", 0) == 0) lambdaDev = std::stod(a.substr(13));
        else if (a.rfind("--lambda-center=", 0) == 0) lambdaCenter = std::stod(a.substr(16));
        else if (a.rfind("--lambda-smooth=", 0) == 0) lambdaSmooth = std::stod(a.substr(16));
        else if (a.rfind("--seed=", 0) == 0) seed = std::stoi(a.substr(7));
    }
    ensureDir(outDir);

    std::cout << "=== rulefit pipeline (OCCT-R1) ===\n";

    // ── Step 0: 曲面 + 网格 ──
    NurbsSurfaceWrapper nurbs = rulefit::createSurface(surface, seed);
    std::cout << "  Surface: " << nurbs.numCtrlU() << "x" << nurbs.numCtrlV() << " ctrl\n";
    Vec3Arr verts;
    FaceArr faces;
    Vec2Arr uvs;
    nurbs.generateMesh(res, res, verts, faces, uvs);
    std::cout << "  Mesh: " << verts.size() << " verts, " << faces.size() << " faces\n";
    NurbsMeshProcessor mesh(verts, faces, nurbs, uvs);
    exportMeshOBJ(outDir + "/mesh.obj", verts, faces);
    std::cout << "[STEP:0:done]\n" << std::flush;

    // ── Step 1: 方向场 ──
    rulefit::DirectionField field = rulefit::computeAsymptoticField(mesh);
    int nHyp = 0;
    for (char h : field.hyperbolic) if (h) ++nHyp;
    std::cout << "  Direction field: " << nHyp << "/" << verts.size() << " hyperbolic\n";
    exportDirectionField(outDir + "/direction_field.txt", verts, field);
    std::cout << "[STEP:1:done]\n" << std::flush;

    // ── Step 2: 分区 EM ──
    Vec3Arr faceCentroids(faces.size());
    for (size_t fi = 0; fi < faces.size(); ++fi) {
        faceCentroids[fi] = (verts[faces[fi].v0] + verts[faces[fi].v1] + verts[faces[fi].v2]) / 3.0;
    }
    std::vector<Vec3> asymPerVertex;
    asymPerVertex.reserve(verts.size() * 2);
    for (size_t i = 0; i < verts.size(); ++i) {
        asymPerVertex.push_back(field.asym1[i]);
        asymPerVertex.push_back(field.asym2[i]);
    }
    std::cout << "  Partitioning K=" << K << " maxIter=" << maxIter
              << " lambdaDev=" << lambdaDev << "...\n";
    rulefit::PartitionResult result = rulefit::partitionByDevelopable(
        verts, faces, faceCentroids, asymPerVertex, K, maxIter, lambdaDev, lambdaCenter, lambdaSmooth);
    std::cout << "  Partitions: " << result.nParts << " (energy=" << result.energy
              << ", iters=" << result.iterations << ")\n";
    exportFaceLabels(outDir + "/face_labels.txt", result.faceLabels);
    exportPatches(outDir + "/patches", result.patches);
    std::cout << "[STEP:2:done]\n" << std::flush;

    // ── Step 3: 评估 ──
    rulefit::Evaluation eval = rulefit::evaluate(result.patches);
    std::cout << "  Evaluation: meanTwist=" << eval.meanTwist
              << " meanFitError=" << eval.meanFitError
              << " developable=" << eval.nDevelopable << "/" << eval.nPatches << "\n";
    {
        std::ofstream out(outDir + "/evaluation.txt");
        out.precision(8);
        out << "mean_twist " << eval.meanTwist << "\n";
        out << "mean_fit_error " << eval.meanFitError << "\n";
        out << "n_developable " << eval.nDevelopable << "\n";
        out << "n_patches " << eval.nPatches << "\n";
        out << "pid twist fit_error\n";
        for (size_t k = 0; k < result.patches.size(); ++k) {
            if (result.patches[k].numPoints < 6) continue;
            out << k << " " << result.patches[k].twist << " "
                << result.patches[k].fitError << "\n";
        }
    }
    std::cout << "[STEP:3:done]\n" << std::flush;
    std::cout << "=== done ===\n";
    return 0;
}
