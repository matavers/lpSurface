#include "distillation/occt_mesh_reconstruct.hpp"

#include <BRepBuilderAPI_MakeFace.hxx>
#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRepBuilderAPI_MakeWire.hxx>
#include <BRepBuilderAPI_MakePolygon.hxx>
#include <BRep_Builder.hxx>
#include <TopoDS_Compound.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Wire.hxx>
#include <TopoDS_Iterator.hxx>
#include <TopExp_Explorer.hxx>
#include <BRepAlgoAPI_Splitter.hxx>
#include <TopTools_ListOfShape.hxx>
#include <BRepMesh_IncrementalMesh.hxx>
#include <BRep_Tool.hxx>
#include <Poly_Triangulation.hxx>
#include <Poly_Triangle.hxx>
#include <gp_Pln.hxx>
#include <gp_Pnt.hxx>
#include <TColgp_Array1OfPnt.hxx>
#include <Precision.hxx>

#include <map>
#include <set>
#include <unordered_map>
#include <algorithm>

namespace distillation {

static inline std::pair<int,int> mkEdge(int a, int b) {
    return (a < b) ? std::make_pair(a, b) : std::make_pair(b, a);
}

static TopoDS_Face buildRectFace(double uMin, double uMax, double vMin, double vMax) {
    BRepBuilderAPI_MakeEdge e1(gp_Pnt(uMin, vMin, 0), gp_Pnt(uMax, vMin, 0));
    BRepBuilderAPI_MakeEdge e2(gp_Pnt(uMax, vMin, 0), gp_Pnt(uMax, vMax, 0));
    BRepBuilderAPI_MakeEdge e3(gp_Pnt(uMax, vMax, 0), gp_Pnt(uMin, vMax, 0));
    BRepBuilderAPI_MakeEdge e4(gp_Pnt(uMin, vMax, 0), gp_Pnt(uMin, vMin, 0));
    BRepBuilderAPI_MakeWire wire;
    wire.Add(e1.Edge()); wire.Add(e2.Edge());
    wire.Add(e3.Edge()); wire.Add(e4.Edge());
    return BRepBuilderAPI_MakeFace(wire.Wire());
}

static void collectBoundaryWires(
    const BoundaryNetwork& net,
    BRep_Builder& bb,
    TopoDS_Compound& comp)
{
    int nbv = static_cast<int>(net.smoothedUVs.size());
    std::vector<bool> used(nbv, false);
    std::vector<IntArr> adj(nbv);
    for (const auto& e : net.edges) {
        int l0 = net.globalToLocal[e.v0];
        int l1 = net.globalToLocal[e.v1];
        if (l0 < 0 || l1 < 0) continue;
        adj[l0].push_back(l1);
        adj[l1].push_back(l0);
    }

    for (int start = 0; start < nbv; ++start) {
        if (used[start] || adj[start].empty()) continue;
        if (adj[start].size() == 1) {
            std::vector<int> chain;
            chain.push_back(start);
            int cur = start, prev = -1;
            while (true) {
                used[cur] = true;
                int next = -1;
                for (int nb : adj[cur]) {
                    if (nb != prev && !used[nb]) { next = nb; break; }
                }
                if (next < 0) break;
                chain.push_back(next);
                prev = cur; cur = next;
            }
            used[cur] = true;
            if (chain.size() >= 2) {
                BRepBuilderAPI_MakePolygon poly;
                for (int vi : chain) {
                    const Vec2& p = net.smoothedUVs[vi];
                    poly.Add(gp_Pnt(p.x(), p.y(), 0));
                }
                if (poly.IsDone() && poly.Wire().IsNull() == false)
                    bb.Add(comp, poly.Wire());
            }
        }
    }

    for (int start = 0; start < nbv; ++start) {
        if (used[start] || adj[start].size() < 2) continue;
        std::vector<int> chain;
        chain.push_back(start);
        used[start] = true;
        int cur = start, prev = -1;
        while (true) {
            int next = -1;
            for (int nb : adj[cur]) {
                if (nb != prev && !used[nb]) { next = nb; break; }
            }
            if (next < 0) break;
            chain.push_back(next);
            used[next] = true;
            prev = cur; cur = next;
        }
        if (chain.size() >= 2) {
            BRepBuilderAPI_MakePolygon poly;
            for (int vi : chain) {
                const Vec2& p = net.smoothedUVs[vi];
                poly.Add(gp_Pnt(p.x(), p.y(), 0));
            }
            if (poly.IsDone() && poly.Wire().IsNull() == false)
                bb.Add(comp, poly.Wire());
        }
    }
}

std::tuple<Vec2Arr, FaceArr, IntArr> occtConstrainedReconstruct(
    const Vec2Arr& originalUVs,
    const FaceArr& originalFaces,
    const IntArr& originalFaceLabels,
    const BoundaryNetwork& net,
    double uMin, double uMax,
    double vMin, double vMax)
{
    int nOrigV = static_cast<int>(originalUVs.size());
    int nOrigF = static_cast<int>(originalFaces.size());

    (void)uMin; (void)uMax; (void)vMin; (void)vMax;

    // Step 1: build rectangular domain face
    TopoDS_Face domainFace = buildRectFace(0.0, 1.0, 0.0, 1.0);

    // Step 2: build compound of boundary edges
    BRep_Builder bb;
    TopoDS_Compound edgeComp;
    bb.MakeCompound(edgeComp);
    collectBoundaryWires(net, bb, edgeComp);

    int nSplitEdges = 0;
    for (TopoDS_Iterator it(edgeComp); it.More(); it.Next()) ++nSplitEdges;
    std::cout << "  [OCCT Split] edges provided: " << nSplitEdges << "\n";

    // Step 3: split face by boundary edges
    TopTools_ListOfShape argList, toolList;
    argList.Append(domainFace);
    toolList.Append(edgeComp);
    BRepAlgoAPI_Splitter splitter;
    splitter.SetArguments(argList);
    splitter.SetTools(toolList);
    splitter.Build();

    if (!splitter.IsDone()) {
        std::cerr << "  [OCCT Split] WARNING: split failed, falling back to original mesh\n";
        Vec2Arr outUVs = originalUVs;
        FaceArr outFaces = originalFaces;
        IntArr outLabels = originalFaceLabels;
        return {outUVs, outFaces, outLabels};
    }

    TopoDS_Shape splitResult = splitter.Shape();
    std::cout << "  [OCCT Split] done\n";

    // Step 4: mesh the split result
    double deflection = 0.0002;
    double angleDeg = 20.0;
    BRepMesh_IncrementalMesh(splitResult, deflection, false, angleDeg, false);
    std::cout << "  [OCCT Mesh] deflection=" << deflection
              << " angle=" << angleDeg << "\n";

    // Step 5: collect all triangles and vertices from each face
    std::vector<Vec2> allUVs;
    std::vector<Face> allFaces;
    std::vector<int> allLabels;

    struct PtHashCpp {
        std::size_t operator()(const std::pair<double,double>& p) const {
            long hx = static_cast<long>(p.first * 1e7);
            long hy = static_cast<long>(p.second * 1e7);
            return static_cast<std::size_t>((hx * 73856093) ^ (hy * 19349663));
        }
    };
    std::unordered_map<std::pair<double,double>, int, PtHashCpp> ptMap;

    auto getOrAddVertex = [&](double x, double y) -> int {
        auto key = std::make_pair(
            std::round(x * 1e7) / 1e7,
            std::round(y * 1e7) / 1e7);
        auto it = ptMap.find(key);
        if (it != ptMap.end()) return it->second;
        int idx = static_cast<int>(allUVs.size());
        allUVs.push_back(Vec2(x, y));
        ptMap[key] = idx;
        return idx;
    };

    int nFacesExtracted = 0;
    TopLoc_Location loc;
    for (TopExp_Explorer faceExp(splitResult, TopAbs_FACE); faceExp.More(); faceExp.Next()) {
        const TopoDS_Shape& sh = faceExp.Current();
        if (sh.ShapeType() != TopAbs_FACE) continue;
        TopoDS_Face face;
        face.TShape(sh.TShape());
        face.Location(sh.Location());
        face.Orientation(sh.Orientation());
        Handle(Poly_Triangulation) tri = BRep_Tool::Triangulation(face, loc);

        if (tri.IsNull()) continue;
        ++nFacesExtracted;

        // Determine partition label: sample a point inside the face
        int faceLabel = -1;
        {
            double cx = 0, cy = 0;
            int nv = tri->NbNodes();
            for (int i = 1; i <= nv; ++i) {
                gp_Pnt p = tri->Node(i);
                cx += p.X(); cy += p.Y();
            }
            if (nv > 0) { cx /= nv; cy /= nv; }

            double bestDist = std::numeric_limits<double>::max();
            for (int fi = 0; fi < nOrigF; ++fi) {
                const Face& of = originalFaces[fi];
                Vec2 c = (originalUVs[of.v0] + originalUVs[of.v1] + originalUVs[of.v2]) / 3.0;
                double d = (c.x() - cx) * (c.x() - cx) + (c.y() - cy) * (c.y() - cy);
                if (d < bestDist) { bestDist = d; faceLabel = originalFaceLabels[fi]; }
            }
        }

        int nTri = tri->NbTriangles();
        for (int i = 1; i <= nTri; ++i) {
            Poly_Triangle t = tri->Triangle(i);
            int n1, n2, n3;
            t.Get(n1, n2, n3);

            gp_Pnt p1 = tri->Node(n1);
            gp_Pnt p2 = tri->Node(n2);
            gp_Pnt p3 = tri->Node(n3);

            int v1 = getOrAddVertex(p1.X(), p1.Y());
            int v2 = getOrAddVertex(p2.X(), p2.Y());
            int v3 = getOrAddVertex(p3.X(), p3.Y());

            allFaces.push_back({v1, v2, v3});
            allLabels.push_back(faceLabel);
        }
    }

    std::cout << "  [OCCT Mesh] " << nFacesExtracted << " faces, "
              << allUVs.size() << " verts, "
              << allFaces.size() << " triangles\n";

    // Propagate unlabeled faces via adjacency
    {
        int nf = static_cast<int>(allFaces.size());
        std::map<std::pair<int,int>, std::vector<int>> edgeToFaces;
        for (int fi = 0; fi < nf; ++fi) {
            const Face& f = allFaces[fi];
            for (int i = 0; i < 3; ++i) {
                edgeToFaces[mkEdge(f[i], f[(i + 1) % 3])].push_back(fi);
            }
        }

        bool changed = true;
        while (changed) {
            changed = false;
            for (int fi = 0; fi < nf; ++fi) {
                if (allLabels[fi] >= 0) continue;
                const Face& f = allFaces[fi];
                for (int i = 0; i < 3; ++i) {
                    for (int nf2 : edgeToFaces[mkEdge(f[i], f[(i + 1) % 3])]) {
                        if (nf2 != fi && allLabels[nf2] >= 0) {
                            allLabels[fi] = allLabels[nf2];
                            changed = true;
                            goto next_face;
                        }
                    }
                }
                next_face:;
            }
        }
    }

    int nUnlabeled = 0;
    for (int lbl : allLabels) if (lbl < 0) ++nUnlabeled;
    std::cout << "  [OCCT Labels] " << (allFaces.size() - nUnlabeled)
              << "/" << allFaces.size() << " labeled\n";

    Vec2Arr outUVs = std::move(allUVs);
    FaceArr outFaces = std::move(allFaces);
    IntArr outLabels = std::move(allLabels);
    return {outUVs, outFaces, outLabels};
}

} // namespace distillation
