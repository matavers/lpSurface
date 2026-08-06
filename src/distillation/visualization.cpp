#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <map>
#include <set>
#include <string>

#include "distillation/visualization.hpp"

#include <BRepBuilderAPI_MakePolygon.hxx>
#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRep_Builder.hxx>
#include <TopoDS_Compound.hxx>
#include <Poly_Triangulation.hxx>
#include <Poly_Triangle.hxx>
#include <NCollection_Array1.hxx>
#include <Quantity_Color.hxx>
#include <Aspect_DisplayConnection.hxx>
#include <OpenGl_GraphicDriver.hxx>
#include <V3d_View.hxx>
#include <V3d_Viewer.hxx>
#include <WNT_Window.hxx>
#include <AIS_Shape.hxx>
#include <AIS_Triangulation.hxx>
#include <AIS_InteractiveContext.hxx>
#include <Graphic3d_CLight.hxx>
#include <Graphic3d_TypeOfLightSource.hxx>

namespace distillation {

static std::vector<Quantity_Color> BOUNDARY_COLORS = {
    Quantity_Color(0.95, 0.23, 0.23, Quantity_TOC_RGB),
    Quantity_Color(0.20, 0.75, 0.95, Quantity_TOC_RGB),
    Quantity_Color(0.95, 0.70, 0.15, Quantity_TOC_RGB),
    Quantity_Color(0.30, 0.85, 0.40, Quantity_TOC_RGB),
    Quantity_Color(0.85, 0.35, 0.70, Quantity_TOC_RGB),
    Quantity_Color(0.60, 0.40, 0.90, Quantity_TOC_RGB),
    Quantity_Color(1.00, 0.60, 0.40, Quantity_TOC_RGB),
    Quantity_Color(0.40, 0.80, 0.80, Quantity_TOC_RGB),
};

struct ViewState {
    Handle(V3d_View) view;
    Handle(AIS_InteractiveContext) context;
    bool rotating = false, panning = false, zooming = false;
    int lastX = 0, lastY = 0;
};

static LRESULT CALLBACK ViewerWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    ViewState* s = reinterpret_cast<ViewState*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
    switch (msg) {
    case WM_LBUTTONDOWN:
        if (s && s->view) { s->rotating = true; s->lastX = LOWORD(lParam); s->lastY = HIWORD(lParam); s->view->StartRotation(s->lastX, s->lastY); SetCapture(hwnd); }
        return 0;
    case WM_LBUTTONUP: if (s) s->rotating = false; ReleaseCapture(); return 0;
    case WM_MBUTTONDOWN:
        if (s && s->view) { s->panning = true; s->lastX = LOWORD(lParam); s->lastY = HIWORD(lParam); SetCapture(hwnd); }
        return 0;
    case WM_MBUTTONUP: if (s) s->panning = false; ReleaseCapture(); return 0;
    case WM_RBUTTONDOWN:
        if (s && s->view) { s->zooming = true; s->lastX = LOWORD(lParam); s->lastY = HIWORD(lParam); s->view->StartZoomAtPoint(s->lastX, s->lastY); SetCapture(hwnd); }
        return 0;
    case WM_RBUTTONUP: if (s) s->zooming = false; ReleaseCapture(); return 0;
    case WM_MOUSEMOVE:
        if (s && s->view) {
            int x = LOWORD(lParam), y = HIWORD(lParam);
            if (s->rotating) { s->view->Rotation(x, y); if (s->context) s->context->UpdateCurrentViewer(); }
            else if (s->panning) { s->view->Pan(x - s->lastX, -(y - s->lastY)); s->lastX = x; s->lastY = y; if (s->context) s->context->UpdateCurrentViewer(); }
            else if (s->zooming) { s->view->ZoomAtPoint(s->lastX, s->lastY, x, y); s->lastX = x; s->lastY = y; if (s->context) s->context->UpdateCurrentViewer(); }
        }
        return 0;
    case WM_MOUSEWHEEL:
        if (s && s->view) { s->view->SetZoom(GET_WHEEL_DELTA_WPARAM(wParam) > 0 ? 0.9 : 1.1); if (s->context) s->context->UpdateCurrentViewer(); }
        return 0;
    case WM_SIZE: if (s && s->view) { s->view->MustBeResized(); return 0; }
    case WM_DESTROY: PostQuitMessage(0); return 0;
    }
    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

PartitionViewer::PartitionViewer(const Vec3Arr& vertices, const FaceArr& faces,
                                 const std::vector<IterationEntry>& iterHistory,
                                 const IntVecSet& partitionsOrdered,
                                 const std::vector<Vec3Arr>& boundaryCurves,
                                 const Vec3Arr& cornerPoints)
    : m_vertices(vertices), m_faces(faces),
      m_iterHistory(iterHistory), m_partitions(partitionsOrdered),
      m_boundaryCurves(boundaryCurves), m_cornerPoints(cornerPoints),
      m_nIters(static_cast<int>(iterHistory.size())),
      m_nParts(static_cast<int>(partitionsOrdered.size())) {}

void PartitionViewer::show() {
    std::cout << "\n=== Partition Viewer (OCCT) ===" << std::endl;
    std::cout << "  Vertices: " << m_vertices.size() << "  Faces: " << m_faces.size() << std::endl;

    Handle(Aspect_DisplayConnection) dispConn = new Aspect_DisplayConnection();
    Handle(OpenGl_GraphicDriver) driver = new OpenGl_GraphicDriver(dispConn);
    m_viewer = new V3d_Viewer(driver);
    {   auto dl = new Graphic3d_CLight(Graphic3d_TypeOfLightSource_Directional);
        dl->SetDirection(gp_Dir(1.0, -1.0, 2.0)); m_viewer->AddLight(dl);
        auto al = new Graphic3d_CLight(Graphic3d_TypeOfLightSource_Ambient);
        al->SetIntensity(0.4f); m_viewer->AddLight(al); m_viewer->SetLightOn(); }
    m_context = new AIS_InteractiveContext(m_viewer);
    m_context->SetDisplayMode(AIS_Shaded, true);

    HINSTANCE hInst = GetModuleHandleW(NULL);
    WNDCLASSW wc = {};
    wc.lpfnWndProc = ViewerWndProc; wc.hInstance = hInst;
    wc.lpszClassName = L"DistillationViewerWnd"; wc.style = CS_OWNDC;
    RegisterClassW(&wc);
    HWND hwnd = CreateWindowExW(0, L"DistillationViewerWnd", L"Partition Viewer",
        WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS,
        100, 100, 1200, 800, NULL, NULL, hInst, NULL);
    if (!hwnd) { std::cerr << "CreateWindowExW failed\n"; return; }

    Handle(WNT_Window) wind = new WNT_Window(hwnd);
    Handle(V3d_View) view = m_viewer->CreateView();
    view->SetWindow(wind); wind->Map();
    view->SetBackgroundColor(Quantity_Color(0.12, 0.15, 0.20, Quantity_TOC_RGB));
    view->SetBackFacingModel(Graphic3d_TypeOfBackfacingModel_DoubleSided);
    view->SetVisualization(V3d_ZBUFFER);
    view->MustBeResized();

    ViewState vs; vs.view = view; vs.context = m_context;
    SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(&vs));

    Vec3 bbMin = m_vertices[0], bbMax = m_vertices[0];
    for (const auto& v : m_vertices)
        for (int d = 0; d < 3; ++d) { bbMin[d] = std::min(bbMin[d], v[d]); bbMax[d] = std::max(bbMax[d], v[d]); }

    // (A) mesh wireframe
    {
        std::set<std::pair<int,int>> edgeSet;
        for (const auto& f : m_faces)
            for (int i = 0; i < 3; ++i) { int a = f[i], b = f[(i+1)%3]; if (a > b) std::swap(a, b); edgeSet.insert({a, b}); }
        TopoDS_Compound comp; BRep_Builder bb; bb.MakeCompound(comp);
        int cnt = 0;
        for (const auto& e : edgeSet) {
            if (++cnt > 4000) break;
            BRepBuilderAPI_MakeEdge me(gp_Pnt(m_vertices[e.first].x(), m_vertices[e.first].y(), m_vertices[e.first].z()),
                                        gp_Pnt(m_vertices[e.second].x(), m_vertices[e.second].y(), m_vertices[e.second].z()));
            if (me.IsDone()) bb.Add(comp, me.Edge());
        }
        Handle(AIS_Shape) s = new AIS_Shape(comp);
        s->SetColor(Quantity_Color(0.45, 0.45, 0.45, Quantity_TOC_RGB));
        m_context->Display(s, false);
    }

    // (B) partition boundaries — smoothed 3D polylines
    {
        int idx = 0;
        for (const auto& pts : m_boundaryCurves) {
            if (pts.size() < 2) continue;
            Quantity_Color bc = BOUNDARY_COLORS[idx % BOUNDARY_COLORS.size()];
            ++idx;
            BRepBuilderAPI_MakePolygon poly;
            for (const auto& p : pts) poly.Add(gp_Pnt(p.x(), p.y(), p.z()));
            if (poly.IsDone()) {
                Handle(AIS_Shape) sh = new AIS_Shape(poly.Wire());
                sh->SetColor(bc);
                m_context->Display(sh, false);
            }
        }
    }

    // (C) corner spheres
    if (!m_cornerPoints.empty()) {
        double r = (bbMax - bbMin).norm() * 0.018;
        int nLat = 12, nLong = 12;
        for (const auto& c : m_cornerPoints) {
            int nV = (nLat + 1) * (nLong + 1);
            NCollection_Array1<gp_Pnt> nodes(1, nV);
            for (int i = 0; i <= nLat; ++i) {
                double theta = 3.14159265358979 * i / nLat;
                for (int j = 0; j <= nLong; ++j) {
                    double phi = 2.0 * 3.14159265358979 * j / nLong;
                    nodes(i * (nLong + 1) + j + 1) = gp_Pnt(
                        c.x() + r * sin(theta) * cos(phi),
                        c.y() + r * sin(theta) * sin(phi),
                        c.z() + r * cos(theta));
                }
            }
            int nT = 2 * nLat * nLong;
            NCollection_Array1<Poly_Triangle> tris(1, nT);
            int ti = 0;
            for (int i = 0; i < nLat; ++i)
                for (int j = 0; j < nLong; ++j) {
                    int a = i * (nLong + 1) + j + 1;
                    int b = a + 1;
                    int c_ = (i + 1) * (nLong + 1) + j + 1;
                    int d = c_ + 1;
                    tris(++ti) = Poly_Triangle(a, b, c_);
                    tris(++ti) = Poly_Triangle(b, d, c_);
                }
            Handle(AIS_Triangulation) ais = new AIS_Triangulation(new Poly_Triangulation(nodes, tris));
            ais->SetColor(Quantity_NOC_GREEN);
            ais->SetMaterial(Graphic3d_NOM_PLASTIC);
            m_context->Display(ais, false);
        }
    }

    ShowWindow(hwnd, SW_SHOW); UpdateWindow(hwnd);
    view->FitAll(); view->TriedronDisplay(Aspect_TOTP_RIGHT_LOWER, Quantity_NOC_WHITE, 0.08);
    view->Redraw(); m_context->UpdateCurrentViewer();
    std::cout << "  Left-drag=rotate  Middle-drag=pan  Right-drag/wheel=zoom\n"
              << "  Close window to exit.\n";
    MSG msg;
    while (GetMessageW(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg); DispatchMessageW(&msg);
        m_context->UpdateCurrentViewer();
    }
    std::cout << "  Viewer closed.\n";
}

} // namespace distillation
