// Test: UV-sphere via Poly_Triangulation + AIS_Triangulation rendering
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <iostream>
#include <cmath>

const double MY_PI = 3.14159265358979323846;

#include <Poly_Triangulation.hxx>
#include <Poly_Triangle.hxx>
#include <NCollection_Array1.hxx>
#include <AIS_Triangulation.hxx>
#include <AIS_Shape.hxx>
#include <BRepBuilderAPI_MakeFace.hxx>
#include <AIS_InteractiveContext.hxx>
#include <V3d_View.hxx>
#include <V3d_Viewer.hxx>
#include <WNT_Window.hxx>
#include <OpenGl_GraphicDriver.hxx>
#include <Aspect_DisplayConnection.hxx>
#include <Quantity_Color.hxx>
#include <gp_Pln.hxx>

struct State {
    Handle(V3d_View) view;
    Handle(AIS_InteractiveContext) ctx;
    bool rotating = false, panning = false, zooming = false;
    int lastX = 0, lastY = 0;
};

static LRESULT CALLBACK WndProc(HWND h, UINT m, WPARAM w, LPARAM l) {
    auto* s = (State*)GetWindowLongPtrW(h, GWLP_USERDATA);
    switch (m) {
    case WM_LBUTTONDOWN:
        if (s && s->view) { s->rotating = true; s->lastX = LOWORD(l); s->lastY = HIWORD(l); s->view->StartRotation(s->lastX, s->lastY); SetCapture(h); }
        return 0;
    case WM_LBUTTONUP: if (s) s->rotating = false; ReleaseCapture(); return 0;
    case WM_MBUTTONDOWN:
        if (s && s->view) { s->panning = true; s->lastX = LOWORD(l); s->lastY = HIWORD(l); SetCapture(h); }
        return 0;
    case WM_MBUTTONUP: if (s) s->panning = false; ReleaseCapture(); return 0;
    case WM_RBUTTONDOWN:
        if (s && s->view) { s->zooming = true; s->lastX = LOWORD(l); s->lastY = HIWORD(l); s->view->StartZoomAtPoint(s->lastX, s->lastY); SetCapture(h); }
        return 0;
    case WM_RBUTTONUP: if (s) s->zooming = false; ReleaseCapture(); return 0;
    case WM_MOUSEMOVE:
        if (s && s->view) {
            int x = LOWORD(l), y = HIWORD(l);
            if (s->rotating) { s->view->Rotation(x, y); if (s->ctx) s->ctx->UpdateCurrentViewer(); }
            else if (s->panning) { s->view->Pan(x - s->lastX, -(y - s->lastY)); s->lastX = x; s->lastY = y; if (s->ctx) s->ctx->UpdateCurrentViewer(); }
            else if (s->zooming) { s->view->ZoomAtPoint(s->lastX, s->lastY, x, y); s->lastX = x; s->lastY = y; if (s->ctx) s->ctx->UpdateCurrentViewer(); }
        }
        return 0;
    case WM_MOUSEWHEEL:
        if (s && s->view) { s->view->SetZoom(GET_WHEEL_DELTA_WPARAM(w) > 0 ? 0.9 : 1.1); if (s->ctx) s->ctx->UpdateCurrentViewer(); }
        return 0;
    case WM_SIZE: if (s && s->view) { s->view->MustBeResized(); } return 0;
    case WM_DESTROY: PostQuitMessage(0); return 0;
    }
    return DefWindowProcW(h, m, w, l);
}

int main() {
    SetConsoleOutputCP(65001);
    std::cout << "=== AIS_Shape rendering test ===\n";

    // graphics
    auto disp = new Aspect_DisplayConnection();
    auto drv = new OpenGl_GraphicDriver(disp);
    auto viewer = new V3d_Viewer(drv);
    viewer->SetDefaultLights();
    viewer->SetLightOn();
    auto ctx = new AIS_InteractiveContext(viewer);
    ctx->SetDisplayMode(AIS_Shaded, true);

    // window
    WNDCLASSW wc = {};
    wc.lpfnWndProc = WndProc; wc.hInstance = GetModuleHandleW(NULL);
    wc.lpszClassName = L"ShapeRenderTest"; wc.style = CS_OWNDC;
    RegisterClassW(&wc);
    HWND hwnd = CreateWindowExW(0, L"ShapeRenderTest", L"AIS_Shape Test",
                                WS_OVERLAPPEDWINDOW | WS_CLIPSIBLINGS | WS_CLIPCHILDREN,
                                100, 100, 800, 600, NULL, NULL, wc.hInstance, NULL);
    if (!hwnd) { std::cerr << "CreateWindowExW failed\n"; return 1; }

    auto wind = new WNT_Window(hwnd);
    auto view = viewer->CreateView();
    view->SetWindow(wind); wind->Map();
    view->SetBackgroundColor(Quantity_Color(0.12, 0.15, 0.20, Quantity_TOC_RGB));
    view->SetVisualization(V3d_ZBUFFER);
    view->MustBeResized();

    State st; st.view = view; st.ctx = ctx;
    SetWindowLongPtrW(hwnd, GWLP_USERDATA, (LONG_PTR)&st);

    // --- sphere via Poly_Triangulation (manually constructed UV-sphere) ---
    std::cout << "=== Sphere (UV sphere via AIS_Triangulation) ===\n";
    {
        int nLat = 20, nLong = 20;
        int nV = (nLat + 1) * (nLong + 1);
        int nT = 2 * nLat * nLong;
        double R = 30.0;

        NCollection_Array1<gp_Pnt> nodes(1, nV);
        for (int i = 0; i <= nLat; ++i) {
            double theta = MY_PI * i / nLat;
            for (int j = 0; j <= nLong; ++j) {
                double phi = 2.0 * MY_PI * j / nLong;
                int idx = i * (nLong + 1) + j + 1;
                nodes(idx) = gp_Pnt(R * sin(theta) * cos(phi),
                                    R * sin(theta) * sin(phi),
                                    R * cos(theta));
            }
        }

        NCollection_Array1<Poly_Triangle> tris(1, nT);
        int ti = 0;
        for (int i = 0; i < nLat; ++i) {
            for (int j = 0; j < nLong; ++j) {
                int a = i * (nLong + 1) + j + 1;
                int b = a + 1;
                int c = (i + 1) * (nLong + 1) + j + 1;
                int d = c + 1;
                tris(++ti) = Poly_Triangle(a, b, c);
                tris(++ti) = Poly_Triangle(b, d, c);
            }
        }

        auto tri = new Poly_Triangulation(nodes, tris);
        std::cout << "  Poly_Triangulation: nodes=" << tri->NbNodes()
                  << " tris=" << tri->NbTriangles() << "\n";

        auto ais = new AIS_Triangulation(tri);
        ais->SetColor(Quantity_Color(1.0, 0.2, 0.2, Quantity_TOC_RGB));
        ais->SetMaterial(Graphic3d_NOM_PLASTIC);
        ctx->Display(ais, false);
        std::cout << "  Displayed OK\n";
    }

    // --- plane face (BRepBuilderAPI_MakeFace from gp_Pln) ---
    std::cout << "=== Plane Face ===\n";
    {
        gp_Pln plane;
        BRepBuilderAPI_MakeFace mf(plane, -40, 40, -40, 40);
        std::cout << "  IsDone=" << (mf.IsDone() ? "true" : "false") << "\n";
        if (mf.IsDone()) {
            auto sh = new AIS_Shape(mf.Shape());
            sh->SetColor(Quantity_Color(0.1, 0.7, 0.3, Quantity_TOC_RGB));
            sh->SetMaterial(Graphic3d_NOM_PLASTIC);
            ctx->Display(sh, false);
            std::cout << "  Displayed OK\n";
        }
    }

    ShowWindow(hwnd, SW_SHOW); UpdateWindow(hwnd);
    view->FitAll(); view->Redraw(); ctx->UpdateCurrentViewer();

    std::cout << "\nClose window to exit.\n";
    MSG msg;
    while (GetMessageW(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
        ctx->UpdateCurrentViewer();
    }
    std::cout << "Done.\n";
    return 0;
}
