#pragma once

#include "common.hpp"

#include <Geom_BSplineSurface.hxx>
#include <Geom_BSplineCurve.hxx>
#include <TColgp_Array2OfPnt.hxx>
#include <TColStd_Array1OfReal.hxx>
#include <TColStd_Array2OfReal.hxx>
#include <gp_Pnt.hxx>
#include <gp_Vec.hxx>
#include <gp_Dir.hxx>

namespace distillation {

class NurbsSurfaceWrapper {
public:
    NurbsSurfaceWrapper() = default;

    NurbsSurfaceWrapper(const Vec3Arr& ctrlPts, int nU, int nV,
                        const std::vector<double>& knotsU,
                        const std::vector<double>& knotsV,
                        int degU, int degV,
                        const std::vector<double>& weights = {});

    explicit NurbsSurfaceWrapper(const Handle(Geom_BSplineSurface)& surf);

    ~NurbsSurfaceWrapper() = default;

    const Handle(Geom_BSplineSurface)& surface() const { return m_surf; }

    Vec3 evaluate(double u, double v) const;

    struct DerivResult {
        Vec3 S, Su, Sv, Suu, Suv, Svv;
    };
    DerivResult evaluateDerivatives(double u, double v) const;

    Vec3 normal(double u, double v) const;

    struct CurvatureResult {
        double k1, k2;
        Vec3 d1, d2;
    };
    CurvatureResult principalCurvatures(double u, double v) const;

    double gaussianCurvature(double u, double v) const;
    double meanCurvature(double u, double v) const;

    struct ProjectResult {
        double u, v;
        bool converged;
    };
    ProjectResult projectToUV(const Vec3& point, double u0 = 0.5, double v0 = 0.5,
                              int maxIter = 30, double tol = 1e-6, double damping = 0.8) const;

    void generateMesh(int resU, int resV,
                      Vec3Arr& vertices, FaceArr& faces, Vec2Arr& uvs) const;

    std::pair<double, double> paramDomainU() const;
    std::pair<double, double> paramDomainV() const;

    int numCtrlU() const { return m_surf->NbUPoles(); }
    int numCtrlV() const { return m_surf->NbVPoles(); }
    int degreeU() const { return m_surf->UDegree(); }
    int degreeV() const { return m_surf->VDegree(); }

private:
    Handle(Geom_BSplineSurface) m_surf;
};

std::vector<double> makeClampedKnots(int nCtrl, int degree, bool nonuniform = false);

} // namespace distillation
