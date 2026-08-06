#include "distillation/tolerance.hpp"

#include <GeomAPI_ProjectPointOnSurf.hxx>
#include <Geom2dAPI_ProjectPointOnCurve.hxx>
#include <Geom_BSplineSurface.hxx>
#include <Geom2d_BSplineCurve.hxx>
#include <gp_Pnt.hxx>
#include <gp_Pnt2d.hxx>

#include <cmath>
#include <algorithm>

namespace distillation {

static double projectToSurface(const gp_Pnt& p, const NurbsSurfaceWrapper& nurbs) {
    Handle(Geom_BSplineSurface) surf = nurbs.surface();
    if (surf.IsNull()) {
        Vec3 v = nurbs.evaluate(0.5, 0.5);
        return (Vec3(p.X(), p.Y(), p.Z()) - v).norm();
    }
    GeomAPI_ProjectPointOnSurf proj(p, surf, 1e-6);
    if (proj.NbPoints() > 0)
        return p.Distance(proj.NearestPoint());
    return 0.0;
}

static bool intersectRayCurve(
    const gp_Pnt2d& origin, double dx, double dy,
    const Handle(Geom2d_BSplineCurve)& curve,
    double u1, double u2,
    double& tHit, gp_Pnt2d& hitPt)
{
    int N = 200;
    double bestCos = -2.0;
    tHit = u1;
    hitPt = curve->Value(u1);
    double lenD = std::sqrt(dx * dx + dy * dy);
    if (lenD < 1e-12) return false;
    double ndx = dx / lenD, ndy = dy / lenD;

    for (int i = 0; i <= N; ++i) {
        double t = u1 + (u2 - u1) * double(i) / N;
        gp_Pnt2d p = curve->Value(t);
        double tox = p.X() - origin.X();
        double toy = p.Y() - origin.Y();
        double len = std::sqrt(tox * tox + toy * toy);
        if (len < 1e-12) continue;
        double cosA = (tox * ndx + toy * ndy) / len;
        if (cosA > bestCos) {
            bestCos = cosA;
            tHit = t;
            hitPt = p;
        }
    }
    return bestCos > 0.98;
}

TolResult computeToleranceParallel(
    const Handle(Geom2d_BSplineCurve)& C0,
    const Handle(Geom2d_BSplineCurve)& C1,
    const NurbsSurfaceWrapper& nurbs,
    const ToleranceSpec& spec)
{
    TolResult result;
    if (C0.IsNull() || C1.IsNull()) return result;

    double u0a = C0->FirstParameter();
    double u0b = C0->LastParameter();
    double u1a = C1->FirstParameter();
    double u1b = C1->LastParameter();

    int ns = spec.samplesPerCurve;
    int nr = spec.samplesPerRuling;
    double bestGlobal = std::numeric_limits<double>::max();

    for (int a = 0; a < spec.angleSteps; ++a) {
        double angle = 2.0 * M_PI * double(a) / spec.angleSteps;
        double dx = cos(angle), dy = sin(angle);

        double maxD = 0.0;
        double sumD = 0.0;
        int count = 0;

        for (int si = 0; si <= ns; ++si) {
            double t0 = u0a + (u0b - u0a) * double(si) / ns;
            gp_Pnt2d p0_2d = C0->Value(t0);
            Vec3 p0_3d = nurbs.evaluate(p0_2d.X(), p0_2d.Y());

            double tHit;
            gp_Pnt2d pHit_2d;
            if (!intersectRayCurve(p0_2d, dx, dy, C1, u1a, u1b, tHit, pHit_2d))
                continue;

            Vec3 p1_3d = nurbs.evaluate(pHit_2d.X(), pHit_2d.Y());

            for (int ri = 0; ri <= nr; ++ri) {
                double lamb = double(ri) / nr;
                gp_Pnt pr(
                    (1.0 - lamb) * p0_3d.x() + lamb * p1_3d.x(),
                    (1.0 - lamb) * p0_3d.y() + lamb * p1_3d.y(),
                    (1.0 - lamb) * p0_3d.z() + lamb * p1_3d.z());
                double d = projectToSurface(pr, nurbs);
                maxD = std::max(maxD, d);
                sumD += d;
                ++count;
            }
        }

        if (count > 0 && maxD < bestGlobal) {
            bestGlobal = maxD;
            result.maxDist = maxD;
            result.rmsDist = std::sqrt(sumD / count);
            result.bestAngleDeg = angle * 180.0 / M_PI;
            result.converged = true;
        }
    }

    return result;
}

TolResult computeToleranceGeneral(
    const Handle(Geom2d_BSplineCurve)& C0,
    const Handle(Geom2d_BSplineCurve)& C1,
    const NurbsSurfaceWrapper& nurbs,
    const ToleranceSpec& spec)
{
    TolResult result;
    if (C0.IsNull() || C1.IsNull()) return result;

    double u0a = C0->FirstParameter();
    double u0b = C0->LastParameter();
    int ns = spec.samplesPerCurve;
    int nr = spec.samplesPerRuling;

    double maxD = 0.0;
    double sumD = 0.0;
    int count = 0;

    for (int si = 0; si <= ns; ++si) {
        double t0 = u0a + (u0b - u0a) * double(si) / ns;
        gp_Pnt2d p0_2d = C0->Value(t0);
        Vec3 p0_3d = nurbs.evaluate(p0_2d.X(), p0_2d.Y());

        Geom2dAPI_ProjectPointOnCurve proj(p0_2d, C1);
        if (proj.NbPoints() == 0) continue;
        double t1 = proj.LowerDistanceParameter();
        gp_Pnt2d p1_2d = C1->Value(t1);
        Vec3 p1_3d = nurbs.evaluate(p1_2d.X(), p1_2d.Y());

        for (int ri = 0; ri <= nr; ++ri) {
            double lamb = double(ri) / nr;
            gp_Pnt pr(
                (1.0 - lamb) * p0_3d.x() + lamb * p1_3d.x(),
                (1.0 - lamb) * p0_3d.y() + lamb * p1_3d.y(),
                (1.0 - lamb) * p0_3d.z() + lamb * p1_3d.z());
            double d = projectToSurface(pr, nurbs);
            maxD = std::max(maxD, d);
            sumD += d;
            ++count;
        }
    }

    result.maxDist = maxD;
    result.rmsDist = (count > 0) ? std::sqrt(sumD / count) : 0.0;
    result.converged = true;
    return result;
}

} // namespace distillation
