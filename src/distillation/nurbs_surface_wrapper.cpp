#include "distillation/nurbs_surface_wrapper.hpp"

#include <GeomAPI_ProjectPointOnSurf.hxx>
#include <GeomLProp_SLProps.hxx>
#include <GeomLib.hxx>

namespace distillation {

// ── makeClampedKnots ─────────────────────────────────────────
std::vector<double> makeClampedKnots(int nCtrl, int degree, bool nonuniform) {
    int nKnots = nCtrl + degree + 1;
    std::vector<double> knots(nKnots, 0.0);
    for (int i = 0; i <= degree; ++i) knots[i] = 0.0;
    for (int i = nKnots - degree - 1; i < nKnots; ++i) knots[i] = 1.0;

    int nInner = nKnots - 2 * (degree + 1);
    if (nInner > 0) {
        for (int j = 0; j < nInner; ++j) {
            double t = (j + 1.0) / (nInner + 1.0);
            if (nonuniform) t = std::pow(t, 0.6);
            knots[degree + 1 + j] = t;
        }
    }
    return knots;
}

// ── Cox-de Boor basis (self-contained, no OCCT dependency) ──
static int findSpan(const std::vector<double>& knots, int degree, double u) {
    int n = static_cast<int>(knots.size()) - degree - 1;
    if (u >= knots[n]) return n - 1;
    if (u <= knots[degree]) return degree;
    int lo = degree, hi = n;
    int mid = (lo + hi) / 2;
    while (u < knots[mid] || u >= knots[mid + 1]) {
        if (u < knots[mid]) hi = mid; else lo = mid;
        mid = (lo + hi) / 2;
    }
    return mid;
}

static ArrX basisFunctionsLocal(const std::vector<double>& knots, int degree, double u) {
    int nCtrl = static_cast<int>(knots.size()) - degree - 1;
    ArrX N = ArrX::Zero(nCtrl);
    int span = findSpan(knots, degree, u);

    std::vector<double> left(degree + 1), right(degree + 1);
    std::vector<double> ndu(degree + 1, 0.0);
    ndu[0] = 1.0;

    for (int j = 1; j <= degree; ++j) {
        left[j] = u - knots[span + 1 - j];
        right[j] = knots[span + j] - u;
        double saved = 0.0;
        for (int r = 0; r < j; ++r) {
            double temp = ndu[r] / (right[r + 1] + left[j - r]);
            ndu[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        ndu[j] = saved;
    }

    for (int j = 0; j <= degree; ++j)
        N[span - degree + j] = ndu[j];
    return N;
}

// ── Convert expanded (repeated) knots to compact (unique + mults) ──
static void compactKnots(const std::vector<double>& expanded,
                          std::vector<double>& unique,
                          std::vector<int>& mults)
{
    unique.clear();
    mults.clear();
    for (size_t i = 0; i < expanded.size(); ) {
        double val = expanded[i];
        int cnt = 1;
        while (i + cnt < expanded.size() && std::abs(expanded[i + cnt] - val) < 1e-12)
            ++cnt;
        unique.push_back(val);
        mults.push_back(cnt);
        i += cnt;
    }
}

// ── Constructor from raw data ───────────────────────────────
NurbsSurfaceWrapper::NurbsSurfaceWrapper(
    const Vec3Arr& ctrlPts, int nU, int nV,
    const std::vector<double>& knotsU,
    const std::vector<double>& knotsV,
    int degU, int degV,
    const std::vector<double>& weights)
{
    NCollection_Array2<gp_Pnt> poles(1, nU, 1, nV);
    NCollection_Array2<double> w(1, nU, 1, nV);

    std::vector<double> uUnique, vUnique;
    std::vector<int> uMults, vMults;
    compactKnots(knotsU, uUnique, uMults);
    compactKnots(knotsV, vUnique, vMults);

    NCollection_Array1<double> ku(1, static_cast<int>(uUnique.size()));
    NCollection_Array1<double> kv(1, static_cast<int>(vUnique.size()));
    NCollection_Array1<int> mu(1, static_cast<int>(uMults.size()));
    NCollection_Array1<int> mv(1, static_cast<int>(vMults.size()));

    for (int i = 0; i < nU; ++i) {
        for (int j = 0; j < nV; ++j) {
            int idx = i * nV + j;
            poles(i + 1, j + 1) = gp_Pnt(ctrlPts[idx].x(), ctrlPts[idx].y(), ctrlPts[idx].z());
            w(i + 1, j + 1) = (weights.empty() ||
                               idx >= static_cast<int>(weights.size())) ? 1.0 : weights[idx];
        }
    }
    for (size_t i = 0; i < uUnique.size(); ++i) ku(i + 1) = uUnique[i];
    for (size_t i = 0; i < vUnique.size(); ++i) kv(i + 1) = vUnique[i];
    for (size_t i = 0; i < uMults.size(); ++i) mu(i + 1) = uMults[i];
    for (size_t i = 0; i < vMults.size(); ++i) mv(i + 1) = vMults[i];

    m_surf = new Geom_BSplineSurface(poles, w, ku, kv, mu, mv, degU, degV);
}

NurbsSurfaceWrapper::NurbsSurfaceWrapper(const Handle(Geom_BSplineSurface)& surf)
    : m_surf(surf) {}

// ── Evaluate ─────────────────────────────────────────────────
Vec3 NurbsSurfaceWrapper::evaluate(double u, double v) const {
    gp_Pnt p;
    m_surf->D0(u, v, p);
    return Vec3(p.X(), p.Y(), p.Z());
}

// ── Evaluate derivatives ────────────────────────────────────
NurbsSurfaceWrapper::DerivResult NurbsSurfaceWrapper::evaluateDerivatives(double u, double v) const {
    gp_Pnt S;
    gp_Vec Su, Sv, Suu, Suv, Svv;
    m_surf->D2(u, v, S, Su, Sv, Suu, Suv, Svv);

    DerivResult r;
    r.S  = Vec3(S.X(), S.Y(), S.Z());
    r.Su = Vec3(Su.X(), Su.Y(), Su.Z());
    r.Sv = Vec3(Sv.X(), Sv.Y(), Sv.Z());
    r.Suu = Vec3(Suu.X(), Suu.Y(), Suu.Z());
    r.Suv = Vec3(Suv.X(), Suv.Y(), Suv.Z());
    r.Svv = Vec3(Svv.X(), Svv.Y(), Svv.Z());
    return r;
}

// ── Normal ──────────────────────────────────────────────────
Vec3 NurbsSurfaceWrapper::normal(double u, double v) const {
    gp_Pnt S;
    gp_Vec Su, Sv;
    m_surf->D1(u, v, S, Su, Sv);
    gp_Vec n = Su.Crossed(Sv);
    double norm = n.Magnitude();
    if (norm < 1e-10)
        return Vec3(0, 0, 1);
    return Vec3(n.X() / norm, n.Y() / norm, n.Z() / norm);
}

// ── Principal curvatures ────────────────────────────────────
NurbsSurfaceWrapper::CurvatureResult NurbsSurfaceWrapper::principalCurvatures(double u, double v) const {
    DerivResult d = evaluateDerivatives(u, v);
    double E = d.Su.dot(d.Su);
    double F = d.Su.dot(d.Sv);
    double G = d.Sv.dot(d.Sv);

    Vec3 n = d.Su.cross(d.Sv);
    double nNorm = n.norm();
    if (nNorm < 1e-10) {
        return {0.0, 0.0, Vec3(1, 0, 0), Vec3(0, 1, 0)};
    }
    n /= nNorm;

    double L = d.Suu.dot(n);
    double Mv = d.Suv.dot(n);
    double N = d.Svv.dot(n);

    double denom = E * G - F * F;
    double H, K;
    if (std::abs(denom) < 1e-10) {
        H = 0.0; K = 0.0;
    } else {
        H = (E * N - 2 * F * Mv + G * L) / (2 * denom);
        K = (L * N - Mv * Mv) / denom;
    }

    double disc = std::max(0.0, H * H - K);
    double sqrtDisc = std::sqrt(disc);
    double k1 = H + sqrtDisc;
    double k2 = H - sqrtDisc;

    Vec3 d1, d2;
    if (std::abs(denom) < 1e-10) {
        d1 = Vec3(1, 0, 0); d2 = Vec3(0, 1, 0);
    } else {
        double W11 = (Mv * F - L * G) / denom;
        double W12 = (L * F - Mv * E) / denom;
        double W21 = (N * F - Mv * G) / denom;
        double W22 = (Mv * F - N * E) / denom;

        Eigen::Matrix2d W;
        W << W11, W12, W21, W22;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigs(W);
        Vec2 e1(eigs.eigenvectors()(0, 1), eigs.eigenvectors()(1, 1));
        Vec2 e2(eigs.eigenvectors()(0, 0), eigs.eigenvectors()(1, 0));

        d1 = e1.x() * d.Su + e1.y() * d.Sv;
        d2 = e2.x() * d.Su + e2.y() * d.Sv;
        d1.normalize();
        d2.normalize();

        double absK1 = std::abs(k1), absK2 = std::abs(k2);
        if (absK2 > absK1) {
            std::swap(k1, k2);
            std::swap(d1, d2);
        }
    }

    return {k1, k2, d1, d2};
}

double NurbsSurfaceWrapper::gaussianCurvature(double u, double v) const {
    auto c = principalCurvatures(u, v);
    return c.k1 * c.k2;
}

double NurbsSurfaceWrapper::meanCurvature(double u, double v) const {
    auto c = principalCurvatures(u, v);
    return (c.k1 + c.k2) * 0.5;
}

// ── Project to UV (Gauss-Newton) ────────────────────────────
NurbsSurfaceWrapper::ProjectResult NurbsSurfaceWrapper::projectToUV(
    const Vec3& point, double u0, double v0,
    int maxIter, double tol, double damping) const
{
    auto [umin, umax] = paramDomainU();
    auto [vmin, vmax] = paramDomainV();
    double u = clamp(u0, umin, umax);
    double v = clamp(v0, vmin, vmax);

    for (int iter = 0; iter < maxIter; ++iter) {
        DerivResult d = evaluateDerivatives(u, v);
        Vec3 r = d.S - point;

        if (r.norm() < tol)
            return {u, v, true};

        double J00 = d.Su.dot(d.Su);
        double J01 = d.Su.dot(d.Sv);
        double J11 = d.Sv.dot(d.Sv);
        double b0  = d.Su.dot(r);
        double b1  = d.Sv.dot(r);

        double det = J00 * J11 - J01 * J01;
        double du, dv;
        if (std::abs(det) < 1e-14) {
            du = -damping * b0 / std::max(J00, 1e-10);
            dv = -damping * b1 / std::max(J11, 1e-10);
        } else {
            du = -damping * (J11 * b0 - J01 * b1) / det;
            dv = -damping * (J00 * b1 - J01 * b0) / det;
        }

        du = clamp(du, -0.5, 0.5);
        dv = clamp(dv, -0.5, 0.5);

        u = clamp(u + du, umin, umax);
        v = clamp(v + dv, vmin, vmax);

        if (std::abs(du) < tol && std::abs(dv) < tol)
            return {u, v, true};
    }
    return {u, v, false};
}

// ── Generate mesh ───────────────────────────────────────────
void NurbsSurfaceWrapper::generateMesh(int resU, int resV,
                                       Vec3Arr& vertices, FaceArr& faces, Vec2Arr& uvs) const
{
    auto [umin, umax] = paramDomainU();
    auto [vmin, vmax] = paramDomainV();
    int nU = resU + 1, nV = resV + 1;

    vertices.resize(nU * nV);
    uvs.resize(nU * nV);

    for (int i = 0; i < nU; ++i) {
        double u = umin + (umax - umin) * i / resU;
        for (int j = 0; j < nV; ++j) {
            double v = vmin + (vmax - vmin) * j / resV;
            int idx = i * nV + j;
            vertices[idx] = evaluate(u, v);
            uvs[idx] = Vec2(u, v);
        }
    }

    faces.clear();
    for (int i = 0; i < nU - 1; ++i) {
        for (int j = 0; j < nV - 1; ++j) {
            int a = i * nV + j;
            faces.push_back({a, a + 1, a + nV});
            faces.push_back({a + 1, a + nV + 1, a + nV});
        }
    }
}

std::pair<double, double> NurbsSurfaceWrapper::paramDomainU() const {
    return {m_surf->UKnot(m_surf->FirstUKnotIndex()), m_surf->UKnot(m_surf->LastUKnotIndex())};
}

std::pair<double, double> NurbsSurfaceWrapper::paramDomainV() const {
    return {m_surf->VKnot(m_surf->FirstVKnotIndex()), m_surf->VKnot(m_surf->LastVKnotIndex())};
}

} // namespace distillation
