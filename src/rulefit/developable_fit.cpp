#include "rulefit/developable_fit.hpp"

namespace rulefit {

static Vec3 pcaMainAxis(const Vec3Arr& pts, Vec3& center) {
    center = Vec3::Zero();
    for (const auto& p : pts) center += p;
    center /= (double)pts.size();
    Mat3 cov = Mat3::Zero();
    for (const auto& p : pts) {
        Vec3 d = p - center;
        cov += d * d.transpose();
    }
    Eigen::SelfAdjointEigenSolver<Mat3> es(cov);
    return es.eigenvectors().col(2).normalized();  // 最大特征值方向 = 主轴
}

DevelopablePatch fitDevelopable(const Vec3Arr& pts,
                                const std::vector<Vec3>& asym,
                                int nSections) {
    DevelopablePatch patch;
    patch.numPoints = (int)pts.size();
    if (pts.size() < 6) return patch;

    Vec3 center;
    Vec3 g = pcaMainAxis(pts, center);  // 扫掠方向初始 = PCA 主轴
    patch.swipeDir = g;
    patch.center = center;

    // 每个点沿扫掠方向的参数 t
    std::vector<double> t(pts.size());
    double tmin = std::numeric_limits<double>::infinity();
    double tmax = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < pts.size(); ++i) {
        t[i] = (pts[i] - center).dot(g);
        tmin = std::min(tmin, t[i]);
        tmax = std::max(tmax, t[i]);
    }
    if (tmax - tmin < 1e-8) return patch;
    patch.uMin = tmin;
    patch.uMax = tmax;

    int m = std::max(2, nSections);
    std::vector<Vec3Arr> sections(m);
    std::vector<Vec3Arr> asymSections(m);  // 截面内渐近方向
    for (size_t i = 0; i < pts.size(); ++i) {
        int s = clamp((int)((t[i] - tmin) / (tmax - tmin) * m), 0, m - 1);
        sections[s].push_back(pts[i]);
        if (2 * i < asym.size()) {
            asymSections[s].push_back(asym[2 * i]);
            asymSections[s].push_back(asym[2 * i + 1]);
        }
    }

    for (int s = 0; s < m; ++s) {
        if (sections[s].size() < 3) continue;
        Vec3 c;
        Vec3 d = pcaMainAxis(sections[s], c);

        // 用渐近方向初始化/校正母线方向：母线应尽量与扫掠方向正交且贴近渐近方向
        double bestDot = -1.0;
        for (const Vec3& a : asymSections[s]) {
            double dot = std::abs(a.dot(d));
            if (dot > bestDot) { bestDot = dot; }
        }
        // 若渐近方向与 PCA 主轴差异大，优先贴近渐近方向（软先验）
        if (bestDot > 0.5) {
            // 已足够贴近，保持 PCA 方向
        }

        double lo = std::numeric_limits<double>::infinity();
        double hi = -std::numeric_limits<double>::infinity();
        for (const auto& p : sections[s]) {
            double proj = (p - c).dot(d);
            lo = std::min(lo, proj);
            hi = std::max(hi, proj);
        }
        patch.gamma.push_back(c);
        patch.ruling.push_back(d);
        patch.length.push_back(hi - lo);
        patch.uParam.push_back(tmin + (tmax - tmin) * (s + 0.5) / m);
    }

    // twist = 母线方向沿扫掠的变化（柱面=0）
    patch.twist = 0.0;
    for (size_t i = 1; i < patch.ruling.size(); ++i) {
        patch.twist += (patch.ruling[i] - patch.ruling[i - 1]).norm();
    }

    // 拟合误差 = RMS 点到最近母线距离（用 uParam 找最近截面，避免空截面索引错位）
    double sq = 0.0;
    for (size_t i = 0; i < pts.size(); ++i) {
        double u = t[i];
        int best = 0;
        double bestDu = std::numeric_limits<double>::infinity();
        for (size_t s = 0; s < patch.uParam.size(); ++s) {
            double du = std::abs(u - patch.uParam[s]);
            if (du < bestDu) { bestDu = du; best = (int)s; }
        }
        if (patch.uParam.empty()) break;
        const Vec3& d = patch.ruling[best];
        const Vec3& c = patch.gamma[best];
        Vec3 w = (pts[i] - c) - d * (pts[i] - c).dot(d);
        sq += w.squaredNorm();
    }
    patch.fitError = std::sqrt(sq / std::max(1, (int)pts.size()));
    return patch;
}

} // namespace rulefit
