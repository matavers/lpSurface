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

    // 扫掠方向初始 = PCA 主轴（条带长轴；渐近方向场用于母线方向）
    Vec3 center;
    Vec3 g = pcaMainAxis(pts, center);
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
    std::vector<Vec3Arr> asymSections(m);  // 截面内渐近方向（asym1/asym2 展开）
    for (size_t i = 0; i < pts.size(); ++i) {
        int s = clamp((int)((t[i] - tmin) / (tmax - tmin) * m), 0, m - 1);
        sections[s].push_back(pts[i]);
        if (2 * i + 1 < asym.size()) {
            asymSections[s].push_back(asym[2 * i]);
            asymSections[s].push_back(asym[2 * i + 1]);
        }
    }

    // 每个截面：母线方向 = 截面 PCA 主轴（渐近方向场尚需 4-rosy 平滑去歧义，
    // 先以 PCA 主轴为母线方向基线，后续方向场平滑后替换为渐近方向）。
    for (int s = 0; s < m; ++s) {
        if (sections[s].size() < 3) continue;
        Vec3 c = Vec3::Zero();
        for (const auto& p : sections[s]) c += p;
        c /= (double)sections[s].size();

        Vec3 dummy;
        Vec3 d = pcaMainAxis(sections[s], dummy);

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

    // #2 软约束可展：母线方向沿扫掠方向平滑（柱面化，减小 twist）
    // 柱面 = 母线方向恒定，是最简单可加工的可展面。
    for (int iter = 0; iter < 2; ++iter) {
        for (size_t i = 1; i + 1 < patch.ruling.size(); ++i) {
            Vec3 sm = patch.ruling[i - 1] + patch.ruling[i + 1];
            if (sm.norm() > 1e-8) patch.ruling[i] = sm.normalized();
        }
    }

    // twist = 母线方向沿扫掠的变化（柱面=0，最可展）
    patch.twist = 0.0;
    for (size_t i = 1; i < patch.ruling.size(); ++i) {
        patch.twist += (patch.ruling[i] - patch.ruling[i - 1]).norm();
    }

    // 拟合误差 = RMS 点到最近母线距离
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
