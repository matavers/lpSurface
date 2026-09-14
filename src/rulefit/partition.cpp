#include "rulefit/partition.hpp"

namespace rulefit {

/// 点到截面母线模型的距离（各向异性 + 局部化）：
/// 横向 = 点到最近截面母线的距离（投影 clamp 到母线长度）；
/// 纵向 = 点超出该分区扫掠参数范围 [uMin,uMax] 的惩罚（狭长且局部）；
/// 中心 = 点到分区中心的距离（打破 EM 吞并正反馈）。
static double distToPatch(const Vec3& p, const DevelopablePatch& patch, double lambdaCenter) {
    if (patch.gamma.empty()) return std::numeric_limits<double>::infinity();
    double u = (p - patch.center).dot(patch.swipeDir);
    int s = 0;
    double best = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < patch.uParam.size(); ++i) {
        double du = std::abs(u - patch.uParam[i]);
        if (du < best) { best = du; s = (int)i; }
    }
    const Vec3& d = patch.ruling[s];
    const Vec3& c = patch.gamma[s];
    double v = clamp((p - c).dot(d), -patch.length[s] * 0.5, patch.length[s] * 0.5);
    Vec3 proj = c + v * d;
    double lateral = (p - proj).norm();
    double uPenalty = 0.0;
    if (u < patch.uMin) uPenalty = patch.uMin - u;
    else if (u > patch.uMax) uPenalty = u - patch.uMax;
    double dCenter = (p - patch.center).norm();
    return lateral + uPenalty + lambdaCenter * dCenter;
}

static void kmeansInit(const Vec3Arr& centroids, int K, IntArr& labels) {
    int n = (int)centroids.size();
    labels.assign(n, 0);
    std::mt19937 rng(42);
    Vec3Arr centers(K);
    centers[0] = centroids[rng() % n];
    for (int k = 1; k < K; ++k) {
        double bestD = -1.0;
        int bestI = 0;
        for (int i = 0; i < n; ++i) {
            double d = std::numeric_limits<double>::infinity();
            for (int j = 0; j < k; ++j) d = std::min(d, (centroids[i] - centers[j]).squaredNorm());
            if (d > bestD) { bestD = d; bestI = i; }
        }
        centers[k] = centroids[bestI];
    }
    for (int i = 0; i < n; ++i) {
        double bestD = std::numeric_limits<double>::infinity();
        for (int k = 0; k < K; ++k) {
            double d = (centroids[i] - centers[k]).squaredNorm();
            if (d < bestD) { bestD = d; labels[i] = k; }
        }
    }
}

PartitionResult partitionByDevelopable(
    const Vec3Arr& verts, const FaceArr& faces,
    const Vec3Arr& faceCentroids,
    const std::vector<Vec3>& asymPerVertex,
    int K, int maxIter, double lambdaDev, double lambdaCenter, double lambdaSmooth) {

    PartitionResult result;
    int nF = (int)faces.size();
    if (nF == 0) return result;

    // 构建面邻接（一次，供 ICM 平滑项使用）
    std::vector<IntArr> faceAdj(nF);
    {
        std::map<std::pair<int, int>, IntArr> edgeFaces;
        for (int fi = 0; fi < nF; ++fi) {
            const Face& f = faces[fi];
            for (int c = 0; c < 3; ++c) {
                int a = f[c], b = f[(c + 1) % 3];
                if (a > b) std::swap(a, b);
                edgeFaces[{a, b}].push_back(fi);
            }
        }
        for (auto& kv : edgeFaces) {
            const IntArr& fl = kv.second;
            for (size_t i = 0; i < fl.size(); ++i)
                for (size_t j = 0; j < fl.size(); ++j)
                    if (i != j) faceAdj[fl[i]].push_back(fl[j]);
        }
    }

    IntArr& labels = result.faceLabels;
    kmeansInit(faceCentroids, K, labels);

    for (int iter = 0; iter < maxIter; ++iter) {
        // ── M-step：拟合每个分区的可展直纹面 ──
        std::vector<Vec3Arr> partPts(K);
        std::vector<std::vector<Vec3>> partAsym(K);
        for (int fi = 0; fi < nF; ++fi) {
            int k = labels[fi];
            if (k < 0 || k >= K) continue;
            partPts[k].push_back(faceCentroids[fi]);
            // 每面平均渐近方向（3 顶点平均），与面质心 1:1 对应
            const Face& f = faces[fi];
            Vec3 a1 = Vec3::Zero(), a2 = Vec3::Zero();
            int cnt = 0;
            for (int c = 0; c < 3; ++c) {
                int v = f[c];
                if (2 * v + 1 < (int)asymPerVertex.size()) {
                    a1 += asymPerVertex[2 * v];
                    a2 += asymPerVertex[2 * v + 1];
                    ++cnt;
                }
            }
            if (cnt > 0) {
                a1 /= (double)cnt;
                a2 /= (double)cnt;
                // 过滤椭圆点的零渐近方向（避免 normalized 产生 NaN）
                if (a1.norm() > 1e-8) partAsym[k].push_back(a1.normalized());
                if (a2.norm() > 1e-8) partAsym[k].push_back(a2.normalized());
            }
        }

        result.patches.assign(K, DevelopablePatch{});
        for (int k = 0; k < K; ++k) {
            if (partPts[k].size() < 6) continue;
            result.patches[k] = fitDevelopable(partPts[k], partAsym[k], 12);
        }

        // ── E-step：ICM（图割近似）——数据项 + 平滑项 ──
        double energy = 0.0;
        for (int icmIter = 0; icmIter < 5; ++icmIter) {
            for (int fi = 0; fi < nF; ++fi) {
                double bestE = std::numeric_limits<double>::infinity();
                int bestK = labels[fi];
                for (int k = 0; k < K; ++k) {
                    if (result.patches[k].numPoints < 6) continue;
                    double data = distToPatch(faceCentroids[fi], result.patches[k], lambdaCenter)
                                + lambdaDev * result.patches[k].twist;
                    double smooth = 0.0;
                    for (int fj : faceAdj[fi]) if (labels[fj] != k) smooth += 1.0;
                    double e = data + lambdaSmooth * smooth;
                    if (e < bestE) { bestE = e; bestK = k; }
                }
                labels[fi] = bestK;
                energy += bestE;
            }
        }
        result.energy = energy;
        result.iterations = iter + 1;

        // 调试：每轮分区规模
        {
            std::vector<int> sizes(K, 0);
            for (int fi = 0; fi < nF; ++fi) sizes[labels[fi]]++;
            int nNonEmpty = 0, maxSize = 0;
            for (int k = 0; k < K; ++k) { if (sizes[k] > 0) nNonEmpty++; maxSize = std::max(maxSize, sizes[k]); }
            std::cout << "    iter " << iter << ": nonEmpty=" << nNonEmpty
                      << " maxSize=" << maxSize << " energy=" << energy << "\n";
        }
    }

    // 统计非空分区数
    result.nParts = 0;
    for (int k = 0; k < K; ++k) {
        if (result.patches[k].numPoints >= 6) result.nParts++;
    }
    return result;
}

} // namespace rulefit
