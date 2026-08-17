#pragma once

#include "common.hpp"
#include "nurbs_surface_wrapper.hpp"

namespace distillation {

struct RuledHistoryEntry {
    int iteration;
    double changeRatio;
    int nEmpty;
    IntArr partitionSizes;
    double totalRMS;
    IntArr labels;
};

class HardEMPartitioner {
public:
    HardEMPartitioner(const NurbsSurfaceWrapper& nurbs, int K = 8,
                      int nCtrl = 6, int degree = 3,
                      double tol = 0.001, int maxIter = 30,
                      double damping = 0.0, bool verbose = true);

    std::tuple<IntVecSet, std::vector<RuledHistoryEntry>>
    partition(const Vec3Arr& vertices, const Vec2Arr& uvs);

    /// 热启动分区：用已有顶点分区 initLabels 拟合各分区直母线（directrix）作为
    /// 初始值，再运行 maxIter 步 EM 优化后输出分区。与冷启动 partition() 的区别是
    /// 跳过 k-means++ 随机种子初始化。
    std::tuple<IntVecSet, std::vector<RuledHistoryEntry>>
    partitionWarmStart(const Vec3Arr& vertices, const Vec2Arr& uvs,
                       const IntArr& initLabels, int maxIter);

    const std::vector<MatX>& controlPoints0() const { return m_cp0; }
    const std::vector<MatX>& controlPoints1() const { return m_cp1; }
    const std::vector<double>& knots() const { return m_knots; }

private:
    ArrX basis(double u);
    Vec3 evalDirectrix(const MatX& cp, double u);
    Vec3 evalRuledSurface(int k, double u, double v);
    double distanceAtU(const Vec3& pt, const MatX& cp0, const MatX& cp1, double uj);

    std::tuple<IntArr, double> assignLabels(const Vec3Arr& vertices, const Vec2Arr& uvs);
    void mStep(const Vec3Arr& vertices, const Vec2Arr& uvs, const IntArr& labels);
    std::pair<MatX, MatX> fitPartitionDirectrices(const Vec3Arr& vertices,
                                                    const Vec2Arr& uvs,
                                                    const IntArr& indices);
    void initializeDirectrices(const Vec3Arr& vertices, const Vec2Arr& uvs);
    double computeTotalRMS(const Vec3Arr& vertices, const Vec2Arr& uvs, const IntArr& labels);

    const NurbsSurfaceWrapper& m_nurbs;
    int m_K;
    int m_nCtrl;
    int m_degree;
    double m_tol;
    int m_maxIter;
    double m_damping;
    bool m_verbose;

    std::vector<double> m_knots;
    std::vector<MatX> m_cp0;
    std::vector<MatX> m_cp1;
    IntArr m_labels;
    std::vector<RuledHistoryEntry> m_history;
    std::map<double, ArrX> m_basisCache;
};

} // namespace distillation
