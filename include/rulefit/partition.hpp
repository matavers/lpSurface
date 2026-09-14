#pragma once

#include "distillation/common.hpp"
#include "rulefit/developable_fit.hpp"

namespace rulefit {

/// 分区结果。
struct PartitionResult {
    IntArr faceLabels;                       // 每个面的分区标签
    int nParts = 0;                          // 实际非空分区数
    std::vector<DevelopablePatch> patches;   // 每个分区的可展直纹面
    double energy = 0.0;                     // 总能量（数据 + 可展）
    int iterations = 0;
};

/// 可展直纹面驱动的分区（EM）：
///  E-step 每面分给最近的可展直纹面（各向异性：沿扫掠方向狭长）；
///  M-step 每个分区软约束可展拟合（fitDevelopable）。
/// @param faceCentroids   每个面的质心（3D）
/// @param asymPerVertex   每顶点两个渐近方向（2*nV 个 Vec3），用于可展拟合先验
PartitionResult partitionByDevelopable(
    const Vec3Arr& verts, const FaceArr& faces,
    const Vec3Arr& faceCentroids,
    const std::vector<Vec3>& asymPerVertex,
    int K, int maxIter = 30, double lambdaDev = 1.0,
    double lambdaCenter = 0.3, double lambdaSmooth = 0.05);

} // namespace rulefit
