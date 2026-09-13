#pragma once

#include "distillation/common.hpp"

namespace rulefit {

/// 一个分区的可展直纹面拟合结果（截面母线模型）。
/// S(u,v) = γ(u) + v·d(u)，γ 是 director curve，d 是母线方向。
struct DevelopablePatch {
    std::vector<Vec3> gamma;      // director curve 顶点（截面中点）
    std::vector<Vec3> ruling;     // 母线方向（单位向量）
    std::vector<double> length;   // 母线长度
    std::vector<double> uParam;   // 截面沿扫掠方向的参数位置
    Vec3 swipeDir;                // 扫掠方向（单位向量）
    Vec3 center;                  // 参考中心（u=0）
    double uMin = 0.0, uMax = 0.0; // 扫掠参数范围
    double twist = 0.0;           // 可展偏差（母线方向沿扫掠的变化）
    double fitError = 0.0;        // 拟合误差（RMS 点到母线距离）
    int numPoints = 0;
};

/// 软约束可展直纹面拟合（截面母线模型）。
/// 沿扫掠方向把点集分成 mSections 个截面，每截面拟合一条母线；
/// twist 度量母线方向沿扫掠的变化（柱面=0，最可展）与偏离渐近方向的程度。
/// @param pts      分区点集（3D）
/// @param asym     每个点的两个渐近方向（asym1, asym2 展开为 2*n 个方向）
/// @param nSections 截面数
DevelopablePatch fitDevelopable(const Vec3Arr& pts,
                                const std::vector<Vec3>& asym,
                                int nSections = 16);

} // namespace rulefit
