#pragma once

#include "distillation/nurbs_mesh_processor.hpp"

namespace rulefit {

/// 逐顶点的渐近方向场（可展直纹面逼近的母线/扫掠方向的几何先验）。
/// 对双曲点（高斯曲率 K<0），曲面上有两个渐近方向，满足 II(d)=0；
/// 可展直纹面的母线沿一族渐近方向、扫掠沿另一族。二者一般不要求正交。
struct DirectionField {
    std::vector<Vec3> asym1;         // 渐近方向 1（单位向量，切平面内）
    std::vector<Vec3> asym2;         // 渐近方向 2
    std::vector<char> hyperbolic;    // 是否双曲点（存在渐近方向）
    std::vector<double> k1, k2;      // 主曲率（诊断）
};

/// 由主曲率方向计算渐近方向场（纯几何先验，用于可展拟合初始化与椭圆区检测）。
DirectionField computeAsymptoticField(const distillation::NurbsMeshProcessor& mesh);

} // namespace rulefit
