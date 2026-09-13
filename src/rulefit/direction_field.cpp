#include "rulefit/direction_field.hpp"

namespace rulefit {

using namespace distillation;

DirectionField computeAsymptoticField(const NurbsMeshProcessor& mesh) {
    DirectionField f;
    int nV = mesh.numVertices();
    const MatX& curv = mesh.principalCurvatures();      // (n,2): k1, k2
    const Vec3Arr& d1 = mesh.principalDirections1();     // 主方向 1（对应 k1）
    const Vec3Arr& d2 = mesh.principalDirections2();     // 主方向 2（对应 k2）
    const ArrX& K = mesh.gaussianCurvatures();           // K = k1*k2

    f.asym1.resize(nV);
    f.asym2.resize(nV);
    f.hyperbolic.assign(nV, 0);
    f.k1.resize(nV);
    f.k2.resize(nV);

    for (int i = 0; i < nV; ++i) {
        double k1 = curv(i, 0), k2 = curv(i, 1);
        f.k1[i] = k1;
        f.k2[i] = k2;

        // 椭圆点(K>0)或抛物点(K=0)：不存在实渐近方向，标记为不可展区
        if (K[i] >= 0.0) {
            f.hyperbolic[i] = 0;
            f.asym1[i] = Vec3(0, 0, 0);
            f.asym2[i] = Vec3(0, 0, 0);
            continue;
        }

        // 双曲点：方向 d = cosθ·e1 ± sinθ·e2，其中 tan²θ = −k1/k2
        double ratio = -k1 / k2;
        if (ratio < 0.0) ratio = 0.0;
        double theta = std::atan(std::sqrt(ratio));
        double c = std::cos(theta), s = std::sin(theta);

        f.hyperbolic[i] = 1;
        f.asym1[i] = (c * d1[i] + s * d2[i]).normalized();
        f.asym2[i] = (c * d1[i] - s * d2[i]).normalized();
    }
    return f;
}

} // namespace rulefit
