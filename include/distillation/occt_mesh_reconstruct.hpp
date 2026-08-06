#pragma once

#include "common.hpp"
#include "boundary_smoother.hpp"

namespace distillation {

std::tuple<Vec2Arr, FaceArr, IntArr> occtConstrainedReconstruct(
    const Vec2Arr& originalUVs,
    const FaceArr& originalFaces,
    const IntArr& originalFaceLabels,
    const BoundaryNetwork& net,
    double uMin, double uMax,
    double vMin, double vMax);

} // namespace distillation
