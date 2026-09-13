#pragma once

#include "distillation/nurbs_surface_wrapper.hpp"

namespace rulefit {

/// 合成测试曲面（与旧 OCCT 相同的 wavy/random/mountain，用于起步验证）
distillation::NurbsSurfaceWrapper createSurface(const std::string& name, int seed = 0);

} // namespace rulefit
