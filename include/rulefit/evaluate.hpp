#pragma once

#include "distillation/common.hpp"
#include "rulefit/developable_fit.hpp"

namespace rulefit {

/// 可展性评估结果。
struct Evaluation {
    std::vector<double> twist;        // 每分区可展偏差
    std::vector<double> fitError;     // 每分区拟合误差
    double meanTwist = 0.0;
    double meanFitError = 0.0;
    int nDevelopable = 0;             // twist < threshold 的分区数
    int nPatches = 0;
};

/// 评估所有分区的可展性与拟合质量。
/// @param twistThreshold 判定"可展"的 twist 阈值
Evaluation evaluate(const std::vector<DevelopablePatch>& patches, double twistThreshold = 0.2);

} // namespace rulefit
