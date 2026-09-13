#include "rulefit/evaluate.hpp"

namespace rulefit {

Evaluation evaluate(const std::vector<DevelopablePatch>& patches, double twistThreshold) {
    Evaluation e;
    e.nPatches = (int)patches.size();
    for (const auto& p : patches) {
        if (p.numPoints < 6) continue;
        e.twist.push_back(p.twist);
        e.fitError.push_back(p.fitError);
        if (p.twist < twistThreshold) e.nDevelopable++;
    }
    auto mean = [](const std::vector<double>& v) {
        if (v.empty()) return 0.0;
        double s = 0.0;
        for (double x : v) s += x;
        return s / (double)v.size();
    };
    e.meanTwist = mean(e.twist);
    e.meanFitError = mean(e.fitError);
    return e;
}

} // namespace rulefit
