#include "rulefit/surface.hpp"

#include <ctime>

namespace rulefit {

using namespace distillation;

static NurbsSurfaceWrapper createRandomSurface(int seed) {
    if (seed == 0) seed = (int)std::time(nullptr);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> amp(0.08, 0.35);
    std::uniform_real_distribution<double> freq(1.2, 4.5);
    std::uniform_real_distribution<double> phase(-1.0, 1.0);

    int nU = 9, nV = 9, degU = 3, degV = 3;
    Vec3Arr cp(nU * nV);
    double uMin = -2.0, uMax = 2.0, vMin = -2.0, vMax = 2.0;

    struct Wave { double ax, ay, fx, fy, px, py; };
    int nWaves = 4 + rng() % 3;
    std::vector<Wave> waves(nWaves);
    for (auto& w : waves) {
        w.ax = amp(rng); w.ay = amp(rng);
        w.fx = freq(rng); w.fy = freq(rng);
        w.px = phase(rng); w.py = phase(rng);
    }

    for (int i = 0; i < nU; ++i) {
        double x = uMin + (uMax - uMin) * i / (nU - 1);
        for (int j = 0; j < nV; ++j) {
            double y = vMin + (vMax - vMin) * j / (nV - 1);
            double z = 0;
            for (auto& w : waves)
                z += w.ax * std::sin(w.fx * x + w.px) * std::cos(w.fy * y + w.py)
                   + w.ay * std::cos(w.fx * x * 0.7 + w.px * 1.3) * std::sin(w.fy * y * 1.1 + w.py * 0.8);
            cp[i * nV + j] = Vec3(x, y, z);
        }
    }
    return NurbsSurfaceWrapper(cp, nU, nV,
        makeClampedKnots(nU, degU, true), makeClampedKnots(nV, degV, true), degU, degV);
}

static NurbsSurfaceWrapper createWavySurface() {
    int nU = 9, nV = 9, degU = 3, degV = 3;
    Vec3Arr cp(nU * nV);
    std::vector<double> xs = { -1.5, -1.125, -0.75, -0.375, 0, 0.375, 0.75, 1.125, 1.5 };
    std::vector<double> ys = { -1.5, -1.125, -0.75, -0.375, 0, 0.375, 0.75, 1.125, 1.5 };
    for (int i = 0; i < nU; ++i)
        for (int j = 0; j < nV; ++j) {
            double x = xs[i], y = ys[j];
            cp[i * nV + j] = Vec3(x, y,
                0.15 * sin(2.5 * x) * cos(3.0 * y) +
                0.10 * sin(5.0 * x + 1.2) * sin(4.0 * y + 0.8) +
                0.08 * cos(7.0 * x) * sin(6.0 * y - 0.5) +
                0.05 * sin(9.0 * x - 1.0) * cos(8.0 * y + 1.5));
        }
    return NurbsSurfaceWrapper(cp, nU, nV,
        makeClampedKnots(nU, degU, true), makeClampedKnots(nV, degV, true), degU, degV);
}

static NurbsSurfaceWrapper createMountainTerrain() {
    int nU = 12, nV = 12, degU = 3, degV = 3;
    Vec3Arr cp(nU * nV);
    for (int i = 0; i < nU; ++i) {
        double x = 4.0 * i / (nU - 1), dx = x - 2.0;
        for (int j = 0; j < nV; ++j) {
            double y = 4.0 * j / (nV - 1), dy = y - 2.0;
            double ridge = std::max(1.2 * exp(-pow((dx + dy) * 0.5, 2) * 2.0),
                                    0.9 * exp(-pow((dx - dy) * 0.4, 2) * 3.0));
            double detail = 0.15 * sin(x * 4.0) * cos(y * 3.7)
                          + 0.10 * sin(x * 7.3 + 1.2) * sin(y * 5.1 + 0.8)
                          + 0.06 * cos(x * 10.0 + 2.0) * sin(y * 8.5 + 1.5)
                          + 0.04 * sin(x * 13.0 * y * 0.5);
            cp[i * nV + j] = Vec3(x, y, ridge + detail);
        }
    }
    return NurbsSurfaceWrapper(cp, nU, nV,
        makeClampedKnots(nU, degU, true), makeClampedKnots(nV, degV, true), degU, degV);
}

NurbsSurfaceWrapper createSurface(const std::string& name, int seed) {
    if (name == "mountain") return createMountainTerrain();
    if (name == "wavy") return createWavySurface();
    return createRandomSurface(seed);
}

} // namespace rulefit
