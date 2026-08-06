#pragma once

#include "common.hpp"
#include "nurbs_surface_wrapper.hpp"

#include <Geom2d_BSplineCurve.hxx>

namespace distillation {

struct TolResult {
    double maxDist = 0.0;
    double rmsDist = 0.0;
    double bestAngleDeg = 0.0;
    int    bestDirection = 0;
    bool   converged = false;
};

struct ToleranceSpec {
    double targetMaxDist  = 0.01;
    double targetRMSDist  = 0.005;
    int    angleSteps     = 36;
    int    samplesPerCurve = 30;
    int    samplesPerRuling = 20;
    int    maxRetries      = 5;
    double sigmaStepRatio  = 0.8;
};

TolResult computeToleranceParallel(
    const Handle(Geom2d_BSplineCurve)& C0,
    const Handle(Geom2d_BSplineCurve)& C1,
    const NurbsSurfaceWrapper& nurbs,
    const ToleranceSpec& spec = ToleranceSpec());

TolResult computeToleranceGeneral(
    const Handle(Geom2d_BSplineCurve)& C0,
    const Handle(Geom2d_BSplineCurve)& C1,
    const NurbsSurfaceWrapper& nurbs,
    const ToleranceSpec& spec = ToleranceSpec());

} // namespace distillation
