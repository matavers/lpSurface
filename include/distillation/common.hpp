#pragma once

#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <queue>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <string>
#include <iostream>
#include <iomanip>
#include <functional>
#include <limits>
#include <optional>
#include <variant>
#include <tuple>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
using VecXi = Eigen::VectorXi;
using Mat3 = Eigen::Matrix3d;
using MatX = Eigen::MatrixXd;
using ArrX = Eigen::ArrayXd;

using Vec3Arr = std::vector<Vec3>;
using Vec2Arr = std::vector<Vec2>;
using IntArr = std::vector<int>;
using IntSet = std::unordered_set<int>;
using IntVecSet = std::vector<IntSet>;

struct Face {
    int v0, v1, v2;
    int& operator[](int i) {
        switch (i) {
            case 0: return v0;
            case 1: return v1;
            case 2: return v2;
            default: throw std::out_of_range("Face index out of range");
        }
    }
    int operator[](int i) const {
        switch (i) {
            case 0: return v0;
            case 1: return v1;
            case 2: return v2;
            default: throw std::out_of_range("Face index out of range");
        }
    }
};

using FaceArr = std::vector<Face>;

constexpr double PI = 3.14159265358979323846;
constexpr double EPS = 1e-12;

inline double degToRad(double deg) { return deg * PI / 180.0; }
inline double radToDeg(double rad) { return rad * 180.0 / PI; }

inline double clamp(double x, double lo, double hi) {
    return std::max(lo, std::min(hi, x));
}

inline int clamp(int x, int lo, int hi) {
    return std::max(lo, std::min(hi, x));
}

struct IterationEntry {
    int iteration;
    int numBenchmarks;
    int uncovered;
    double totalOverlap;
    double avgCoverage;
    IntArr benchmarks;
    ArrX coverage;
};
