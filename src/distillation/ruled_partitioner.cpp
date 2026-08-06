#include "distillation/ruled_partitioner.hpp"

namespace distillation {

static int findSpanRuled(const std::vector<double>& knots, int degree, double u) {
    int n = static_cast<int>(knots.size()) - degree - 1;
    if (u >= knots[n]) return n - 1;
    if (u <= knots[degree]) return degree;
    int lo = degree, hi = n;
    int mid = (lo + hi) / 2;
    while (u < knots[mid] || u >= knots[mid + 1]) {
        if (u < knots[mid]) hi = mid; else lo = mid;
        mid = (lo + hi) / 2;
    }
    return mid;
}

static ArrX basisFunctionsRuled(const std::vector<double>& knots, int degree, double u) {
    int nCtrl = static_cast<int>(knots.size()) - degree - 1;
    ArrX N = ArrX::Zero(nCtrl);
    int span = findSpanRuled(knots, degree, u);

    std::vector<double> left(degree + 1), right(degree + 1), ndu(degree + 1, 0.0);
    ndu[0] = 1.0;
    for (int j = 1; j <= degree; ++j) {
        left[j] = u - knots[span + 1 - j];
        right[j] = knots[span + j] - u;
        double saved = 0.0;
        for (int r = 0; r < j; ++r) {
            double temp = ndu[r] / (right[r + 1] + left[j - r]);
            ndu[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        ndu[j] = saved;
    }
    for (int j = 0; j <= degree; ++j)
        N[span - degree + j] = ndu[j];
    return N;
}

HardEMPartitioner::HardEMPartitioner(const NurbsSurfaceWrapper& nurbs, int K,
                                     int nCtrl, int degree,
                                     double tol, int maxIter,
                                     double damping, bool verbose)
    : m_nurbs(nurbs), m_K(std::min(K, 32)),
      m_nCtrl(std::max(degree + 1, nCtrl)),
      m_degree(degree), m_tol(tol), m_maxIter(maxIter),
      m_damping(clamp(damping, 0.0, 0.9)), m_verbose(verbose)
{
    m_knots = makeClampedKnots(m_nCtrl, degree, false);
}

ArrX HardEMPartitioner::basis(double u) {
    double key = std::round(u * 1e12) / 1e12;
    auto it = m_basisCache.find(key);
    if (it != m_basisCache.end()) return it->second;
    ArrX b = basisFunctionsRuled(m_knots, m_degree, u);
    m_basisCache[key] = b;
    return b;
}

Vec3 HardEMPartitioner::evalDirectrix(const MatX& cp, double u) {
    ArrX b = basis(u);
    Vec3 r(0, 0, 0);
    for (int i = 0; i < m_nCtrl; ++i)
        r += b(i) * cp.row(i);
    return r;
}

Vec3 HardEMPartitioner::evalRuledSurface(int k, double u, double v) {
    Vec3 c0 = evalDirectrix(m_cp0[k], u);
    Vec3 c1 = evalDirectrix(m_cp1[k], u);
    return (1.0 - v) * c0 + v * c1;
}

double HardEMPartitioner::distanceAtU(const Vec3& pt, const MatX& cp0,
                                       const MatX& cp1, double uj) {
    Vec3 c0 = evalDirectrix(cp0, uj);
    Vec3 c1 = evalDirectrix(cp1, uj);
    Vec3 dVec = c1 - c0;
    double d2 = dVec.squaredNorm();
    if (d2 < 1e-16)
        return (c0 - pt).squaredNorm();
    double vRaw = (pt - c0).dot(dVec) / d2;
    double vOpt = clamp(vRaw, 0.0, 1.0);
    Vec3 proj = c0 + vOpt * dVec;
    return (proj - pt).squaredNorm();
}

std::tuple<IntArr, double> HardEMPartitioner::assignLabels(
    const Vec3Arr& vertices, const Vec2Arr& uvs)
{
    int N = static_cast<int>(vertices.size());
    IntArr labels(N, 0);

    for (int j = 0; j < N; ++j) {
        double uj = uvs[j].x();
        int bestK = 0;
        double bestD2 = std::numeric_limits<double>::infinity();
        for (int k = 0; k < m_K; ++k) {
            double d2 = distanceAtU(vertices[j], m_cp0[k], m_cp1[k], uj);
            if (d2 < bestD2) { bestD2 = d2; bestK = k; }
        }
        labels[j] = bestK;
    }

    if (!m_labels.empty() && m_damping > 0.0) {
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (int j = 0; j < N; ++j) {
            if (dist(rng) < m_damping) labels[j] = m_labels[j];
        }
    }

    // majority filter: eliminate single-vertex noise strips
    {
        // determine grid size from UV regular grid (e.g. 61x61)
        int nU = static_cast<int>(std::sqrt(N) + 0.5);
        int nV = N / nU;
        for (int pass = 0; pass < 5; ++pass) {
            IntArr filtered = labels;
            for (int i = 0; i < nU; ++i) {
                for (int j = 0; j < nV; ++j) {
                    int idx = i * nV + j;
                    std::vector<int> neighborPids;
                    if (i > 0)      neighborPids.push_back(labels[(i-1)*nV + j]);
                    if (i < nU-1)   neighborPids.push_back(labels[(i+1)*nV + j]);
                    if (j > 0)      neighborPids.push_back(labels[i*nV + (j-1)]);
                    if (j < nV-1)   neighborPids.push_back(labels[i*nV + (j+1)]);

                    std::unordered_map<int,int> cnt;
                    int bestCnt = 0, bestPid = labels[idx];
                    for (int nb : neighborPids) {
                        int c = ++cnt[nb];
                        if (c > bestCnt) { bestCnt = c; bestPid = nb; }
                    }
                    if (bestPid != labels[idx]) {
                        // only flip if at least 3 of 4 neighbors agree
                        if (bestCnt >= 3)
                            filtered[idx] = bestPid;
                    }
                }
            }
            labels = filtered;
        }
    }

    double changeRatio = m_labels.empty() ? 1.0
        : static_cast<double>((Eigen::Map<const VecXi>(labels.data(), N).array()
                                != Eigen::Map<const VecXi>(m_labels.data(), N).array()).count()) / N;

    m_labels = labels;
    return {labels, changeRatio};
}

std::pair<MatX, MatX> HardEMPartitioner::fitPartitionDirectrices(
    const Vec3Arr& vertices, const Vec2Arr& uvs, const IntArr& indices)
{
    int nPts = static_cast<int>(indices.size());
    if (nPts < m_degree + 1) {
        Vec3 mean(0, 0, 0);
        for (int idx : indices) mean += vertices[idx];
        mean /= nPts;
        MatX cp0 = MatX::Zero(m_nCtrl, 3);
        MatX cp1 = MatX::Zero(m_nCtrl, 3);
        for (int i = 0; i < m_nCtrl; ++i) { cp0.row(i) = mean; cp1.row(i) = mean; }
        return {cp0, cp1};
    }

    MatX A(nPts, 2 * m_nCtrl);
    for (int row = 0; row < nPts; ++row) {
        int j = indices[row];
        double uj = uvs[j].x(), vj = uvs[j].y();
        ArrX N = basis(uj);
        for (int i = 0; i < m_nCtrl; ++i) {
            A(row, i) = (1.0 - vj) * N(i);
            A(row, m_nCtrl + i) = vj * N(i);
        }
    }

    MatX cp(2 * m_nCtrl, 3);
    for (int dim = 0; dim < 3; ++dim) {
        Eigen::VectorXd b(nPts);
        for (int row = 0; row < nPts; ++row) b(row) = vertices[indices[row]](dim);
        cp.col(dim) = A.colPivHouseholderQr().solve(b);
    }

    return {cp.topRows(m_nCtrl), cp.bottomRows(m_nCtrl)};
}

void HardEMPartitioner::mStep(const Vec3Arr& vertices, const Vec2Arr& uvs,
                               const IntArr& labels) {
    for (int k = 0; k < m_K; ++k) {
        IntArr idx;
        for (int j = 0; j < static_cast<int>(labels.size()); ++j)
            if (labels[j] == k) idx.push_back(j);
        if (idx.empty()) continue;
        auto [cp0, cp1] = fitPartitionDirectrices(vertices, uvs, idx);
        m_cp0[k] = cp0;
        m_cp1[k] = cp1;
    }
}

void HardEMPartitioner::initializeDirectrices(const Vec3Arr& vertices,
                                                const Vec2Arr& uvs) {
    int N = static_cast<int>(vertices.size());
    int K = std::min(m_K, N);

    std::mt19937 rng(42);
    IntArr seeds;
    seeds.push_back(static_cast<int>(rng() % N));
    ArrX dists = ArrX::Constant(N, std::numeric_limits<double>::infinity());

    for (int k = 1; k < K; ++k) {
        Vec2 lastUV = uvs[seeds.back()];
        for (int i = 0; i < N; ++i)
            dists[i] = std::min(dists[i], (uvs[i] - lastUV).squaredNorm());
        int maxI = 0; double maxD = dists[0];
        for (int i = 1; i < N; ++i)
            if (dists[i] > maxD) { maxD = dists[i]; maxI = i; }
        seeds.push_back(maxI);
    }

    m_cp0.resize(m_K, MatX::Zero(m_nCtrl, 3));
    m_cp1.resize(m_K, MatX::Zero(m_nCtrl, 3));
    for (int k = 0; k < K; ++k) {
        Vec3 pt = vertices[seeds[k]];
        for (int i = 0; i < m_nCtrl; ++i) {
            m_cp0[k].row(i) = pt;
            m_cp1[k].row(i) = pt;
        }
    }
    for (int k = K; k < m_K; ++k) {
        Vec3 pt = vertices[seeds.back()];
        for (int i = 0; i < m_nCtrl; ++i) {
            m_cp0[k].row(i) = pt;
            m_cp1[k].row(i) = pt;
        }
    }
}

double HardEMPartitioner::computeTotalRMS(const Vec3Arr& vertices,
                                            const Vec2Arr& uvs,
                                            const IntArr& labels) {
    double totalD2 = 0.0;
    int N = static_cast<int>(vertices.size());
    for (int j = 0; j < N; ++j) {
        double d2 = distanceAtU(vertices[j], m_cp0[labels[j]],
                                m_cp1[labels[j]], uvs[j].x());
        totalD2 += d2;
    }
    return std::sqrt(totalD2 / std::max(1, N));
}

std::tuple<IntVecSet, std::vector<RuledHistoryEntry>>
HardEMPartitioner::partition(const Vec3Arr& vertices, const Vec2Arr& uvs) {
    int N = static_cast<int>(vertices.size());
    if (m_verbose) {
        std::cout << "Hard-EM v2: K=" << m_K << ", nCtrl=" << m_nCtrl
                  << ", degree=" << m_degree << ", tol=" << m_tol
                  << ", damping=" << m_damping << ", vertices=" << N << std::endl;
    }

    m_basisCache.clear();
    initializeDirectrices(vertices, uvs);
    m_labels.clear();
    m_history.clear();

    for (int it = 0; it < m_maxIter; ++it) {
        auto [labels, change] = assignLabels(vertices, uvs);

        int nEmpty = 0;
        IntArr sizes(m_K, 0);
        for (int j = 0; j < N; ++j) sizes[labels[j]]++;
        for (int k = 0; k < m_K; ++k) if (sizes[k] == 0) nEmpty++;

        double totalRMS = computeTotalRMS(vertices, uvs, labels);

        m_history.push_back({it + 1, change, nEmpty, sizes, totalRMS, labels});

        if (m_verbose) {
            std::cout << "  iter " << (it + 1) << ": change=" << change
                      << " empty=" << nEmpty << " RMS=" << totalRMS << " sizes=[";
            for (int k = 0; k < m_K; ++k) std::cout << sizes[k] << " ";
            std::cout << "]" << std::endl;
        }

        if (it > 1 && change < m_tol) {
            if (m_verbose) std::cout << "  => converged: change " << change
                                     << " < " << m_tol << std::endl;
            break;
        }

        mStep(vertices, uvs, labels);
    }

    IntVecSet partitions(m_K);
    for (int k = 0; k < m_K; ++k) partitions[k] = IntSet{};

    if (!m_labels.empty()) {
        for (int j = 0; j < N; ++j)
            partitions[m_labels[j]].insert(j);
    }

    if (m_verbose) {
        std::cout << "\n  Final ruled surfaces:" << std::endl;
        for (int k = 0; k < m_K; ++k) {
            if (!partitions[k].empty()) {
                double span0 = (m_cp0[k].row(0) - m_cp0[k].row(m_nCtrl - 1)).norm();
                double span1 = (m_cp1[k].row(0) - m_cp1[k].row(m_nCtrl - 1)).norm();
                double diff = (m_cp1[k] - m_cp0[k]).rowwise().norm().mean();
                std::cout << "    #" << k << ": " << partitions[k].size() << " verts"
                          << " |C0|=" << span0 << " |C1|=" << span1
                          << " |C1-C0|=" << diff << std::endl;
            }
        }
    }

    return {partitions, m_history};
}

} // namespace distillation
