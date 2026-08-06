"""
Sandwich bounds: optimal correspondences + monotone smooth envelope curves.

Algorithm:
  1. Sample curve A uniformly at N points: (u_i, P_i, UV_i)
  2. For each P_i, golden-section search on curve B for optimal v_i*
     minimizing the single-ruling NURBS geodesic integral ∫‖L(t)-G(t)‖² dt
  3. PAV isotonic regression: enforce v_i* monotonic (increasing)
  4. Build two Fritsch-Carlson monotonic cubic splines:
       φ_lower(u):  through (u_i, max(0, v_i* - delta))
       φ_upper(u):  through (u_i, min(1, v_i* + delta))

These form a monotone corridor [φ_lower(u), φ_upper(u)] that constrains
the mapping φ during ruled-surface optimization.
"""

import numpy as np

# Gauss-Legendre 5-point on [0,1]
_gl_t = np.array([0.046910077030668, 0.2307653449471585, 0.5, 0.7692346550528415, 0.953089922969332], dtype=np.float64)
_gl_w = np.array([0.1184634425280945, 0.239314335249683, 0.2844444444444445, 0.239314335249683, 0.1184634425280945], dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════
# Low-level helpers (polyline, NURBS, golden-section)
# ═══════════════════════════════════════════════════════════════════

def _arc_length_param(pts):
    segs = np.diff(pts, axis=0)
    lens = np.linalg.norm(segs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(lens)])
    if cum[-1] < 1e-12:
        return np.linspace(0, 1, len(pts))
    return cum / cum[-1]


def _resample_polyline(pts, params, t_new):
    idx = np.searchsorted(params[1:-1], t_new, side='right')
    idx = np.clip(idx, 0, len(pts) - 2)
    t0 = params[idx]; t1 = params[idx + 1]
    denom = np.where(t1 - t0 < 1e-12, 1.0, t1 - t0)
    alpha = ((t_new - t0) / denom).clip(0, 1)
    return pts[idx] * (1 - alpha[:, None]) + pts[idx + 1] * alpha[:, None]


def _ruling_integral(nurbs_surf, p_3d, p_uv, q_3d, q_uv):
    """Single-ruling error ∫₀¹ ‖(1-t)P+tQ - S(u(t),v(t))‖² dt (5-pt GL)."""
    u_p, v_p = p_uv
    u_q, v_q = q_uv

    L_pts = p_3d + np.outer(_gl_t, q_3d - p_3d)          # (5, 3)
    G_pts = np.zeros((5, 3))
    for k in range(5):
        u_mid = u_p + _gl_t[k] * (u_q - u_p)
        v_mid = v_p + _gl_t[k] * (v_q - v_p)
        G_pts[k] = nurbs_surf.evaluate(u_mid, v_mid)

    diff = L_pts - G_pts
    sq = np.sum(diff * diff, axis=1)
    return float(np.dot(sq, _gl_w))


def _golden_search(f, a, b, tol=1e-3, max_iter=50):
    phi = (np.sqrt(5) - 1) / 2
    c = b - phi * (b - a); d = a + phi * (b - a)
    fc = f(c); fd = f(d)
    for _ in range(max_iter):
        if b - a < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a); fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a); fd = f(d)
    return (a + b) / 2


def extract_curves(loop_3d, loop_uv, t_P, t_Q):
    """Split closed polygon loop at parameters t_P, t_Q into two arcs.

    Returns:
        arc1_3d, arc1_uv  – P→Q forward  (Curve A)
        arc2_3d, arc2_uv  – P→Q backward (Curve B, wraps around)
    """
    N = len(loop_3d)
    if t_P > t_Q:
        t_P, t_Q = t_Q, t_P

    idx_p = int(round(t_P * (N - 1)))
    idx_q = int(round(t_Q * (N - 1)))
    idx_p = max(0, min(N - 2, idx_p))
    idx_q = max(idx_p + 1, min(N - 1, idx_q))

    grid = np.linspace(0, 1, N)
    P_3d = _resample_polyline(loop_3d, grid, np.array([t_P]))[0]
    Q_3d = _resample_polyline(loop_3d, grid, np.array([t_Q]))[0]
    P_uv = _resample_polyline(loop_uv,  grid, np.array([t_P]))[0]
    Q_uv = _resample_polyline(loop_uv,  grid, np.array([t_Q]))[0]

    pts1_3d = [P_3d] + [loop_3d[i] for i in range(idx_p + 1, idx_q)] + [Q_3d]
    pts1_uv = [P_uv] + [loop_uv[i] for i in range(idx_p + 1, idx_q)] + [Q_uv]

    pts2_3d = [P_3d] + [loop_3d[i] for i in range(idx_p - 1, -1, -1)] + \
              [loop_3d[i] for i in range(N - 1, idx_q, -1)] + [Q_3d]
    pts2_uv = [P_uv] + [loop_uv[i] for i in range(idx_p - 1, -1, -1)] + \
              [loop_uv[i] for i in range(N - 1, idx_q, -1)] + [Q_uv]

    return (np.array(pts1_3d), np.array(pts1_uv),
            np.array(pts2_3d), np.array(pts2_uv))


# ═══════════════════════════════════════════════════════════════════
# Optimal correspondence search
# ═══════════════════════════════════════════════════════════════════

def find_optimal_correspondences(nurbs_surf, curve_a_3d, curve_a_uv,
                                 curve_b_3d, curve_b_uv, N_samples=100,
                                 tol=1e-3, search_window=0.30):
    """For each uniform sample on Curve A, find the best v on Curve B.

    The search is constrained to v ∈ [max(0, u_i - window), min(1, u_i + window)]
    to avoid spurious local minima in the flat NURBS error landscape.

    Returns:
        u_grid: (N,) uniform parameters on [0,1]
        v_opt:  (N,) optimal Curve B parameters
    """
    params_a = _arc_length_param(curve_a_3d)
    u_grid = np.linspace(0, 1, N_samples)
    P_list = _resample_polyline(curve_a_3d, params_a, u_grid)
    uv_list = _resample_polyline(curve_a_uv,  params_a, u_grid)

    params_b = _arc_length_param(curve_b_3d)

    v_opt = np.zeros(N_samples)

    for i in range(N_samples):
        P_i = P_list[i]; uv_i = uv_list[i]

        def objective(v):
            v = float(np.clip(v, 0, 1))
            Q = _resample_polyline(curve_b_3d, params_b, np.array([v]))[0]
            Q_uv = _resample_polyline(curve_b_uv,  params_b, np.array([v]))[0]
            return _ruling_integral(nurbs_surf, P_i, uv_i, Q, Q_uv)

        v_opt[i] = _golden_search(objective,
                                   max(0.0, u_grid[i] - search_window),
                                   min(1.0, u_grid[i] + search_window),
                                   tol=tol)

    return u_grid, v_opt


# ═══════════════════════════════════════════════════════════════════
# PAV isotonic regression
# ═══════════════════════════════════════════════════════════════════

def isotonic_regression(y):
    """Pool Adjacent Violators – project y onto monotone increasing cone."""
    n = len(y)
    pools = [(float(y[i]), 1.0) for i in range(n)]  # (sum, count)

    i = 0
    while i < len(pools) - 1:
        mean_i = pools[i][0] / pools[i][1]
        mean_j = pools[i + 1][0] / pools[i + 1][1]
        if mean_i > mean_j:
            pools[i] = (pools[i][0] + pools[i + 1][0],
                        pools[i][1] + pools[i + 1][1])
            pools.pop(i + 1)
            if i > 0:
                i -= 1
        else:
            i += 1

    result = np.zeros(n)
    idx = 0
    for s, c in pools:
        result[idx:idx + int(c)] = s / c
        idx += int(c)
    return result


# ═══════════════════════════════════════════════════════════════════
# Fritsch-Carlson monotone cubic spline (zero dependency)
# ═══════════════════════════════════════════════════════════════════

def _fritsch_carlson_slopes(x, y):
    n = len(x)
    if n < 2:
        return np.zeros_like(y)
    dx = np.diff(x); dy = np.diff(y)
    m = np.zeros(n)
    if n == 2:
        s = dy[0] / dx[0] if dx[0] > 1e-12 else 0
        m[0] = m[1] = s
        return m

    for i in range(1, n - 1):
        d1 = dy[i - 1] / dx[i - 1] if dx[i - 1] > 1e-12 else 0
        d2 = dy[i]     / dx[i]     if dx[i]     > 1e-12 else 0
        if d1 * d2 <= 0:
            m[i] = 0
        else:
            m[i] = 2.0 / (1.0 / d1 + 1.0 / d2) if d1 + d2 != 0 else 0

    d0 = dy[0] / dx[0] if dx[0] > 1e-12 else 0
    d1 = dy[1] / dx[1] if n > 2 and dx[1] > 1e-12 else d0
    m[0] = ((3 * d0 - d1) / 2) if abs((3 * d0 - d1) / 2) < 2 * abs(d0) else 0

    d_n1 = dy[-1] / dx[-1] if dx[-1] > 1e-12 else 0
    d_n2 = dy[-2] / dx[-2] if n > 2 and dx[-2] > 1e-12 else d_n1
    m[-1] = ((3 * d_n1 - d_n2) / 2) if abs((3 * d_n1 - d_n2) / 2) < 2 * abs(d_n1) else 0

    return m


class MonotoneCubicSpline:
    """Monotonicity-preserving cubic Hermite spline (Fritsch-Carlson)."""

    def __init__(self, x, y):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.m = _fritsch_carlson_slopes(self.x, self.y)

    def __call__(self, u):
        u = np.asarray(u, dtype=float)
        scalar = u.ndim == 0
        if scalar:
            u = u.reshape(1)

        idx = np.searchsorted(self.x[1:-1], u, side='right')
        idx = np.clip(idx, 0, len(self.x) - 2)

        h = self.x[idx + 1] - self.x[idx]
        h = np.where(h < 1e-12, 1.0, h)
        t = (u - self.x[idx]) / h

        h00 = (1 + 2 * t) * (1 - t) ** 2
        h10 = t * (1 - t) ** 2
        h01 = t ** 2 * (3 - 2 * t)
        h11 = t ** 2 * (t - 1)

        result = (h00 * self.y[idx] +
                  h10 * h * self.m[idx] +
                  h01 * self.y[idx + 1] +
                  h11 * h * self.m[idx + 1])

        return float(result[0]) if scalar else result


# ═══════════════════════════════════════════════════════════════════
# Sandwich construction
# ═══════════════════════════════════════════════════════════════════

def build_sandwich_curves(u_grid, v_opt, delta=0.05):
    """Build φ_lower, φ_upper from optimal correspondences.

    Both are MonotoneCubicSpline instances. Guarantees:
      - φ_lower(u) < φ_upper(u) for all u in [0,1]
      - Both are non-decreasing (if input v_opt is non-decreasing)
      - φ_lower(0) = 0  (implicit, via prepended knot)
      - φ_upper(1) = 1  (implicit, via appended knot)
    """
    v_lower = np.clip(v_opt - delta, 0.0, 1.0)
    v_upper = np.clip(v_opt + delta, 0.0, 1.0)

    # Ensure strict separation at every knot
    eps = 0.002
    for i in range(len(u_grid)):
        if v_upper[i] - v_lower[i] < eps:
            mid = (v_lower[i] + v_upper[i]) / 2
            v_lower[i] = max(0.0, mid - eps / 2)
            v_upper[i] = min(1.0, mid + eps / 2)

    # Prepend / append knots so bounds contain φ(0)=0 and φ(1)=1
    ug = np.concatenate([[0.0], u_grid, [1.0]])
    vl = np.concatenate([[0.0], v_lower, v_lower[-1:]])
    vu = np.concatenate([v_upper[:1], v_upper, [1.0]])

    phi_lower = MonotoneCubicSpline(ug, vl)
    phi_upper = MonotoneCubicSpline(ug, vu)
    return phi_lower, phi_upper


# ═══════════════════════════════════════════════════════════════════
# Full pipeline
# ═══════════════════════════════════════════════════════════════════

def compute_sandwich_bounds(nurbs_surf, loop_3d, loop_uv, t_P, t_Q,
                            N_samples=100, delta=0.05):
    """End-to-end: split loop → find optimal correspondences → PAV → splines.

    Returns:
        u_grid:      (N,)  curve A sample parameters
        v_opt_raw:   (N,)  raw optimal B parameters
        v_opt_mono:  (N,)  monotonic PAV-corrected
        phi_lower:   callable  lower sandwich curve
        phi_upper:   callable  upper sandwich curve
    """
    arc1_3d, arc1_uv, arc2_3d, arc2_uv = extract_curves(loop_3d, loop_uv, t_P, t_Q)

    u_grid, v_opt = find_optimal_correspondences(
        nurbs_surf, arc1_3d, arc1_uv, arc2_3d, arc2_uv,
        N_samples, tol=1e-3, search_window=0.25)

    v_mono = isotonic_regression(v_opt)

    phi_lower, phi_upper = build_sandwich_curves(u_grid, v_mono, delta)

    return u_grid, v_opt, v_mono, phi_lower, phi_upper


# ═══════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import os, sys, time

    data_dir = sys.argv[1] if len(sys.argv) > 1 else '../results/retry_0'
    from nurbs_eval import NurbsSurface

    ns_path = os.path.join(data_dir, 'nurbs_surface.txt')
    surf = NurbsSurface(ns_path)
    print(f"NURBS: {surf.nU}x{surf.nV} deg {surf.degU}x{surf.degV}")

    pid = 0
    while pid < 50:
        lf3 = os.path.join(data_dir, f'part_{pid}_loop.txt')
        lfuv = os.path.join(data_dir, f'part_{pid}_loop_uv.txt')
        if not os.path.exists(lf3):
            pid += 1; continue
        if not os.path.exists(lfuv):
            print(f"  [Partition {pid}] no UV file, skipping")
            pid += 1; continue

        loop_3d = np.loadtxt(lf3); loop_uv = np.loadtxt(lfuv)
        print(f"\n[Partition {pid}] {len(loop_3d)} verts")

        t0 = time.time()
        u_grid, v_raw, v_mono, plo, pup = compute_sandwich_bounds(
            surf, loop_3d, loop_uv, 0.3, 0.7, N_samples=50, delta=0.05)
        dt = time.time() - t0

        cross = sum(1 for i in range(len(u_grid)) if plo(u_grid[i]) >= pup(u_grid[i]))
        print(f"  v_range=[{v_mono[0]:.4f}, {v_mono[-1]:.4f}]  "
              f"crossings={cross}  time={dt:.1f}s")
        pid += 1
