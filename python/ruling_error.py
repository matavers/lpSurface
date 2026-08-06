"""
Ruled-surface error via line-vs-geodesic deviation on a NURBS surface.

For a matching pair (P on curve A, Q on curve B), the error is:
    ∫₀¹ || L(t) - G(t) ||² dt

where:
    L(t) = (1-t)·P + t·Q                              – straight ruling in 3D
    G(t) = S( u_p + t·(u_q-u_p),  w_p + t·(w_q-w_p) ) – geodesic on NURBS surface

The integral uses 5-point Gauss-Legendre quadrature along each ruling,
summed over N uniformly-spaced sample points on Curve A with trapezoidal weights.
"""

import numpy as np

# ═══════════════════════════════════════════════════════════════════
# Gauss-Legendre 5-point quadrature on [-1, 1]
# ═══════════════════════════════════════════════════════════════════
_gx = np.array([
    -0.906179845938664,
    -0.538469310105683,
     0.0,
     0.538469310105683,
     0.906179845938664,
], dtype=np.float64)
_gw = np.array([
    0.236926885056189,
    0.478628670499366,
    0.568888888888889,
    0.478628670499366,
    0.236926885056189,
], dtype=np.float64)

# Mapped to [0, 1]
_gl_t = 0.5 * (_gx + 1.0)          # shape (5,)
_gl_w = 0.5 * _gw                  # shape (5,)


def _lerp_uv(u0, v0, u1, v1, t):
    """Linear interpolate UV parameters. t is scalar or array."""
    u = u0 + t * (u1 - u0)
    v = v0 + t * (v1 - v0)
    return u, v


def _lerp_3d(p0, p1, t):
    """Linear interpolate 3D points. p0, p1: (3,), t: scalar or (n,)."""
    return p0 + np.outer(t, p1 - p0)  # (n, 3) if t is (n,), else (3,)


# ═══════════════════════════════════════════════════════════════════
# Preprocessing: re-parameterize a polyline arc from [0, 1]
# ═══════════════════════════════════════════════════════════════════

def _arc_length_param(pts):
    """Compute cumulative arc-length parameters for a polyline. Returns (N,) [0,1]."""
    segs = np.diff(pts, axis=0)
    lens = np.linalg.norm(segs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(lens)])
    if cum[-1] < 1e-12:
        return np.linspace(0, 1, len(pts))
    return cum / cum[-1]


def _resample_polyline(pts, params, t_new):
    """Resample a polyline at new parameters t_new ∈ [0,1]."""
    idx = np.searchsorted(params[1:-1], t_new, side='right')
    idx = np.clip(idx, 0, len(pts) - 2)
    t0 = params[idx]
    t1 = params[idx + 1]
    denom = t1 - t0
    denom = np.where(denom < 1e-12, 1.0, denom)
    alpha = ((t_new - t0) / denom).clip(0, 1)
    return pts[idx] * (1 - alpha[:, None]) + pts[idx + 1] * alpha[:, None]


# ═══════════════════════════════════════════════════════════════════
# Curve preparation
# ═══════════════════════════════════════════════════════════════════

def extract_curves(loop_3d, loop_uv, t_P, t_Q):
    """Split a closed polygon loop at parameters t_P, t_Q into two arcs.

    Returns:
        arc1_3d: polyline P→Q (forward)     — Curve A
        arc1_uv: UV params for arc1 vertices
        arc2_3d: polyline P→Q (backward)    — Curve B
        arc2_uv: UV params for arc2 vertices
    """
    N = len(loop_3d)
    if t_P > t_Q:
        t_P, t_Q = t_Q, t_P

    idx_p = int(round(t_P * (N - 1)))
    idx_q = int(round(t_Q * (N - 1)))
    idx_p = max(0, min(N - 2, idx_p))
    idx_q = max(idx_p + 1, min(N - 1, idx_q))

    P_3d = _resample_polyline(loop_3d,
                              np.linspace(0, 1, N),
                              np.array([t_P]))[0]
    Q_3d = _resample_polyline(loop_3d,
                              np.linspace(0, 1, N),
                              np.array([t_Q]))[0]
    P_uv = _resample_polyline(loop_uv,
                              np.linspace(0, 1, N),
                              np.array([t_P]))[0]
    Q_uv = _resample_polyline(loop_uv,
                              np.linspace(0, 1, N),
                              np.array([t_Q]))[0]

    # Arc 1: P → Q forward
    pts1_3d = [P_3d] + [loop_3d[i] for i in range(idx_p + 1, idx_q)] + [Q_3d]
    pts1_uv = [P_uv] + [loop_uv[i] for i in range(idx_p + 1, idx_q)] + [Q_uv]

    # Arc 2: P → Q backward (wrapping)
    pts2_3d = [P_3d]
    pts2_uv = [P_uv]
    for i in range(idx_p - 1, -1, -1):
        pts2_3d.append(loop_3d[i])
        pts2_uv.append(loop_uv[i])
    for i in range(N - 1, idx_q, -1):
        pts2_3d.append(loop_3d[i])
        pts2_uv.append(loop_uv[i])
    pts2_3d.append(Q_3d)
    pts2_uv.append(Q_uv)

    return (np.array(pts1_3d), np.array(pts1_uv),
            np.array(pts2_3d), np.array(pts2_uv))


# ═══════════════════════════════════════════════════════════════════
# Error computation
# ═══════════════════════════════════════════════════════════════════

def _ruling_integral(nurbs_surf, p_3d, p_uv, q_3d, q_uv):
    """Compute ∫₀¹ ||L(t)-G(t)||² dt between point P and Q.

    5-point Gauss-Legendre quadrature.
    """
    u_p, v_p = p_uv
    u_q, v_q = q_uv

    # Straight line points at GL nodes: L(t_k)
    L_pts = _lerp_3d(p_3d, q_3d, _gl_t)  # (5, 3)

    # NURBS surface points at GL nodes: G(t_k) = S(u_mid, v_mid)
    G_pts = np.zeros((5, 3))
    for k in range(5):
        u_mid, v_mid = _lerp_uv(u_p, v_p, u_q, v_q, _gl_t[k])
        G_pts[k] = nurbs_surf.evaluate(u_mid, v_mid)

    # Squared distance at each node
    diff = L_pts - G_pts  # (5, 3)
    sq_norm = np.sum(diff * diff, axis=1)  # (5,)

    # Weighted sum
    return float(np.dot(sq_norm, _gl_w))


def precompute_curve_a(curve_a_3d, curve_a_uv, N_samples=200):
    """Pre-sample Curve A at N uniformly-spaced arc-length positions.

    Returns:
        P_list:  (N, 3) 3D points on Curve A
        uv_list: (N, 2) UV parameters at each sample
        weights: (N,)  trapezoidal integration weights
    """
    params = _arc_length_param(curve_a_3d)
    t_vals = np.linspace(0, 1, N_samples)

    P_list = _resample_polyline(curve_a_3d, params, t_vals)
    uv_list = _resample_polyline(curve_a_uv, params, t_vals)

    # Trapezoidal weights: h = 1/(N-1), first/last = h/2, rest = h
    h = 1.0 / (N_samples - 1)
    weights = np.full(N_samples, h)
    weights[0] = h / 2.0
    weights[-1] = h / 2.0

    return P_list, uv_list, weights


def compute_error_for_point(nurbs_surf, curve_a_pre, q_3d, q_uv):
    """Compute total error for a candidate point Q on Curve B.

    Args:
        nurbs_surf: NurbsSurface evaluator
        curve_a_pre: tuple (P_list, uv_list, weights) from precompute_curve_a
        q_3d: (3,) 3D position of candidate Q on Curve B
        q_uv: (2,) UV parameters of Q

    Returns:
        float: E_total = Σ w_i · ∫₀¹ ||L_i(t) - G_i(t)||² dt
    """
    P_list, uv_list, weights = curve_a_pre
    total = 0.0
    for i in range(len(P_list)):
        e_local = _ruling_integral(nurbs_surf, P_list[i], uv_list[i], q_3d, q_uv)
        total += weights[i] * e_local
    return total


def compute_nurbs_error(nurbs_surf, loop_3d, loop_uv, t_P, t_Q, N_samples=200):
    """Top-level: build curves, pre-sample A, compute error for current (t_P, t_Q).

    Args:
        nurbs_surf:   NurbsSurface evaluator
        loop_3d:      (M, 3) 3D polygon loop
        loop_uv:      (M, 2) UV coords for loop vertices
        t_P:          parameter on loop for point P
        t_Q:          parameter on loop for point Q
        N_samples:    number of samples on Curve A (default 200)

    Returns:
        float: total NURBS-geodesic error for this (t_P, t_Q).
    """
    arc1_3d, arc1_uv, arc2_3d, arc2_uv = \
        extract_curves(loop_3d, loop_uv, t_P, t_Q)

    # Curve A = arc1 (P→Q), Curve B = arc2 (P→Q via reverse)
    curve_a_3d = arc1_3d
    curve_a_uv = arc1_uv
    curve_b_3d = arc2_3d
    curve_b_uv = arc2_uv

    # Pre-sample Curve A
    pre_a = precompute_curve_a(curve_a_3d, curve_a_uv, N_samples)

    # Q is the endpoint shared by both arcs (last point of both)
    q_3d = curve_b_3d[-1]
    q_uv = curve_b_uv[-1]

    return compute_error_for_point(nurbs_surf, pre_a, q_3d, q_uv)


# ═══════════════════════════════════════════════════════════════════
# 1D optimization over t_Q (golden-section search)
# ═══════════════════════════════════════════════════════════════════

def _golden_section_search(f, a, b, tol=1e-4, max_iter=50):
    """Minimize scalar function f(x) on [a,b] via golden-section search."""
    phi = (np.sqrt(5) - 1) / 2
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc = f(c)
    fd = f(d)
    for _ in range(max_iter):
        if b - a < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = f(d)
    return (a + b) / 2


def optimize_t_Q_nurbs(nurbs_surf, loop_3d, loop_uv, t_P, N_samples=200,
                       t_Q_min=0.01, t_Q_max=0.99, tol=1e-4):
    """Find optimal t_Q for fixed t_P using NURBS-geodesic error.

    Returns:
        (best_t_Q, best_error)
    """

    def objective(t_Q_val):
        return compute_nurbs_error(nurbs_surf, loop_3d, loop_uv,
                                   t_P, t_Q_val, N_samples)

    best_t = _golden_section_search(objective, t_Q_min, t_Q_max, tol)
    best_e = objective(best_t)
    return best_t, best_e


# ═══════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import os, sys, time

    if len(sys.argv) < 2:
        data_dir = '../results/retry_0'
    else:
        data_dir = sys.argv[1]

    from nurbs_eval import NurbsSurface
    ns_path = os.path.join(data_dir, 'nurbs_surface.txt')
    if not os.path.exists(ns_path):
        print(f"NURBS surface not found: {ns_path}")
        sys.exit(1)

    surf = NurbsSurface(ns_path)
    print(f"Loaded NURBS: {surf.nU}x{surf.nV} degree {surf.degU}x{surf.degV}")

    # Find all partitions
    pid = 0
    while pid < 50:
        lf3 = os.path.join(data_dir, f'part_{pid}_loop.txt')
        lfuv = os.path.join(data_dir, f'part_{pid}_loop_uv.txt')
        if not os.path.exists(lf3) or not os.path.exists(lfuv):
            pid += 1
            continue
        loop_3d = np.loadtxt(lf3)
        loop_uv = np.loadtxt(lfuv)
        print(f"\n[Partition {pid}] {len(loop_3d)} loop vertices")

        t_P = 0.3
        t0 = time.time()
        err = compute_nurbs_error(surf, loop_3d, loop_uv, t_P, 0.5, N_samples=150)
        dt = time.time() - t0
        print(f"  t_P=0.3 t_Q=0.5  error={err:.6f}  ({dt*1000:.1f} ms)")

        t0 = time.time()
        best_tq, best_e = optimize_t_Q_nurbs(surf, loop_3d, loop_uv, t_P,
                                             N_samples=150, tol=0.01)
        dt = time.time() - t0
        print(f"  optimal t_Q={best_tq:.4f}  error={best_e:.6f}  ({dt*1000:.1f} ms)")
        pid += 1
