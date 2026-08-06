"""
NURBS-geodesic ruled-surface optimizer with sandwich-bounded φ mapping.

Pipeline (per partition):
  Phase 1 – Coarse (t_P, t_Q) grid search with identity φ
  Phase 2 – Sandwich bounds from optimal per-sample correspondences
  Phase 3 – L-BFGS-B optimization of φ coefficients within bounds
  Phase 4 – (opt) Joint refinement of (t_P, t_Q, φ) with stale bounds

Usage:
  python fit_ruled_nurbs.py <data_dir> [--max-iter 200] [--n-ctrl 4]
"""

import os, sys, time
import numpy as np
from scipy.optimize import minimize

# ── Reuse core helpers from sandwich_bounds ──
from sandwich_bounds import (
    _arc_length_param, _resample_polyline, _ruling_integral,
    _golden_search, extract_curves,
    compute_sandwich_bounds, isotonic_regression,
)

from nurbs_eval import NurbsSurface

# ═══════════════════════════════════════════════════════════════════
# B-spline basis (numpy)
# ═══════════════════════════════════════════════════════════════════

def _bspline_basis(u, degree, n_ctrl):
    """Cox-de Boor. Returns (n_ctrl,) array."""
    n_knots = n_ctrl + degree + 1
    knots = np.zeros(n_knots)
    knots[:degree + 1] = 0.0
    knots[-(degree + 1):] = 1.0
    inner = n_knots - 2 * (degree + 1)
    if inner > 0:
        knots[degree + 1:-degree - 1] = np.linspace(0, 1, inner + 2)[1:-1]

    u = np.clip(u, 0.0, 1.0)

    span = degree
    for k in range(degree, n_knots - degree - 1):
        if knots[k] <= u < knots[k + 1]:
            span = k
            break
    if u >= knots[-degree - 1]:
        span = n_knots - degree - 2

    N = np.zeros(degree + 1)
    N[0] = 1.0
    left = np.zeros(degree + 1, dtype=int)
    right = np.zeros(degree + 1, dtype=int)

    for j in range(1, degree + 1):
        left[j] = span + 1 - j
        right[j] = span + j
        saved = 0.0
        for r in range(j):
            temp = N[r] / (knots[right[r + 1]] - knots[left[r + 1]] + 1e-14)
            N[r] = saved + (knots[right[r + 1]] - u) * temp
            saved = (u - knots[left[r + 1]]) * temp
        N[j] = saved

    result = np.zeros(n_ctrl)
    result[span - degree:span + 1] = N[:degree + 1]
    return result


def _eval_phi(u, coeffs_inner, degree, n_ctrl):
    """φ(u) = u + Σ cⱼ·Bⱼ(u)  (c₀=c_{n-1}=0 → φ(0)=0, φ(1)=1)."""
    B = _bspline_basis(u, degree, n_ctrl)
    full = np.concatenate([[0.0], coeffs_inner, [0.0]])
    return u + np.dot(B, full)


def _clamped_phi(u, coeffs_inner, phi_lower, phi_upper, degree, n_ctrl):
    """φ(u) clamped to [φ_lower(u), φ_upper(u)]."""
    raw = _eval_phi(u, coeffs_inner, degree, n_ctrl)
    lo = float(phi_lower(u))
    hi = float(phi_upper(u))
    return np.clip(raw, lo, hi)


# ═══════════════════════════════════════════════════════════════════
# Phase 1: coarse (t_P, t_Q) search (identity φ, single Q)
# ═══════════════════════════════════════════════════════════════════

def _simple_nurbs_error(t_P, t_Q, loop_3d, loop_uv, nurbs_surf, N_samples=80):
    """Error with identity φ: all Pᵢ connect to the single endpoint Q."""
    arc1_3d, arc1_uv, arc2_3d, arc2_uv = extract_curves(loop_3d, loop_uv, t_P, t_Q)

    params_a = _arc_length_param(arc1_3d)
    u_grid = np.linspace(0, 1, N_samples)
    P_list = _resample_polyline(arc1_3d, params_a, u_grid)
    uv_list = _resample_polyline(arc1_uv,  params_a, u_grid)

    Q = arc2_3d[-1]; Q_uv = arc2_uv[-1]

    h = 1.0 / (N_samples - 1)
    weights = np.full(N_samples, h)
    weights[0] = h / 2.0; weights[-1] = h / 2.0

    total = 0.0
    for i in range(N_samples):
        e = _ruling_integral(nurbs_surf, P_list[i], uv_list[i], Q, Q_uv)
        total += weights[i] * e
    return total


def _coarse_tP_tQ(loop_3d, loop_uv, nurbs_surf, N_samples=80, n_grid=6):
    """Grid t_P, golden-section t_Q at each. Return best (t_P, t_Q, error)."""
    tP_grid = np.linspace(0.15, 0.45, n_grid)
    best_err = float('inf')
    best = (0.3, 0.7)

    for tP in tP_grid:
        def f(tQ):
            return _simple_nurbs_error(tP, tQ, loop_3d, loop_uv, nurbs_surf, N_samples)
        lo = max(tP + 0.05, 0.10)
        tQ_opt = _golden_search(f, lo, 0.95, tol=0.01)
        err = f(tQ_opt)
        if err < best_err:
            best_err = err
            best = (tP, tQ_opt)

    return best[0], best[1], best_err


# ═══════════════════════════════════════════════════════════════════
# Phase 3/4: NURBS loss with hard-clamped φ
# ═══════════════════════════════════════════════════════════════════

def _make_nurbs_loss(loop_3d, loop_uv, nurbs_surf,
                     phi_lower, phi_upper,
                     degree, n_ctrl, N_samples,
                     t_P_fixed, t_Q_fixed, refine_t=False):
    """Return a callable loss(x)→float for scipy.optimize.

    φ is HARD-CLAMPED to [φ_lower(u), φ_upper(u)].
    Only the NURBS geodesic integral contributes to the loss.

    x layout when refine_t=False:  [c₁, c₂, ..., c_{k-2}]        (n_ctrl-2 vars)
    x layout when refine_t=True:   [t_P, t_Q, c₁, ..., c_{k-2}]  (2 + n_ctrl-2 vars)
    """

    def loss(x):
        if refine_t:
            t_P = x[0]; t_Q = x[1]; coeffs = x[2:]
        else:
            t_P = t_P_fixed; t_Q = t_Q_fixed; coeffs = x

        arc1_3d, arc1_uv, arc2_3d, arc2_uv = extract_curves(loop_3d, loop_uv, t_P, t_Q)
        curve_a_3d, curve_a_uv = arc1_3d, arc1_uv
        curve_b_3d, curve_b_uv = arc2_3d, arc2_uv

        params_a = _arc_length_param(curve_a_3d)
        u_grid = np.linspace(0, 1, N_samples)
        P_list = _resample_polyline(curve_a_3d, params_a, u_grid)
        uv_list = _resample_polyline(curve_a_uv,  params_a, u_grid)

        h = 1.0 / (N_samples - 1)
        weights = np.full(N_samples, h)
        weights[0] = h / 2.0; weights[-1] = h / 2.0

        params_b = _arc_length_param(curve_b_3d)

        total = 0.0
        for i in range(N_samples):
            phic = float(_clamped_phi(u_grid[i], coeffs, phi_lower, phi_upper, degree, n_ctrl))
            Q = _resample_polyline(curve_b_3d, params_b, np.array([phic]))[0]
            Q_uv = _resample_polyline(curve_b_uv,  params_b, np.array([phic]))[0]
            e = _ruling_integral(nurbs_surf, P_list[i], uv_list[i], Q, Q_uv)
            total += weights[i] * e
        return total

    return loss


# ═══════════════════════════════════════════════════════════════════
# Master optimizer for one partition
# ═══════════════════════════════════════════════════════════════════

def optimize_partition(loop_3d, loop_uv, nurbs_surf,
                       n_ctrl=4, degree=3, N_samples=100,
                       delta=0.05, max_iter=200, tol=1e-6,
                       do_refine=False):
    """Full Phase-1→3 pipeline.

    Returns dict with keys:
      t_P, t_Q, coeffs, loss, loss_phase1, v_opt_raw, v_opt_mono,
      phi_lower, phi_upper, u_grid
    """

    # ── Phase 1: coarse (t_P, t_Q) ──
    tP_best, tQ_best, err1 = _coarse_tP_tQ(loop_3d, loop_uv, nurbs_surf,
                                            N_samples=min(80, N_samples))
    print(f"  Phase1: t_P={tP_best:.4f} t_Q={tQ_best:.4f} err={err1:.6f}")

    # ── Phase 2: sandwich bounds ──
    u_grid, v_raw, v_mono, plo, pup = compute_sandwich_bounds(
        nurbs_surf, loop_3d, loop_uv, tP_best, tQ_best,
        N_samples=N_samples, delta=delta)
    print(f"  Phase2: v* range [{v_mono[0]:.4f}, {v_mono[-1]:.4f}]")

    # ── Phase 3: optimize φ coefficients ──
    loss_fn = _make_nurbs_loss(loop_3d, loop_uv, nurbs_surf, plo, pup,
                               degree, n_ctrl, N_samples,
                               tP_best, tQ_best, refine_t=False)

    x0 = np.zeros(n_ctrl - 2)
    bounds = [(-0.5, 0.5)] * (n_ctrl - 2)

    res = minimize(loss_fn, x0, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': max_iter, 'ftol': tol})
    coeffs_opt = res.x
    loss_opt = res.fun
    print(f"  Phase3: loss={loss_opt:.6f}  nit={res.nit}  coeffs={np.round(coeffs_opt, 4)}")

    # ── Phase 4 (optional): joint refinement ──
    if do_refine:
        loss_fn4 = _make_nurbs_loss(loop_3d, loop_uv, nurbs_surf, plo, pup,
                                    degree, n_ctrl, N_samples,
                                    tP_best, tQ_best, refine_t=True)
        x0_4 = np.concatenate([[tP_best], [tQ_best], coeffs_opt])
        bounds4 = ([(0.05, 0.45), (0.55, 0.95)] +
                   [(-0.5, 0.5)] * (n_ctrl - 2))
        res4 = minimize(loss_fn4, x0_4, method='L-BFGS-B', bounds=bounds4,
                        options={'maxiter': max_iter, 'ftol': tol})
        tP_best, tQ_best = res4.x[0], res4.x[1]
        coeffs_opt = res4.x[2:]
        loss_opt = res4.fun
        print(f"  Phase4: loss={loss_opt:.6f}  t_P={tP_best:.4f} t_Q={tQ_best:.4f}")

    return {
        't_P': tP_best, 't_Q': tQ_best,
        'coeffs': coeffs_opt,
        'loss': loss_opt,
        'loss_phase1': err1,
        'v_opt_raw': v_raw,
        'v_opt_mono': v_mono,
        'phi_lower': plo,
        'phi_upper': pup,
        'u_grid': u_grid,
    }


# ═══════════════════════════════════════════════════════════════════
# Batch runner (like fit_ruled_grad.py main)
# ═══════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print("Usage: python fit_ruled_nurbs.py <data_dir> [--max-iter 200] [--n-ctrl 4] [--delta 0.05] [--refine]")
        return 1

    data_dir = sys.argv[1]
    max_iter, n_ctrl, delta = 200, 4, 0.05
    do_refine = False
    args = sys.argv[2:]; i = 0
    while i < len(args):
        if args[i] == '--max-iter' and i + 1 < len(args):
            max_iter = int(args[i + 1]); i += 2
        elif args[i] == '--n-ctrl' and i + 1 < len(args):
            n_ctrl = int(args[i + 1]); i += 2
        elif args[i] == '--delta' and i + 1 < len(args):
            delta = float(args[i + 1]); i += 2
        elif args[i] == '--refine':
            do_refine = True; i += 1
        else:
            i += 1

    ns_path = os.path.join(data_dir, 'nurbs_surface.txt')
    if not os.path.exists(ns_path):
        print(f"NURBS surface not found: {ns_path}")
        return 1
    surf = NurbsSurface(ns_path)
    print(f"NURBS: {surf.nU}x{surf.nV} degree {surf.degU}x{surf.degV}")

    pid = 0; results = []; max_pid = 50
    while pid < max_pid:
        lf3 = os.path.join(data_dir, f'part_{pid}_loop.txt')
        lfuv = os.path.join(data_dir, f'part_{pid}_loop_uv.txt')
        if not os.path.exists(lf3):
            pid += 1; continue
        if not os.path.exists(lfuv):
            print(f"\n[Partition {pid}] no UV — skipping")
            pid += 1; continue

        loop_3d = np.loadtxt(lf3); loop_uv = np.loadtxt(lfuv)
        if loop_uv.ndim == 1:
            loop_uv = loop_uv.reshape(-1, 2)
        print(f"\n[Partition {pid}] {len(loop_3d)} verts")

        t0 = time.time()
        try:
            res = optimize_partition(loop_3d, loop_uv, surf,
                                     n_ctrl=n_ctrl, N_samples=100,
                                     delta=delta, max_iter=max_iter,
                                     do_refine=do_refine)
            dt = time.time() - t0
            print(f"  total time: {dt:.1f}s")
            results.append({'pid': pid, **{k: res[k] for k in
                ['t_P', 't_Q', 'coeffs', 'loss', 'loss_phase1']}})
        except Exception as e:
            print(f"  FAILED: {e}")

        pid += 1

    # ── Write summary ──
    if results:
        out_f = os.path.join(data_dir, 'tolerance_nurbs.txt')
        with open(out_f, 'w') as f:
            f.write("pid t_P t_Q loss loss_phase1 coeffs\n")
            for r in results:
                f.write(f"{r['pid']} {r['t_P']:.6f} {r['t_Q']:.6f} "
                        f"{r['loss']:.6f} {r['loss_phase1']:.6f} "
                        f"{' '.join(f'{c:.6f}' for c in r['coeffs'])}\n")
        print(f"\n=== {len(results)} partitions → {out_f} ===")
    else:
        print("\nNo partitions processed.")


if __name__ == '__main__':
    main()
