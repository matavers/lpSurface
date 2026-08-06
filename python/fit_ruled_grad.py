"""
Partition ruled-surface fitting via gradient optimization with autodiff.

Uses implicit differentiation: Newton iteration (no_grad) to find closest
(u,v) for each data point, then differentiates through S(u*,v*) w.r.t.
t_P, t_Q, and B-spline correspondence coefficients.

Usage: python fit_ruled_grad.py <data_dir> [--max-iter 200] [--n-ctrl 8]
"""
import sys, os
import numpy as np
import torch

EPS = 1e-12
TORCH_DTYPE = torch.float64

# ═══════════════════════════════════════════════════════════════════
# B-spline basis (Cox-de Boor recurrence, autograd-compatible)
# ═══════════════════════════════════════════════════════════════════

def bspline_basis(u, degree=3, n_ctrl=8):
    """u: (n_pts,) in [0,1]. Returns (n_pts, n_ctrl)."""
    n_pts = len(u)
    n_knots = n_ctrl + degree + 1
    knots = torch.zeros(n_knots, dtype=TORCH_DTYPE)
    knots[:degree+1] = 0.0
    knots[-(degree+1):] = 1.0
    inner = n_knots - 2*(degree+1)
    if inner > 0:
        knots[degree+1:-degree-1] = torch.linspace(0, 1, inner+2, dtype=TORCH_DTYPE)[1:-1]

    # Degree 0: N_{i,0}(u) = 1 if u in [k_i, k_{i+1}), else 0
    N = torch.zeros(n_pts, n_knots - 1, dtype=TORCH_DTYPE)
    for i in range(n_knots - 1):
        N[:, i] = ((u >= knots[i]) & (u < knots[i+1])).to(TORCH_DTYPE)
    # Handle u==1
    one_mask = (u >= 1.0 - 1e-10)
    N[one_mask, -1] = 1.0
    N[one_mask, -2] = 0.0

    for p in range(1, degree+1):
        N_new = torch.zeros(n_pts, n_knots - 1 - p, dtype=TORCH_DTYPE)
        for i in range(n_knots - 1 - p):
            d1 = knots[i+p] - knots[i]
            d2 = knots[i+p+1] - knots[i+1]
            left = ((u - knots[i]) / (d1 + EPS)) * N[:, i]
            right = ((knots[i+p+1] - u) / (d2 + EPS)) * N[:, i+1]
            left = torch.where(d1 > 1e-14, left, torch.zeros_like(left))
            right = torch.where(d2 > 1e-14, right, torch.zeros_like(right))
            N_new[:, i] = left + right
        N = N_new
    return N[:, :n_ctrl]

# ═══════════════════════════════════════════════════════════════════
# Correspondence function φ(u)
# ═══════════════════════════════════════════════════════════════════

class Correspondence:
    """φ(u) = u + Σc_j·B_j(u), with φ(0)=0, φ(1)=1 enforced."""
    def __init__(self, n_ctrl=8, degree=3):
        self.n_ctrl = n_ctrl
        self.degree = degree
        self.c_inner = torch.zeros(n_ctrl - 2, dtype=TORCH_DTYPE)
        self.c_inner.requires_grad_(True)

    def full_coeffs(self):
        return torch.cat([torch.zeros(1, dtype=TORCH_DTYPE),
                          self.c_inner,
                          torch.zeros(1, dtype=TORCH_DTYPE)])

    def evaluate(self, u):
        B = bspline_basis(u, self.degree, self.n_ctrl)
        return u + B @ self.full_coeffs()

    def derivative(self, u):
        u_d = u.detach().clone().requires_grad_(True)
        phi = self.evaluate(u_d)
        grad = torch.autograd.grad(phi.sum(), u_d, create_graph=True)[0]
        return grad

    def params(self):
        return [self.c_inner]

# ═══════════════════════════════════════════════════════════════════
# Director curve γ(t) — piecewise linear from polygon loop
# ═══════════════════════════════════════════════════════════════════

class DirectorCurve:
    def __init__(self, points):
        """points: (N,3) tensor. First & last points are differentiable (P,Q)."""
        self.pts = points
        segs = points[1:] - points[:-1]
        lengths = segs.norm(dim=1)
        self.total = lengths.sum().clamp(min=EPS)
        self.cum = torch.cat([torch.zeros(1, dtype=TORCH_DTYPE),
                              torch.cumsum(lengths, dim=0)]) / self.total

    def evaluate(self, u):
        u = u.clamp(0.0, 1.0)
        idx = torch.searchsorted(self.cum[1:-1], u.detach()).clamp(0, len(self.pts)-2)
        t0 = self.cum[idx]; t1 = self.cum[idx+1]
        alpha = ((u - t0) / (t1 - t0 + EPS)).clamp(0, 1)
        return self.pts[idx] * (1 - alpha.unsqueeze(-1)) + self.pts[idx+1] * alpha.unsqueeze(-1)

    def derivative(self, u):
        """Tangent direction. Approximate via central difference."""
        h = 1e-4
        up = self.evaluate(u + h)
        um = self.evaluate(u - h)
        return (up - um) / (2 * h)

# ═══════════════════════════════════════════════════════════════════
# Ruled surface S(u,v)
# ═══════════════════════════════════════════════════════════════════

class RuledSurface:
    def __init__(self, loop, t_P, t_Q, corr):
        self.loop = loop
        self.t_P = t_P; self.t_Q = t_Q; self.corr = corr
        self._bounds = None  # (N, 2) tensor: [phi_lower, phi_upper]
        self._build()

    def _point_at(self, t):
        """Point at parameter t on full N-vertex loop."""
        t = t.clamp(0, 1)
        N = len(self.loop)
        idx_f = t * (N - 1)
        idx = int(idx_f.detach().clamp(0, N-2).item())
        alpha = (idx_f - idx).clamp(0, 1)
        return self.loop[idx] * (1 - alpha) + self.loop[idx+1] * alpha

    def _build(self):
        N = len(self.loop)
        tp = self.t_P.clamp(0, 1); tq = self.t_Q.clamp(0, 1)
        if tp > tq: tp, tq = tq, tp

        P = self._point_at(tp); Q = self._point_at(tq)
        idx_p = int(tp.detach() * (N-1)); idx_p = max(0, min(N-2, idx_p))
        idx_q = int(tq.detach() * (N-1)); idx_q = max(idx_p+1, min(N-1, idx_q))

        # γ1: P → Q
        pts1 = [P]
        for i in range(idx_p+1, idx_q):
            pts1.append(self.loop[i].detach().clone())
        pts1.append(Q)
        self.g1 = DirectorCurve(torch.stack(pts1))

        # γ2: P → Q via the OTHER side (reverse order for opposite orientation)
        pts2 = [P]
        for i in range(idx_p-1, -1, -1):
            pts2.append(self.loop[i].detach().clone())
        for i in range(N-1, idx_q, -1):
            pts2.append(self.loop[i].detach().clone())
        pts2.append(Q)
        self.g2 = DirectorCurve(torch.stack(pts2))

    def _clamped_phi(self, u):
        """φ(u) clamped to [φ_lower(u), φ_upper(u)] if bounds are set."""
        phi = self.corr.evaluate(u)
        if self._bounds is None:
            return phi
        N = self._bounds.shape[0]
        # Convert u to index in bounds array
        u_idx = (u.detach() * (N - 1)).long().clamp(0, N - 1)
        lo = self._bounds[u_idx, 0]; hi = self._bounds[u_idx, 1]
        return torch.clamp(phi, lo, hi)

    def evaluate(self, u, v):
        """S(u,v) = (1-v)·γ1(u) + v·γ2(φ_clamped(u)). u,v can be 0-d or 1-d."""
        scalar = u.dim() == 0
        if scalar: u = u.unsqueeze(0); v = v.unsqueeze(0)
        p1 = self.g1.evaluate(u)
        p2 = self.g2.evaluate(self._clamped_phi(u))
        v = v.clamp(0, 1).unsqueeze(-1)
        result = p1 * (1 - v) + p2 * v
        return result[0] if scalar else result

    def derivatives(self, u, v, h=1e-4):
        """Finite-difference Jacobian [S_u, S_v]."""
        S0 = self.evaluate(u, v)
        Su = (self.evaluate(u + h, v) - self.evaluate(u - h, v)) / (2 * h)
        Sv = (self.evaluate(u, v + h) - self.evaluate(u, v - h)) / (2 * h)
        return Su, Sv, S0

    def sample_grid(self, nu=30, nv=12):
        """Returns (nu*nv, 3) grid points on the ruled surface."""
        u = torch.linspace(0.001, 0.999, nu, dtype=TORCH_DTYPE)
        v = torch.linspace(0.001, 0.999, nv, dtype=TORCH_DTYPE)
        p1 = self.g1.evaluate(u)
        p2 = self.g2.evaluate(self._clamped_phi(u))
        grid = p1.unsqueeze(1)*(1-v.unsqueeze(0).unsqueeze(-1)) + \
               p2.unsqueeze(1)*v.unsqueeze(0).unsqueeze(-1)
        return u, v, grid.reshape(-1, 3)

    def set_bounds(self, phi_lower, phi_upper):
        self._bounds = torch.tensor(
            np.column_stack([phi_lower, phi_upper]), dtype=TORCH_DTYPE)

    def export_obj(self, path, nu=30, nv=12):
        """Export ruled surface as OBJ quads using current parameters."""
        u, v, grid = self.sample_grid(nu, nv)
        grid_np = grid.reshape(nu, nv, 3).detach().numpy()

        with open(path, 'w') as f:
            for i in range(nu):
                for j in range(nv):
                    p = grid_np[i, j]
                    f.write(f"v {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")
            for i in range(nu - 1):
                for j in range(nv - 1):
                    a = i * nv + j + 1
                    b = a + 1
                    c = (i+1) * nv + j + 1
                    d = c + 1
                    f.write(f"f {a} {b} {d} {c}\n")
        return True

    def distances_batch(self, pts, nu=30, nv=12):
        """Grid + envelope theorem: (u*,v*) from argmin (no grad), err from S(u*,v*)."""
        M = len(pts)
        u, v, grid = self.sample_grid(nu, nv)
        G = nu * nv

        with torch.no_grad():
            diff = pts.unsqueeze(1) - grid.unsqueeze(0)
            sq = diff.pow(2).sum(dim=2)
            best = sq.argmin(dim=1)

        u_idx = best // nv
        v_idx = best % nv
        u_star = u[u_idx].detach()
        v_star = v[v_idx].detach()

        dists = torch.zeros(M, dtype=TORCH_DTYPE)
        for i in range(M):
            S_i = self.evaluate(u_star[i], v_star[i])
            dists[i] = (pts[i] - S_i).norm()
        return dists.mean()

# ═══════════════════════════════════════════════════════════════════
# Loss
# ═══════════════════════════════════════════════════════════════════

def compute_loss(surface, pts, corr, lam_arc=8.0):
    n = len(pts)
    if n == 0:
        return torch.tensor(0.0, dtype=TORCH_DTYPE, requires_grad=True)

    loss_data = surface.distances_batch(pts, nu=60, nv=24)

    len1 = surface.g1.total; len2 = surface.g2.total
    ratio = len1 / (len2 + EPS)
    loss_arc = lam_arc * ((ratio - 1.0).pow(2) + (1.0/(ratio+EPS) - 1.0).pow(2))

    tp = surface.t_P.clamp(0,1); tq = surface.t_Q.clamp(0,1)
    gap = (tq - tp).abs()
    loss_gap = 3.0 * (1.0 / (gap + 0.01) - 1.0/1.01)

    return loss_data + loss_arc + loss_gap

# ═══════════════════════════════════════════════════════════════════
# Optimization
# ═══════════════════════════════════════════════════════════════════

def optimize_ruled(loop_verts, interior_pts, n_ctrl=8, max_iter=200, lr=0.02, use_bounds=True):
    loop = torch.tensor(loop_verts, dtype=TORCH_DTYPE)
    pts = torch.tensor(interior_pts, dtype=TORCH_DTYPE)
    if pts.dim() == 1:
        pts = pts.unsqueeze(0)

    # Import bounds module if needed
    have_bounds = use_bounds
    if have_bounds:
        from ruling_bounds import compute_bounds
        loop_np = loop_verts

    tp_raw = torch.tensor(0.0, dtype=TORCH_DTYPE, requires_grad=True)
    tq_raw = torch.tensor(1.0, dtype=TORCH_DTYPE, requires_grad=True)
    corr = Correspondence(n_ctrl=n_ctrl)

    # Initial bounds at starting t_P, t_Q
    if have_bounds:
        current_lo, current_hi = compute_bounds(loop_np, 0.5, 0.73, interior_pts)

    params = [tp_raw, tq_raw] + corr.params()
    optimizer = torch.optim.LBFGS(params, lr=lr, max_iter=20,
        line_search_fn='strong_wolfe',
        tolerance_grad=1e-9, tolerance_change=1e-11)
    best_tP, best_tQ = 0.25, 0.75
    best_loss = float('inf')
    history = []

    # Mutable bounds state (updated outside closure to avoid L-BFGS overhead)
    current_lo = None; current_hi = None

    def closure():
        optimizer.zero_grad()
        tp = torch.sigmoid(tp_raw)
        tq = torch.sigmoid(tq_raw)
        surf = RuledSurface(loop, tp, tq, corr)
        if current_lo is not None:
            surf.set_bounds(current_lo, current_hi)
        loss = compute_loss(surf, pts, corr)
        if not torch.isnan(loss):
            loss.backward()
        return loss

    for it in range(max_iter):
        loss = optimizer.step(closure)
        lv = float(loss.detach())
        history.append(lv)
        tp = float(torch.sigmoid(tp_raw).detach())
        tq = float(torch.sigmoid(tq_raw).detach())

        # Update bounds only once (costly, good enough at initial t_P/t_Q)
        if have_bounds and it == 0:
            lo, hi = compute_bounds(loop_np, tp, tq, interior_pts)
            current_lo = lo; current_hi = hi

        if lv < best_loss and not np.isnan(lv):
            best_loss = lv; best_tP = tp; best_tQ = tq

        if it % 10 == 0 or it == max_iter-1:
            print(f"    iter {it:4d}  loss={lv:.8f}  t_P={tp:.4f} t_Q={tq:.4f}")

        if it > 10 and len(history) > 1 and abs(history[-1] - history[-2]) < 1e-10:
            break

    return best_tP, best_tQ, corr.c_inner.detach().numpy(), history

# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./results"
    data_dir = os.path.abspath(data_dir)
    if not os.path.isdir(data_dir):
        print(f"  [ERR] data dir not found: {data_dir}")
        return 1
    os.makedirs(data_dir, exist_ok=True)

    max_iter, lr, n_ctrl = 15, 0.02, 2
    args = sys.argv[2:]; i = 0
    while i < len(args):
        if args[i] == '--max-iter' and i+1 < len(args):
            max_iter = int(args[i+1]); i += 2
        elif args[i] == '--lr' and i+1 < len(args):
            lr = float(args[i+1]); i += 2
        elif args[i] == '--n-ctrl' and i+1 < len(args):
            n_ctrl = int(args[i+1]); i += 2
        else: i += 1

    # ── Scan base partition files (consecutive PID) ──
    pid = 0; results = []; max_pid = 50
    candidate_pids = []
    while pid < max_pid:
        lf = os.path.join(data_dir, f"part_{pid}_loop.txt")
        if os.path.exists(lf):
            candidate_pids.append(pid)
        pid += 1
    total_parts = len(candidate_pids)
    print(f"[PIPELINE:done:count:{total_parts}]")
    print(f"[PIPELINE:stage:Ruled]")

    for pid in candidate_pids:
        lf = os.path.join(data_dir, f"part_{pid}_loop.txt")
        pf = os.path.join(data_dir, f"part_{pid}_points.txt")
        loop_v = np.loadtxt(lf)
        pts_v = np.loadtxt(pf) if os.path.exists(pf) else np.zeros((0,3))
        if pts_v.ndim == 1: pts_v = pts_v.reshape(1, -1)
        print(f"\n[PIPELINE:stage:part_{pid}]")
        print(f"[Partition {pid}] loop={loop_v.shape} pts={pts_v.shape if len(pts_v)>0 else '(empty)'}")

        # Check if bounds module is available
        use_bounds = False
        try:
            from ruling_bounds import compute_bounds
            if len(pts_v) > 0:
                lo, hi = compute_bounds(loop_v, 0.5, 0.73, pts_v)
                if lo is not None:
                    print(f"    bounds available ({len(lo)} pts)")
                    use_bounds = True
        except Exception as e:
            print(f"    bounds disabled: {e}")

        tP, tQ, coeffs, hist = optimize_ruled(loop_v, pts_v,
            n_ctrl=n_ctrl, max_iter=max_iter, lr=lr, use_bounds=use_bounds)
        out_f = os.path.join(data_dir, f"part_{pid}_ruled.txt")
        with open(out_f, 'w') as f:
            f.write(f"t_P={tP:.8f}\nt_Q={tQ:.8f}\n")
            f.write(f"coeffs={' '.join(f'{c:.8f}' for c in coeffs)}\n")
            if hist: f.write(f"final_loss={hist[-1]:.8f}\n")

        loop_t = torch.tensor(loop_v, dtype=TORCH_DTYPE)
        tp_t = torch.tensor(tP, dtype=TORCH_DTYPE)
        tq_t = torch.tensor(tQ, dtype=TORCH_DTYPE)
        corr2 = Correspondence(n_ctrl=n_ctrl)
        with torch.no_grad():
            corr2.c_inner.copy_(torch.tensor(coeffs, dtype=TORCH_DTYPE))
        surf = RuledSurface(loop_t, tp_t, tq_t, corr2)
        obj_f = os.path.join(data_dir, f"ruled_surf_{pid}.obj")
        surf.export_obj(obj_f, nu=80, nv=32)
        print(f"    wrote {obj_f}")
        print(f"[PIPELINE:file:ruled:{obj_f}]")

        _, _, surf_grid = surf.sample_grid(80, 32)
        surf_grid = surf_grid.detach().numpy()
        maxD, rmsD = 0.0, 0.0
        if len(pts_v) > 0:
            for p in pts_v:
                d = np.min(np.linalg.norm(surf_grid - p, axis=1))
                maxD = max(maxD, d); rmsD += d * d
            rmsD = np.sqrt(rmsD / max(1, len(pts_v)))

        print(f"[PIPELINE:done:part:{pid}:maxDist={maxD:.6f}:rmsDist={rmsD:.6f}]")
        results.append({'pid': pid, 't_P': tP, 't_Q': tQ,
                        'maxDist': maxD, 'rmsDist': rmsD})

    print(f"\n[PIPELINE:done:all:{len(results)}]")
    print(f"=== {len(results)} partitions optimized ===")
    for r in results:
        md = r.get('maxDist', float('nan'))
        rd = r.get('rmsDist', float('nan'))
        print(f"  part {r['pid']}: t_P={r['t_P']:.4f} t_Q={r['t_Q']:.4f} "
              f"maxDist={md:.6f} rmsDist={rd:.6f}")

    # Write tolerance.txt
    tol_f = os.path.join(data_dir, "tolerance.txt")
    with open(tol_f, 'w') as f:
        f.write("pid t_P t_Q maxDist rmsDist\n")
        for r in results:
            md = r.get('maxDist', 0)
            rd = r.get('rmsDist', 0)
            f.write(f"{r['pid']} {r['t_P']:.6f} {r['t_Q']:.6f} {md:.6f} {rd:.6f}\n")
    print(f"  wrote {tol_f}")
    print(f"[PIPELINE:file:tolerance:{tol_f}]")

if __name__ == "__main__":
    main()
