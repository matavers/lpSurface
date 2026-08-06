"""
Ruling bounds via 2D-projected visibility matrix (fast + correct).

Algorithm:
  1. Project polygon loop onto its local plane (PCA normal).
  2. Build γ₁ and γ₂ from split points t_P, t_Q — also projected.
  3. For each sampled u on γ₁, binary search for φ_lower/φ_upper on γ₂
     using 2D segment-polygon intersection check on the projected plane.
  4. Enforce monotonicity of φ_lower, φ_upper.
"""
import numpy as np

TOL_RULING = 0.02   # ignored if using visibility — use MASK instead
N_SAMPLES = 30


def _polygon_normal(pts):
    """Newell normal."""
    nv = np.zeros(3); n = len(pts)
    for i in range(n):
        j = (i+1)%n
        nv[0] += (pts[i][1]-pts[j][1])*(pts[i][2]+pts[j][2])
        nv[1] += (pts[i][2]-pts[j][2])*(pts[i][0]+pts[j][0])
        nv[2] += (pts[i][0]-pts[j][0])*(pts[i][1]+pts[j][1])
    return nv / (np.linalg.norm(nv)+1e-12)


def _project_2d(pts, normal):
    if abs(normal[2]) < 0.9:
        u = np.cross(normal, [0,0,1.0])
    else:
        u = np.cross(normal, [1,0,0])
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    return np.column_stack([np.dot(pts, u), np.dot(pts, v)])


def _seg_intersect_2d(a0, a1, b0, b1):
    """2D segment intersection (no endpoint sharing)."""
    d1 = (b1[0]-b0[0])*(a0[1]-b0[1]) - (b1[1]-b0[1])*(a0[0]-b0[0])
    d2 = (b1[0]-b0[0])*(a1[1]-b0[1]) - (b1[1]-b0[1])*(a1[0]-b0[0])
    d3 = (a1[0]-a0[0])*(b0[1]-a0[1]) - (a1[1]-a0[1])*(b0[0]-a0[0])
    d4 = (a1[0]-a0[0])*(b1[1]-a0[1]) - (a1[1]-a0[1])*(b1[0]-a0[0])
    return d1*d2 < -1e-10 and d3*d4 < -1e-10


def _arc_length_param(curve_pts):
    diffs = np.diff(curve_pts, axis=0)
    lengths = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0], np.cumsum(lengths)])
    total = cum[-1] if cum[-1] > 1e-12 else 1.0
    return cum / total, total


def _interpolate_curve(curve_pts, cum, t):
    idx = min(np.searchsorted(cum, t), len(curve_pts) - 2)
    alpha = (t - cum[idx]) / (cum[idx+1] - cum[idx] + 1e-12)
    return curve_pts[idx] * (1 - alpha) + curve_pts[idx+1] * alpha


def _seg_intersect_3d(a0, a1, b0, b1):
    """Check 3D segment intersection."""
    for d in range(3):
        if max(a0[d],a1[d]) < min(b0[d],b1[d]) or max(b0[d],b1[d]) < min(a0[d],a1[d]):
            return False
    ab = a1 - a0; cd = b1 - b0
    n = np.abs(np.cross(ab, cd)); drop = int(np.argmax(n))
    ax = [d for d in range(3) if d != drop]
    def c2(u,v): return u[ax[0]]*v[ax[1]] - u[ax[1]]*v[ax[0]]
    d1 = c2(cd, a0-b0); d2 = c2(cd, a1-b0)
    d3 = c2(ab, b0-a0); d4 = c2(ab, b1-a0)
    return d1*d2 < -1e-10 and d3*d4 < -1e-10


def compute_bounds(loop, t_P, t_Q, interior_pts=None, n_samples=N_SAMPLES):
    """
    Compute φ_lower, φ_upper using 2D-projected polygon edge intersection.
    Ruling is VALID if its 2D projection doesn't cross any polygon edge.
    """
    n = len(loop)
    tp = min(t_P, t_Q); tq = max(t_P, t_Q)
    idx_p = max(0, min(n-2, int(tp * (n-1))))
    idx_q = max(idx_p+1, min(n-1, int(tq * (n-1))))

    # Build γ₁ and γ₂ (3D)
    pts1 = [loop[idx_p] * (1-(tp*(n-1)-idx_p)) + loop[idx_p+1] * (tp*(n-1)-idx_p)]
    for i in range(idx_p+1, idx_q): pts1.append(loop[i])
    pts1.append(loop[idx_q]*(1-(tq*(n-1)-idx_q)) + loop[(idx_q+1)%n]*(tq*(n-1)-idx_q))
    pts1 = np.array(pts1); cum1, _ = _arc_length_param(pts1)

    pts2 = [pts1[0]]
    for i in range(idx_p-1, -1, -1): pts2.append(loop[i])
    for i in range(n-1, idx_q, -1): pts2.append(loop[i])
    pts2.append(pts1[-1])
    pts2 = np.array(pts2); cum2, _ = _arc_length_param(pts2)

    # Use 3D edges for correct concavity detection
    edges3d = [(loop[i], loop[(i+1)%n]) for i in range(n)]

    def ruling_valid_3d(u1, u2):
        """Check if 3D segment γ₁(u1)—γ₂(u2) crosses non-adjacent polygon edges."""
        p1 = _interpolate_curve(pts1, cum1, u1)
        p2 = _interpolate_curve(pts2, cum2, u2)
        for ei, (e0, e1) in enumerate(edges3d):
            # Skip edges sharing an endpoint with p1 or p2
            if max(abs(e0-p1).max(), abs(e1-p1).max(), abs(e0-p2).max(), abs(e1-p2).max()) < 1e-8:
                continue
            if _seg_intersect_3d(p1, p2, e0, e1):
                return False
        return True

    phi_lower = np.zeros(n_samples)
    phi_upper = np.zeros(n_samples)

    for si in range(n_samples):
        u = si / (n_samples - 1)
        u_guess = u

        # Binary search for lower bound
        lo, hi = 0.0, u_guess
        for _ in range(12):
            mid = (lo + hi) / 2
            if ruling_valid_3d(u, mid):
                hi = mid
            else:
                lo = mid
        phi_lower[si] = hi

        # Binary search for upper bound
        lo, hi = u_guess, 1.0
        for _ in range(12):
            mid = (lo + hi) / 2
            if ruling_valid_3d(u, mid):
                lo = mid
            else:
                hi = mid
        phi_upper[si] = lo

    # Monotonicity
    for i in range(1, n_samples):
        phi_lower[i] = max(phi_lower[i], phi_lower[i-1])
        phi_upper[i] = max(phi_upper[i], phi_upper[i-1])
    for i in range(n_samples):
        if phi_lower[i] > phi_upper[i]:
            phi_lower[i] = phi_upper[i]

    return phi_lower, phi_upper
