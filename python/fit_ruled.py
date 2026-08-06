"""
Partition ruled-surface fitting (working version).

Method: grid-search over split points t_P, t_Q on the polygon loop.
For each candidate, build a ruled surface with linear correspondence (phi=identity),
measure the max distance to interior points, and pick the best pair.

Full gradient-based optimization (B-spline correspondence + autodiff) is in
fit_ruled_grad.py for research use.

Usage: python fit_ruled.py <data_dir> [--n-splits 20]
"""
import sys, os
import numpy as np

def build_ruled_surface_samples(loop, t_P, t_Q, nu=30, nv=12):
    """
    loop: (N,3) closed polygon
    t_P, t_Q: split parameters in [0,1]
    Returns (nu*nv, 3) sampled points on the ruled surface.
    """
    N = len(loop)
    if t_P > t_Q:
        t_P, t_Q = t_Q, t_P

    # Point at parameter t along full loop
    def point_at(t):
        t = np.clip(t, 0.0, 1.0)
        idx_f = t * (N - 1)
        idx = int(idx_f)
        idx = max(0, min(N-2, idx))
        alpha = idx_f - idx
        return loop[idx] * (1 - alpha) + loop[idx+1] * alpha

    tp, tq = float(t_P), float(t_Q)
    idx_p = max(0, min(N-2, int(tp * (N-1))))
    idx_q = max(idx_p+1, min(N-1, int(tq * (N-1))))

    # gamma1: P -> Q
    pts1 = [point_at(tp)]
    for i in range(idx_p+1, idx_q):
        pts1.append(loop[i])
    pts1.append(point_at(tq))
    pts1 = np.array(pts1)

    # gamma2: Q -> P (wrapping)
    pts2 = [point_at(tq)]
    for i in range(idx_q+1, N):
        pts2.append(loop[i])
    for i in range(0, idx_p):
        pts2.append(loop[i])
    pts2.append(point_at(tp))
    pts2 = np.array(pts2)

    # Arc-length parameterize
    def arc_len_param(pts):
        segs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0], np.cumsum(segs)])
        total = cum[-1] or 1.0
        cum = cum / total
        return cum

    c1 = arc_len_param(pts1)
    c2 = arc_len_param(pts2)

    # Sample u in [0,1], interpolate pts1 and pts2
    u = np.linspace(0, 1, nu)
    p1 = np.zeros((nu, 3))
    p2 = np.zeros((nu, 3))
    for i, ui in enumerate(u):
        idx1 = min(np.searchsorted(c1, ui), len(pts1)-2)
        a1 = (ui - c1[idx1]) / (c1[idx1+1] - c1[idx1] + 1e-12)
        p1[i] = pts1[idx1] * (1-a1) + pts1[idx1+1] * a1

        idx2 = min(np.searchsorted(c2, ui), len(pts2)-2)
        a2 = (ui - c2[idx2]) / (c2[idx2+1] - c2[idx2] + 1e-12)
        p2[i] = pts2[idx2] * (1-a2) + pts2[idx2+1] * a2

    # Build grid
    v = np.linspace(0, 1, nv)
    surf = np.zeros((nu * nv, 3))
    for i in range(nu):
        for j in range(nv):
            surf[i*nv + j] = p1[i] * (1-v[j]) + p2[i] * v[j]
    return surf

def max_distance(surf_samples, pts):
    """Maximum distance from each point in pts to nearest sample in surf_samples."""
    max_d = 0.0
    for p in pts:
        d = np.min(np.linalg.norm(surf_samples - p, axis=1))
        if d > max_d:
            max_d = d
    return max_d

def rms_distance(surf_samples, pts):
    sq = 0.0
    for p in pts:
        d = np.min(np.linalg.norm(surf_samples - p, axis=1))
        sq += d * d
    return np.sqrt(sq / max(1, len(pts)))

def arc_length_between(loop, t_a, t_b):
    """Compute arc length along polygon loop between parameters t_a and t_b."""
    N = len(loop)
    if t_a > t_b:
        t_a, t_b = t_b, t_a
    segs = np.linalg.norm(np.diff(loop, axis=0), axis=1)
    total = np.sum(segs)
    ia = max(0, min(N-2, int(t_a * (N-1))))
    ib = max(ia+1, min(N-1, int(t_b * (N-1))))
    frac_a = t_a * (N-1) - ia
    frac_b = t_b * (N-1) - ib
    length = (1-frac_a) * segs[ia] + np.sum(segs[ia+1:ib]) + frac_b * segs[ib]
    return length, total

def optimize_ruled(loop_verts, interior_pts, n_splits=20):
    best_max = float('inf')
    best_rms = float('inf')
    best_tP = 0.25; best_tQ = 0.75
    nu, nv = 30, 12
    N = len(loop_verts)
    total_len = np.sum(np.linalg.norm(np.diff(loop_verts, axis=0), axis=1))
    min_len = total_len * 0.2

    candidates = np.linspace(0.05, 0.95, n_splits)
    total_candidates = 0; valid_candidates = 0

    for i, tp in enumerate(candidates):
        for j, tq in enumerate(candidates):
            total_candidates += 1
            if abs(tp - tq) < 0.15: continue
            len1, _ = arc_length_between(loop_verts, tp, tq)
            len2 = total_len - len1
            if len1 < min_len or len2 < min_len: continue
            if len1 > len2: len1, len2 = len2, len1
            if len2 < 1e-12 or len1 / len2 < 0.3: continue

            valid_candidates += 1
            surf = build_ruled_surface_samples(loop_verts, tp, tq, nu, nv)
            max_d = max_distance(surf, interior_pts)
            if max_d < best_max:
                best_max = max_d
                best_rms = rms_distance(surf, interior_pts)
                best_tP = tp; best_tQ = tq

    print(f"    candidates: {total_candidates} total, {valid_candidates} valid  "
          f"best t_P={best_tP:.4f} t_Q={best_tQ:.4f} max={best_max:.6f}")
    return best_tP, best_tQ, best_max, best_rms

def export_ruled_surface_obj(loop_verts, t_P, t_Q, path, nu=30, nv=12):
    """Export ruled surface as OBJ."""
    surf = build_ruled_surface_samples(loop_verts, t_P, t_Q, nu, nv)
    with open(path, 'w') as f:
        for i in range(nu * nv):
            p = surf[i]
            f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        for i in range(nu - 1):
            for j in range(nv - 1):
                a = i * nv + j + 1
                b = a + 1
                c = (i+1) * nv + j + 1
                d = c + 1
                f.write(f"f {a} {b} {d} {c}\n")
    print(f"    wrote {path}")

def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./results"
    n_splits = 20
    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == '--n-splits' and i+1 < len(args):
            n_splits = int(args[i+1]); i += 2
        else:
            i += 1

    pid = 0
    all_results = []
    while True:
        lf = os.path.join(data_dir, f"part_{pid}_loop.txt")
        pf = os.path.join(data_dir, f"part_{pid}_points.txt")
        if not os.path.exists(lf) or not os.path.exists(pf):
            break
        loop = np.loadtxt(lf)
        pts = np.loadtxt(pf)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        print(f"\n[Partition {pid}] loop={loop.shape} pts={pts.shape}")
        tP, tQ, maxD, rmsD = optimize_ruled(loop, pts, n_splits)

        out_f = os.path.join(data_dir, f"part_{pid}_ruled.txt")
        with open(out_f, 'w') as f:
            f.write(f"t_P={tP:.8f}\nt_Q={tQ:.8f}\n")
            f.write(f"maxDist={maxD:.8f}\nrmsDist={rmsD:.8f}\n")

        # Export ruled surface OBJ
        obj_f = os.path.join(data_dir, f"ruled_surf_{pid}.obj")
        export_ruled_surface_obj(loop, tP, tQ, obj_f)

        all_results.append({
            'pid': pid, 't_P': tP, 't_Q': tQ,
            'maxDist': maxD, 'rmsDist': rmsD
        })
        pid += 1

    # Write tolerance summary
    tol_f = os.path.join(data_dir, "tolerance.txt")
    with open(tol_f, 'w') as f:
        f.write("pid t_P t_Q maxDist rmsDist\n")
        for r in all_results:
            f.write(f"{r['pid']} {r['t_P']:.6f} {r['t_Q']:.6f} "
                    f"{r['maxDist']:.6f} {r['rmsDist']:.6f}\n")

    print(f"\n=== Tolerance Summary ({len(all_results)} partitions) ===")
    for r in all_results:
        print(f"  part {r['pid']}: max={r['maxDist']:.6f} rms={r['rmsDist']:.6f} "
              f"t_P={r['t_P']:.4f} t_Q={r['t_Q']:.4f}")


if __name__ == "__main__":
    main()
