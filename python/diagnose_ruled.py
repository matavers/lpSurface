"""
Diagnose ruled surface quality: twist, self-intersection, normal consistency,
coverage, arc-length ratio. Reads ruled_surf_N.obj + part_N_loop.txt.
"""
import sys, os
import numpy as np

def load_obj(path):
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            if not p: continue
            if p[0] == 'v':
                verts.append([float(p[1]), float(p[2]), float(p[3])])
            elif p[0] == 'f':
                idxs = [int(x.split('/')[0]) - 1 for x in p[1:]]
                if len(idxs) == 4:
                    faces.append([idxs[0], idxs[1], idxs[2]])
                    faces.append([idxs[0], idxs[2], idxs[3]])
                else:
                    faces.append(idxs)
    return np.array(verts), np.array(faces)

def triangle_normal(v0, v1, v2):
    e1 = v1 - v0; e2 = v2 - v0
    n = np.cross(e1, e2)
    l = np.linalg.norm(n)
    return n / l if l > 1e-12 else np.zeros(3)

def check_normal_consistency(verts, faces):
    """Check if triangle normals point in the same direction (no sign flips)."""
    normals = np.array([triangle_normal(verts[f[0]], verts[f[1]], verts[f[2]]) for f in faces])
    mean_n = normals.mean(axis=0)
    mean_n = mean_n / (np.linalg.norm(mean_n) + 1e-12)
    dots = np.dot(normals, mean_n)
    n_flipped = int(np.sum(dots < -0.3))  # more than 90° from mean = likely flipped
    return n_flipped, dots

def check_self_intersection(verts, faces):
    """Simple check: bounding boxes of edge pairs. Returns count of potential intersections."""
    n_faces = len(faces)
    if n_faces > 500:
        return -1, "too many faces for naive check"
    edges = set()
    for f in faces:
        for i in range(3):
            a, b = f[i], f[(i+1)%3]
            edges.add((min(a,b), max(a,b)))
    edge_list = list(edges)
    n_edges = len(edge_list)
    n_cross = 0
    for i in range(n_edges):
        for j in range(i+1, n_edges):
            a0, a1 = verts[edge_list[i][0]], verts[edge_list[i][1]]
            b0, b1 = verts[edge_list[j][0]], verts[edge_list[j][1]]
            # Skip shared vertices
            if edge_list[i][0] in edge_list[j] or edge_list[i][1] in edge_list[j]:
                continue
            # Bbox quick reject
            if (max(a0[0],a1[0]) < min(b0[0],b1[0]) or max(b0[0],b1[0]) < min(a0[0],a1[0]) or
                max(a0[1],a1[1]) < min(b0[1],b1[1]) or max(b0[1],b1[1]) < min(a0[1],a1[1])):
                continue
            n_cross += 1
    return n_cross, "potential edge crossings"

def check_coverage(verts, faces, loop_verts):
    """How well does the ruled surface cover the loop's bounding box?"""
    if loop_verts is None or len(loop_verts) < 3:
        return None, "no loop data"
    loop_bb_min = loop_verts.min(axis=0)
    loop_bb_max = loop_verts.max(axis=0)
    surf_bb_min = verts.min(axis=0)
    surf_bb_max = verts.max(axis=0)

    loop_diag = np.linalg.norm(loop_bb_max - loop_bb_min)
    if loop_diag < 1e-8: return None, "degenerate loop"

    # Coverage ratio per axis
    loop_range = loop_bb_max - loop_bb_min
    surf_range = surf_bb_max - surf_bb_min
    cov = np.where(loop_range > 1e-8, surf_range / loop_range, 1.0)
    return cov, f"X={cov[0]:.2f} Y={cov[1]:.2f} Z={cov[2]:.2f}"

def load_loop(path):
    if not os.path.exists(path): return None
    return np.loadtxt(path)

def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./results"
    pid = 0
    summary = []

    while True:
        surf_f = os.path.join(data_dir, f"ruled_surf_{pid}.obj")
        loop_f = os.path.join(data_dir, f"part_{pid}_loop.txt")
        if not os.path.exists(surf_f): break

        verts, faces = load_obj(surf_f)
        loop = load_loop(loop_f)
        n_flipped, dots = check_normal_consistency(verts, faces)
        n_cross, cross_msg = check_self_intersection(verts, faces)
        cov, cov_msg = check_coverage(verts, faces, loop)

        nf = len(faces)
        flip_pct = n_flipped / max(1, nf) * 100

        # Arc length ratio from tolerance.txt
        arc_ratio = None
        tol_f = os.path.join(data_dir, "tolerance.txt")
        if os.path.exists(tol_f):
            with open(tol_f) as f:
                f.readline()
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3 and int(parts[0]) == pid:
                        tp, tq = float(parts[1]), float(parts[2])
                        arc_ratio = abs(tq - tp) / (1.0 - abs(tq - tp)) if abs(tq - tp) < 0.5 else None
                        break

        issues = []
        if n_flipped > nf * 0.3: issues.append(f"FLIP({flip_pct:.0f}%)")
        if isinstance(n_cross, int) and n_cross > nf * 0.5: issues.append(f"CROSS({n_cross})")
        if cov is not None and np.any(cov < 0.4): issues.append(f"GAP({cov_msg})")

        status = "ISSUES: " + ",".join(issues) if issues else "OK"
        print(f"  part {pid}: {nf} tris, {n_flipped} flipped ({flip_pct:.0f}%), "
              f"cross={cross_msg}, cov={cov_msg}, arc={f'{arc_ratio:.3f}' if arc_ratio else '?'} "
              f"[{status}]")

        summary.append({
            'pid': pid, 'n_flipped': n_flipped, 'flip_pct': flip_pct,
            'n_cross': n_cross, 'cov': cov, 'issues': issues
        })
        pid += 1

    print(f"\n=== Summary ({pid} surfaces) ===")
    bad = [s for s in summary if s['issues']]
    if bad:
        print(f"  {len(bad)} surfaces have issues:")
        for s in bad:
            print(f"    part {s['pid']}: {s['issues']}")
    else:
        print("  All surfaces pass basic checks.")

    # Root cause analysis
    print("\n=== Root cause hypothesis ===")
    print("  'Twisted' appearance = normal flips.")
    print("  Cause: gamma1 and gamma2 built from polygon arcs that wrap")
    print("  in the SAME direction, causing the linear interpolation")
    print("  to cross through the interior (diagonal rulings).")
    print("  Check: if flip_pct > 30%, the arc ordering is likely wrong.")
    print("  Fix: ensure gamma1 (P->Q) and gamma2 (Q->P) form an oriented")
    print("  loop around the partition, then the rulings stay on the boundary.")

if __name__ == "__main__":
    main()
