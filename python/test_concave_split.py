"""
凹性分割快速测试 + 可视化

用法: python test_concave_split.py <data_dir>
直接读 part_N_loop.txt / part_N_loop_uv.txt，不调用优化器。
"""

import sys, os, time
import numpy as np
import pyvista as pv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TAB10 = np.array([
    [0.122, 0.467, 0.706], [1.000, 0.498, 0.055],
    [0.173, 0.627, 0.173], [0.839, 0.153, 0.157],
    [0.580, 0.404, 0.741], [0.549, 0.337, 0.294],
    [0.890, 0.467, 0.761], [0.498, 0.498, 0.498],
    [0.738, 0.738, 0.131], [0.090, 0.745, 0.812],
])


# ═══════════════════════════════════════════════════════════════════
# 2D concavity (Python version)
# ═══════════════════════════════════════════════════════════════════

def _cross2d(o, a, b):
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

def convex_hull_2d(pts):
    n = len(pts); 
    if n < 3: return list(range(n))
    idx = sorted(range(n), key=lambda i: (pts[i][0], pts[i][1]))
    hull = []
    for i in idx:
        while len(hull) >= 2 and _cross2d(pts[hull[-2]], pts[hull[-1]], pts[i]) <= 1e-10:
            hull.pop()
        hull.append(i)
    lower = len(hull)
    for i in reversed(idx[:-1]):
        while len(hull) > lower and _cross2d(pts[hull[-2]], pts[hull[-1]], pts[i]) <= 1e-10:
            hull.pop()
        hull.append(i)
    if len(hull) > 1: hull.pop()
    return hull

def point_to_line_dist(p, a, b):
    ab = np.array(b) - np.array(a)
    l2 = np.dot(ab, ab)
    if l2 < 1e-12: return np.linalg.norm(np.array(p) - np.array(a))
    t = np.clip(np.dot(np.array(p) - np.array(a), ab) / l2, 0, 1)
    return np.linalg.norm(np.array(p) - (np.array(a) + t * ab))

def detect_pockets(poly):
    n = len(poly); 
    if n < 4: return []
    hull = convex_hull_2d(poly)
    h = len(hull)
    if h < 3: return []
    pockets = []
    for hi in range(h):
        hnext = (hi + 1) % h
        pA, pB = hull[hi], hull[hnext]
        arc = []; cur = (pA + 1) % n
        while cur != pB:
            arc.append(cur); cur = (cur + 1) % n
            if len(arc) > n: break
        if not arc: continue
        max_d = max(point_to_line_dist(poly[vi], poly[pA], poly[pB]) for vi in arc)
        pockets.append({'start':pA,'end':pB,'arc':arc,'max_depth':max_d})
    return pockets


def split_tip(poly, pocket):
    arc = pocket['arc']; m = len(arc)
    if m < 4: return None
    best_ratio = 1e9; best_i = best_j = -1
    for i in range(m):
        for j in range(i + 3, m):
            chord = np.array(poly[arc[j]]) - np.array(poly[arc[i]])
            cl = np.linalg.norm(chord)
            if cl < 1e-8: continue
            max_d = max(point_to_line_dist(poly[arc[k]], poly[arc[i]], poly[arc[j]]) for k in range(i+1, j))
            if max_d > 0:
                r = cl / max_d
                if r < best_ratio: best_ratio = r; best_i, best_j = i, j
    if best_i < 0: return None
    return (poly[arc[best_i]], poly[arc[best_j]])


def split_corner(poly, pocket):
    arc = pocket['arc']
    if not arc: return None
    max_d = -1; deepest = -1
    for vi in arc:
        d = point_to_line_dist(poly[vi], poly[pocket['start']], poly[pocket['end']])
        if d > max_d: max_d = d; deepest = vi
    if deepest < 0: return None

    n = len(poly)
    vd = np.array(poly[deepest])
    din = vd - np.array(poly[(deepest + n - 1) % n])
    dout = np.array(poly[(deepest + 1) % n]) - vd
    din /= (np.linalg.norm(din) + 1e-12)
    dout /= (np.linalg.norm(dout) + 1e-12)
    bisector = din + dout
    bisector /= (np.linalg.norm(bisector) + 1e-12)

    max_ray = np.linalg.norm(np.array(poly[pocket['end']]) - np.array(poly[pocket['start']])) * 2.0
    best_t = 1e9; best_hit = None
    for i in range(n):
        j = (i + 1) % n
        if i == deepest or j == deepest: continue
        s1, e1 = np.array(poly[i]), np.array(poly[j])
        d1 = e1 - s1; d2 = bisector * max_ray
        cross = d1[0]*d2[1] - d1[1]*d2[0]
        if abs(cross) < 1e-12: continue
        t = ((vd[0]-s1[0])*d2[1] - (vd[1]-s1[1])*d2[0]) / cross
        u = ((vd[0]-s1[0])*d1[1] - (vd[1]-s1[1])*d1[0]) / cross
        if 0 <= t <= 1 and u > 1e-3 and u < best_t:
            best_t = u; best_hit = vd + u * d2
    if best_hit is None: return None
    return (tuple(vd), tuple(best_hit))


def analyze_concavity(poly_uv):
    n = len(poly_uv)
    if n < 4: return False, None, ''

    pts = np.array(poly_uv)
    diag = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
    if diag < 1e-8: return False, None, ''

    pockets = detect_pockets(poly_uv)
    best = None; best_ratio = 0.0
    for pk in pockets:
        r = pk['max_depth'] / diag
        if r > best_ratio: best_ratio = r; best = pk

    THRESHOLD = 0.03
    if best is None or best_ratio < THRESHOLD: return False, None, ''

    pA, pB = np.array(poly_uv[best['start']]), np.array(poly_uv[best['end']])
    area2 = pA[0]*pB[1] - pA[1]*pB[0]
    prev = pB
    for vi in best['arc']:
        p = np.array(poly_uv[vi]); area2 += prev[0]*p[1] - prev[1]*p[0]; prev = p

    stype = 'tip' if area2 > 0 else 'corner'
    line = split_tip(poly_uv, best) if area2 > 0 else split_corner(poly_uv, best)
    return (True, line, stype) if line else (True, None, stype)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_concave_split.py <data_dir>")
        return 1
    data_dir = sys.argv[1]
    if not os.path.isdir(data_dir):
        print(f"Not a directory: {data_dir}")
        return 1

    # Scan part_N_loop.txt and part_N_loop_uv.txt
    results = []
    max_pid = 50
    for pid in range(max_pid):
        f3d = os.path.join(data_dir, f'part_{pid}_loop.txt')
        fuv = os.path.join(data_dir, f'part_{pid}_loop_uv.txt')
        if not os.path.exists(f3d): continue
        loop_3d = np.loadtxt(f3d)
        loop_uv = np.loadtxt(fuv) if os.path.exists(fuv) else None
        if loop_3d.ndim == 1: loop_3d = loop_3d.reshape(-1, 3)
        if loop_uv is None:
            continue

        is_c, split_line, stype = analyze_concavity(loop_uv.tolist())
        results.append({
            'pid': pid, 'concave': is_c, 'split_line': split_line,
            'stype': stype, 'n_verts': len(loop_3d),
            'poly_3d': loop_3d, 'poly_uv': loop_uv,
        })

    n_c = sum(1 for r in results if r['concave'])
    n_split = sum(1 for r in results if r['split_line'])
    print(f"Partitions: {len(results)} | Concave: {n_c} | Can-split: {n_split}")
    for r in results:
        status = f"{r['stype']:6s}" if r['concave'] else "OK"
        print(f"  part {r['pid']:2d}: {r['n_verts']:3d}v  {status}")

    # ── Visualization ──
    pv.set_plot_theme("document")
    pl = pv.Plotter()
    pl.add_text("Red=concave  Yellow=split-line  Green=OK | Partitions only",
                position="upper_left", font_size=10)

    # Load NURBS surface mesh
    mesh_path = os.path.join(data_dir, 'mesh.obj')
    if os.path.exists(mesh_path):
        verts, faces = [], []
        with open(mesh_path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                if parts[0] == 'v':
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif parts[0] == 'f':
                    fv = [v.split('/')[0] for v in parts[1:]]
                    if len(fv) == 3:
                        faces.append([int(v) - 1 for v in fv])
        if verts:
            farr = np.array(faces, dtype=int)
            mesh = pv.PolyData(np.array(verts),
                               np.hstack([np.full((len(faces), 1), 3), farr]))
            pl.add_mesh(mesh, color='lightgray', opacity=0.15, show_edges=False)

    for r in results:
        poly = r['poly_3d']
        closed = np.vstack([poly, poly[0:1]])
        # Open polyline for connected rendering
        n = len(closed)
        idx_arr = np.arange(n)
        lines = np.hstack([[n], idx_arr])

        if r['split_line']:
            color = 'red'
            polyd = pv.PolyData(closed, lines=lines)
            pl.add_mesh(polyd, color=color, line_width=4)
        elif r['concave']:
            color = 'orange'
            polyd = pv.PolyData(closed, lines=lines)
            pl.add_mesh(polyd, color=color, line_width=3)
        else:
            polyd = pv.PolyData(closed, lines=lines)
            pl.add_mesh(polyd, color=TAB10[2], line_width=2)

        # Draw split line (UV -> closest 3D correspondence)
        if r['split_line']:
            p0_uv, p1_uv = r['split_line']
            uv_arr = r['poly_uv']
            d0 = np.linalg.norm(uv_arr - np.array(p0_uv), axis=1)
            d1 = np.linalg.norm(uv_arr - np.array(p1_uv), axis=1)
            i0, i1 = np.argmin(d0), np.argmin(d1)
            seg = np.array([poly[i0], poly[i1]])
            pl.add_lines(seg, color='yellow', width=8, connected=False)

    pl.show_grid()
    pl.camera_position = 'xy'
    pl.show()


if __name__ == '__main__':
    main()
