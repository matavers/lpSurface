"""
独立凹性分割测试：从 NURBS 曲面生成 → 分区 → 平滑 → 模拟直纹方向 → 检测 + 分割 → 可视化

用法: python test_concave_split.py
无外部依赖（除 nurbs_eval, PyVista, numpy）
"""

import sys, os, random
import numpy as np
import pyvista as pv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nurbs_eval import NurbsSurface

TAB10 = np.array([
    [0.122,0.467,0.706],[1.000,0.498,0.055],[0.173,0.627,0.173],[0.839,0.153,0.157],
    [0.580,0.404,0.741],[0.549,0.337,0.294],[0.890,0.467,0.761],[0.498,0.498,0.498],
    [0.738,0.738,0.131],[0.090,0.745,0.812],
])

# ═══════════════════════════════════════════════════════════════════
# 1. NURBS surface + mesh generation (Python only)
# ═══════════════════════════════════════════════════════════════════

def make_wavy_surface():
    import tempfile, subprocess
    # Write surface to temp file, reload with NurbsSurface
    # Alternatively, build control points directly:
    from collections import namedtuple
    # Use the C++ pipeline to generate — but for standalone, build inline:
    nU = nV = 9; degU = degV = 3
    xs = np.linspace(-1.5, 1.5, 9)
    ys = np.linspace(-1.5, 1.5, 9)
    cp = np.zeros((nU, nV, 4))
    for i in range(nU):
        for j in range(nV):
            x, y = xs[i], ys[j]
            z = 0.15*np.sin(2.5*x)*np.cos(3.0*y) + 0.10*np.sin(5.0*x+1.2)*np.sin(4.0*y+0.8) \
                + 0.08*np.cos(7.0*x)*np.sin(6.0*y-0.5) + 0.05*np.sin(9.0*x-1.0)*np.cos(8.0*y+1.5)
            cp[i, j] = [x, y, z, 1.0]

    # Knots: clamped degree-3 with 9 CPs → 13 knots
    nK = nU + degU + 1
    knots = np.zeros(nK); knots[:degU+1] = 0; knots[-(degU+1):] = 1
    inner = nK - 2*(degU+1)
    if inner > 0: knots[degU+1:-degU-1] = np.linspace(0,1,inner+2)[1:-1]

    s = NurbsSurface.__new__(NurbsSurface)
    s.nU, s.nV = nU, nV
    s.degU, s.degV = degU, degV
    s.knotsU, s.knotsV = knots.copy(), knots.copy()
    s.cp = cp
    return s


def generate_mesh(surf, res):
    """Generate (vertices, faces, uvs) from NURBS surface."""
    u_vals = np.linspace(0, 1, res)
    v_vals = np.linspace(0, 1, res)
    verts = []
    uvs = []
    idx_map = {}
    k = 0
    for i, u in enumerate(u_vals):
        for j, v in enumerate(v_vals):
            p = surf.evaluate(float(u), float(v))
            verts.append(p)
            uvs.append([u, v])
            idx_map[(i, j)] = k
            k += 1

    faces = []
    for i in range(res - 1):
        for j in range(res - 1):
            a = idx_map[(i, j)]
            b = idx_map[(i+1, j)]
            c = idx_map[(i+1, j+1)]
            d = idx_map[(i, j+1)]
            faces.append([a, b, c])
            faces.append([a, c, d])
    return np.array(verts), np.array(faces, dtype=int), np.array(uvs)


# ═══════════════════════════════════════════════════════════════════
# 2. Simple grid-based partition in UV domain
#    Divide UV [0,1]x[0,1] into a grid, each cell = one partition
# ═══════════════════════════════════════════════════════════════════

def grid_partition(uvs, faces, n_u=4, n_v=4):
    """Assign each face to a grid-based partition based on centroid UV."""
    n_faces = len(faces)
    labels = np.full(n_faces, -1, dtype=int)
    for fi in range(n_faces):
        f = faces[fi]
        cu = (uvs[f[0],0] + uvs[f[1],0] + uvs[f[2],0]) / 3
        cv = (uvs[f[0],1] + uvs[f[1],1] + uvs[f[2],1]) / 3
        gu = min(n_u-1, int(cu * n_u))
        gv = min(n_v-1, int(cv * n_v))
        labels[fi] = gu + gv * n_u
    return labels, n_u * n_v


# ═══════════════════════════════════════════════════════════════════
# 3. Boundary extraction + simple smoothing
# ═══════════════════════════════════════════════════════════════════

def extract_partition_loops(face_labels, faces, verts, uvs, n_parts):
    loops_3d, loops_uv = [], []
    for pid in range(n_parts):
        # Gather faces for this partition
        pfaces = [fi for fi in range(len(faces)) if face_labels[fi] == pid]
        if len(pfaces) < 3:
            loops_3d.append(np.zeros((0,3)))
            loops_uv.append(np.zeros((0,2)))
            continue

        # Find boundary edges (appear exactly once in partition)
        edge_count = {}
        for fi in pfaces:
            f = faces[fi]
            for k in range(3):
                a, b = f[k], f[(k+1)%3]
                if a > b: a, b = b, a
                edge_count[(a,b)] = edge_count.get((a,b), 0) + 1
        bedges = [e for e, c in edge_count.items() if c == 1]

        if len(bedges) < 3:
            loops_3d.append(np.zeros((0,3)))
            loops_uv.append(np.zeros((0,2)))
            continue

        # Build adjacency and trace loop
        adj = {}
        for a, b in bedges:
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)

        # Trace
        loop_idx = []
        cur = list(adj.keys())[0]; prev = -1
        for _ in range(10000):
            loop_idx.append(cur)
            nbs = adj[cur]
            nxt = -1
            for nb in nbs:
                if nb != prev: nxt = nb; break
            if nxt < 0 or nxt == loop_idx[0]: break
            prev, cur = cur, nxt

        if len(loop_idx) < 3:
            loops_3d.append(np.zeros((0,3)))
            loops_uv.append(np.zeros((0,2)))
            continue

        loops_3d.append(verts[loop_idx])
        loops_uv.append(uvs[loop_idx])

    return loops_3d, loops_uv


def smooth_loop(uv_loop, iters=5, lam=0.3):
    """Simple Laplacian smoothing on a closed 2D polyline."""
    pts = uv_loop.copy()
    n = len(pts)
    if n < 3: return pts
    for _ in range(iters):
        new_pts = pts.copy()
        for i in range(n):
            prev = pts[(i-1)%n]; nxt = pts[(i+1)%n]
            new_pts[i] = pts[i] + lam * ((prev + nxt)/2 - pts[i])
        pts = new_pts
    return pts


# ═══════════════════════════════════════════════════════════════════
# 4. Ruling direction simulation (pick t_P, t_Q)
# ═══════════════════════════════════════════════════════════════════

def assign_ruling(loop):
    """Pick two opposite-ish points on the loop as ruling endpoints."""
    n = len(loop)
    if n < 4: return 0.25, 0.75
    # Use bounding box long axis direction
    diag = loop.max(axis=0) - loop.min(axis=0)
    axis_dir = np.argmax(diag[:2])  # 0=u, 1=v
    tP, tQ = 0.2, 0.5 + random.uniform(0.25, 0.35)
    return tP, tQ


# ═══════════════════════════════════════════════════════════════════
# 5. Concavity detection (same as C++ version)
# ═══════════════════════════════════════════════════════════════════

def convex_hull_2d(pts):
    n = len(pts);
    if n < 3: return list(range(n))
    def cross(o,a,b): return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    idx = sorted(range(n), key=lambda i: (pts[i][0], pts[i][1]))
    hull = []
    for i in idx:
        while len(hull)>=2 and cross(pts[hull[-2]], pts[hull[-1]], pts[i]) <= 1e-10: hull.pop()
        hull.append(i)
    lo = len(hull)
    for i in reversed(idx[:-1]):
        while len(hull)>lo and cross(pts[hull[-2]], pts[hull[-1]], pts[i]) <= 1e-10: hull.pop()
        hull.append(i)
    if len(hull)>1: hull.pop()
    return hull

def detect_concave(uv_loop):
    n = len(uv_loop); 
    if n < 6: return False, None
    pts = np.array(uv_loop)
    diag = np.linalg.norm(pts.max(axis=0)-pts.min(axis=0))
    if diag < 1e-8: return False, None

    hull = convex_hull_2d(pts)
    if len(hull) < 3: return False, None

    best_ratio = 0; best_pk = None
    for hi in range(len(hull)):
        hnext = (hi+1)%len(hull)
        pA, pB = hull[hi], hull[hnext]
        arc = []; cur = (pA+1)%n
        while cur != pB: arc.append(cur); cur = (cur+1)%n; 
        if not arc: continue

        ab = pts[pB] - pts[pA]; l2 = np.dot(ab,ab)
        max_d = 0
        for vi in arc:
            t = np.dot(pts[vi]-pts[pA], ab)/l2 if l2>1e-12 else 0
            t = np.clip(t,0,1)
            proj = pts[pA] + t*ab
            d = np.linalg.norm(pts[vi]-proj)
            if d > max_d: max_d = d
        r = max_d/diag
        if r > best_ratio: best_ratio = r; best_pk = (pA,pB,arc)

    if best_ratio < 0.03: return False, None

    pA,pB,arc = best_pk
    # Determine tip vs corner
    area2 = pts[pA,0]*pts[pB,1]-pts[pA,1]*pts[pB,0]
    prev = pts[pB]
    for vi in arc: area2 += prev[0]*pts[vi,1]-prev[1]*pts[vi,0]; prev = pts[vi]

    if area2 > 0:  # tip
        m = len(arc)
        best_r = 1e9; bi = bj = -1
        for i in range(m):
            for j in range(i+3,m):
                ch = pts[arc[j]]-pts[arc[i]]; cl = np.linalg.norm(ch)
                if cl<1e-8: continue
                md = max(np.linalg.norm(pts[arc[k]] - (pts[arc[i]] + np.dot(pts[arc[k]]-pts[arc[i]],ch)/cl**2*ch)) for k in range(i+1,j))
                if md>0 and cl/md < best_r: best_r=cl/md; bi,bj=i,j
        if bi<0: return True, None
        return True, (tuple(pts[arc[bi]]), tuple(pts[arc[bj]]))
    else:  # corner
        max_d = -1; deepest = -1
        for vi in arc:
            ab = pts[pB]-pts[pA]; l2=np.dot(ab,ab)
            t = np.dot(pts[vi]-pts[pA],ab)/l2 if l2>1e-12 else 0
            d = np.linalg.norm(pts[vi]-(pts[pA]+np.clip(t,0,1)*ab))
            if d>max_d: max_d=d; deepest=vi
        if deepest<0: return True, None
        vd=pts[deepest]; din=vd-pts[(deepest+n-1)%n]; dout=pts[(deepest+1)%n]-vd
        din/=np.linalg.norm(din)+1e-12; dout/=np.linalg.norm(dout)+1e-12
        bis=din+dout; bis/=np.linalg.norm(bis)+1e-12
        mr=np.linalg.norm(pts[pB]-pts[pA])*2
        bt=1e9; bh=None
        for i in range(n):
            j=(i+1)%n
            if i==deepest or j==deepest: continue
            s1,e1=pts[i],pts[j]; d1=e1-s1; d2=bis*mr
            cr=d1[0]*d2[1]-d1[1]*d2[0]
            if abs(cr)<1e-12: continue
            t=((vd[0]-s1[0])*d2[1]-(vd[1]-s1[1])*d2[0])/cr
            u=((vd[0]-s1[0])*d1[1]-(vd[1]-s1[1])*d1[0])/cr
            if 0<=t<=1 and u>1e-3 and u<bt: bt=u; bh=vd+u*d2
        if bh is None: return True, None
        return True, (tuple(vd), tuple(bh))


# ═══════════════════════════════════════════════════════════════════
# 6. Apply split to face labels
# ═══════════════════════════════════════════════════════════════════

def apply_split(labels, faces, uvs, split_uv, src_pid):
    """Assign faces on each side of UV split line to src_pid/new_pid."""
    p0, p1 = np.array(split_uv[0]), np.array(split_uv[1])
    ab = p1 - p0
    n_faces = len(faces)
    new_pid = labels.max() + 1

    left_count = right_count = 0
    side = np.zeros(n_faces, dtype=int)
    for fi in range(n_faces):
        if labels[fi] != src_pid: continue
        f = faces[fi]
        cu = (uvs[f[0],0] + uvs[f[1],0] + uvs[f[2],0]) / 3
        cv = (uvs[f[0],1] + uvs[f[1],1] + uvs[f[2],1]) / 3
        cross = ab[0]*(cv-p0[1]) - ab[1]*(cu-p0[0])
        if cross > 1e-12:
            side[fi] = 1; right_count += 1
        else:
            left_count += 1

    if left_count < 3 or right_count < 3:
        return labels

    new_labels = labels.copy()
    for fi in range(n_faces):
        if labels[fi] == src_pid and side[fi] == 1:
            new_labels[fi] = new_pid
    return new_labels


# ═══════════════════════════════════════════════════════════════════
# 7. Main pipeline + visualization
# ═══════════════════════════════════════════════════════════════════

def main():
    print("1. Generating wavy NURBS surface...")
    surf = make_wavy_surface()
    verts, faces, uvs = generate_mesh(surf, 60)
    print(f"   Mesh: {len(verts)} verts, {len(faces)} faces")

    print("2. Grid partition (4x4 = 16 partitions)...")
    labels, n_parts = grid_partition(uvs, faces, 4, 4)
    print(f"   {n_parts} partitions")

    print("3. Extracting boundary loops...")
    loops_3d, loops_uv = extract_partition_loops(labels, faces, verts, uvs, n_parts)
    valid = [i for i in range(n_parts) if len(loops_3d[i]) >= 6]
    print(f"   {len(valid)} valid partitions (>= 6 vertices)")

    print("4. Smoothing boundaries (5 iters)...")
    loops_uv_smoothed = [smooth_loop(loops_uv[i]) if len(loops_uv[i]) >= 6
                         else loops_uv[i] for i in range(n_parts)]

    print("5. Assigning ruling directions + detecting concavity...")
    concave_info = []
    for pid in valid:
        is_c, line = detect_concave(loops_uv_smoothed[pid])
        tp, tq = assign_ruling(loops_3d[pid])
        concave_info.append({'pid':pid,'concave':is_c,'split_line':line,'tP':tp,'tQ':tq})

    n_c = sum(1 for c in concave_info if c['concave'])
    n_s = sum(1 for c in concave_info if c['split_line'])
    print(f"   Concave: {n_c}/{len(valid)}, can-split: {n_s}")

    print("6. Splitting concave partitions...")
    new_labels_before = labels.copy()
    n_splits = 0
    split_details = []
    for c in concave_info:
        if c['split_line']:
            new = apply_split(labels, faces, uvs, c['split_line'], c['pid'])
            if new.max() > labels.max():
                labels = new
                n_splits += 1
                split_details.append(c)
    print(f"   Applied {n_splits} splits, partitions: {labels.max()+1} (was {n_parts})")

    print("7. Re-extracting boundaries after split...")
    loops3d_new, loops_uv_new = extract_partition_loops(
        labels, faces, verts, uvs, labels.max() + 1)

    # ── PyVista ──────────────────────────────────────────────────
    pv.set_plot_theme("document")
    pl = pv.Plotter(shape=(1, 2))
    pl.subplot(0,0); pl.add_text("BEFORE (smoothed)", font_size=10)
    pl.subplot(0,1); pl.add_text("AFTER (split + re-extract)", font_size=10)

    # Background mesh
    farr = np.array(faces, dtype=int)
    mesh_obj = pv.PolyData(verts, np.hstack([np.full((len(faces),1),3), farr]))

    for sp in [0, 1]:
        pl.subplot(0, sp)
        pl.add_mesh(mesh_obj, color='lightgray', opacity=0.08, show_edges=False)

        cur_loops = loops_3d if sp == 0 else loops3d_new
        cur_uvs = loops_uv_smoothed if sp == 0 else loops_uv_new
        cur_concave = [c for c in concave_info] if sp == 0 else []

        for pid in range(len(cur_loops)):
            poly = cur_loops[pid]
            if len(poly) < 3: continue
            closed = np.vstack([poly, poly[0:1]])

            # Check if this pid was concave
            cinfo = next((c for c in concave_info if c['pid'] == pid), None)
            is_conc = cinfo['concave'] if cinfo else False
            has_split = cinfo['split_line'] is not None if cinfo else False

            if sp == 0:
                color = TAB10[pid % 10]
                lw = 1
            else:
                if has_split:
                    color = 'red'; lw = 3
                elif is_conc:
                    color = 'orange'; lw = 2
                else:
                    color = TAB10[2]; lw = 1

            polyd = pv.PolyData(closed, lines=np.hstack([[len(closed)], np.arange(len(closed))]))
            pl.add_mesh(polyd, color=color, line_width=lw)

            # Split lines on BEFORE view
            if sp == 0 and cinfo and cinfo['split_line']:
                uv_arr = cur_uvs[pid]; poly3 = cur_loops[pid]
                if len(uv_arr) >= 2:
                    p0, p1 = cinfo['split_line']
                    d0 = np.linalg.norm(uv_arr-np.array(p0),axis=1)
                    d1 = np.linalg.norm(uv_arr-np.array(p1),axis=1)
                    i0, i1 = np.argmin(d0), np.argmin(d1)
                    if i0 < len(poly3) and i1 < len(poly3):
                        pl.add_lines(np.array([poly3[i0],poly3[i1]]),
                                     color='yellow', width=6, connected=False)

    pl.link_views()
    pl.show()


if __name__ == '__main__':
    main()
