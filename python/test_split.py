"""
简化分割测试：从原始网格到拉普拉斯平滑 → 凹性检测 + 分割可视化
跳过优化器和容差迭代，快速验证凹性检测和分割效果

用法:
  python test_split.py [--surface wavy] [--K 16] [--threshold 0.30] [--smooth-iters 16]

步骤:
  1. C++ 管线运行一次 (无优化器) → 生成分区 + 拉普拉斯平滑后的边界
     (拉普拉斯平滑复用主流程 C++ 实现，--smooth-iters 控制迭代次数)
  2. Python 实现凹性检测 (作用于平滑后边界) → 标记分割区域 + 分割线
  3. PyVista 交互可视化: 原始/平滑边界对比 + 分割线 + 凹部标记
"""

import os, sys, time, subprocess, tempfile, argparse
import numpy as np
import pyvista as pv
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXE = SCRIPT_DIR.parent / "build" / "Release" / "distillation.exe"

TAB10 = np.array([
    [0.122,0.467,0.706],[1.000,0.498,0.055],[0.173,0.627,0.173],[0.839,0.153,0.157],
    [0.580,0.404,0.741],[0.549,0.337,0.294],[0.890,0.467,0.761],[0.498,0.498,0.498],
    [0.738,0.738,0.131],[0.090,0.745,0.812],
])


# ═══════════════════════════════════════════════════════════════
# 2D Convex Hull (Andrew's monotone chain)
# ═══════════════════════════════════════════════════════════════
def cross2d(o, a, b):
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])


def convex_hull_2d(pts):
    n = len(pts)
    if n < 3: return list(range(n))
    idx = sorted(range(n), key=lambda i: (pts[i][0], pts[i][1]))
    hull = []
    for i in range(n):
        while len(hull) >= 2 and cross2d(pts[hull[-2]], pts[hull[-1]], pts[idx[i]]) <= 0:
            hull.pop()
        hull.append(idx[i])
    lower = len(hull)
    for i in range(n-2, -1, -1):
        while len(hull) > lower and cross2d(pts[hull[-2]], pts[hull[-1]], pts[idx[i]]) <= 0:
            hull.pop()
        hull.append(idx[i])
    if len(hull) > 1: hull.pop()
    return hull


def point_to_line_dist(p, a, b):
    ab = np.array(b) - np.array(a)
    l2 = ab.dot(ab)
    if l2 < 1e-12: return np.linalg.norm(np.array(p) - np.array(a))
    t = max(0, min(1, np.dot(np.array(p) - np.array(a), ab) / l2))
    proj = np.array(a) + t * ab
    return np.linalg.norm(np.array(p) - proj)


def detect_pockets(poly):
    n = len(poly)
    if n < 4: return []
    hull = convex_hull_2d(poly)
    h = len(hull)
    if h < 3: return []
    hull_set = set(hull)

    pockets = []
    for hi in range(h):
        hnext = (hi + 1) % h
        pA, pB = hull[hi], hull[hnext]
        arc = []
        cur = (pA + 1) % n
        while cur != pB:
            arc.append(cur)
            cur = (cur + 1) % n
            if len(arc) > n: break
        if not arc: continue
        max_depth = max(point_to_line_dist(poly[vi], poly[pA], poly[pB]) for vi in arc) if arc else 0
        width = np.linalg.norm(np.array(poly[pA]) - np.array(poly[pB]))
        pockets.append({'start': pA, 'end': pB, 'arc': arc,
                        'depth': max_depth, 'width': width})
    return pockets


def compute_poly_diam(poly):
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    return np.sqrt((max(xs)-min(xs))**2 + (max(ys)-min(ys))**2)


def split_tip(poly, pocket):
    arc = pocket['arc']
    m = len(arc)
    if m < 4: return None
    best_ratio = None
    best_pair = None
    for i in range(m):
        for j in range(i+3, m):
            a, b = poly[arc[i]], poly[arc[j]]
            chord = np.array(b) - np.array(a)
            cl = np.linalg.norm(chord)
            if cl < 1e-8: continue
            maxd = max(point_to_line_dist(poly[arc[k]], a, b) for k in range(i+1, j))
            if maxd < 1e-8: maxd = 1e-8
            ratio = cl / maxd
            if best_ratio is None or ratio < best_ratio:
                best_ratio = ratio; best_pair = (arc[i], arc[j])
    if best_pair is None: return None
    return (np.array(poly[best_pair[0]]), np.array(poly[best_pair[1]]))


def split_corner(poly, pocket):
    arc, pA, pB = pocket['arc'], pocket['start'], pocket['end']
    if not arc: return None
    deepest = max(arc, key=lambda vi: point_to_line_dist(poly[vi], poly[pA], poly[pB]))
    vd = np.array(poly[deepest]); n = len(poly)
    vp = np.array(poly[(deepest+n-1)%n])
    vn = np.array(poly[(deepest+1)%n])
    indir = vd - vp; outdir = vn - vd
    indir = indir / (np.linalg.norm(indir)+1e-12)
    outdir = outdir / (np.linalg.norm(outdir)+1e-12)
    bis = indir + outdir
    bl = np.linalg.norm(bis)
    if bl < 1e-12: return None
    bis = bis / bl
    edge = np.array(poly[pB]) - np.array(poly[pA])
    if bis[0]*edge[1] - bis[1]*edge[0] < 0: bis = -bis
    max_ray = np.linalg.norm(np.array(poly[pA]) - np.array(poly[pB])) * 2
    best_t = None; best_hit = None
    for i in range(n):
        j = (i+1)%n
        if i == deepest or j == deepest: continue
        if i == (deepest+n-1)%n or j == (deepest+1)%n: continue
        s1, e1 = np.array(poly[i]), np.array(poly[j])
        d1 = e1 - s1; d2 = bis * max_ray
        cross = d1[0]*d2[1] - d1[1]*d2[0]
        if abs(cross) < 1e-12: continue
        t = ((vd[0]-s1[0])*d2[1] - (vd[1]-s1[1])*d2[0]) / cross
        u = ((vd[0]-s1[0])*d1[1] - (vd[1]-s1[1])*d1[0]) / cross
        if 0 <= t <= 1 and u > 1e-3:
            hit = vd + u * d2
            if best_t is None or u < best_t:
                best_t = u; best_hit = hit
    if best_hit is None: return None
    return (vd, best_hit)


def classify_pocket(poly, pocket):
    """Returns (split_type, split_line) or (0, None). 1=tip, 2=corner."""
    pA, pB = pocket['start'], pocket['end']
    arc = pocket['arc']
    area2 = np.array(poly[pA])[0]*np.array(poly[pB])[1] - np.array(poly[pA])[1]*np.array(poly[pB])[0]
    prev = np.array(poly[pB])
    for vi in arc:
        p = np.array(poly[vi])
        area2 += prev[0]*p[1] - prev[1]*p[0]
        prev = p
    if area2 > 0:
        sl = split_tip(poly, pocket)
        return (1, sl) if sl else (0, None)
    else:
        sl = split_corner(poly, pocket)
        return (2, sl) if sl else (0, None)


# ═══════════════════════════════════════════════════════════════
# Load C++ output
# ═══════════════════════════════════════════════════════════════
def load_boundaries_uv(path):
    polys = []
    with open(path) as f:
        while True:
            header = f.readline()
            if not header: break
            parts = header.strip().split()
            if len(parts) < 1: continue
            try:
                npts = int(parts[0])
            except ValueError:
                continue
            poly = []
            for _ in range(npts):
                line = f.readline()
                if not line: break
                coords = line.strip().split()
                if len(coords) >= 2:
                    poly.append((float(coords[0]), float(coords[1])))
            if len(poly) >= 3:
                polys.append(poly)
    return polys


def load_boundaries_3d(path):
    """Load 3D boundaries: header `npts pid`, then npts lines `x y z`."""
    polys = []
    with open(path) as f:
        while True:
            header = f.readline()
            if not header: break
            parts = header.strip().split()
            if len(parts) < 2: continue
            try:
                npts = int(parts[0])
            except ValueError:
                continue
            poly = []
            for _ in range(npts):
                line = f.readline()
                if not line: break
                coords = line.strip().split()
                if len(coords) >= 3:
                    poly.append((float(coords[0]), float(coords[1]), float(coords[2])))
            if len(poly) >= 3:
                polys.append(poly)
    return polys


def load_face_labels(path):
    return np.loadtxt(path, dtype=int)


def load_obj(path):
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            if parts[0] == 'v':
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f':
                fv = [v.split('/')[0] for v in parts[1:4]]
                faces.append([int(v)-1 for v in fv])
    return np.array(verts), np.array(faces)


# ═══════════════════════════════════════════════════════════════
# Concavity detection (area-deficit based)
# ═══════════════════════════════════════════════════════════════

def polygon_area(poly):
    """Absolute area of closed polygon."""
    n = len(poly)
    a = 0.0
    for i in range(n):
        j = (i+1)%n
        a += poly[i][0]*poly[j][1] - poly[j][0]*poly[i][1]
    return abs(a) / 2.0

def convex_hull_area(poly):
    """Area of convex hull."""
    hull = convex_hull_2d(poly)
    hpts = [poly[i] for i in hull]
    return polygon_area(hpts)

def detect_concave(poly, threshold=0.05):
    """Returns (is_concave, deficit, split_info).
    split_info = (split_type, split_line) or None."""
    pa = polygon_area(poly)
    ha = convex_hull_area(poly)
    if ha < 1e-12: return False, 0, None
    deficit = 1.0 - pa / ha
    if deficit <= threshold: return False, deficit, None

    pockets = detect_pockets(poly)
    if not pockets: return True, deficit, None
    diam = compute_poly_diam(poly)
    best_ratio = 0; best_pkt = None
    for pi, pkt in enumerate(pockets):
        r = pkt['depth'] / diam
        if r > best_ratio: best_ratio = r; best_pkt = pi
    if best_pkt is None: return True, deficit, None
    split_type, split_line = classify_pocket(poly, pockets[best_pkt])
    return True, deficit, (split_type, split_line)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--surface', default='wavy')
    parser.add_argument('--K', type=int, default=16)
    parser.add_argument('--threshold', type=float, default=0.30)
    parser.add_argument('--smooth-iters', type=int, default=16,
                        help='Laplacian smoothing iterations in C++ pipeline '
                             '(16 ~ auto K=ceil((2*sigma/h)^2))')
    args = parser.parse_args()

    EXE_PATH = str(EXE)
    if not os.path.exists(EXE_PATH):
        EXE_PATH = EXE_PATH.replace('\\build\\', '\\OCCT\\build\\')
        if not os.path.exists(EXE_PATH):
            print(f"distillation.exe not found, build first")
            return 1

    results_dir = str(SCRIPT_DIR / "test_results")

    # ── Step 1: Run C++ pipeline once (no optimizer) ──
    if os.path.exists(results_dir):
        import shutil; shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    cmd = [EXE_PATH, f'--surface={args.surface}',
           f'--export-dir={results_dir}',
           f'--smooth-iters={args.smooth_iters}',
           f'--max-retries=1']
    print(f"Running: {' '.join(cmd)}")
    env = os.environ.copy()
    env['PATH'] = str(SCRIPT_DIR.parent / 'build' / 'Release') + ';' + env.get('PATH', '')
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env,
                            cwd=str(SCRIPT_DIR.parent))
    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    if result.returncode != 0:
        print("C++ error:", result.stderr[-500:])
        return 1

    # ── Step 2: Find retry dir ──
    retry_dirs = sorted(Path(results_dir).glob("retry_*"))
    if not retry_dirs:
        print("No retry dir found, checking top-level...")
        data_dir = Path(results_dir)
    else:
        data_dir = retry_dirs[0]
    print(f"Loading from: {data_dir}")

    # ── Step 3: Load data ──
    mfile = data_dir / "mesh.obj"
    lfile = data_dir / "face_labels.txt"

    verts3d, faces = load_obj(str(mfile)) if mfile.exists() else (None, None)
    if mfile.exists():
        print(f"Mesh: {len(verts3d)} verts, {len(faces)} faces")

    face_labels = load_face_labels(str(lfile)) if lfile.exists() else None
    n_parts = int(face_labels.max()) + 1 if face_labels is not None else 0

    # Load per-partition loop files (1:1 with partitions), may have gaps
    boundaries_uv = {}  # pid → list of [(x,y), ...]
    boundaries_3d = {}  # pid → list of [(x,y,z), ...]
    max_pid = 0

    for pid in range(50):
        luv = data_dir / f"part_{pid}_loop_uv.txt"
        l3d = data_dir / f"part_{pid}_loop.txt"
        if not luv.exists():
            continue
        # UV loop: each line "x y"
        uv_pts = []
        with open(luv) as f:
            for line in f:
                s = line.strip().split()
                if len(s) >= 2:
                    uv_pts.append((float(s[0]), float(s[1])))
        if len(uv_pts) >= 3:
            boundaries_uv[pid] = uv_pts
        else:
            boundaries_uv[pid] = []
        # 3D loop: each line "x y z"
        pts3d = []
        if l3d.exists():
            with open(l3d) as f:
                for line in f:
                    s = line.strip().split()
                    if len(s) >= 3:
                        pts3d.append((float(s[0]), float(s[1]), float(s[2])))
        boundaries_3d[pid] = pts3d if len(pts3d) >= 3 else uv_pts
        max_pid = max(max_pid, pid + 1)

    if not boundaries_uv:
        print("No part_N_loop_uv.txt files found")
        return 1
    n_parts = max(n_parts, max_pid)
    pids = sorted(boundaries_uv.keys())
    print(f"Loaded {len(pids)} partition loops (pids={pids[:5]}{'...' if len(pids)>5 else ''}), "
          f"{sum(1 for b in boundaries_3d.values() if len(b) >= 3)} 3D loops, "
          f"{n_parts} partitions")

    # Raw (pre-smoothing) boundary polylines — exported by the C++ pipeline
    # before its Laplacian smoothing, for side-by-side comparison in the view.
    raw_bnds_3d = []
    rfile = data_dir / "boundaries_iter_000.txt"
    if rfile.exists():
        raw_bnds_3d = load_boundaries_3d(str(rfile))
        print(f"Raw boundary polylines (pre-smoothing): {len(raw_bnds_3d)}")

    # ── Step 4: Concave detection ──
    concave_results = []
    all_split_lines = []

    colors = np.zeros((n_parts, 3))
    for i in range(n_parts):
        colors[i] = TAB10[i % 10]

    for pid in pids:
        poly = boundaries_uv[pid]
        if not poly or len(poly) < 3:
            concave_results.append({'pid': pid, 'deficit': 0, 'concave': False,
                                    'split_type': 0, 'split_line': None, 'poly': poly})
            continue
        is_concave, deficit, split_info = detect_concave(poly, args.threshold)
        split_type, split_line = split_info if split_info else (0, None)

        concave_results.append({
            'pid': pid, 'deficit': deficit,
            'concave': is_concave, 'split_type': split_type,
            'split_line': split_line, 'poly': poly,
        })

        status = f"CONCAVE (deficit={deficit:.3f}, {'tip' if split_type==1 else 'corner'}, split={'OK' if split_line else 'FAILED'})" if is_concave else f"convex (deficit={deficit:.5f})"
        print(f"  part {pid}: {status}")
        if split_line:
            all_split_lines.append((pid, split_line))

    deficits = [r['deficit'] for r in concave_results]
    n_concave = sum(1 for r in concave_results if r['concave'])
    print(f"\nConcavity summary (threshold={args.threshold}): "
          f"{n_concave}/{len(concave_results)} concave, "
          f"deficit range=[{min(deficits):.4f}, {max(deficits):.4f}], "
          f"mean={np.mean(deficits):.4f}")
    print(f"(deficit = 1 - polygon_area/convex_hull_area)")
    print(f"(lower deficit → closer to convex; threshold={args.threshold}: deficit > this → concave)")

    # ── Step 5: Interactive Visualization ──
    pv.set_plot_theme("document")
    pl = pv.Plotter()
    pl.add_title("Split Test — slider to switch views", font_size=11)
    view_actors = {i: [] for i in range(5)}

    # shared: plain surface (used as bg in views 0,2,3,4)
    if verts3d is not None and faces is not None:
        mesh_bg = pv.PolyData(
            verts3d,
            np.hstack([np.full((len(faces), 1), 3), faces]).astype(int))
        pl.add_mesh(mesh_bg, color='lightgray', show_edges=False,
                    opacity=0.35, name='bg_surf')
        view_actors[0].append('bg_surf')
        view_actors[2].append('bg_surf')
        view_actors[3].append('bg_surf')
        view_actors[4].append('bg_surf')

    # shared: partition-colored surface (used in view 1)
    face_cols = np.ones((len(faces), 3)) * 0.7
    if face_labels is not None:
        for fi in range(len(faces)):
            lbl = face_labels[fi]
            if 0 <= lbl < n_parts:
                face_cols[fi] = colors[lbl]
    if verts3d is not None and faces is not None:
        mesh_part = pv.PolyData(
            verts3d,
            np.hstack([np.full((len(faces), 1), 3), faces]).astype(int))
        pl.add_mesh(mesh_part, scalars=face_cols, rgb=True,
                    show_edges=False, opacity=0.88, name='part_surf')
        view_actors[1].append('part_surf')
    # raw (pre-smoothing) boundary — thin gray for comparison
    for cid, raw_pts in enumerate(raw_bnds_3d):
        rp = np.array(raw_pts)
        if len(rp) < 2:
            continue
        rbp = pv.PolyData(rp)
        nr = len(rp)
        rbp.lines = np.array([nr] + list(range(nr)), dtype=np.int64)
        rname = f'v2_raw_{cid}'
        pl.add_mesh(rbp, color='gray', line_width=1.0,
                    render_lines_as_tubes=True, opacity=0.5, name=rname)
        view_actors[2].append(rname)

    for pid in pids:
        if pid not in boundaries_3d:
            continue
        bnd3d = boundaries_3d[pid]
        pts = np.array(bnd3d)
        cr = concave_results[pid] if pid < len(concave_results) else None
        is_concave = bool(cr and cr['concave'])
        color = 'red' if is_concave else TAB10[pid % 10]
        lw = 3 if is_concave else 1.5
        bname = f'v2_bnd3d_{pid}'
        bp = pv.PolyData(pts)
        nv = len(pts)
        bp.lines = np.array([nv] + list(range(nv)), dtype=np.int64)
        pl.add_mesh(bp, color=color, line_width=lw,
                    render_lines_as_tubes=True, name=bname)
        view_actors[2].append(bname)
        # Ruling direction line in 3D (from arc midpoint to chord midpoint)
        if is_concave and pid in boundaries_uv:
            poly_uv = boundaries_uv[pid]
            diams = compute_poly_diam(poly_uv) + 1e-12
            best_pkt = None; best_r = 0
            for pkt in detect_pockets(poly_uv):
                r = pkt['depth'] / diams
                if r > best_r:
                    best_r = r; best_pkt = pkt
            if best_pkt and best_pkt['arc']:
                ia = best_pkt['start']
                ib = best_pkt['end']
                im = best_pkt['arc'][len(best_pkt['arc']) // 2]
                if ia < len(pts) and ib < len(pts) and im < len(pts):
                    va3 = pts[ia]
                    vb3 = pts[ib]
                    vm3 = pts[im]
                    vc3 = (va3 + vb3) * 0.5
                    rd = pv.PolyData(np.array([vm3, vc3]))
                    rd.lines = np.array([2, 0, 1], dtype=np.int64)
                    rname = f'v2_rule_{pid}'
                    pl.add_mesh(rd, color='cyan', line_width=2.5, name=rname)
                    view_actors[2].append(rname)

    # --- View 3: Surface bg + boundary + cutting lines (3D) ---
    for pid in pids:
        if pid not in boundaries_3d:
            continue
        bnd3d = boundaries_3d[pid]
        if not bnd3d or len(bnd3d) < 3:
            continue
        pts = np.array(bnd3d)
        bname = f'v3_bnd_{pid}'
        bp = pv.PolyData(pts)
        nv = len(pts)
        bp.lines = np.array([nv] + list(range(nv)), dtype=np.int64)
        pl.add_mesh(bp, color=TAB10[pid % 10], line_width=1.5,
                    render_lines_as_tubes=True, name=bname)
        view_actors[3].append(bname)
    # 3D split lines: map UV split endpoints → nearest 3D boundary vertex
    for idx, (pid, (p0, p1)) in enumerate(all_split_lines):
        if pid not in boundaries_uv or pid not in boundaries_3d:
            continue
        poly_uv = boundaries_uv[pid]
        pts3d = np.array(boundaries_3d[pid])
        if len(poly_uv) != len(pts3d):
            continue
        # find nearest UV vertex to each split endpoint
        uv_arr = np.array(poly_uv)
        d0 = np.linalg.norm(uv_arr - np.array(p0), axis=1)
        d1 = np.linalg.norm(uv_arr - np.array(p1), axis=1)
        i0, i1 = np.argmin(d0), np.argmin(d1)
        seg = np.array([pts3d[i0], pts3d[i1]])
        s = pv.PolyData(seg)
        s.lines = np.array([2, 0, 1], dtype=int)
        sname = f'v3_cut_{idx}'
        pl.add_mesh(s, color='black', line_width=4, name=sname)
        view_actors[3].append(sname)

    # --- View 4: Warm-start (locally split 3D boundaries) ---
    # build new boundary set after applying splits
    from collections import defaultdict
    split_groups = defaultdict(list)  # pid → [split_line, ...]
    for pid, sl in all_split_lines:
        split_groups[pid].append(sl)
    # Also try to load retry_1 for real warm-start data
    warm_bnds = []
    if len(retry_dirs) >= 2:
        wfile = retry_dirs[1] / "boundaries.txt"
        if wfile.exists():
            warm_bnds = load_boundaries_3d(str(wfile))
    if not warm_bnds and boundaries_3d:
        # apply locally computed splits
        for pid in pids:
            if pid in split_groups:
                continue
            warm_bnds.append(boundaries_3d[pid])
    for cid, bnd in enumerate(warm_bnds):
        pts = np.array(bnd)
        wn = f'v4_ws_{cid}'
        bp = pv.PolyData(pts)
        nv = len(pts)
        bp.lines = np.array([nv] + list(range(nv)), dtype=np.int64)
        pl.add_mesh(bp, color=TAB10[cid % 10], line_width=2,
                    render_lines_as_tubes=True, name=wn)
        view_actors[4].append(wn)
    if not warm_bnds:
        ta = pl.add_text("No warm-start data", position='upper_right',
                         font_size=14, color='gray', name='v4_nodata')
        view_actors[4].append('v4_nodata')

    # ── Slider ──
    def set_view_visibility(v):
        sel = int(v)
        select_names = set(view_actors.get(sel, []))
        try:
            for a in pl.renderer._actors:
                if hasattr(a, '_name'):
                    a.SetVisibility(a._name in select_names)
        except Exception:
            pass
        pl.render()

    pl.add_slider_widget(
        set_view_visibility, rng=[0, 4], value=1, title="View",
        pointa=(0.02, 0.93), pointb=(0.28, 0.93),
        style='modern')
    pl.add_text("0:原曲面  1:初始分区  2:凹性(灰=原始/彩=平滑)  3:切割线  4:再分区",
                position='lower_edge', font_size=10, color='black', name='v_legend')
    set_view_visibility(1)
    pl.show()


if __name__ == '__main__':
    main()
