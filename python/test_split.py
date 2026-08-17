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
# Polygon area / convex hull helpers
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


# ═══════════════════════════════════════════════════════════════
# Vertex-level concavity (cross product §2.1.1) + warm-start split
# ═══════════════════════════════════════════════════════════════

def orient_loops_ccw(poly_uv, poly_3d=None):
    """Ensure the UV loop is CCW (signed area > 0), flipping the 3D loop in sync.
    Returns (uv_ccw, p3d_ccw_or_None)."""
    poly_uv = np.asarray(poly_uv, dtype=np.float64)
    n = len(poly_uv)
    if n < 3:
        p3 = np.asarray(poly_3d, dtype=np.float64) if poly_3d is not None else None
        return poly_uv, p3
    area2 = sum(poly_uv[i][0] * poly_uv[(i + 1) % n][1]
                - poly_uv[i][1] * poly_uv[(i + 1) % n][0] for i in range(n))
    if area2 >= 0:
        p3 = np.asarray(poly_3d, dtype=np.float64) if poly_3d is not None else None
        return poly_uv, p3
    p3 = None
    if poly_3d is not None:
        arr = np.asarray(poly_3d, dtype=np.float64)
        if len(arr) == n:
            p3 = arr[::-1]
    return poly_uv[::-1], p3


def classify_vertices(poly):
    """§2.1.1 cross-product vertex convexity test (CCW closed polygon).
    Returns list: +1 convex, -1 concave, 0 collinear."""
    poly = np.asarray(poly, dtype=np.float64)
    n = len(poly)
    cls = [0] * n
    if n < 3:
        return cls
    mean_len_sq = sum(np.linalg.norm(poly[(i + 1) % n] - poly[i]) ** 2 for i in range(n)) / n
    eps = mean_len_sq * 1e-12
    for i in range(n):
        p0, p1, p2 = poly[(i - 1) % n], poly[i], poly[(i + 1) % n]
        cr = (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0])
        if cr > eps:
            cls[i] = 1
        elif cr < -eps:
            cls[i] = -1
        else:
            cls[i] = 0
    return cls


def concave_runs(cls):
    """Contiguous runs of concave (-1) vertices, circular-aware.
    Returns [(start, end), ...] (closed interval indices)."""
    n = len(cls)
    if not any(c < 0 for c in cls):
        return []
    runs = []
    start = None
    for i in range(n):
        if cls[i] < 0:
            if start is None:
                start = i
        elif start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, n - 1))
    # merge the run that wraps around the loop start/end
    if len(runs) >= 2 and runs[0][0] == 0 and runs[-1][1] == n - 1:
        runs = [(runs[-1][0], runs[0][1])] + runs[1:-1]
    return runs


def analyze_partition(poly_ccw, threshold=0.30, min_run_len=3):
    """Concavity analysis on a CCW closed polygon (smoothed boundary).
    Returns dict: cls / runs / deficit / split_lines.
    min_run_len filters out micro-concavities (isolated concave vertices left
    by Laplacian smoothing residue) so only macro concave segments remain."""
    cls = classify_vertices(poly_ccw)
    n = len(cls)
    runs = []
    for s, e in concave_runs(cls):
        ln = (e - s + 1) if e >= s else (n - s + e + 1)
        if ln >= min_run_len:
            runs.append((s, e))

    pa = polygon_area(poly_ccw)
    ha = convex_hull_area(poly_ccw)
    deficit = (1.0 - pa / ha) if ha > 1e-12 else 0.0

    split_lines = []
    if runs:
        pockets = detect_pockets(poly_ccw)
        for pkt in pockets:
            if pkt['width'] < 1e-8:
                continue
            # normalized concavity depth/width as significance gate
            if pkt['depth'] / pkt['width'] < 0.3:
                continue
            st, sl = classify_pocket(poly_ccw, pkt)
            if sl:
                split_lines.append((st, sl[0], sl[1]))

    return {'cls': cls, 'runs': runs, 'deficit': deficit, 'split_lines': split_lines}


def split_loop_by_line(poly_uv, poly_3d, p0, p1):
    """Split a closed loop into two 3D sub-loops along the chord p0-p1 (warm-start).
    poly_uv/poly_3d are 1:1 (same vertex order); p0/p1 are in UV space.
    Returns [subA_3d, subB_3d] (implicitly closed), or None on failure."""
    uv = np.asarray(poly_uv, dtype=np.float64)
    p3d = np.asarray(poly_3d, dtype=np.float64)
    n = len(uv)
    if n < 4 or len(p3d) != n:
        return None
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    i0 = int(np.argmin(np.linalg.norm(uv - p0, axis=1)))
    i1 = int(np.argmin(np.linalg.norm(uv - p1, axis=1)))
    if i0 == i1:
        return None
    if i0 > i1:
        i0, i1 = i1, i0
    subA = p3d[i0:i1 + 1]
    subB = np.vstack([p3d[i1:], p3d[:i0 + 1]])
    if len(subA) < 4 or len(subB) < 4:
        return None
    return [subA, subB]


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

    # ── Step 4: Concave detection (vertex-level, on smoothed boundaries) ──
    concave_results = {}   # pid → analysis dict
    all_split_lines = []   # [(pid, (p0, p1)), ...]

    colors = np.zeros((n_parts, 3))
    for i in range(n_parts):
        colors[i] = TAB10[i % 10]

    for pid in pids:
        poly = boundaries_uv[pid]
        if not poly or len(poly) < 3:
            concave_results[pid] = {'cls': [], 'runs': [], 'deficit': 0, 'split_lines': []}
            continue

        # Orient CCW (UV + 3D in sync) so the cross-product sign is meaningful
        uv_ccw, b3d_ccw = orient_loops_ccw(poly, boundaries_3d.get(pid))
        boundaries_uv[pid] = uv_ccw
        if b3d_ccw is not None:
            boundaries_3d[pid] = b3d_ccw

        res = analyze_partition(uv_ccw, args.threshold)
        res['poly'] = uv_ccw
        concave_results[pid] = res

        n_runs = len(res['runs'])
        n_splits = len(res['split_lines'])
        print(f"  part {pid}: deficit={res['deficit']:.4f}, "
              f"{n_runs} concave segment(s), {n_splits} split line(s)")
        for st, p0, p1 in res['split_lines']:
            all_split_lines.append((pid, (p0, p1)))

    n_concave = sum(1 for r in concave_results.values() if r['runs'])
    deficits = [r['deficit'] for r in concave_results.values()]
    print(f"\nConcavity summary: {n_concave}/{len(concave_results)} partitions with concave segments, "
          f"{len(all_split_lines)} split line(s) generated")
    if deficits:
        print(f"deficit range=[{min(deficits):.4f}, {max(deficits):.4f}], "
              f"mean={np.mean(deficits):.4f}")

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
        cr = concave_results.get(pid)
        runs = cr['runs'] if cr else []
        nv = len(pts)

        # smoothed boundary loop in partition color (thin)
        bp = pv.PolyData(pts)
        bp.lines = np.array([nv] + list(range(nv)), dtype=np.int64)
        bname = f'v2_bnd3d_{pid}'
        pl.add_mesh(bp, color=TAB10[pid % 10], line_width=1.5,
                    render_lines_as_tubes=True, name=bname)
        view_actors[2].append(bname)

        # mark concave segments (vertex clusters judged concave) in red
        for ri, (s, e) in enumerate(runs):
            if s <= e:
                seg_idx = list(range(s, e + 1))
            else:
                seg_idx = list(range(s, nv)) + list(range(0, e + 1))
            seg_pts = pts[seg_idx]
            ns = len(seg_pts)
            if ns < 2:
                continue
            seg_bp = pv.PolyData(seg_pts)
            seg_bp.lines = np.array([ns] + list(range(ns)), dtype=np.int64)
            rname = f'v2_concave_{pid}_{ri}'
            pl.add_mesh(seg_bp, color='red', line_width=4,
                        render_lines_as_tubes=True, name=rname)
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

    # --- View 4: Warm-start (apply concave splits → re-partition) ---
    split_map = {}  # pid → list of (p0, p1) split lines
    for pid, sl in all_split_lines:
        split_map.setdefault(pid, []).append(sl)

    warm_bnds = []
    for pid in pids:
        uv_poly = boundaries_uv.get(pid)
        b3d = boundaries_3d.get(pid)
        if b3d is None:
            continue
        # split concave partitions along the first generated split line
        if pid in split_map and uv_poly is not None:
            subs = split_loop_by_line(uv_poly, b3d, *split_map[pid][0])
            if subs:
                warm_bnds.extend(subs)
                continue
        warm_bnds.append(np.array(b3d))

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
    pl.add_text("0:原曲面  1:初始分区  2:凹线段(灰=原始/彩=平滑/红=凹)  3:切割线  4:热启动再分区",
                position='lower_edge', font_size=10, color='black', name='v_legend')
    set_view_visibility(1)
    pl.show()


if __name__ == '__main__':
    main()
