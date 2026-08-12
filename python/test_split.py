"""
简化分割测试：从原始网格到凹性检测+分割可视化
跳过优化器和容差迭代，快速验证凹性检测和分割效果

用法:
  python test_split.py [--surface wavy] [--K 16] [--threshold 0.03] [--smooth-iters 1]

步骤:
  1. C++ 管线运行一次 (无优化器) → 生成分区 + 平滑边界
  2. Python 实现凹性检测 → 标记分割区域 + 分割线
  3. PyVista 交互可视化: 边界/分割线/凹部标记
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
    parser.add_argument('--threshold', type=float, default=0.03)
    parser.add_argument('--smooth-iters', type=int, default=1)
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
    bfile = data_dir / "boundaries_uv.txt"
    mfile = data_dir / "mesh.obj"
    lfile = data_dir / "face_labels.txt"
    if not bfile.exists():
        print(f"Missing: {bfile}")
        return 1

    boundaries = load_boundaries_uv(str(bfile))
    print(f"Loaded {len(boundaries)} boundary polylines")

    verts3d, faces = load_obj(str(mfile)) if mfile.exists() else (None, None)
    if mfile.exists():
        print(f"Mesh: {len(verts3d)} verts, {len(faces)} faces")

    face_labels = load_face_labels(str(lfile)) if lfile.exists() else None
    n_parts = int(face_labels.max()) + 1 if face_labels is not None else len(boundaries)
    print(f"Partitions: {n_parts}")

    # ── Step 4: Concave detection ──
    concave_results = []
    all_split_lines = []

    colors = np.zeros((n_parts, 3))
    for i in range(n_parts):
        colors[i] = TAB10[i % 10]

    for pid in range(min(n_parts, len(boundaries))):
        poly = boundaries[pid]
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

    # ── Step 5: Visualize ──
    pv.set_plot_theme("document")
    pl = pv.Plotter(shape=(1, 2))
    pl.subplot(0, 0); pl.add_title("Partition Boundaries + Split Lines (UV space)")
    pl.subplot(0, 1); pl.add_title("3D Mesh + Split Lines")

    # --- Left: UV space ---
    pl.subplot(0, 0)
    for pid in range(min(n_parts, len(boundaries))):
        poly = boundaries[pid]
        if len(poly) < 3: continue
        # Convert 2D -> 3D for PyVista (z=0)
        pts_3d = np.array([[p[0], p[1], 0.0] for p in poly])
        cr = concave_results[pid] if pid < len(concave_results) else None
        status_color = 'red' if (cr and cr['concave']) else None
        color = status_color or TAB10[pid % 10]
        poly_obj = pv.PolyData(pts_3d)
        nv = len(pts_3d)
        poly_obj.lines = np.array([nv] + list(range(nv)) + [0], dtype=int)
        pl.add_mesh(poly_obj, color=color, line_width=2, name=f'uv_{pid}')

    for pid, (p0, p1) in all_split_lines:
        seg_pts = np.array([[p0[0], p0[1], 0.0], [p1[0], p1[1], 0.0]])
        seg_obj = pv.PolyData(seg_pts)
        seg_obj.lines = np.array([2, 0, 1], dtype=int)
        pl.add_mesh(seg_obj, color='black', line_width=3, name=f'split_uv_{pid}')

    # --- Right: 3D Mesh ---
    pl.subplot(0, 1)
    if verts3d is not None and faces is not None:
        # Color faces by partition
        face_colors_3d = np.ones((len(faces), 3)) * 0.7  # default gray
        if face_labels is not None:
            for fi in range(len(faces)):
                lbl = face_labels[fi]
                if 0 <= lbl < n_parts:
                    face_colors_3d[fi] = colors[lbl]
        mesh = pv.PolyData(verts3d, np.hstack([np.full((len(faces), 1), 3), faces]).astype(int))
        pl.add_mesh(mesh, scalars=face_colors_3d, rgb=True, show_edges=False,
                    opacity=0.8, name='mesh3d')
    else:
        # Show split lines as annotations
        for pid, (p0, p1) in all_split_lines:
            pl.add_points(np.array([p0, p1]), point_size=8, name=f'split_3d_{pid}')

    pl.link_views()
    pl.show()


if __name__ == '__main__':
    main()
