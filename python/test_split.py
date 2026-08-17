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

import concavity_analysis as ca

SCRIPT_DIR = Path(__file__).resolve().parent
EXE = SCRIPT_DIR.parent / "build" / "Release" / "distillation.exe"

TAB10 = np.array([
    [0.122,0.467,0.706],[1.000,0.498,0.055],[0.173,0.627,0.173],[0.839,0.153,0.157],
    [0.580,0.404,0.741],[0.549,0.337,0.294],[0.890,0.467,0.761],[0.498,0.498,0.498],
    [0.738,0.738,0.131],[0.090,0.745,0.812],
])


# 2D 凸包 / 口袋 / 分割线 / 顶点凹凸等凹性分析算法
# 已迁移到 concavity_analysis 模块（多尺度凹区域分析）。


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
# UV/3D 同步定向 + 热启动切分（3D 相关，保留在 test_split 侧；
# 纯 2D 凹性分析算法在 concavity_analysis 模块）
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


def map_subpoly_to_3d(sub_poly, uv_poly, pts3d):
    """把 ACD 分解出的 2D 子多边形顶点映射到 3D（用最近 UV 顶点匹配）。"""
    uv = np.asarray(uv_poly, dtype=np.float64)
    out = []
    for p in sub_poly:
        j = int(np.argmin(np.linalg.norm(uv - np.asarray(p, dtype=np.float64), axis=1)))
        out.append(pts3d[j])
    return np.array(out)


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

    # ── Step 4: Multi-scale concavity analysis (concavity_analysis module) ──
    concave_results = {}   # pid → analysis dict (from ca.analyze_boundary)
    all_split_lines = []   # [(pid, (p0, p1)), ...]

    colors = np.zeros((n_parts, 3))
    for i in range(n_parts):
        colors[i] = TAB10[i % 10]

    for pid in pids:
        poly = boundaries_uv[pid]
        if not poly or len(poly) < 3:
            concave_results[pid] = {'runs': [], 'segments': [], 'deficit': 0,
                                    'split_lines': [], 'sub_polygons': []}
            continue

        # Orient CCW (UV + 3D in sync) so the cross-product sign is meaningful
        uv_ccw, b3d_ccw = orient_loops_ccw(poly, boundaries_3d.get(pid))
        boundaries_uv[pid] = uv_ccw
        if b3d_ccw is not None:
            boundaries_3d[pid] = b3d_ccw

        res = ca.analyze_boundary(uv_ccw, tau=args.threshold)
        concave_results[pid] = res

        n_seg = len(res['segments'])
        n_splits = len(res['split_lines'])
        n_sub = len(res['sub_polygons'])
        ph = res.get('ph')
        if ph and ph['intervals'] is not None:
            ph_str = f"PH cycles={len(ph['intervals'])}"
            if ph['intervals']:
                ph_str += f" (top persist={ph['intervals'][0][2]:.4f})"
        elif ph and ph['note']:
            ph_str = f"PH: {ph['note']}"
        else:
            ph_str = "PH: n/a"
        print(f"  part {pid}: deficit={res['deficit']:.4f}, "
              f"{n_seg} concave segment(s), {n_splits} split line(s), "
              f"{n_sub} ACD sub-polygon(s), {ph_str}")
        for st, p0, p1 in res['split_lines']:
            all_split_lines.append((pid, (p0, p1)))

    n_concave = sum(1 for r in concave_results.values() if r['segments'])
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
        segments = cr['segments'] if cr else []
        nv = len(pts)

        # smoothed boundary loop in partition color (thin)
        bp = pv.PolyData(pts)
        bp.lines = np.array([nv] + list(range(nv)), dtype=np.int64)
        bname = f'v2_bnd3d_{pid}'
        pl.add_mesh(bp, color=TAB10[pid % 10], line_width=1.5,
                    render_lines_as_tubes=True, name=bname)
        view_actors[2].append(bname)

        # mark macro concave segments (multi-scale filtered) in red
        for ri, seg in enumerate(segments):
            seg_idx = seg['indices']
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

    # --- View 4: Warm-start (ACD recursive decomposition → re-partition) ---
    warm_bnds = []
    for pid in pids:
        uv_poly = boundaries_uv.get(pid)
        b3d = boundaries_3d.get(pid)
        if b3d is None:
            continue
        res = concave_results.get(pid)
        subs = res.get('sub_polygons') if res else None
        if subs and len(subs) > 1:
            pts3d = np.array(b3d)
            for sp in subs:
                warm_bnds.append(map_subpoly_to_3d(sp, uv_poly, pts3d))
        else:
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
