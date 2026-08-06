"""
OCCT partition + boundary smoothing visualizer (PyVista)
Usage: python visualize.py <data_directory> [--iter N]
"""
import sys, os, glob
import numpy as np
import pyvista as pv

TAB10_RGB = np.array([
    [0.122, 0.467, 0.706], [1.000, 0.498, 0.055],
    [0.173, 0.627, 0.173], [0.839, 0.153, 0.157],
    [0.580, 0.404, 0.741], [0.549, 0.337, 0.294],
    [0.890, 0.467, 0.761], [0.498, 0.498, 0.498],
    [0.738, 0.738, 0.131], [0.090, 0.745, 0.812],
])

def load_obj(path):
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            if parts[0] == 'v':
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f':
                faces.append([int(parts[1]) - 1, int(parts[2]) - 1, int(parts[3]) - 1])
    return np.array(verts), np.array(faces)

def load_labels(path):
    return np.loadtxt(path, dtype=int)

def load_boundaries(path):
    boundaries = []
    if not os.path.exists(path): return boundaries
    with open(path) as f:
        line = f.readline()
        while line:
            parts = line.strip().split()
            if len(parts) < 2:
                line = f.readline(); continue
            n_pts = int(parts[0])
            cid = int(parts[1])
            pts = []
            for _ in range(n_pts):
                line = f.readline()
                if not line: break
                pts.append([float(x) for x in line.strip().split()])
            boundaries.append((cid, np.array(pts)))
            line = f.readline()
    return boundaries

def load_corners(path):
    pts = []
    if not os.path.exists(path): return np.empty((0, 3))
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.array(pts) if pts else np.empty((0, 3))

def load_smooth_history(path):
    history = []
    if not os.path.exists(path): return history
    with open(path) as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                history.append({
                    'iter': int(parts[0]),
                    'max_disp': float(parts[1]),
                    'avg_disp': float(parts[2]),
                    'n_verts': int(parts[3]),
                })
    return history

def load_run_meta(path):
    meta = {}
    if not os.path.exists(path): return meta
    with open(path) as f:
        for line in f:
            parts = line.strip().split('=')
            if len(parts) == 2:
                meta[parts[0]] = parts[1]
    return meta

def find_iter_boundary_files(data_dir):
    pattern = os.path.join(data_dir, "boundaries_iter_*.txt")
    files = sorted(glob.glob(pattern))
    iters = []
    for f in files:
        base = os.path.basename(f)
        try:
            iters.append(int(base.replace("boundaries_iter_", "").replace(".txt", "")))
        except ValueError:
            pass
    return iters, files

def load_tolerance(path):
    tolerances = []
    if not os.path.exists(path): return tolerances
    with open(path) as f:
        f.readline()
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                tolerances.append({
                    'pair_id': int(parts[0]),
                    'maxDist': float(parts[1]),
                    'rmsDist': float(parts[2]),
                    'bestAngleDeg': float(parts[3]),
                })
    return tolerances

def load_ruled_surfaces(data_dir):
    surfs = []
    idx = 0
    while True:
        path = os.path.join(data_dir, f"ruled_surf_{idx}.obj")
        if not os.path.exists(path):
            break
        try:
            mesh = pv.read(path)
            surfs.append(mesh)
        except Exception:
            pass
        idx += 1
    return surfs

def main():
    data_dir = "."
    show_iter = None
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--iter" and i + 1 < len(args):
            show_iter = int(args[i + 1]); i += 2
        elif not args[i].startswith("--"):
            data_dir = args[i]; i += 1
        else:
            i += 1

    mesh_path = os.path.join(data_dir, "mesh.obj")
    labels_path = os.path.join(data_dir, "partition_labels.txt")

    for p in [mesh_path, labels_path]:
        if not os.path.exists(p):
            print(f"  [ERR] {p} not found -- run distillation.exe first")
            return 1

    verts, faces = load_obj(mesh_path)
    labels = load_labels(labels_path)
    pyvista_faces = np.column_stack([np.full(len(faces), 3), faces]).ravel()

    history = load_smooth_history(os.path.join(data_dir, "smooth_history.txt"))
    meta = load_run_meta(os.path.join(data_dir, "run_meta.txt"))

    if show_iter is not None:
        if show_iter < 0:
            print(f"  [ERR] --iter must be >= 0, got {show_iter}")
            return 1
        target_file = os.path.join(data_dir, f"boundaries_iter_{show_iter:03d}.txt")
        if not os.path.exists(target_file):
            max_iter = int(meta.get("actual_iters", "?"))
            avail = find_iter_boundary_files(data_dir)[0]
            print(f"  [ERR] iteration {show_iter} not found. Valid range: 0..{max_iter}")
            if avail:
                print(f"  Available files: {avail}")
            return 1
        boundaries = load_boundaries(target_file)
        title = f"Partitions + Boundary iter {show_iter} (K={meta.get('K','?')})"
    else:
        boundaries_path = os.path.join(data_dir, "boundaries.txt")
        boundaries = load_boundaries(boundaries_path)
        title = f"Partitions + NURBS curves (K={meta.get('K','?')})"

    corners = load_corners(os.path.join(data_dir, "corners.txt"))
    tolerances = load_tolerance(os.path.join(data_dir, "tolerance.txt"))
    ruled_surfs = load_ruled_surfaces(data_dir) if show_iter is None else []

    has_ruled = len(ruled_surfs) > 0
    n_cols = 2 if has_ruled else 1
    pl = pv.Plotter(shape=(1, n_cols), title=title)
    pl.background_color = 'lightblue'

    # ── Left pane: mesh + boundary curves + corners ──
    pl.subplot(0, 0)
    mesh = pv.PolyData(verts, pyvista_faces)
    face_colors = np.zeros((len(faces), 3))
    for idx, t in enumerate(faces):
        pids = set([labels[t[0]], labels[t[1]], labels[t[2]]])
        pids.discard(-1)
        if len(pids) == 1:
            face_colors[idx] = TAB10_RGB[pids.pop() % 10]
        elif len(pids) > 1:
            avg = np.sum([TAB10_RGB[p % 10] for p in pids], axis=0) / len(pids)
            face_colors[idx] = avg
        else:
            face_colors[idx] = [0.7, 0.7, 0.7]
    mesh.cell_data["rgb"] = face_colors
    pl.add_mesh(mesh, scalars="rgb", rgb=True,
                show_edges=True, edge_color="gray", opacity=1.0)

    if not boundaries:
        print("  [WARN] No boundary data to display")
    else:
        for cid, pts in boundaries:
            if len(pts) < 2: continue
            color = TAB10_RGB[cid % 10]
            curve = pv.PolyData()
            curve.points = pts
            n = len(pts)
            curve.lines = np.array([n] + list(range(n)), dtype=np.int64)
            pl.add_mesh(curve, color=color, line_width=6, opacity=0.98,
                        render_lines_as_tubes=True)

    if len(corners) > 0:
        corner_cloud = pv.PolyData(np.array(corners))
        pl.add_mesh(corner_cloud, color="yellow", point_size=12,
                    render_points_as_spheres=True, opacity=1.0)

    # left info
    left_info = []
    if meta:
        left_info.append(f"surface={meta.get('surface','?')}  "
                        f"sigma={meta.get('sigma','?')}  K={meta.get('K','?')}")
    if show_iter is not None:
        left_info.append(f"iter {show_iter}  (raw polylines)")
    else:
        left_info.append(f"NURBS curves: {len(boundaries)}")
    pl.add_text("\n".join(left_info), position="upper_left", font_size=9, color="black")

    # ── Right pane: ruled surfaces ──
    if has_ruled:
        pl.subplot(0, 1)
        for idx, surf in enumerate(ruled_surfs):
            color = TAB10_RGB[idx % 10]
            pl.add_mesh(surf, color=color, opacity=0.4,
                        show_edges=True, edge_color="gray", smooth_shading=True,
                        label=f"pair {idx}")
        right_info = [f"{len(ruled_surfs)} ruled surfaces"]
        if tolerances:
            worst = max(tolerances, key=lambda x: x['maxDist'])
            right_info.append(f"max err={worst['maxDist']:.4f}")
            right_info.append(f"angle={worst['bestAngleDeg']:.0f}deg")
            for t in tolerances:
                right_info.append(f"  pair {t['pair_id']}: {t['maxDist']:.4f} @ {t['bestAngleDeg']:.0f}deg")
        pl.add_text("\n".join(right_info), position="upper_left", font_size=9, color="black")

    # global history
    if history:
        hist_lines = ["Smoothing history:"]
        for h in history:
            hist_lines.append(f"  iter {h['iter']:3d}: max={h['max_disp']:.4f}  avg={h['avg_disp']:.4f}")
        pl.add_text("\n".join(hist_lines), position="lower_left", font_size=8, color="black")

    pl.show()
    return 0

if __name__ == "__main__":
    sys.exit(main())
