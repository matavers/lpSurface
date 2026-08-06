"""
Partition quality analysis: point cloud vs surface comparison
Usage: python analyze.py <data_directory>
"""
import sys, os
import numpy as np
import pyvista as pv

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

def detect_single_vertex_strips(verts, faces, labels):
    """Find vertices that are isolated single-point outliers in their partition."""
    n = len(labels)
    adj = [set() for _ in range(n)]
    for tri in faces:
        for i in range(3):
            a, b = tri[i], tri[(i+1)%3]
            adj[a].add(b); adj[b].add(a)
    # count neighbors with same/different label
    strips = []
    for vi in range(n):
        same = sum(1 for nb in adj[vi] if labels[nb] == labels[vi])
        total = len(adj[vi])
        if total > 0 and same / total < 0.4:
            strips.append(vi)
    return np.array(strips)

def main():
    if len(sys.argv) < 2:
        data_dir = "."
    else:
        data_dir = sys.argv[1]

    mesh_path = os.path.join(data_dir, "mesh.obj")
    labels_path = os.path.join(data_dir, "partition_labels.txt")

    for p in [mesh_path, labels_path]:
        if not os.path.exists(p):
            print(f"[ERR] {p} not found")
            return 1

    verts, faces = load_obj(mesh_path)
    labels = load_labels(labels_path)
    n = len(verts)
    n_uniq = len(np.unique(labels[labels >= 0]))

    # compute adjacency and strip detection
    adj = [set() for _ in range(n)]
    for tri in faces:
        for i in range(3):
            a, b = tri[i], tri[(i+1)%3]
            adj[a].add(b); adj[b].add(a)

    # count border vertices and edges
    border_verts = set()
    border_edges = 0
    for tri in faces:
        for i in range(3):
            a, b = tri[i], tri[(i+1)%3]
            if labels[a] != labels[b]:
                border_verts.add(a); border_verts.add(b)
                border_edges += 1

    strips = detect_single_vertex_strips(verts, faces, labels)
    strip_pids = {}
    for vi in strips:
        pid = labels[vi]
        strip_pids.setdefault(pid, []).append(vi)

    print(f"=== Partition Analysis ===")
    print(f"  vertices: {n}, faces: {len(faces)}, partitions: {n_uniq}")
    print(f"  border vertices: {len(border_verts)} ({100*len(border_verts)/n:.1f}%)")
    print(f"  border edges: {border_edges}")
    print(f"  single-vertex outlier strips: {len(strips)} ({100*len(strips)/n:.1f}%)")
    for pid, vlist in sorted(strip_pids.items()):
        print(f"    pid={pid}: {len(vlist)} outliers")

    for pid in sorted(np.unique(labels[labels >= 0])):
        mask = labels == pid
        count = mask.sum()
        bcount = sum(1 for vi in range(n) if mask[vi] and any(labels[nb] != pid for nb in adj[vi]))
        pct = 100 * bcount / count if count > 0 else 0
        print(f"  pid={pid:2d}: size={count:4d} border={bcount:4d} ({pct:.0f}%)")

    pyvista_faces = np.column_stack([np.full(len(faces), 3), faces]).ravel()

    # tab10 RGB (0-1)
    TAB10_RGB = np.array([
        [0.122, 0.467, 0.706], [1.000, 0.498, 0.055],
        [0.173, 0.627, 0.173], [0.839, 0.153, 0.157],
        [0.580, 0.404, 0.741], [0.549, 0.337, 0.294],
        [0.890, 0.467, 0.761], [0.498, 0.498, 0.498],
        [0.738, 0.738, 0.131], [0.090, 0.745, 0.812],
    ])

    pl = pv.Plotter(shape=(1, 2), title="Partition Analysis")

    # Left: point cloud
    pl.subplot(0, 0)
    cloud = pv.PolyData(verts)
    cloud.point_data["partition"] = labels
    pl.add_mesh(cloud, style="points", point_size=15,
                scalars="partition", cmap="tab10",
                show_scalar_bar=False)
    pl.add_text("Point Cloud (per vertex)", position="upper_edge", font_size=10)

    # Right: per-face with RGB averaging for mixed triangles
    pl.subplot(0, 1)
    mesh_surf = pv.PolyData(verts, pyvista_faces)
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
    mesh_surf.cell_data["rgb"] = face_colors
    pl.add_mesh(mesh_surf, scalars="rgb", rgb=True,
                show_edges=True, edge_color="gray")
    pl.add_text("Surface (per-face, boundary avg)", position="upper_edge", font_size=10)

    # white boundary lines on both subplots
    edge_set = set()
    for t in faces:
        for i in range(3):
            a, b = t[i], t[(i + 1) % 3]
            if labels[a] != labels[b]:
                edge_set.add(tuple(sorted((a, b))))
    if edge_set:
        lines_arr = np.empty(len(edge_set) * 3, dtype=np.int64)
        for i, (a, b) in enumerate(edge_set):
            lines_arr[i * 3] = 2
            lines_arr[i * 3 + 1] = a
            lines_arr[i * 3 + 2] = b
        bounds = pv.PolyData(verts, lines=lines_arr)

        pl.subplot(0, 0)
        pl.add_mesh(bounds.copy(), color="white", line_width=1, opacity=0.6)
        pl.subplot(0, 1)
        pl.add_mesh(bounds.copy(), color="white", line_width=1, opacity=0.6)

    # Link cameras
    pl.link_views()

    print(f"\n  Close the window to exit.")
    pl.show()
    return 0

if __name__ == "__main__":
    sys.exit(main())
