"""
Concave split test with visualization — standalone, no C++ dependency.

Usage: python test_concave_split.py

Flow:
  1. Generate wavy mesh surface + UV grid
  2. K-means partition (16 clusters on UV centroids)
  3. Extract partition boundaries
  4. Simple Laplacian smooth (5 iters)
  5. Concave detection (convex hull pockets, depth/diameter threshold)
  6. Split tip pockets, re-assign faces
  7. Visualize before/after

Keys in viewer:
  1  - original mesh
  2  - partition boundaries (before split)
  3  - macro-concave partitions highlighted
  4  - split lines
  5  - boundaries after split
  6  - new partitions (after-split coloring)
  q  - quit
"""

import sys, os, time, random
import numpy as np

# Try to use PyVista
try:
    import pyvista as pv
    HAS_PV = True
except ImportError:
    HAS_PV = False
    print("Warning: pyvista not available, visualization disabled")
    print("  Install: pip install pyvista")

# ═══════════════════════════════════════════════════════════════════
# 1. Mesh generation — wavy surface + UV grid
# ═══════════════════════════════════════════════════════════════════

def wavy_z(x, y):
    """Wavy surface height function."""
    return (0.15 * np.sin(2.5 * x) * np.cos(3.0 * y) +
            0.10 * np.sin(5.0 * x + 1.2) * np.sin(4.0 * y + 0.8) +
            0.08 * np.cos(7.0 * x) * np.sin(6.0 * y - 0.5) +
            0.05 * np.sin(9.0 * x - 1.0) * np.cos(8.0 * y + 1.5))


def generate_mesh(res=30, x_range=(-1.5, 1.5), y_range=(-1.5, 1.5)):
    """Generate triangulated mesh from wavy surface on a regular UV grid."""
    xs = np.linspace(x_range[0], x_range[1], res)
    ys = np.linspace(y_range[0], y_range[1], res)

    verts = []
    uvs = []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            z = wavy_z(x, y)
            verts.append([x, y, z])
            uvs.append([(x - x_range[0]) / (x_range[1] - x_range[0]),
                        (y - y_range[0]) / (y_range[1] - y_range[0])])
    verts = np.array(verts, dtype=np.float64)
    uvs = np.array(uvs, dtype=np.float64)

    faces = []
    for i in range(res - 1):
        for j in range(res - 1):
            a = i * res + j
            b = a + 1
            c = (i + 1) * res + j
            d = c + 1
            faces.append([a, b, d])
            faces.append([a, d, c])
    faces = np.array(faces, dtype=np.int32)

    return verts, faces, uvs, xs, ys


# ═══════════════════════════════════════════════════════════════════
# 2. Simple K-means partition (on UV centroids)
# ═══════════════════════════════════════════════════════════════════

def kmeans_partition(uvs, faces, n_clusters=16, n_iter=20, seed=42):
    """K-means clustering of face UV centroids."""
    rng = np.random.RandomState(seed)
    nf = len(faces)

    # Face centroids in UV space
    centroids = np.array([(uvs[f[0]] + uvs[f[1]] + uvs[f[2]]) / 3.0 for f in faces])

    # Init centers randomly
    centers = centroids[rng.choice(nf, n_clusters, replace=False)]

    for _ in range(n_iter):
        # Assign
        dists = np.sum((centroids[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)

        # Update
        for k in range(n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                centers[k] = centroids[mask].mean(axis=0)

    return labels


# ═══════════════════════════════════════════════════════════════════
# 3. Boundary extraction
# ═══════════════════════════════════════════════════════════════════

def extract_partition_boundary(face_labels, faces, n_parts):
    """Extract the boundary polygon (closed loop) for each partition."""
    boundaries = []
    for pid in range(n_parts):
        edge_count = {}
        for fi in np.where(face_labels == pid)[0]:
            f = faces[fi]
            for k in range(3):
                a, b = f[k], f[(k + 1) % 3]
                if a > b:
                    a, b = b, a
                edge_count[(a, b)] = edge_count.get((a, b), 0) + 1

        # Boundary edges appear exactly once
        boundary_edges = [e for e, c in edge_count.items() if c == 1]
        if len(boundary_edges) < 3:
            boundaries.append(None)
            continue

        # Trace edges into a loop
        adj = {}
        for a, b in boundary_edges:
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)

        loop = [boundary_edges[0][0]]
        prev = -1
        for _ in range(len(boundary_edges) + 2):
            cur = loop[-1]
            nbs = adj.get(cur, [])
            nxt = None
            for n in nbs:
                if n != prev:
                    nxt = n
                    break
            if nxt is None or nxt == loop[0]:
                break
            prev = cur
            loop.append(nxt)

        if len(loop) >= 3 and loop[0] == loop[-1]:
            loop = loop[:-1]
        boundaries.append(loop if len(loop) >= 3 else None)

    return boundaries


# ═══════════════════════════════════════════════════════════════════
# 4. Simple Laplacian smoothing
# ═══════════════════════════════════════════════════════════════════

def laplacian_smooth_polygon(points, n_iters=5, lam=0.3):
    """Smooth a closed 2D polygon via Laplacian."""
    pts = points.copy().astype(np.float64)
    n = len(pts)
    if n < 4:
        return pts
    for _ in range(n_iters):
        new_pts = pts.copy()
        for i in range(n):
            prev = pts[(i - 1) % n]
            nxt = pts[(i + 1) % n]
            avg = (prev + nxt) / 2.0
            new_pts[i] = pts[i] + lam * (avg - pts[i])
        pts = new_pts
    return pts


# ═══════════════════════════════════════════════════════════════════
# 5. 2D Convex hull (Andrew's monotone chain)
# ═══════════════════════════════════════════════════════════════════

def convex_hull_2d(pts):
    """Returns indices of points on the convex hull."""
    n = len(pts)
    if n < 3:
        return list(range(n))
    idx = sorted(range(n), key=lambda i: (pts[i][0], pts[i][1]))

    lower = []
    for i in idx:
        while len(lower) >= 2:
            o, a = pts[lower[-2]], pts[lower[-1]]
            b = pts[i]
            if (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]) > 0:
                break
            lower.pop()
        lower.append(i)

    upper = []
    for i in reversed(idx):
        while len(upper) >= 2:
            o, a = pts[upper[-2]], pts[upper[-1]]
            b = pts[i]
            if (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]) > 0:
                break
            upper.pop()
        upper.append(i)

    return lower[:-1] + upper[:-1]


def point_to_line_dist(p, a, b):
    ab = np.array(b) - a
    len2 = ab @ ab
    if len2 < 1e-12:
        return np.linalg.norm(np.array(p) - a)
    t = max(0.0, min(1.0, (np.array(p) - a) @ ab / len2))
    proj = a + t * ab
    return np.linalg.norm(np.array(p) - proj)


# ═══════════════════════════════════════════════════════════════════
# 6. Pocket detection and concave classification
# ═══════════════════════════════════════════════════════════════════

def detect_pockets(poly):
    """Find all convex hull pockets in a 2D polygon."""
    n = len(poly)
    if n < 4:
        return []

    hull = convex_hull_2d(poly)
    h = len(hull)
    if h < 3:
        return []

    hull_set = set(hull)
    pockets = []

    for hi in range(h):
        hnext = (hi + 1) % h
        pA, pB = hull[hi], hull[hnext]

        # Walk from pA to pB along polygon
        arc = []
        cur = (pA + 1) % n
        while cur != pB:
            arc.append(cur)
            cur = (cur + 1) % n
            if len(arc) > n:
                break

        if not arc:
            continue

        max_depth = max(point_to_line_dist(poly[vi], poly[pA], poly[pB]) for vi in arc)
        width = np.linalg.norm(np.array(poly[pA]) - poly[pB])

        pockets.append({
            'pA': pA, 'pB': pB, 'arc': arc,
            'max_depth': max_depth, 'width': width
        })

    return pockets


def classify_split(poly, pocket):
    """Attempt to find a split line for a pocket. Returns (p0, p1) or (None, None)."""
    arc = pocket['arc']
    m = len(arc)
    if m < 4:
        return None, None

    # Compute polygon diameter for normalization
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    diam = np.sqrt((max(xs)-min(xs))**2 + (max(ys)-min(ys))**2)
    ratio = pocket['max_depth'] / diam if diam > 1e-8 else 0

    if ratio < 0.03:
        return None, None

    # Compute pocket area sign (positive = tip outward, negative = corner inward)
    idxA = int(pocket['pA'])
    idxB = int(pocket['pB'])
    posA = np.array(poly[idxA])
    posB = np.array(poly[idxB])
    area2 = posA[0]*posB[1] - posA[1]*posB[0]
    prev = posB
    for vi in arc:
        p = np.array(poly[int(vi)])
        area2 += prev[0]*p[1] - prev[1]*p[0]
        prev = p

    if area2 > 0:
        # Tip: find neck
        best_ratio = 1e9
        best_i, best_j = -1, -1
        for i in range(m):
            vi = int(arc[i])
            for j in range(i + 3, m):
                vj = int(arc[j])
                chord = np.array(poly[vj]) - np.array(poly[vi])
                clen = np.linalg.norm(chord)
                if clen < 1e-8:
                    continue
                max_d = max(point_to_line_dist(poly[int(arc[k])], poly[vi], poly[vj])
                           for k in range(i + 1, j))
                if max_d > 0:
                    r = clen / max_d
                    if r < best_ratio:
                        best_ratio = r
                        best_i, best_j = vi, vj
        if best_i >= 0:
            return np.array(poly[best_i]), np.array(poly[best_j])
    else:
        # Corner: angle bisector
        # Find deepest arc point
        deepest_i = max((int(vi) for vi in arc),
                        key=lambda vi: point_to_line_dist(poly[vi], posA, posB))
        vDeep = np.array(poly[deepest_i])
        n = len(poly)
        vPrev = np.array(poly[(deepest_i - 1) % n])
        vNext = np.array(poly[(deepest_i + 1) % n])
        d1 = vDeep - vPrev; d1 /= max(np.linalg.norm(d1), 1e-12)
        d2 = vNext - vDeep; d2 /= max(np.linalg.norm(d2), 1e-12)
        bisector = d1 + d2
        blen = np.linalg.norm(bisector)
        if blen < 1e-12:
            return None, None
        bisector /= blen

        # Ray cast
        ray_end = vDeep + bisector * diam * 2
        best_u = 1e9
        best_hit = None
        for i in range(n):
            j = (i + 1) % n
            if i == deepest_i or j == deepest_i:
                continue
            if (deepest_i - i) % n <= 1 or (i - deepest_i) % n <= 1:
                continue
            s1, e1 = np.array(poly[i]), np.array(poly[j])
            d_seg = e1 - s1
            d_ray = ray_end - vDeep
            cross = d_seg[0]*d_ray[1] - d_seg[1]*d_ray[0]
            if abs(cross) < 1e-12:
                continue
            t = ((vDeep[0]-s1[0])*d_ray[1] - (vDeep[1]-s1[1])*d_ray[0]) / cross
            u = ((vDeep[0]-s1[0])*d_seg[1] - (vDeep[1]-s1[1])*d_seg[0]) / cross
            if 0 <= t <= 1 and u > 1e-3 and u < best_u:
                best_u = u
                best_hit = vDeep + u * d_ray
        if best_hit is not None:
            return vDeep, best_hit

    return None, None


# ═══════════════════════════════════════════════════════════════════
# 7. Face re-assignment after split
# ═══════════════════════════════════════════════════════════════════

def split_faces(face_labels, faces, uvs, p0, p1, src_pid, n_parts):
    """Assign faces on each side of line p0->p1 to different partitions."""
    nf = len(faces)
    sides = np.zeros(nf, dtype=int)

    def line_side(pt):
        return np.sign((p1[0]-p0[0])*(pt[1]-p0[1]) - (p1[1]-p0[1])*(pt[0]-p0[0]))

    left_cnt = right_cnt = 0
    for fi in range(nf):
        if face_labels[fi] != src_pid:
            continue
        c = (uvs[faces[fi][0]] + uvs[faces[fi][1]] + uvs[faces[fi][2]]) / 3.0
        s = line_side(c)
        sides[fi] = s
        if s > 0:
            right_cnt += 1
        else:
            left_cnt += 1

    if left_cnt < 3 or right_cnt < 3:
        return 0

    new_pid = n_parts
    face_labels[faces[fi] if sides[fi] > 0 else None] = None  # no-op
    for fi in range(nf):
        if face_labels[fi] != src_pid:
            continue
        if sides[fi] > 0:
            face_labels[fi] = new_pid

    return 1


# ═══════════════════════════════════════════════════════════════════
# 8. Main test flow
# ═══════════════════════════════════════════════════════════════════

def run_test():
    print("=== Concave Split Test ===")

    # ── Generate mesh ──
    print("1. Generating wavy mesh (30×30 grid)...")
    verts, faces, uvs, xs, ys = generate_mesh(30)
    print(f"   {len(verts)} vertices, {len(faces)} faces")

    # ── K-means partition ──
    n_parts = 16
    print(f"2. K-means partition ({n_parts} clusters)...")
    face_labels = kmeans_partition(uvs, faces, n_clusters=n_parts)
    # Count actual non-empty partitions
    active = len(set(face_labels))
    print(f"   {active} non-empty partitions")

    # ── Extract boundaries ──
    print("3. Extracting partition boundaries...")
    boundaries_before = extract_partition_boundary(face_labels, faces, n_parts)

    # ── Smooth boundaries ──
    print("4. Smoothing boundaries (5 iterations)...")
    boundaries_smooth = []
    for bidx, b in enumerate(boundaries_before):
        if b is None:
            boundaries_smooth.append(None)
            continue
        poly2d = np.array([uvs[v] for v in b])
        smoothed = laplacian_smooth_polygon(poly2d, n_iters=5)
        boundaries_smooth.append(smoothed)
    boundaries_before = boundaries_smooth  # use smoothed for analysis

    # ── Concave detection ──
    print("5. Concave detection + split attempt...")
    concave_pids = []
    split_lines = []  # (pid, p0, p1, polyline_p0, polyline_p1)
    n_splits = 0

    for pid in range(n_parts):
        poly = boundaries_before[pid]
        if poly is None or len(poly) < 4:
            continue

        pockets = detect_pockets(poly)
        if not pockets:
            continue

        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
        diam = np.sqrt((max(xs)-min(xs))**2 + (max(ys)-min(ys))**2)
        if diam < 1e-8:
            continue

        best_pkt = max(pockets, key=lambda pk: pk['max_depth'])
        ratio = best_pkt['max_depth'] / diam

        if ratio >= 0.03:
            concave_pids.append(pid)
            print(f"   part {pid}: depth/diam={ratio:.4f} (pockets={len(pockets)}) -> macro-concave")

            p0, p1 = classify_split(poly, best_pkt)
            if p0 is not None:
                # Build 3D positions for the split line
                # p0 and p1 are in UV space — need to evaluate wavy surface
                x0 = xs[0] + p0[0] * (xs[-1] - xs[0]) if False else \
                     p0[0] * (1.5 - (-1.5)) + (-1.5)
                # Actually p0/p1 are 2D polygon points in UV space [0,1]
                # Map UV to 3D via the wavy function
                u0, v0 = p0[0], p0[1]
                x0 = u0 * 3.0 - 1.5
                y0 = v0 * 3.0 - 1.5
                z0 = wavy_z(x0, y0)

                u1, v1 = p1[0], p1[1]
                x1 = u1 * 3.0 - 1.5
                y1 = v1 * 3.0 - 1.5
                z1 = wavy_z(x1, y1)

                split_lines.append((pid, p0, p1,
                                   np.array([x0, y0, z0]),
                                   np.array([x1, y1, z1])))

                # Apply split
                n = split_faces(face_labels, faces, uvs, p0, p1, pid, n_parts)
                if n > 0:
                    n_splits += n
                    n_parts += 1
                    print(f"   -> Split! {n} faces -> new partition {n_parts-1}")

    print(f"   Total: {len(concave_pids)} macro-concave, {n_splits} splits")

    # ── Extract boundaries after split ──
    print("6. Extracting boundaries after split...")
    boundaries_after = extract_partition_boundary(face_labels, faces, n_parts)

    # ── Visualization ──
    if not HAS_PV:
        return

    print("7. Visualizing...")
    pv.set_plot_theme("document")
    pl = pv.Plotter()
    pl.add_text("Concave Split Test  |  1=Mesh 2=Boundaries 3=Concave 4=Splits 5=After  q=Quit",
                position="upper_left", font_size=10)

    # --- Mesh ---
    tri_faces = np.hstack([np.full((len(faces), 1), 3), faces]).astype(int)
    mesh = pv.PolyData(verts, tri_faces)
    pl.add_mesh(mesh, color='lightblue', opacity=0.4, show_edges=True, edge_color='gray',
                name='mesh', label='Surface Mesh')

    # --- Boundaries before split ---
    for pid in range(min(n_parts, len(boundaries_before))):
        poly = boundaries_before[pid]
        if poly is None or len(poly) < 3:
            continue
        poly_3d = np.array([[p[0]*3.0-1.5, p[1]*3.0-1.5, wavy_z(p[0]*3.0-1.5, p[1]*3.0-1.5)]
                           for p in poly])
        # Close the loop
        if np.linalg.norm(poly_3d[0] - poly_3d[-1]) > 1e-6:
            poly_3d = np.vstack([poly_3d, poly_3d[0:1]])
        pl.add_lines(poly_3d, color='black', width=2, name=f'boundary_{pid}',
                     label='Boundaries (before)')

    # --- Concave partitions highlighted ---
    concave_actors = []
    for pid in concave_pids:
        if pid >= len(boundaries_before):
            continue
        poly = boundaries_before[pid]
        if poly is None:
            continue
        poly_3d = np.array([[p[0]*3.0-1.5, p[1]*3.0-1.5, wavy_z(p[0]*3.0-1.5, p[1]*3.0-1.5)]
                           for p in poly])
        if np.linalg.norm(poly_3d[0] - poly_3d[-1]) > 1e-6:
            poly_3d = np.vstack([poly_3d, poly_3d[0:1]])
        a = pl.add_lines(poly_3d, color='red', width=4, name=f'concave_{pid}',
                         label='Macro-Concave')
        concave_actors.append(a)

    # --- Split lines ---
    split_actors = []
    for pid, p0_uv, p1_uv, p0_3d, p1_3d in split_lines:
        line_3d = np.array([p0_3d, p1_3d])
        a = pl.add_lines(line_3d, color='orange', width=4, name=f'split_{pid}',
                         label='Split Lines')
        split_actors.append(a)

    # --- Boundaries after split ---
    after_actors = []
    colors = ['cyan', 'magenta', 'lime', 'yellow']
    for pid in range(min(n_parts, len(boundaries_after))):
        poly = boundaries_after[pid]
        if poly is None or len(poly) < 3:
            continue
        poly_3d = np.array([[p[0]*3.0-1.5, p[1]*3.0-1.5, wavy_z(p[0]*3.0-1.5, p[1]*3.0-1.5)]
                           for p in poly])
        if np.linalg.norm(poly_3d[0] - poly_3d[-1]) > 1e-6:
            poly_3d = np.vstack([poly_3d, poly_3d[0:1]])
        color = colors[pid % len(colors)]
        a = pl.add_lines(poly_3d, color=color, width=3, name=f'after_{pid}',
                         label='After Split')
        after_actors.append(a)

    # Initially hide after-split boundaries
    for a in after_actors:
        a.SetVisibility(False)

    # --- Toggle callbacks ---
    all_boundary = []
    for pid in range(min(n_parts, len(boundaries_before))):
        all_boundary.append(f'boundary_{pid}')

    def toggle_by_name(names):
        for a in pl.renderer.actors.values():
            if a.GetObjectName() in names:
                a.SetVisibility(not a.GetVisibility())

    pl.add_key_event('1', lambda: pl.renderer.actors['mesh'].SetVisibility(
        not pl.renderer.actors['mesh'].GetVisibility()))
    pl.add_key_event('2', lambda: toggle_by_name(all_boundary))
    pl.add_key_event('3', lambda: [a.SetVisibility(not a.GetVisibility()) for a in concave_actors])
    pl.add_key_event('4', lambda: [a.SetVisibility(not a.GetVisibility()) for a in split_actors])
    pl.add_key_event('5', lambda: [a.SetVisibility(not a.GetVisibility()) for a in after_actors])

    pl.add_key_event('q', pl.close)

    pl.show_grid()
    pl.camera_position = 'iso'
    pl.show()


if __name__ == '__main__':
    run_test()
