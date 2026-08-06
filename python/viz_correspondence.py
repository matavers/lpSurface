"""
可视化：分区边界 + 对应关系线段 + 夹逼曲线

用法:
  python viz_correspondence.py <data_dir> [--pid 0] [--samples 40] [--delta 0.10]

操作:
  1  - 切换 NURBS 曲面网格
  2  - 切换分区边界 loop
  3  - 切换对应关系线段 (P_i → Q_i*)
  4  - 切换夹逼走廊曲线 (φ_lower, φ_upper, φ)
  5  - 切换曲线 A / 曲线 B 标识
  q  - 退出
"""

import sys, os, time
import numpy as np
import pyvista as pv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nurbs_eval import NurbsSurface
from sandwich_bounds import (
    compute_sandwich_bounds, extract_curves,
    _arc_length_param, _resample_polyline, _ruling_integral,
)

TAB10 = np.array([
    [0.122, 0.467, 0.706], [1.000, 0.498, 0.055],
    [0.173, 0.627, 0.173], [0.839, 0.153, 0.157],
    [0.580, 0.404, 0.741], [0.549, 0.337, 0.294],
    [0.890, 0.467, 0.761], [0.498, 0.498, 0.498],
    [0.738, 0.738, 0.131], [0.090, 0.745, 0.812],
])


def _add_loop_lines(plotter, pts, color, width, name):
    """Add a closed polyline loop."""
    poly = pv.PolyData(pts)
    n = len(pts)
    # Closed: 0-1-2-...-(n-1)-0
    poly.lines = np.array([n + 1] + list(range(n)) + [0], dtype=int)
    return plotter.add_mesh(poly, color=color, line_width=width, name=name)


def _add_polyline(plotter, pts, color, width, name):
    """Add an open polyline."""
    poly = pv.PolyData(pts)
    n = len(pts)
    poly.lines = np.array([n] + list(range(n)), dtype=int)
    return plotter.add_mesh(poly, color=color, line_width=width, name=name)


def _add_segments(plotter, starts, ends, color, width, name):
    """Add disconnected line segments (each start[i]->end[i])."""
    n = len(starts)
    pts = np.zeros((n * 2, 3))
    pts[0::2] = starts; pts[1::2] = ends
    segs = np.array([[2 * i, 2 * i + 1] for i in range(n)])
    return plotter.add_lines(pts, color=color, width=width, name=name, connected=False)


def load_obj(path):
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f':
                fv = [parts[1], parts[2], parts[3]]
                fv = [v.split('/')[0] for v in fv]
                faces.append([int(v) - 1 for v in fv])
    return np.array(verts), np.array(faces)


def _bspline_basis(u, degree, n_ctrl):
    n_knots = n_ctrl + degree + 1
    knots = np.zeros(n_knots)
    knots[:degree + 1] = 0.0
    knots[-(degree + 1):] = 1.0
    inner = n_knots - 2 * (degree + 1)
    if inner > 0:
        knots[degree + 1:-degree - 1] = np.linspace(0, 1, inner + 2)[1:-1]
    u = np.clip(u, 0.0, 1.0)
    span = degree
    for k in range(degree, n_knots - degree - 1):
        if knots[k] <= u < knots[k + 1]:
            span = k; break
    if u >= knots[-degree - 1]:
        span = n_knots - degree - 2
    N = np.zeros(degree + 1); N[0] = 1.0
    left = np.zeros(degree + 1, dtype=int)
    right = np.zeros(degree + 1, dtype=int)
    for j in range(1, degree + 1):
        left[j] = span + 1 - j; right[j] = span + j
        saved = 0.0
        for r in range(j):
            temp = N[r] / (knots[right[r + 1]] - knots[left[r + 1]] + 1e-14)
            N[r] = saved + (knots[right[r + 1]] - u) * temp
            saved = (u - knots[left[r + 1]]) * temp
        N[j] = saved
    result = np.zeros(n_ctrl)
    result[span - degree:span + 1] = N[:degree + 1]
    return result


def _eval_phi_np(u_arr, coeffs, degree, n_ctrl):
    """Vectorized φ(u) = u + Σ cⱼ·Bⱼ(u)."""
    full = np.concatenate([[0.0], coeffs, [0.0]])
    result = np.zeros_like(u_arr)
    for idx, u in enumerate(u_arr):
        B = _bspline_basis(float(u), degree, n_ctrl)
        result[idx] = float(u) + np.dot(B, full)
    return result


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print("Usage: python viz_correspondence.py <data_dir> [--pid 0] [--samples 40] [--delta 0.10]")
        return 1

    data_dir = sys.argv[1]
    pid = 0; N_samples = 40; delta = 0.10
    args = sys.argv[2:]; i = 0
    while i < len(args):
        if args[i] == '--pid' and i + 1 < len(args):
            pid = int(args[i + 1]); i += 2
        elif args[i] == '--samples' and i + 1 < len(args):
            N_samples = int(args[i + 1]); i += 2
        elif args[i] == '--delta' and i + 1 < len(args):
            delta = float(args[i + 1]); i += 2
        else:
            i += 1

    # ── Load data ──
    ns_path = os.path.join(data_dir, 'nurbs_surface.txt')
    lf3 = os.path.join(data_dir, f'part_{pid}_loop.txt')
    lfuv = os.path.join(data_dir, f'part_{pid}_loop_uv.txt')
    mesh_path = os.path.join(data_dir, 'mesh.obj')

    if not os.path.exists(lf3):
        print(f"Loop file not found: {lf3}")
        return 1
    if not os.path.exists(lfuv):
        print(f"UV loop file not found: {lfuv}")
        return 1
    if not os.path.exists(ns_path):
        print(f"NURBS surface not found: {ns_path}")
        return 1

    surf = NurbsSurface(ns_path)
    loop_3d = np.loadtxt(lf3)
    loop_uv = np.loadtxt(lfuv)
    print(f"NURBS: {surf.nU}x{surf.nV} deg={surf.degU}x{surf.degV}")
    print(f"Partition {pid}: {len(loop_3d)} loop vertices")

    # ── Compute sandwich bounds ──
    t_P, t_Q = 0.3, 0.7  # default split
    print(f"Computing sandwich bounds (t_P={t_P:.3f}, t_Q={t_Q:.3f}, N={N_samples}, delta={delta})...")
    t0 = time.time()
    u_grid, v_raw, v_mono, plo, pup = compute_sandwich_bounds(
        surf, loop_3d, loop_uv, t_P, t_Q, N_samples=N_samples, delta=delta)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  v_raw range: [{v_raw.min():.4f}, {v_raw.max():.4f}]")
    print(f"  v_mono range: [{v_mono[0]:.4f}, {v_mono[-1]:.4f}]")

    # ── Extract curves ──
    arc1_3d, arc1_uv, arc2_3d, arc2_uv = extract_curves(loop_3d, loop_uv, t_P, t_Q)

    # ── Build correspondence segments P_i -> Q_i* ──
    params_a = _arc_length_param(arc1_3d)
    P_list = _resample_polyline(arc1_3d, params_a, u_grid)
    params_b = _arc_length_param(arc2_3d)
    Q_raw = _resample_polyline(arc2_3d, params_b, v_raw)
    Q_mono = _resample_polyline(arc2_3d, params_b, v_mono)

    print(f"  Segments: {len(P_list)} P points, {len(Q_raw)} Q_raw, {len(Q_mono)} Q_mono")

    # ── Plotter ──
    pv.set_plot_theme("document")
    pl = pv.Plotter()
    pl.add_text(f"Partition {pid}  |  1=Surf 2=Loop 3=Corresp 4=Bounds  q=Quit",
                position="upper_left", font_size=10)

    actors = {}  # key -> (actor, visible_bool)

    # --- NURBS surface mesh ---
    if os.path.exists(mesh_path):
        v, f = load_obj(mesh_path)
        mesh = pv.PolyData(v, np.hstack([np.full((len(f), 1), 3), f]).astype(int))
        actors['surf'] = [pl.add_mesh(mesh, color='lightblue', opacity=0.35,
                                       show_edges=False, name='nurbs_surf'), True]
    else:
        print(f"  Warning: {mesh_path} not found, skipping surface")

    # --- Partition boundary loop (closed) ---
    actors['loop'] = [_add_loop_lines(pl, loop_3d, 'black', 3, 'loop'), True]

    # --- Curve A (P->Q) in red, Curve B (Q->P) in blue ---
    actors['curveA'] = [_add_polyline(pl, arc1_3d, TAB10[3], 4, 'curve_A'), False]
    actors['curveB'] = [_add_polyline(pl, arc2_3d, TAB10[0], 4, 'curve_B'), False]

    # --- Split point markers ---
    actors['splitP'] = [pl.add_points(arc1_3d[0:1], color=TAB10[3], point_size=12, name='P'), False]
    actors['splitQ'] = [pl.add_points(arc2_3d[-1:], color=TAB10[0], point_size=12, name='Q'), False]

    # --- Correspondence segments: P_i -> Q*(v_raw) ---
    actors['corresp_raw'] = [_add_segments(pl, P_list, Q_raw, TAB10[1], 2, 'corresp_raw'), True]

    # --- Correspondence segments: P_i -> Q*(v_mono, after PAV) ---
    actors['corresp_mono'] = [_add_segments(pl, P_list, Q_mono, TAB10[2], 2, 'corresp_mono'), False]

    # --- Sandwich corridor curves (3D) ---
    u_dense = np.linspace(0, 1, 200)
    v_lo = np.array([float(plo(u)) for u in u_dense])
    v_hi = np.array([float(pup(u)) for u in u_dense])

    curve_b_3d_lo = _resample_polyline(arc2_3d, params_b, v_lo)
    actors['sandwich_lo'] = [_add_polyline(pl, curve_b_3d_lo, TAB10[3], 3, 'phi_lower'), False]
    curve_b_3d_hi = _resample_polyline(arc2_3d, params_b, v_hi)
    actors['sandwich_hi'] = [_add_polyline(pl, curve_b_3d_hi, TAB10[0], 3, 'phi_upper'), False]
    v_ident = u_dense.copy()
    curve_b_3d_id = _resample_polyline(arc2_3d, params_b, v_ident)
    actors['sandwich_id'] = [_add_polyline(pl, curve_b_3d_id, TAB10[8], 2, 'phi_identity'), False]

    # --- Toggle callback ---
    key_map = {'1': 'surf', '2': 'loop', '3': 'corresp_raw', '4': 'sandwich_lo',
               '5': 'corresp_mono', '6': 'curveA', '7': 'sandwich_hi',
               '8': 'sandwich_id', '0': 'splitP'}

    def _toggle_actor(actor_list):
        for item in actor_list:
            if isinstance(item, pv.Actor):
                item.SetVisibility(not item.GetVisibility())

    def key_callback():
        # We'll use add_key_event instead of a standalone callback
        pass

    def make_toggle(key):
        def handler():
            if key == '3':
                # Toggle both raw corresp and mono corresp together
                _toggle_actor([actors['corresp_raw'][0]])
            elif key == '4':
                # Toggle all sandwich curves together
                for k in ['sandwich_lo', 'sandwich_hi', 'sandwich_id']:
                    _toggle_actor([actors[k][0]])
            elif key == '0':
                for k in ['splitP', 'splitQ', 'curveA', 'curveB']:
                    _toggle_actor([actors[k][0]])
            elif key in key_map:
                k = key_map[key]
                _toggle_actor([actors[k][0]])
        return handler

    pl.add_key_event('1', make_toggle('1'))
    pl.add_key_event('2', make_toggle('2'))
    pl.add_key_event('3', make_toggle('3'))
    pl.add_key_event('4', make_toggle('4'))
    pl.add_key_event('5', make_toggle('5'))
    pl.add_key_event('6', make_toggle('6'))
    pl.add_key_event('7', make_toggle('7'))
    pl.add_key_event('8', make_toggle('8'))
    pl.add_key_event('0', make_toggle('0'))
    pl.add_key_event('q', lambda: pl.close())

    # ── Show ──
    pl.show_grid()
    pl.camera_position = 'iso'
    pl.show()


if __name__ == '__main__':
    main()
