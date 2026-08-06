"""
Compare point-cloud vs mesh visualization of partition labels.
Usage: python analyze_labels.py <data_dir>
"""
import sys, os
import numpy as np
import pyvista as pv
from visualize import load_obj, load_labels

def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    verts, faces = load_obj(os.path.join(data_dir, "mesh.obj"))
    labels = load_labels(os.path.join(data_dir, "partition_labels.txt"))
    pyvista_faces = np.column_stack([np.full(len(faces), 3), faces]).ravel()

    pl = pv.Plotter(shape=(1,2))

    # Left: point cloud
    pl.subplot(0,0)
    cloud = pv.PolyData(verts)
    cloud["label"] = labels
    pl.add_points(cloud, scalars="label", cmap="tab10",
                  point_size=30, render_points_as_spheres=True,
                  show_scalar_bar=False)
    pl.add_text("Point cloud (per-vertex)", font_size=14)
    pl.reset_camera()

    # Right: mesh surface
    pl.subplot(0,1)
    mesh = pv.PolyData(verts, pyvista_faces)
    pl.add_mesh(mesh, scalars=labels, cmap="tab10",
                show_edges=True, edge_color="gray", show_scalar_bar=False)
    pl.add_text("Mesh surface", font_size=14)
    pl.reset_camera()

    pl.link_views()
    pl.show()

if __name__ == "__main__":
    main()
