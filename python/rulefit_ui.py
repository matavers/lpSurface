"""
rulefit UI — 可交互可视化 + 逐步调试（OCCT-R1）。

架构：
  - PipelineRunner(QThread) 后台线程：运行 rulefit.exe（subprocess），逐行读输出，
    解析 [STEP:N:done] 标记，并读取对应 step 的中间结果文件（文件读取在后台线程）。
  - MainWindow 主线程：QtInteractor(pyvistaqt) 3D 显示 + 参数控件 + 控制台。
    通过 signal 接收后台线程的数据，更新 3D 显示（UI 交互在主线程）。

用法: python rulefit_ui.py
"""
import sys, os, subprocess
from pathlib import Path
import numpy as np

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
                             QFormLayout, QComboBox, QSpinBox, QDoubleSpinBox, QPushButton,
                             QTextEdit, QLabel, QGroupBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

from pyvistaqt import QtInteractor
import pyvista as pv

PROJECT_DIR = Path(__file__).resolve().parent.parent
EXE = PROJECT_DIR / "build" / "Release" / "rulefit.exe"
DEFAULT_OUT = PROJECT_DIR / "out"

# 每分区颜色（TAB10 前 16）
TAB10 = np.array([
    [0.122, 0.467, 0.706], [1.0, 0.498, 0.055], [0.173, 0.627, 0.173],
    [0.839, 0.153, 0.157], [0.580, 0.404, 0.741], [0.549, 0.337, 0.294],
    [0.890, 0.467, 0.761], [0.498, 0.498, 0.498], [0.737, 0.741, 0.133],
    [0.090, 0.745, 0.812], [0.647, 0.165, 0.165], [0.161, 0.616, 0.561],
    [0.929, 0.694, 0.125], [0.941, 0.941, 0.941], [0.800, 0.800, 0.800],
    [0.300, 0.300, 0.300]])


# ═══════════════════════════════════════════════════════════════
# 文件读取（后台线程调用）
# ═══════════════════════════════════════════════════════════════

def load_mesh_obj(path):
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            if p[0] == 'v':
                verts.append([float(p[1]), float(p[2]), float(p[3])])
            elif p[0] == 'f':
                faces.append([int(x.split('/')[0]) - 1 for x in p[1:4]])
    return np.array(verts), np.array(faces, dtype=np.int64)


def load_direction_field(path):
    """返回 dict: verts, asym1, asym2, hyperbolic, k1, k2"""
    data = np.loadtxt(path, skiprows=1)
    return {
        'verts': data[:, 0:3],
        'asym1': data[:, 3:6],
        'asym2': data[:, 6:9],
        'hyperbolic': data[:, 9].astype(bool),
        'k1': data[:, 10],
        'k2': data[:, 11],
    }


def load_face_labels(path):
    return np.loadtxt(path, dtype=int)


def load_patches(dirpath):
    """返回 patches: list of dict(gamma, ruling, length)"""
    patches = []
    i = 0
    while True:
        p = Path(dirpath) / f"patch_{i}.txt"
        if not p.exists():
            break
        d = np.loadtxt(p, skiprows=1)
        if d.ndim == 1:
            d = d.reshape(1, -1)
        patches.append({'gamma': d[:, 0:3], 'ruling': d[:, 3:6], 'length': d[:, 6]})
        i += 1
    return patches


def load_evaluation(path):
    res = {}
    with open(path) as f:
        for line in f:
            p = line.split()
            if p and p[0] in ('mean_twist', 'mean_fit_error', 'n_developable', 'n_patches'):
                res[p[0]] = float(p[1])
    return res


# ═══════════════════════════════════════════════════════════════
# 后台线程：运行 pipeline + 读取中间结果文件
# ═══════════════════════════════════════════════════════════════

class PipelineRunner(QThread):
    line_signal = pyqtSignal(str)            # 控制台输出行
    step_data_signal = pyqtSignal(int, object)  # (step 编号, 该 step 的数据 dict)
    finished_signal = pyqtSignal(int)

    def __init__(self, args, out_dir):
        super().__init__()
        self.args = args
        self.out_dir = str(out_dir)

    def run(self):
        cmd = [str(EXE)] + self.args + [f"--out={self.out_dir}"]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in iter(proc.stdout.readline, ''):
                line = line.rstrip()
                if line:
                    self.line_signal.emit(line)
                if line.startswith('[STEP:'):
                    n = int(line.split(':')[1])
                    data = self._read_step(n)
                    self.step_data_signal.emit(n, data)
            proc.wait()
            self.finished_signal.emit(proc.returncode)
        except Exception as e:
            self.line_signal.emit(f"Error: {e}")
            self.finished_signal.emit(1)

    def _read_step(self, n):
        """后台线程读取 step n 的中间结果文件。"""
        d = self.out_dir
        if n == 0:
            verts, faces = load_mesh_obj(os.path.join(d, 'mesh.obj'))
            return {'verts': verts, 'faces': faces}
        if n == 1:
            return load_direction_field(os.path.join(d, 'direction_field.txt'))
        if n == 2:
            labels = load_face_labels(os.path.join(d, 'face_labels.txt'))
            patches = load_patches(os.path.join(d, 'patches'))
            return {'labels': labels, 'patches': patches}
        if n == 3:
            return load_evaluation(os.path.join(d, 'evaluation.txt'))
        return {}


# ═══════════════════════════════════════════════════════════════
# 主窗口
# ═══════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("rulefit — Developable Surface Partitioning (OCCT-R1)")
        self.resize(1400, 900)
        self._runner = None
        self._step_data = {}
        self._mesh_actor = None
        self._field_actors = []
        self._patch_actors = []

        self._setup_ui()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Horizontal)
        self._plotter = QtInteractor(self, shape=(1, 1))
        self._plotter.set_background('lightgray')
        splitter.addWidget(self._plotter)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(4, 0, 4, 0)
        self._setup_params(rl)
        self._setup_steps(rl)
        splitter.addWidget(right)
        splitter.setSizes([1000, 400])
        root.addWidget(splitter, 1)

        self._console = QTextEdit()
        self._console.setReadOnly(True)
        self._console.setMaximumHeight(220)
        self._console.setFont(QFont("Consolas", 10))
        root.addWidget(self._console)

    def _setup_params(self, layout):
        g = QGroupBox("Parameters")
        form = QFormLayout(g)
        self._cmb_surface = QComboBox()
        self._cmb_surface.addItems(["wavy", "random", "mountain"])
        form.addRow("Surface:", self._cmb_surface)
        self._spn_res = QSpinBox(); self._spn_res.setRange(10, 120); self._spn_res.setValue(40)
        form.addRow("Mesh res:", self._spn_res)
        self._spn_k = QSpinBox(); self._spn_k.setRange(2, 64); self._spn_k.setValue(16)
        form.addRow("K partitions:", self._spn_k)
        self._spn_iter = QSpinBox(); self._spn_iter.setRange(1, 200); self._spn_iter.setValue(20)
        form.addRow("Max iter:", self._spn_iter)
        self._spn_ldev = QDoubleSpinBox(); self._spn_ldev.setRange(0, 100); self._spn_ldev.setValue(1.0)
        form.addRow("lambda dev:", self._spn_ldev)
        self._spn_lcen = QDoubleSpinBox(); self._spn_lcen.setRange(0, 100); self._spn_lcen.setValue(0.3)
        form.addRow("lambda center:", self._spn_lcen)
        self._btn_run = QPushButton("Run Pipeline")
        self._btn_run.clicked.connect(self._on_run)
        form.addRow(self._btn_run)
        layout.addWidget(g)

    def _setup_steps(self, layout):
        g = QGroupBox("Steps")
        form = QFormLayout(g)
        self._step_labels = {}
        for n, name in [(0, "Mesh"), (1, "Direction field"), (2, "Partition"), (3, "Evaluation")]:
            lb = QLabel("pending")
            lb.setStyleSheet("color: gray")
            form.addRow(f"[{n}] {name}:", lb)
            self._step_labels[n] = lb
        layout.addWidget(g)
        layout.addStretch(1)

    # ── 运行 ──
    def _on_run(self):
        if self._runner and self._runner.isRunning():
            return
        for lb in self._step_labels.values():
            lb.setText("pending")
            lb.setStyleSheet("color: gray")
        self._console.clear()
        self._clear_actors()
        self._step_data = {}

        args = [f"--surface={self._cmb_surface.currentText()}",
                f"--res={self._spn_res.value()}",
                f"--K={self._spn_k.value()}",
                f"--max-iter={self._spn_iter.value()}",
                f"--lambda-dev={self._spn_ldev.value()}",
                f"--lambda-center={self._spn_lcen.value()}"]
        self._runner = PipelineRunner(args, DEFAULT_OUT)
        self._runner.line_signal.connect(self._on_line)
        self._runner.step_data_signal.connect(self._on_step_data)
        self._runner.finished_signal.connect(self._on_finished)
        self._runner.start()

    def _on_line(self, line):
        self._console.append(line)

    def _on_step_data(self, n, data):
        self._step_data[n] = data
        self._step_labels[n].setText("done")
        self._step_labels[n].setStyleSheet("color: green")
        self._render_step(n, data)

    def _on_finished(self, code):
        self._console.append(f"--- pipeline finished (code={code}) ---")

    # ── 渲染各 step ──
    def _clear_actors(self):
        for a in self._field_actors + self._patch_actors:
            try:
                self._plotter.remove_actor(a)
            except Exception:
                pass
        if self._mesh_actor is not None:
            try:
                self._plotter.remove_actor(self._mesh_actor)
            except Exception:
                pass
        self._field_actors = []
        self._patch_actors = []
        self._mesh_actor = None

    def _render_step(self, n, data):
        if n == 0:
            self._render_mesh(data)
        elif n == 1:
            self._render_field(data)
        elif n == 2:
            self._render_partition(data)
        elif n == 3:
            self._render_evaluation(data)
        self._plotter.reset_camera()
        self._plotter.render()

    def _render_mesh(self, data):
        verts, faces = data['verts'], data['faces']
        mesh = pv.PolyData(verts, np.hstack([np.full((len(faces), 1), 3), faces]).astype(int))
        self._mesh_actor = self._plotter.add_mesh(
            mesh, color='lightblue', show_edges=True, edge_color='gray',
            name='mesh', label='Surface mesh')

    def _render_field(self, data):
        verts = data['verts']
        asym1 = data['asym1']
        asym2 = data['asym2']
        hyper = data['hyperbolic']
        if self._mesh_actor is None:
            self._mesh_actor = self._plotter.add_mesh(
                verts, color='lightblue', opacity=0.6, name='mesh')
        # 渐近方向 1（红色箭头）和 2（蓝色箭头），只画双曲点
        for tag, arr, color in [('a1', asym1, 'red'), ('a2', asym2, 'blue')]:
            pts = verts[hyper]
            vec = arr[hyper]
            if len(pts) == 0:
                continue
            g = self._plotter.add_arrows(pts, vec, mag=0.15, color=color,
                                         name=f'field_{tag}', label=f'asymptotic {tag}')
            self._field_actors.append(g)

    def _render_partition(self, data):
        labels = data['labels']
        patches = data['patches']
        n_part = len(patches)
        # 分区着色：需要 mesh 顶点+面。用 step 0 的 mesh。
        if 0 in self._step_data:
            verts = self._step_data[0]['verts']
            faces = self._step_data[0]['faces']
            colors = np.zeros((len(verts), 3))
            face_colors = np.array([TAB10[l % 16] for l in labels])
            cnt = np.zeros(len(verts))
            for fi, f in enumerate(faces):
                for v in f:
                    colors[v] += face_colors[fi]
                    cnt[v] += 1
            cnt[cnt == 0] = 1
            colors = colors / cnt[:, None]
            mesh = pv.PolyData(verts, np.hstack([np.full((len(faces), 1), 3), faces]).astype(int))
            mesh.point_data['rgb'] = colors
            if self._mesh_actor is None:
                self._mesh_actor = self._plotter.add_mesh(
                    mesh, scalars='rgb', rgb=True, name='partition', label='Partition')
        # 直纹面 patch（director curve + 母线）
        for k, p in enumerate(patches):
            if len(p['gamma']) == 0:
                continue
            # director curve 折线
            self._plotter.add_lines(np.array(p['gamma']), color=TAB10[k % 16], width=3,
                                    name=f'gamma_{k}', label='director curve')
            # 母线（每截面一条线段：gamma -> gamma + ruling*length）
            segs = []
            for s in range(len(p['gamma'])):
                a = p['gamma'][s]
                b = a + p['ruling'][s] * p['length'][s]
                segs.append([a, b])
            if segs:
                self._plotter.add_lines(np.array(segs).reshape(-1, 3), color='black',
                                        width=1, name=f'ruling_{k}', label='ruling')

    def _render_evaluation(self, data):
        self._console.append("--- Evaluation ---")
        for k, v in data.items():
            self._console.append(f"  {k} = {v}")


def main():
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
