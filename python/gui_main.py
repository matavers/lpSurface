"""
Distillation GUI - NURBS Surface Partitioning + Boundary Smoothing + Ruled Fitting

Tree hierarchy:
  Retry 0
    ├── Mesh
    ├── Boundary iterations
    │     ├── iter 000
    │     └── ...
    ├── Final boundaries
    └── Ruled surfaces
          ├── part_0
          └── ...

Output prefix mapping (fixed 14char width):
  <Hard-EM     >  Hard-EM partitioning
  <Merge       >  Tiny region merge
  <Laplacian   >  Laplacian smoothing
  <Harmonic    >  Harmonic mesh update
  <OCCT        >  OCCT split / mesh
  <Export      >  File writes
  <Ruled       >  Ruled surface fitting
  <Tolerance   >  Tolerance check
"""

import sys, os, glob, subprocess, re, math, json, copy
from pathlib import Path
from datetime import datetime
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QTextEdit,
    QTreeWidget, QTreeWidgetItem, QSplitter, QFormLayout,
    QSpinBox, QDoubleSpinBox, QMessageBox, QFileDialog,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QColor, QTextCursor

try:
    from pyvistaqt import QtInteractor
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
BUILD_EXE = PROJECT_DIR / "build" / "Release" / "distillation.exe"
CONFIG_PATH = SCRIPT_DIR / ".gui_config.json"

TAB10_RGB = np.array([
    [0.122, 0.467, 0.706], [1.000, 0.498, 0.055],
    [0.173, 0.627, 0.173], [0.839, 0.153, 0.157],
    [0.580, 0.404, 0.741], [0.549, 0.337, 0.294],
    [0.890, 0.467, 0.761], [0.498, 0.498, 0.498],
    [0.738, 0.738, 0.131], [0.090, 0.745, 0.812],
])

PIPELINE_EVENT_RE = re.compile(
    r'\[PIPELINE:(retry|stage|progress|file|done):([^\]]+)\]')
# Pipeline stages in order:
PIPELINE_STAGES = [
    'Init', 'Hard-EM', 'Merge', 'Laplacian', 'Harmonic',
    'OCCT', 'Export', 'Ruled', 'Tolerance', 'Complete'
]

CONSOLE_CSS = """
QTextEdit {
    background: #1e1e1e; color: #d4d4d4;
    font-family: Consolas; font-size: 20px; padding: 4px;
}
"""

_TAG_W = 14  # prefix width for alignment

# ── Output prefix tagger ────────────────────────────

def _tag_line(line, force_tag=None):
    """Return (prefix, line) based on content matching."""
    if force_tag: return force_tag, line
    s = line.strip()
    if not s: return '', line
    if 'Hard-EM' in s or ('iter ' in s and 'change=' in s): return 'Hard-EM', line
    if 'Final ruled surfaces' in s: return 'Hard-EM', line
    if 'Partitions:' in s:        return 'Hard-EM', line
    if 'MergeTiny' in s:          return 'Merge', line
    if 'Boundary:' in s:         return 'Export', line
    if 'Laplacian' in s:         return 'Laplacian', line
    if 'Harmonic' in s:          return 'Harmonic', line
    if 'OCCT Split' in s or 'OCCT Mesh' in s or 'OCCT Labels' in s: return 'OCCT', line
    if 'Step 3' in s or 'Split]' in s: return 'OCCT', line
    if 'wrote ' in s:            return 'Export', line
    if 'Concave pass' in s or 'Concave split' in s: return 'Concave', line
    if '  part ' in s and ('cands' in s or 'regions' in s): return 'Concave', line
    if s.startswith('    r') and 'depth=' in s: return 'Concave', line
    if 'Partition part_' in s and 'loop=' in s: return 'Ruled', line
    if '    iter ' in s and 'loss=' in s: return 'Ruled', line
    if '=== ' in s and 'optimized ===' in s: return 'Ruled', line
    if '  part ' in s and 'maxDist=' in s: return 'Tolerance', line
    if '[Retry' in s:             return 'Retry', line
    if '[Tol]' in s:              return 'Tolerance', line
    if 'Done' in s and '===' in s: return 'Pipeline', line
    if 'Random surface seed' in s or 'Surface:' in s: return 'Init', line
    return '', line


class _ProcRunner(QThread):
    output_signal = pyqtSignal(str, str)    # (tag, line)
    finished_signal = pyqtSignal(int)

    def __init__(self, cmd, cwd, base_tag=''):
        super().__init__()
        self.cmd = cmd
        self.cwd = str(cwd)
        self.base_tag = base_tag

    def run(self):
        try:
            self._proc = subprocess.Popen(
                self.cmd, shell=True, cwd=self.cwd,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1
            )
            for line in iter(self._proc.stdout.readline, ''):
                tag, line = _tag_line(line.rstrip(), self.base_tag)
                self.output_signal.emit(tag, line)
            self._proc.wait()
            self.finished_signal.emit(self._proc.returncode)
        except Exception as e:
            self.output_signal.emit('Error', f"Error: {e}")
            self.finished_signal.emit(1)

    def terminate(self):
        if hasattr(self, '_proc') and self._proc:
            self._proc.terminate()


def load_boundaries(path):
    bnds = []
    if not os.path.exists(path): return bnds
    with open(path) as f:
        line = f.readline()
        while line:
            p = line.strip().split()
            if len(p) >= 2:
                n_pts, cid = int(p[0]), int(p[1])
                pts = []
                for _ in range(n_pts):
                    line = f.readline()
                    if not line: break
                    pts.append([float(x) for x in line.strip().split()])
                if pts: bnds.append((cid, np.array(pts)))
            line = f.readline()
    return bnds


class MainWindow(QMainWindow):
    # State machine constants
    EMPTY_CFG = {'mesh': False, 'finalBoundary': False, 'selectedIter': None, 'selectedSurfs': set()}
    DEFAULT_CFG = {'mesh': True, 'finalBoundary': True, 'selectedIter': None, 'selectedSurfs': set()}

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Distillation - NURBS Surface Partitioning")
        self.resize(1400, 900)
        self._export_dir = str(PROJECT_DIR / "results")
        self._proc = None; self._py_runner = None; self._poll_timer = None
        self._retry_idx = 0; self._active_retry = None
        self._item_actors = {}; self._loaded_paths = set()
        self._retry_nodes = {}; self._retry_configs = {}
        self._bnd_group = {}; self._ruled_group = {}
        self._ruled_names = {}
        self._suppress_check = False
        self._follow_mode = True
        self._latest_retry_id = None
        self._active_stage = None
        self._stage_progress = ""
        self._part_count = 0
        self._part_done = 0
        self._current_part_label = None

        self._setup_menu(); self._setup_ui(); self._setup_status()

    def _setup_menu(self):
        m = self.menuBar().addMenu("File")
        m.addAction("Import Data...", self._on_import_data)
        m.addAction("Set Export Dir...", self._on_set_export_dir)
        m.addAction("Clear Results", self._on_clear_results)
        m.addSeparator(); m.addAction("Exit", self.close)

    def _setup_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        ml = QVBoxLayout(central); ml.setContentsMargins(4,4,4,4)
        splitter = QSplitter(Qt.Horizontal)
        if HAS_PYVISTA:
            self._plotter = QtInteractor(self, shape=(1,1))
            self._plotter.set_background('lightblue')
            splitter.addWidget(self._plotter)
        else:
            splitter.addWidget(QLabel("pyvistaqt not installed"))
        right = QWidget(); rl = QVBoxLayout(right)
        rl.setContentsMargins(4,0,4,0)
        self._setup_params(rl); self._setup_tree(rl)
        splitter.addWidget(right); splitter.setSizes([900,500])
        ml.addWidget(splitter, 1)
        self._console = QTextEdit(); self._console.setReadOnly(True)
        self._console.setMaximumHeight(250)
        self._console.setStyleSheet(CONSOLE_CSS)
        self._console.setFont(QFont("Consolas", 16))
        ml.addWidget(self._console)

    def _setup_params(self, layout):
        form = QFormLayout(); form.setSpacing(6)
        self._cmb_surface = QComboBox()
        self._cmb_surface.addItems(["random","wavy","mountain"])
        form.addRow("Surface:", self._cmb_surface)
        self._spn_smooth = QSpinBox(); self._spn_smooth.setRange(-1,200)
        self._spn_smooth.setValue(1); self._spn_smooth.setSpecialValueText("auto")
        form.addRow("Smooth Iters:", self._spn_smooth)
        self._spn_sigma = QDoubleSpinBox(); self._spn_sigma.setRange(-1,10)
        self._spn_sigma.setValue(-1); self._spn_sigma.setDecimals(4)
        self._spn_sigma.setSpecialValueText("auto")
        form.addRow("Sigma:", self._spn_sigma)
        self._spn_tol = QDoubleSpinBox(); self._spn_tol.setRange(-1,100)
        self._spn_tol.setValue(-1); self._spn_tol.setDecimals(4)
        self._spn_tol.setSpecialValueText("no check")
        form.addRow("Tol Target:", self._spn_tol)
        self._spn_retries = QSpinBox(); self._spn_retries.setRange(1,10)
        self._spn_retries.setValue(5)
        form.addRow("Max Retries:", self._spn_retries)
        row = QHBoxLayout()
        self._txt_dir = QLineEdit(self._export_dir); row.addWidget(self._txt_dir)
        btn = QPushButton("..."); btn.setMaximumWidth(30)
        btn.clicked.connect(self._on_set_export_dir); row.addWidget(btn)
        form.addRow("Export Dir:", row)
        bar = QHBoxLayout()
        self._btn_run = QPushButton("Run Algorithm")
        self._btn_run.setStyleSheet(
            "QPushButton{background:#4CAF50;color:white;font-size:20px;padding:8px}")
        self._btn_run.clicked.connect(self._on_run); bar.addWidget(self._btn_run)
        self._btn_stop = QPushButton("Stop")
        self._btn_stop.setEnabled(False); self._set_stop_style(False)
        self._btn_stop.clicked.connect(self._stop); bar.addWidget(self._btn_stop)
        form.addRow(bar); layout.addLayout(form)
        self._load_config()

    def _set_stop_style(self, active):
        self._btn_stop.setStyleSheet(
            "QPushButton{background:#c33;color:white;font-size:20px;padding:8px}"
            if active else
            "QPushButton{background:#888;color:#ddd;font-size:20px;padding:8px}")

    def _setup_tree(self, layout):
        hdr = QHBoxLayout()
        hdr.addWidget(QLabel("Visualization"))
        hdr.addStretch()
        self._btn_follow = QPushButton("Resume Follow")
        self._btn_follow.setMaximumHeight(22)
        self._btn_follow.setStyleSheet("font-size:10px;padding:1px 6px;")
        self._btn_follow.clicked.connect(self._on_enable_follow)
        self._btn_follow.setVisible(False)
        hdr.addWidget(self._btn_follow)
        layout.addLayout(hdr)
        self._tree = QTreeWidget(); self._tree.setHeaderHidden(True)
        self._tree.itemChanged.connect(self._on_check)
        layout.addWidget(self._tree)

    # ── Tree building ──────────────────────────────

    def _add_retry_node(self, label):
        node = QTreeWidgetItem([label])
        node.setFlags(node.flags() | Qt.ItemIsUserCheckable)
        node.setExpanded(True)
        node.setData(1, Qt.UserRole, label)
        self._tree.addTopLevelItem(node)
        self._retry_nodes[label] = node
        self._item_actors[id(node)] = []
        self._retry_configs[label] = copy.deepcopy(self.EMPTY_CFG)
        self._latest_retry_id = label
        return node

    def _set_node_checked(self, node, checked):
        """Set check state without triggering _on_check."""
        self._suppress_check = True
        node.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)
        self._suppress_check = False

    def _new_data_handler(self, label):
        """State machine: NEW_DATA event."""
        self._retry_configs[label] = copy.deepcopy(self.EMPTY_CFG)
        self._latest_retry_id = label

        if self._follow_mode:
            old = self._active_retry
            if old and old in self._retry_configs:
                self._retry_configs[old] = copy.deepcopy(self.EMPTY_CFG)
            self._retry_configs[label] = copy.deepcopy(self.DEFAULT_CFG)
            self._sync_retry_checkboxes(label)
            self._active_retry = label
            self._rebuild_3d()
            # Config applied when children arrive via _on_file_written

    def _apply_retry_config(self, rlabel):
        """Push config dict to tree items for one retry."""
        cfg = self._retry_configs.get(rlabel)
        if not cfg: return
        rnode = self._retry_nodes.get(rlabel)
        if not rnode: return
        for i in range(rnode.childCount()):
            child = rnode.child(i)
            ct = child.text(0)
            if ct == 'Mesh':
                self._set_node_checked(child, cfg['mesh'])
            elif ct == 'Final boundaries':
                self._set_node_checked(child, cfg['finalBoundary'])
            elif ct == 'Boundary iterations':
                has_iter = cfg['selectedIter'] is not None
                self._set_node_checked(child, has_iter)
                for j in range(child.childCount()):
                    gc = child.child(j)
                    self._set_node_checked(gc, gc.text(0) == cfg.get('selectedIter', ''))
            elif ct == 'Ruled surfaces':
                sel = cfg.get('selectedSurfs', set())
                all_checked = True
                for j in range(child.childCount()):
                    gc = child.child(j)
                    chk = gc.text(0) in sel
                    self._set_node_checked(gc, chk)
                    if not chk: all_checked = False
                self._set_node_checked(child, len(sel) > 0)

    def _sync_retry_checkboxes(self, active_label):
        """Only one retry node checked."""
        for i in range(self._tree.topLevelItemCount()):
            t = self._tree.topLevelItem(i)
            self._set_node_checked(t, t.data(1, Qt.UserRole) == active_label)

    def _get_group(self, rlabel, gname, cache):
        if rlabel in cache: return cache[rlabel]
        p = self._retry_nodes.get(rlabel)
        if not p: return None
        n = QTreeWidgetItem([gname])
        n.setFlags(n.flags() | Qt.ItemIsUserCheckable)
        n.setExpanded(True)
        p.addChild(n); self._item_actors[id(n)] = []; cache[rlabel] = n
        return n

    def _add_leaf(self, parent, label, path, tag, idx):
        item = QTreeWidgetItem([label])
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setData(1, Qt.UserRole, path)
        item.setData(2, Qt.UserRole, tag)
        item.setData(3, Qt.UserRole, idx)
        parent.addChild(item); parent.setExpanded(True)
        self._item_actors[id(item)] = []
        return item

    def _setup_status(self):
        self._status = self.statusBar(); self._status.showMessage("Ready")

    def _set_status(self, msg):
        self._status.showMessage(msg)

    def _load_config(self):
        try:
            with open(CONFIG_PATH) as f:
                cfg = json.load(f)
            self._cmb_surface.setCurrentText(cfg.get("surface","random"))
            self._spn_smooth.setValue(cfg.get("smooth_iters",1))
            self._spn_sigma.setValue(cfg.get("sigma",-1))
            self._spn_tol.setValue(cfg.get("tol_target",-1))
            self._spn_retries.setValue(cfg.get("max_retries",5))
            self._txt_dir.setText(cfg.get("export_dir",self._export_dir))
            self._export_dir = self._txt_dir.text()
        except: pass

    def _save_config(self):
        cfg = {"surface":self._cmb_surface.currentText(),
               "smooth_iters":self._spn_smooth.value(),
               "sigma":self._spn_sigma.value(),
               "tol_target":self._spn_tol.value(),
               "max_retries":self._spn_retries.value(),
               "export_dir":self._txt_dir.text()}
        try:
            with open(CONFIG_PATH,'w') as f: json.dump(cfg,f,indent=2)
        except: pass

    # ── Menu actions ───────────────────────────────

    def _on_set_export_dir(self):
        d = QFileDialog.getExistingDirectory(self,"Export Dir",self._export_dir)
        if d: self._export_dir = d; self._txt_dir.setText(d)

    def _on_import_data(self):
        d = QFileDialog.getExistingDirectory(self, "Import Results Directory", self._export_dir)
        if not d: return
        d = os.path.normpath(d)
        self._export_dir = d
        self._txt_dir.setText(d)
        self._tree.clear(); self._item_actors.clear(); self._loaded_paths.clear()
        self._retry_nodes.clear(); self._bnd_group.clear(); self._ruled_group.clear()
        self._ruled_names.clear(); self._retry_configs.clear()
        self._clear_3d(); self._console.clear()
        self._log("GUI", f"Importing data from: {d}", "")

        # Scan for retry directories
        retry_dirs = []
        for fn in sorted(os.listdir(d)):
            p = os.path.join(d, fn)
            if os.path.isdir(p) and re.match(r'retry_\d+', fn):
                retry_dirs.append((int(fn.replace('retry_','')), fn))
        if not retry_dirs:
            # No retry dirs found, treat the directory itself as retry_0
            self._retry_idx = 0
            label = "Retry 0"
            self._add_retry_node(label)
            self._new_data_handler_import(label)
            self._import_dir_files(d, label)
        else:
            retry_dirs.sort()
            self._retry_idx = retry_dirs[-1][0]
            for ridx, rfn in retry_dirs:
                label = f"Retry {ridx}"
                self._add_retry_node(label)
                self._new_data_handler_import(label)
                self._import_dir_files(os.path.join(d, rfn), label)
            # Activate the latest retry
            latest = f"Retry {retry_dirs[-1][0]}"
            self._active_retry = latest
            self._retry_configs[latest] = copy.deepcopy(self.DEFAULT_CFG)
            self._sync_retry_checkboxes(latest)

        self._rebuild_3d()
        self._log("GUI", "Import complete.", "")

    def _new_data_handler_import(self, label):
        """Like _new_data_handler but exits follow mode on import."""
        self._retry_configs[label] = copy.deepcopy(self.EMPTY_CFG)
        self._latest_retry_id = label
        # Don't auto-activate; let the caller decide

    def _import_dir_files(self, dirpath, rlabel):
        """Load all visualization files from one retry directory."""
        for fn in sorted(os.listdir(dirpath)):
            p = os.path.normpath(os.path.join(dirpath, fn))
            self._on_file_written(p)

    def _on_clear_results(self):
        import shutil
        d = Path(self._export_dir)
        for sub in d.glob("retry_*"): shutil.rmtree(str(sub))
        for f in d.glob("*"): 
            if f.is_file(): f.unlink()
        self._clear_3d(); self._tree.clear()
        self._item_actors.clear(); self._loaded_paths.clear()
        self._retry_nodes.clear(); self._bnd_group.clear(); self._ruled_group.clear()
        self._ruled_names.clear()
        self._retry_idx = 0; self._active_retry = None; self._follow_mode = True
        self._active_stage = None; self._stage_progress = ""
        self._part_count = 0; self._part_done = 0
        self._current_part_label = None
        self._console.clear(); self._log("", "Results cleared.", "")

    # ── Run / Stop ─────────────────────────────────

    def _on_run(self):
        if self._proc and self._proc.isRunning():
            QMessageBox.information(self, "Running", "Already running."); return
        self._export_dir = self._txt_dir.text()
        Path(self._export_dir).mkdir(parents=True, exist_ok=True)
        self._tree.clear(); self._item_actors.clear(); self._loaded_paths.clear()
        self._retry_nodes.clear(); self._bnd_group.clear(); self._ruled_group.clear()
        self._ruled_names.clear()
        self._retry_idx = 0; self._active_retry = None; self._follow_mode = True
        self._active_stage = None; self._stage_progress = ""
        self._part_count = 0; self._part_done = 0
        self._current_part_label = None
        self._clear_3d(); self._console.clear()
        self._log("Pipeline", "=== Starting distillation pipeline ===", "")
        # Create initial Retry 0 and activate it
        label = "Retry 0"
        self._add_retry_node(label)
        self._new_data_handler(label)

        parts = [
            f'"{BUILD_EXE}"',
            f'--surface={self._cmb_surface.currentText()}',
            f'--export-dir="{self._export_dir}"',
        ]
        si = self._spn_smooth.value()
        if si > 0: parts.append(f'--smooth-iters={si}')
        s = self._spn_sigma.value()
        if s > 0: parts.append(f'--sigma={s:.4f}')
        t = self._spn_tol.value()
        if t > 0: parts.append(f'--tol-target={t:.4f}')
        parts.append(f'--max-retries={self._spn_retries.value()}')
        cmd = ' '.join(parts)
        self._log("Pipeline", f"Running: {cmd}", "")

        self._btn_run.setEnabled(False)
        self._btn_stop.setEnabled(True); self._set_stop_style(True)

        self._proc = _ProcRunner(cmd, PROJECT_DIR)
        self._proc.output_signal.connect(self._on_output)
        self._proc.finished_signal.connect(self._on_cpp_done)
        self._proc.start()
        self._set_status("Running: Pipeline")

        self._poll_timer = QTimer()
        self._poll_timer.timeout.connect(self._poll_files)
        self._poll_timer.start(1500)
        self._save_config()

    def _stop(self):
        for proc_attr in ['_proc', '_py_runner']:
            runner = getattr(self, proc_attr, None)
            if runner:
                try:
                    pid = runner._proc.pid
                    # Kill entire process tree via taskkill
                    subprocess.run(
                        f'taskkill /F /T /PID {pid}',
                        shell=True, capture_output=True)
                    # Fallback: kill child python processes via wmic
                    subprocess.run(
                        f'wmic process where "ParentProcessId={pid}" delete 2>nul',
                        shell=True, capture_output=True)
                except: pass
                runner.wait(1000)
        self._proc = None; self._py_runner = None
        if self._poll_timer: self._poll_timer.stop()
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False); self._set_stop_style(False)
        self._set_status("Stopped")
        self._log("GUI", "Stopped.", "")

    # ── Output logging with aligned tags ────────────

    def _on_output(self, tag, line):
        """Log a line with its detected tag."""
        self._log(tag, line, "")

    def _log(self, tag, text, extra):
        """Add timestamped line with aligned prefix."""
        ts = datetime.now().strftime("[%H:%M:%S] ")
        prefix = f"<{tag:<{_TAG_W-2}}>" if tag else " " * _TAG_W
        self._console.append(ts + prefix + text)
        c = self._console.textCursor(); c.movePosition(QTextCursor.End)
        self._console.setTextCursor(c)
        # Update status bar based on tag + tree stage label
        if tag in ('Hard-EM','Merge','Laplacian','Harmonic','OCCT'):
            self._set_status(f"Running: {tag}")
            self._active_stage = tag
            self._update_retry_stage_label(self._active_retry, tag)
        elif tag == 'Concave':
            self._set_status("Running: Concave split")
            self._active_stage = 'Concave'
            self._update_retry_stage_label(self._active_retry, 'Concave')
        elif tag == 'Ruled':
            self._set_status("Running: Ruled fitting")
            self._active_stage = 'Ruled'
        elif tag == 'Tolerance':
            self._set_status("Running: Tolerance check")
            self._active_stage = 'Tolerance'
        elif tag == 'Retry':
            self._set_status("Running: Pipeline (retry)")
            self._active_stage = 'Retry'
        elif tag == 'Pipeline' and 'complete' in text.lower():
            self._update_retry_stage_label(self._active_retry, "Complete")
        # parse for retries and file writes
        if text:
            self._parse_line(text)

    def _parse_line(self, line):
        pe = PIPELINE_EVENT_RE.search(line)
        if pe:
            self._on_pipeline_event(pe.group(1), pe.group(2))
            return

        m = re.search(r'\[Retry\s+(\d+)\]', line)
        if m:
            n = int(m.group(1))
            if n > self._retry_idx:
                self._retry_idx = n
                label = f"Retry {n}"
                if label not in self._retry_nodes:
                    self._add_retry_node(label)
                self._new_data_handler(label)
            self._active_stage = 'Retry'
            self._update_retry_stage_label(f"Retry {n}", "Retry")
            return

        m2 = re.search(r'\[Partition\s+(part_\S+)\]', line)
        if m2:
            self._current_part_label = m2.group(1)

        if 'wrote' in line.lower():
            m = re.search(r'wrote\s+(\S+)', line)
            if m: self._on_file_written(m.group(1))

    def _on_pipeline_event(self, ptype, pdata):
        if ptype == 'retry':
            try:
                n = int(pdata)
            except ValueError:
                return
            label = f"Retry {n}"
            if label not in self._retry_nodes:
                self._add_retry_node(label)
            self._new_data_handler(label)
            self._retry_idx = max(self._retry_idx, n)
            self._active_stage = 'Retry'
            self._part_count = 0
            self._part_done = 0
            self._update_retry_stage_label(label, "Pipeline start")

        elif ptype == 'stage':
            stage = pdata.strip()
            self._active_stage = stage
            self._stage_progress = ""
            active = self._active_retry
            if stage in ('Hard-EM', 'Merge', 'Laplacian', 'Harmonic', 'OCCT'):
                self._set_status(f"Running: {stage}")
            self._update_retry_stage_label(active, stage)

        elif ptype == 'progress':
            self._stage_progress = pdata.strip()
            active = self._active_retry
            stage = self._active_stage or ""
            label = f"{stage} {self._stage_progress}" if self._stage_progress else stage
            self._update_retry_stage_label(active, label)

        elif ptype == 'file':
            parts = pdata.split(':', 1)
            if len(parts) >= 2:
                self._on_file_written(parts[1].strip())
            elif len(parts) == 1:
                self._on_file_written(parts[0].strip())

        elif ptype == 'done':
            parts = pdata.split(':', 1)
            sub = parts[0].strip() if parts else ""
            if sub == 'part':
                self._part_done += 1
                detail = parts[1] if len(parts) > 1 else ""
                m = re.search(r'maxDist=([\d.]+)', detail)
                md = m.group(1) if m else ""
                info = f"{self._part_done}/{self._part_count}"
                if md:
                    info += f" max={md}"
                self._update_retry_stage_label(
                    self._active_retry, f"Ruled [{info}]")
            elif sub == 'all':
                self._update_retry_stage_label(
                    self._active_retry, "Complete")
            elif sub == 'count':
                try:
                    self._part_count = int(parts[1])
                except (ValueError, IndexError):
                    pass
            elif sub == 'candidates':
                try:
                    self._part_count = int(parts[1])
                except (ValueError, IndexError):
                    pass

    def _update_retry_stage_label(self, rlabel, stage_text):
        if not rlabel:
            return
        rnode = self._retry_nodes.get(rlabel)
        if not rnode:
            return
        if stage_text:
            rnode.setText(0, f"{rlabel}  [{stage_text}]")
        else:
            rnode.setText(0, rlabel)

    def _on_file_written(self, path):
        path = os.path.normpath(path)
        if path in self._loaded_paths:
            return
        self._loaded_paths.add(path)
        fn = os.path.basename(path)
        tag = None
        if fn == 'mesh.obj': tag = 'mesh'
        elif fn == 'boundaries.txt': tag = 'boundary'
        elif 'boundaries_iter_' in fn and fn.endswith('.txt'): tag = 'boundary'
        elif 'ruled_surf_' in fn and fn.endswith('.obj'): tag = 'ruled'
        elif fn == 'tolerance.txt': tag = 'tolerance'
        else: return

        m = re.search(r'retry_(\d+)', path)
        rn = int(m.group(1)) if m else self._retry_idx
        rlabel = f"Retry {rn}"
        is_new_retry = rlabel not in self._retry_nodes
        if is_new_retry:
            self._add_retry_node(rlabel)
            self._new_data_handler(rlabel)
        rnode = self._retry_nodes[rlabel]
        cfg = self._retry_configs.get(rlabel, {})
        idx = rn; leaf = None; it = 0

        if tag == 'mesh':
            leaf = self._add_leaf(rnode, "Mesh", path, tag, rn)
            self._set_node_checked(leaf, cfg.get('mesh', False))
            idx = rn
        elif tag == 'boundary':
            if fn == 'boundaries.txt':
                leaf = self._add_leaf(rnode, "Final boundaries", path, tag, rn)
                self._set_node_checked(leaf, cfg.get('finalBoundary', False))
                idx = rn
            else:
                try: it = int(fn.replace('boundaries_iter_','').replace('.txt',''))
                except: it = 0
                grp = self._get_group(rlabel, "Boundary iterations", self._bnd_group)
                if grp:
                    lbl = f"iter {it:03d}"
                    leaf = self._add_leaf(grp, lbl, path, tag, rn*1000+it)
                    self._set_node_checked(leaf, cfg.get('selectedIter') == lbl)
                    # Update group parent state
                    has_sel = cfg.get('selectedIter') is not None
                    self._set_node_checked(grp, has_sel)
                    idx = rn*1000+it
        elif tag == 'ruled':
            try: sidx = int(fn.replace('ruled_surf_','').replace('.obj',''))
            except: sidx = 0
            if hasattr(self, '_current_part_label') and self._current_part_label:
                self._ruled_names[sidx] = self._current_part_label
            lbl = self._ruled_names.get(sidx, f"surf {sidx}")
            grp = self._get_group(rlabel, "Ruled surfaces", self._ruled_group)
            if grp:
                leaf = self._add_leaf(grp, lbl, path, tag, sidx)
                surfs = cfg.setdefault('selectedSurfs', set())
                if self._follow_mode:
                    surfs.add(lbl)
                self._set_node_checked(leaf, lbl in surfs)
                self._set_node_checked(grp, len(surfs) > 0)
                idx = sidx
            if self._follow_mode and self._part_count > 0:
                done = len(cfg.get('selectedSurfs', set()))
                self._update_retry_stage_label(rlabel,
                    f"Ruled [{done}/{self._part_count}]")
        elif tag == 'tolerance':
            pass

        if rlabel == self._active_retry and leaf:
            names = self._load_path(path, tag, idx)
            self._item_actors[id(leaf)] = names
            self._apply_visibility()

    # ── File polling ──────────────────────────────

    def _poll_files(self):
        for n in range(self._retry_idx + 5):
            d = os.path.join(self._export_dir, f"retry_{n}")
            if not os.path.isdir(d): continue
            rlabel = f"Retry {n}"
            if rlabel not in self._retry_nodes:
                self._add_retry_node(rlabel)
            for fn in sorted(os.listdir(d)):
                p = os.path.normpath(os.path.join(d, fn))
                if p not in self._loaded_paths:
                    self._on_file_written(p)

    # ── C++ done → post-processing ────────────────

    def _on_cpp_done(self, code):
        self._poll_files()
        if code != 0:
            self._log("Pipeline", f"Failed (exit={code})", "")
            self._finish_run(); return

        self._log("Pipeline", "=== C++ pipeline complete ===", "")

        tol_target = self._spn_tol.value()
        if tol_target > 0:
            self._log("GUI", "Tolerance mode: post-processing skipped (C++ handles it)", "")
            self._update_retry_stage_label(self._active_retry, "Complete")
            self._finish_run()
            return

        # No tolerance → run fit_ruled_grad directly
        retry_dir = self._export_dir
        for n in range(self._retry_idx, -1, -1):
            d = os.path.join(self._export_dir, f"retry_{n}")
            if os.path.isdir(d): retry_dir = d; break
        self._log("Ruled", f"Starting ruled fitting on {retry_dir}", "")
        cmd = f'python "{SCRIPT_DIR/"fit_ruled_grad.py"}" "{retry_dir}" --max-iter 8'
        self._py_runner = _ProcRunner(cmd, PROJECT_DIR, 'Ruled')
        self._py_runner.output_signal.connect(self._on_output)
        self._py_runner.finished_signal.connect(self._on_py_done)
        self._py_runner.start()

    def _on_py_done(self, code):
        self._poll_files()
        self._log("Ruled", "=== Python optimizer complete ===", "")
        self._update_retry_stage_label(self._active_retry, "Complete")
        self._finish_run()

    def _finish_run(self):
        if self._poll_timer: self._poll_timer.stop()
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False); self._set_stop_style(False)
        self._active_stage = None
        self._set_status("Finished")

    # ── State machine event handlers ──────────────

    def _on_enable_follow(self):
        """ENABLE_FOLLOW event: reset to latest retry with default config."""
        self._follow_mode = True
        self._btn_follow.setVisible(False)
        lid = self._latest_retry_id
        if not lid: return
        if self._active_retry and self._active_retry in self._retry_configs:
            self._retry_configs[self._active_retry] = copy.deepcopy(self.EMPTY_CFG)
        self._retry_configs[lid] = copy.deepcopy(self.DEFAULT_CFG)
        self._apply_retry_config(lid)
        self._sync_retry_checkboxes(lid)
        self._active_retry = lid
        self._rebuild_3d()

    def _on_check(self, item, col):
        """State machine TOGGLE_NODE event."""
        if self._suppress_check: return
        txt = item.text(0)
        parent = item.parent()
        rnode = self._find_retry_parent(item)
        if not rnode: return
        rlabel = rnode.data(1, Qt.UserRole)
        checked = item.checkState(0) == Qt.Checked

        # If operating on a non-active retry, first switch
        if rlabel != self._active_retry:
            self._handle_switch_retry(rlabel)
            # After switch, our item may have been reset; re-read state
            checked = item.checkState(0) == Qt.Checked

        cfg = self._retry_configs.get(rlabel)
        if not cfg: return

        parent = item.parent()
        pname = parent.text(0) if parent else ""

        if txt.startswith("Retry "):
            if checked:
                self._handle_switch_retry(rlabel)
            else:
                self._set_node_checked(item, checked)
            self._apply_visibility()
            return

        # Mesh
        if txt == 'Mesh':
            cfg['mesh'] = checked
        # Final boundaries
        elif txt == 'Final boundaries':
            cfg['finalBoundary'] = checked
            if checked:
                cfg['selectedIter'] = None
                self._apply_retry_config(rlabel)
        # Boundary iteration child
        elif pname == "Boundary iterations":
            if checked:
                cfg['finalBoundary'] = False
                cfg['selectedIter'] = txt
                self._apply_retry_config(rlabel)
            else:
                if cfg['selectedIter'] == txt:
                    cfg['selectedIter'] = None
        # Boundary iterations group parent
        elif txt == "Boundary iterations":
            if not checked:
                cfg['selectedIter'] = None
                self._apply_retry_config(rlabel)
        # Ruled surface child
        elif pname == "Ruled surfaces":
            sel = cfg.setdefault('selectedSurfs', set())
            if checked: sel.add(txt)
            else: sel.discard(txt)
        # Ruled surfaces group parent
        elif txt == "Ruled surfaces":
            if not checked:
                cfg.setdefault('selectedSurfs', set()).clear()
                self._apply_retry_config(rlabel)

        self._follow_mode = False
        self._btn_follow.setVisible(True)
        self._apply_visibility()

    def _find_retry_parent(self, item):
        """Walk up tree to find the Retry node containing this item."""
        while item:
            if item.text(0).startswith("Retry "):
                return item
            item = item.parent()
        return None

    def _handle_switch_retry(self, target_label):
        """SWITCH_RETRY event: copy old config to new retry, clear old."""
        old_label = self._active_retry
        if old_label == target_label: return

        # Copy old config to target
        if old_label and old_label in self._retry_configs:
            src_cfg = copy.deepcopy(self._retry_configs[old_label])
            self._retry_configs[old_label] = copy.deepcopy(self.EMPTY_CFG)
        else:
            src_cfg = copy.deepcopy(self.EMPTY_CFG)
        self._retry_configs[target_label] = src_cfg
        self._apply_retry_config(target_label)

        # Update retry checkboxes
        self._sync_retry_checkboxes(target_label)
        self._active_retry = target_label
        self._follow_mode = False
        self._btn_follow.setVisible(True)
        self._rebuild_3d()

    def _apply_visibility(self):
        if not HAS_PYVISTA: return
        # Collect which actors should be visible
        visible_set = set()
        def walk(node, inherited_vis):
            vis = inherited_vis and node.checkState(0) == Qt.Checked
            for aname in self._item_actors.get(id(node), []):
                if vis: visible_set.add(aname)
            for i in range(node.childCount()):
                walk(node.child(i), vis)
        for i in range(self._tree.topLevelItemCount()):
            top = self._tree.topLevelItem(i)
            walk(top, top.checkState(0) == Qt.Checked)

        # Apply all at once
        try:
            for a in self._plotter.renderer._actors:
                if hasattr(a, '_name'):
                    a.SetVisibility(a._name in visible_set)
            self._plotter.render()
        except: pass

    def _set_actor_visible(self, name, vis):
        if not HAS_PYVISTA: return
        try:
            for a in self._plotter.renderer._actors:
                if hasattr(a, '_name') and a._name == name:
                    a.SetVisibility(vis)
        except: pass

    # ── Retry switch rebuild ───────────────────────

    def _rebuild_3d(self):
        self._clear_3d()
        if HAS_PYVISTA:
            self._plotter.disable_render = True
        try:
            rlabel = self._active_retry
            if not rlabel: return
            m = re.search(r'Retry\s+(\d+)', rlabel)
            if not m: return
            n = int(m.group(1))
            d = os.path.join(self._export_dir, f"retry_{n}")
            if not os.path.isdir(d): return
            for fn in sorted(os.listdir(d)):
                p = os.path.normpath(os.path.join(d, fn))
                tag = None; idx = n
                if fn == 'mesh.obj' and 'original' not in fn and 'recon' not in fn:
                    tag = 'mesh'
                elif fn == 'boundaries.txt':
                    tag = 'boundary'
                elif 'boundaries_iter_' in fn and fn.endswith('.txt'):
                    tag = 'boundary'
                    try: idx=n*1000+int(fn.replace('boundaries_iter_','').replace('.txt',''))
                    except: pass
                elif 'ruled_surf_' in fn and fn.endswith('.obj'):
                    tag = 'ruled'
                    try: idx=int(fn.replace('ruled_surf_','').replace('.obj',''))
                    except: pass
                if tag is None: continue
                names = self._load_path(p, tag, idx)
                rnode = self._retry_nodes.get(rlabel)
                if rnode:
                    for tnode in self._find_tree_items(rnode, p):
                        self._item_actors[id(tnode)] = names
                self._loaded_paths.add(p)
        finally:
            if HAS_PYVISTA:
                self._plotter.disable_render = False
        self._apply_visibility()

    def _find_tree_items(self, root, path):
        """Find all tree items under root that reference this path."""
        found = []
        def walk(node):
            if node.data(1, Qt.UserRole) == path:
                found.append(node)
            for i in range(node.childCount()):
                walk(node.child(i))
        walk(root)
        return found

    # ── 3D View ────────────────────────────────────

    def _clear_3d(self):
        if HAS_PYVISTA:
            self._plotter.clear()
            self._plotter.set_background('lightblue')

    def _load_path(self, path, tag, idx=0, render_now=True):
        if not HAS_PYVISTA: return []
        names = []
        try:
            import pyvista as pv
            if tag == 'mesh':
                m = pv.read(path)
                lp = os.path.join(os.path.dirname(path), "partition_labels.txt")
                if os.path.exists(lp):
                    labels = np.loadtxt(lp, dtype=int)
                    nc = m.n_cells; fcols = np.full((nc,3),[0.7]*3)
                    for i in range(nc):
                        try:
                            c=m.get_cell(i); pts=c.point_ids
                            if len(pts)==3:
                                ps={labels[p] for p in pts if 0<=p<len(labels)}; ps.discard(-1)
                                if len(ps)==1: fcols[i]=TAB10_RGB[ps.pop()%10]
                                elif len(ps)>1: fcols[i]=np.mean([TAB10_RGB[p%10] for p in ps],axis=0)
                        except: pass
                    m.cell_data['rgb']=fcols
                    nm=f"mesh_{idx}"
                    self._plotter.add_mesh(m,scalars='rgb',rgb=True,name=nm,opacity=0.85,show_edges=True,edge_color='gray')
                else:
                    nm=f"mesh_{idx}"
                    self._plotter.add_mesh(m,name=nm,opacity=0.85,show_edges=True,edge_color='gray')
                names.append(nm)
            elif tag == 'boundary':
                bnds = load_boundaries(path)
                for j,(cid,pts) in enumerate(bnds):
                    if len(pts)<2: continue
                    nm = f"bnd_{idx}_{j}"
                    c = pv.PolyData(); c.points = pts
                    n = len(pts); c.lines = np.array([n]+list(range(n)),dtype=np.int64)
                    self._plotter.add_mesh(c,name=nm,color=TAB10_RGB[cid%10],line_width=4,render_lines_as_tubes=True)
                    names.append(nm)
            elif tag == 'ruled':
                s = pv.read(path)
                nm = f"ruled_{idx}"
                self._plotter.add_mesh(s,name=nm,color=TAB10_RGB[idx%10],opacity=0.4,show_edges=True,edge_color='gray')
                names.append(nm)
            if render_now:
                self._plotter.render()
        except Exception as e:
            self._log("GUI", f"load error: {os.path.basename(path)}: {e}", "")
        return names

    def closeEvent(self, event):
        self._stop(); event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow(); w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
