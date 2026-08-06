# NURBS Surface Partitioning + Boundary Smoothing + Ruled Fitting

## 环境

- Windows 10/11 x64, VS2022, CMake ≥ 3.16
- Python 3.8+, PyTorch, pyvista, numpy, pyvistaqt

```powershell
pip install torch pyvista numpy pyvistaqt python-docx
```

## GUI 前端

```powershell
cd OCCT
python .\python\gui_main.py
```

功能：
- 左侧参数面板：曲面类型、σ、平滑次数、容差目标、L-BFGS 参数
- 中间 3D 视口：实时同步 `./results` 中的网格和边界
- 右侧步骤树：Hard-EM → 凹分割 → 拉普拉斯 → 调和更新 → 直纹面拟合 → 容差
- 底部命令行：算法输出流式显示
- 预览曲面：生成预览并自动加载 3D 视图

## 构建

```powershell
cd OCCT
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

## 工作流

**所有命令必须在 `OCCT` 目录下执行**，数据目录统一用 `./results`。

### 基础运行

```powershell
cd OCCT
# 1. C++ 管线 (分区 + 平滑 + 调和更新 + 导出)
.\build\Release\distillation.exe --surface=random --export-dir=./results

# 2. 直纹面构建 (梯度优化)
python .\python\fit_ruled_grad.py .\results --max-iter 10

# 3. 可视化 (左: 网格+边界折线 | 右: 直纹面)
python .\python\visualize.py .\results
```

### 容差驱动迭代 (自动调 σ + K)

```powershell
cd OCCT
.\build\Release\distillation.exe --surface=random --export-dir=./results --tol-target 0.3 --smooth-iters=1
```

迭代输出示例：
```
[Tol] running: python fit_ruled_grad.py ...
[Tol] 6/8 exceed 0.3, sigma->0.026
[Retry 1] K=8 sigma=0.026
[Tol] 6/8 exceed 0.3, K->10
[Retry 2] K=10 sigma=0.026
...
```

迭代输出示例：
```
[Tol] running: python fit_ruled_grad.py ...
[Tol] 6/8 exceed 0.3, sigma->0.026
[Retry 1] K=8 sigma=0.026
[Tol] 6/8 exceed 0.3, K->10              ← σ 无效，增加分区数
[Retry 2] K=10 sigma=0.026
...
[Tol] all within 0.3, done
```

### 查看迭代过程

```powershell
python python\visualize.py .\results --iter 0   # 平滑前
python python\visualize.py .\results --iter 5   # 第5次迭代
```

## 参数

| 参数 | 说明 | 默认 |
|------|------|------|
| `--surface=<random\|wavy\|mountain>` | 曲面 | random |
| `--export-dir=<path>` | 导出目录 | ./results |
| `--sigma=<val>` | UV域平滑尺度 | 2×avgEdgeLen |
| `--smooth-iters=<n>` | 1=单次平滑 | K=ceil((2σ/h)²) |
| `--tol-target=<val>` | 容差目标，超标自动迭代 | 无(不迭代) |

## 诊断

```powershell
python python\diagnose_ruled.py .\results
```

## 导出文件

| 文件 | 内容 |
|------|------|
| `mesh.obj` | 调和更新后网格 |
| `boundaries.txt` | 平滑后边界折线 (3D) |
| `boundaries_iter_NNN.txt` | 迭代边界快照 |
| `part_N_loop.txt` | 分区 N 边界环 (3D) |
| `part_N_points.txt` | 分区 N 内部采样点 |
| `part_N_ruled.txt` | 分区 N 优化结果 |
| `ruled_surf_N.obj` | 分区 N 直纹面 (80×32 quad) |
| `tolerance.txt` | 容差 (pid, t_P, t_Q, maxDist, rmsDist) |
| `smooth_history.txt` | 平滑历史 |
| `run_meta.txt` | 元数据 |

## 算法流程

```
Hard-EM 分区 (K 可调) → 极小区域合并
    → 凹检测+分割循环 (≤5 轮, 深宽比κ>0.08)
    → 图拉普拉斯平滑 → 调和网格更新 (Δv=0)
    → 边界分段 → 直纹面拟合 (L-BFGS)
    → 容差 = 直纹面点到分区内点的最大距离
    → [超标: σ×0.7, σ无效则 K+=2, 重跑]
```

## 管道命令

```powershell
cd OCCT

# 基础运行 (导出到 results/retry_0/)
.\build\Release\distillation.exe --surface=random --export-dir=./results --smooth-iters=1

# 直纹面拟合 (处理 retry_0 目录)
python .\python\fit_ruled_grad.py .\results\retry_0 --max-iter 10

# 可视化
python .\python\visualize.py .\results\retry_0

# GUI
python .\python\gui_main.py
```

## 容差驱动迭代

```powershell
.\build\Release\distillation.exe --surface=random --export-dir=./results --tol-target 0.3 --max-retries 5
# 每轮输出到 results/retry_0/, retry_1/ ...
# 容差超标 → σ×0.7 重跑; σ无效 → K+=2 重跑
```

## 容差定义

直纹面 80×32 网格采样 → 分区内部点到最近采样点的最大距离 = maxDist
