"""
多尺度凹区域分析（依据《凹区域多尺度分析》报告）

对拉普拉斯平滑后的闭合边界环（CCW 2D 多边形，边数多、边长近似）做完整的
多尺度凹性分析。覆盖报告的核心方法：

  §2.1.1 叉积顶点凹凸判别 (cross product)
  §2.1.2 离散转向角 / 曲率符号
  §2.2   凸包 + 凸包凹度 + 面积比
  §3.1.1 桥 (Bridge) 与口袋 (Pocket)
  §3.1.2 直线凹度 (SL-Concavity) 与最短路径凹度 (SP-Concavity)
  §3.1.3 归一化凹度 (depth^a / width^b，含对数阻尼)
  §3.2.1 多尺度邻域曲率
  §3.2.2 形状指数 (Shape Index) / 曲率度 (Curvedness) 的 2D 退化
  §3.2.3 转向角表示 (Turn Angle Representation, TAR)
  §4.1   近似凸分解 (ACD)：递归切分 + 自适应阈值
  §4.2   曲率尺度空间 (CSS)：TAR 多尺度平滑 + 特征生命周期
  §4.3   持久性同调 (PH)：以多尺度生命周期作为持久性代理（免重库）

所有函数作用于 numpy (n,2) 的 CCW 闭合多边形。主入口 analyze_boundary()。
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════
# 基础几何工具
# ═══════════════════════════════════════════════════════════════

def ensure_ccw(poly):
    """确保闭合多边形 CCW（有向面积 > 0）。"""
    poly = np.asarray(poly, dtype=np.float64)
    n = len(poly)
    if n < 3:
        return poly
    area2 = sum(poly[i][0] * poly[(i + 1) % n][1]
                - poly[i][1] * poly[(i + 1) % n][0] for i in range(n))
    return poly[::-1] if area2 < 0 else poly


def cross2d(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull_2d(pts):
    """2D 凸包（Andrew 单调链），返回 CCW 顶点索引。"""
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    if n < 3:
        return list(range(n))
    idx = sorted(range(n), key=lambda i: (pts[i][0], pts[i][1]))
    hull = []
    for i in range(n):
        while len(hull) >= 2 and cross2d(pts[hull[-2]], pts[hull[-1]], pts[idx[i]]) <= 0:
            hull.pop()
        hull.append(idx[i])
    lower = len(hull)
    for i in range(n - 2, -1, -1):
        while len(hull) > lower and cross2d(pts[hull[-2]], pts[hull[-1]], pts[idx[i]]) <= 0:
            hull.pop()
        hull.append(idx[i])
    if len(hull) > 1:
        hull.pop()
    return hull


def point_to_line_dist(p, a, b):
    """点到线段的最小距离（SL 凹度的基础）。"""
    p, a, b = np.asarray(p), np.asarray(a), np.asarray(b)
    ab = b - a
    l2 = ab @ ab
    if l2 < 1e-12:
        return float(np.linalg.norm(p - a))
    t = max(0.0, min(1.0, (p - a) @ ab / l2))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def polygon_area(poly):
    """闭合多边形绝对面积。"""
    poly = np.asarray(poly, dtype=np.float64)
    n = len(poly)
    a = 0.0
    for i in range(n):
        j = (i + 1) % n
        a += poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1]
    return abs(a) / 2.0


def convex_hull_area(poly):
    hull = convex_hull_2d(poly)
    return polygon_area([poly[i] for i in hull])


# ═══════════════════════════════════════════════════════════════
# §2.1.1 叉积顶点凹凸判别
# ═══════════════════════════════════════════════════════════════

def classify_vertices(poly):
    """叉积顶点凹凸判别（CCW 闭合多边形）。
    返回 list: +1 凸 (左转), -1 凹 (右转), 0 共线。"""
    poly = np.asarray(poly, dtype=np.float64)
    n = len(poly)
    cls = [0] * n
    if n < 3:
        return cls
    mean_len_sq = sum(np.linalg.norm(poly[(i + 1) % n] - poly[i]) ** 2 for i in range(n)) / n
    eps = mean_len_sq * 1e-12
    for i in range(n):
        p0, p1, p2 = poly[(i - 1) % n], poly[i], poly[(i + 1) % n]
        cr = (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0])
        if cr > eps:
            cls[i] = 1
        elif cr < -eps:
            cls[i] = -1
        else:
            cls[i] = 0
    return cls


def concave_runs(cls):
    """凹顶点(-1)的连续段（环状感知）。返回 [(start, end), ...]（闭区间）。"""
    n = len(cls)
    if not any(c < 0 for c in cls):
        return []
    runs = []
    start = None
    for i in range(n):
        if cls[i] < 0:
            if start is None:
                start = i
        elif start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, n - 1))
    # 合并跨环段
    if len(runs) >= 2 and runs[0][0] == 0 and runs[-1][1] == n - 1:
        runs = [(runs[-1][0], runs[0][1])] + runs[1:-1]
    return runs


def run_indices(s, e, n):
    """凹段 (s, e) 的顶点索引列表（环状展开）。"""
    if s <= e:
        return list(range(s, e + 1))
    return list(range(s, n)) + list(range(0, e + 1))


# ═══════════════════════════════════════════════════════════════
# §2.1.2 / §3.2.3 转向角表示 (TAR)
# ═══════════════════════════════════════════════════════════════

def turn_angles(poly):
    """每顶点转向角（弧度，带符号）。CCW：正=左转(凸)，负=右转(凹)。
    TAR 曲线上波谷（负值）对应凹顶点，波谷深度量化凹陷尖锐度。"""
    poly = np.asarray(poly, dtype=np.float64)
    n = len(poly)
    if n < 3:
        return np.zeros(n)
    e = poly - np.roll(poly, 1, axis=0)      # 入边 i-1 -> i
    f = np.roll(poly, -1, axis=0) - poly     # 出边 i -> i+1
    cross = e[:, 0] * f[:, 1] - e[:, 1] * f[:, 0]
    dot = np.einsum('ij,ij->i', e, f)
    return np.arctan2(cross, dot)


def gaussian_smooth_1d(x, sigma):
    """环状一维高斯平滑（CSS §4.2 的核心算子）。"""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if sigma <= 0 or n == 0:
        return x.copy()
    r = int(np.ceil(3 * sigma))
    if r >= n:
        r = n - 1
    ks = np.arange(-r, r + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (ks / sigma) ** 2)
    kernel /= kernel.sum()
    out = np.zeros_like(x)
    for k in range(-r, r + 1):
        out += kernel[k + r] * np.roll(x, k)
    return out


def smooth_tar(tar, sigma):
    """TAR 转向角序列的多尺度高斯平滑（§3.2.3 + §4.2）。
    在尺度 sigma 下观察转向角，波谷（负值）深度随尺度增大而减小；
    宏观凹在大尺度下仍保留显著波谷，微观锯齿则被抹平。"""
    return gaussian_smooth_1d(tar, sigma)


# ═══════════════════════════════════════════════════════════════
# §3.1.1 桥 (Bridge) 与口袋 (Pocket)
# ═══════════════════════════════════════════════════════════════

def detect_pockets(poly):
    """检测凸包口袋：对每条凸包边（桥）追踪其覆盖的边界弧（凹区域）。
    返回 [{start, end, arc, depth(sl), width}, ...]。"""
    poly = np.asarray(poly, dtype=np.float64)
    n = len(poly)
    if n < 4:
        return []
    hull = convex_hull_2d(poly)
    h = len(hull)
    if h < 3:
        return []

    pockets = []
    for hi in range(h):
        hnext = (hi + 1) % h
        pA, pB = hull[hi], hull[hnext]
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
        width = float(np.linalg.norm(poly[pA] - poly[pB]))
        pockets.append({'start': pA, 'end': pB, 'arc': arc,
                        'depth': max_depth, 'width': width})
    return pockets


# ═══════════════════════════════════════════════════════════════
# §3.1.2 SL / SP 凹度
# ═══════════════════════════════════════════════════════════════

def sl_concavity(poly, pocket):
    """直线凹度 (SL-Concavity)：凹顶点到桥的最大垂直欧氏距离。"""
    poly = np.asarray(poly, dtype=np.float64)
    pA, pB = pocket['start'], pocket['end']
    if not pocket['arc']:
        return 0.0
    return max(point_to_line_dist(poly[vi], poly[pA], poly[pB]) for vi in pocket['arc'])


def _boundary_arc_len(poly, a, b):
    """沿闭合多边形边界从 a 到 b（正向）的弧长。"""
    poly = np.asarray(poly, dtype=np.float64)
    n = len(poly)
    if a == b:
        return 0.0
    total = 0.0
    i = a
    steps = 0
    while i != b and steps < n:
        total += float(np.linalg.norm(poly[(i + 1) % n] - poly[i]))
        i = (i + 1) % n
        steps += 1
    return total


def sp_concavity(poly, pocket):
    """最短路径凹度 (SP-Concavity)：最深凹顶点沿边界到最近桥端点的弧长。
    路径被限制在多边形边界内，比 SL 更能反映内部深度。"""
    poly = np.asarray(poly, dtype=np.float64)
    pA, pB = pocket['start'], pocket['end']
    arc = pocket['arc']
    if not arc:
        return 0.0
    deepest = max(arc, key=lambda vi: point_to_line_dist(poly[vi], poly[pA], poly[pB]))
    dA = _boundary_arc_len(poly, pA, deepest)
    dB = _boundary_arc_len(poly, deepest, pB)
    return min(dA, dB)


# ═══════════════════════════════════════════════════════════════
# §3.1.3 归一化凹度
# ═══════════════════════════════════════════════════════════════

def normalized_concavity(depth, width, a=1.0, b=1.0, use_log=False):
    """归一化凹度：concavity ∝ depth^a / width^b。
    深而窄的凹（depth 大 / width 小）得分高，浅而宽的凹得分低。
    use_log=True 时用对数阻尼宽度（报告 §3.1.3 提到 log 抑制宽度影响）。"""
    if width < 1e-9 or depth < 1e-12:
        return 0.0
    if use_log:
        return depth ** a / (np.log1p(width) ** b)
    return depth ** a / width ** b


def pocket_concavity(poly, pocket):
    """口袋的归一化凹度：深度取 SL 与 SP 凹度的较大者（§3.1.2）。
    SP 凹度对"宽而深"的凹更准确（SL 垂直距离会低估），因此取 max(sl, sp)。"""
    depth_sl = sl_concavity(poly, pocket)
    depth_sp = sp_concavity(poly, pocket)
    depth = max(depth_sl, depth_sp)
    return normalized_concavity(depth, pocket['width'])


# ═══════════════════════════════════════════════════════════════
# §3.2.1 多尺度邻域曲率
# ═══════════════════════════════════════════════════════════════

def multiscale_curvature(poly, radius):
    """多尺度邻域曲率：邻域半径 radius（顶点数）下的累积转向角。
    小 radius 反映局部高频（微小锯齿），大 radius 反映宏观趋势（大尺度凹凸）。"""
    poly = np.asarray(poly, dtype=np.float64)
    n = len(poly)
    tar = turn_angles(poly)
    curv = np.zeros(n)
    for i in range(n):
        acc = 0.0
        for k in range(-radius, radius + 1):
            acc += tar[(i + k) % n]
        curv[i] = acc
    return curv


# ═══════════════════════════════════════════════════════════════
# §3.2.2 形状指数 (Shape Index) / 曲率度 (Curvedness)
# ═══════════════════════════════════════════════════════════════

def shape_index_curvedness(curvature):
    """形状指数与曲率度的 2D 退化。
    SI = (2/pi)*arctan(kappa) ∈ [-1,1]：负值=凹（接近 -1 = 深杯状），正值=凸。
    Curvedness = |kappa|：弯曲剧烈程度。"""
    curvature = np.asarray(curvature, dtype=np.float64)
    si = (2.0 / np.pi) * np.arctan(curvature)
    curvedness = np.abs(curvature)
    return si, curvedness


# ═══════════════════════════════════════════════════════════════
# 口袋分割线（tip=切尖, corner=切角）
# ═══════════════════════════════════════════════════════════════

def split_tip(poly, pocket):
    """切尖：在口袋弧上找最窄"颈"，横切。返回 (p0, p1) 或 None。"""
    poly = np.asarray(poly, dtype=np.float64)
    arc = pocket['arc']
    m = len(arc)
    if m < 4:
        return None
    best_ratio = None
    best_pair = None
    for i in range(m):
        for j in range(i + 3, m):
            a, b = poly[arc[i]], poly[arc[j]]
            chord = b - a
            cl = float(np.linalg.norm(chord))
            if cl < 1e-8:
                continue
            maxd = max(point_to_line_dist(poly[arc[k]], a, b) for k in range(i + 1, j))
            if maxd < 1e-8:
                maxd = 1e-8
            ratio = cl / maxd
            if best_ratio is None or ratio < best_ratio:
                best_ratio = ratio
                best_pair = (arc[i], arc[j])
    if best_pair is None:
        return None
    return np.array(poly[best_pair[0]]), np.array(poly[best_pair[1]])


def split_corner(poly, pocket):
    """切角：从最深点沿角平分线发射射线到对面。返回 (p0, p1) 或 None。"""
    poly = np.asarray(poly, dtype=np.float64)
    arc, pA, pB = pocket['arc'], pocket['start'], pocket['end']
    if not arc:
        return None
    n = len(poly)
    deepest = max(arc, key=lambda vi: point_to_line_dist(poly[vi], poly[pA], poly[pB]))
    vd = poly[deepest]
    vp = poly[(deepest + n - 1) % n]
    vn = poly[(deepest + 1) % n]
    indir = vd - vp
    outdir = vn - vd
    indir = indir / (np.linalg.norm(indir) + 1e-12)
    outdir = outdir / (np.linalg.norm(outdir) + 1e-12)
    bis = indir + outdir
    bl = np.linalg.norm(bis)
    if bl < 1e-12:
        return None
    bis = bis / bl
    edge = poly[pB] - poly[pA]
    if bis[0] * edge[1] - bis[1] * edge[0] < 0:
        bis = -bis
    max_ray = np.linalg.norm(poly[pA] - poly[pB]) * 2
    ray_end = vd + bis * max_ray

    best_u = None
    best_hit = None
    for i in range(n):
        j = (i + 1) % n
        if i == deepest or j == deepest:
            continue
        if i == (deepest + n - 1) % n or j == (deepest + 1) % n:
            continue
        s1, e1 = poly[i], poly[j]
        d1 = e1 - s1
        d2 = ray_end - vd
        cr = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(cr) < 1e-12:
            continue
        t = ((vd[0] - s1[0]) * d2[1] - (vd[1] - s1[1]) * d2[0]) / cr
        u = ((vd[0] - s1[0]) * d1[1] - (vd[1] - s1[1]) * d1[0]) / cr
        if 0 <= t <= 1 and u > 1e-3:
            if best_u is None or u < best_u:
                best_u = u
                best_hit = vd + u * d2
    if best_hit is None:
        return None
    return vd, best_hit


def classify_pocket(poly, pocket, concavity_threshold=0.3):
    """对口袋判断 tip/corner 并生成分割线。返回 (split_type, (p0,p1)) 或 (0, None)。
    1=tip（切尖），2=corner（切角）。

    显著性门限用 pocket_concavity（max(SL,SP) / width）；单顶点/双顶点伪口袋
    （arc<3）直接忽略。"""
    poly = np.asarray(poly, dtype=np.float64)
    arc = pocket['arc']
    if len(arc) < 3:
        return 0, None

    if pocket_concavity(poly, pocket) < concavity_threshold:
        return 0, None

    pA, pB = pocket['start'], pocket['end']
    area2 = poly[pA][0] * poly[pB][1] - poly[pA][1] * poly[pB][0]
    prev = poly[pB]
    for vi in arc:
        p = poly[vi]
        area2 += prev[0] * p[1] - prev[1] * p[0]
        prev = p

    if area2 > 0:
        sl = split_tip(poly, pocket)
        return (1, sl) if sl else (0, None)
    else:
        sl = split_corner(poly, pocket)
        return (2, sl) if sl else (0, None)


# ═══════════════════════════════════════════════════════════════
# §4.2 / §4.3 CSS / PH 简化：多尺度生命周期持久性
# ═══════════════════════════════════════════════════════════════

def concave_segment_persistence(poly, runs, scales=(0, 1, 2, 4, 8)):
    """凹段多尺度持久性（CSS §4.2 零交叉生命周期 + PH §4.3 持久性的简化代理）。

    对给定凹段列表，在不同尺度 sigma 下对 TAR 做高斯平滑，计算该段的"深度"
    （累积负转向角）。宏观凹在大尺度平滑后仍保留显著深度（生命周期长），
    微观锯齿很快被抹平（生命周期短）。

    返回 [{run, depths:{sigma:depth}, persistence}, ...]，按 persistence 降序。"""
    poly = np.asarray(poly, dtype=np.float64)
    n = len(poly)
    tar = turn_angles(poly)
    results = []
    for (s, e) in runs:
        idx = run_indices(s, e, n)
        depths = {}
        for sigma in scales:
            star = smooth_tar(tar, sigma) if sigma > 0 else tar
            depths[sigma] = float(abs(sum(star[j] for j in idx if star[j] < 0)))
        # 持久性 = 最大尺度下仍存在的深度
        persistence = depths[max(scales)]
        results.append({'run': (s, e), 'depths': depths, 'persistence': persistence})
    results.sort(key=lambda r: r['persistence'], reverse=True)
    return results


def ph_persistence(poly, max_points=200):
    """PH 持久同调（§4.3）：对边界顶点做 Vietoris-Rips 复形，计算 H1 持久图。

    对闭合凹多边形，H1 持久图中的环对应两类：
      1) 主环：多边形整体围成的内部区域（death 尺度最大，接近轮廓直径）；
      2) 凹区域环：每个凹区域（凸包桥 + 凹边界围成的洞）对应一个环，
         其寿命 (death - birth) 量化该凹区域的显著性。

    需要 ripser 库（pip install ripser）。未安装时返回 (None, note)。
    返回 (intervals, note)：
      intervals: [(birth, death, persistence), ...] 按 persistence 降序（已剔除主环）
    """
    try:
        from ripser import ripser
    except ImportError:
        return None, "ripser not installed (pip install ripser)"

    poly = np.asarray(poly, dtype=np.float64)
    n = len(poly)
    if n < 4:
        return [], "too few vertices"

    # 下采样避免 ripser 复杂度爆炸
    pts = poly
    if n > max_points:
        step = int(np.ceil(n / max_points))
        pts = poly[::step]
        if len(pts) < 4:
            pts = poly

    dgms = ripser(pts, maxdim=1)['dgms']
    h1 = dgms[1]
    finite = h1[np.isfinite(h1[:, 1])]
    if len(finite) == 0:
        return [], "no finite H1 cycles"

    # 主环 = death 尺度最大（多边形整体内部区域）
    main_idx = int(np.argmax(finite[:, 1]))
    rest = np.delete(finite, main_idx, axis=0)

    intervals = [(float(b), float(d), float(d - b)) for b, d in rest]
    intervals.sort(key=lambda x: x[2], reverse=True)
    return intervals, None


# ═══════════════════════════════════════════════════════════════
# §4.1 近似凸分解 (ACD)
# ═══════════════════════════════════════════════════════════════

def split_polygon_by_chord(poly, i0, i1, p0=None, p1=None):
    """用分割线 (p0, p1) 把闭合多边形切成两个子多边形。
    i0/i1 是分割线端点在 poly 上的最近顶点索引（确定边界分段点）。
    子多边形顶点显式含真实分割线端点 p0/p1，闭合边即分割线。
    返回 [subA, subB] 或 None。"""
    poly = np.asarray(poly, dtype=np.float64)
    n = len(poly)
    if n < 4 or i0 == i1:
        return None
    if i0 > i1:
        i0, i1 = i1, i0
    q0 = np.asarray(p0, dtype=np.float64) if p0 is not None else poly[i0]
    q1 = np.asarray(p1, dtype=np.float64) if p1 is not None else poly[i1]
    # 子多边形 A：q0 -> poly[i0+1..i1-1] -> q1（闭合边 q1->q0 即分割线）
    subA = np.vstack([q0, poly[i0 + 1:i1], q1])
    # 子多边形 B：q1 -> poly[i1+1..n-1, 0..i0-1] -> q0（闭合边 q0->q1 即分割线）
    subB = np.vstack([q1, poly[i1 + 1:], poly[:i0], q0])
    if len(subA) < 4 or len(subB) < 4:
        return None
    return [subA, subB]


def _single_decompose(poly, tau):
    """单次 ACD 分解：找归一化凹度最大的口袋，若 > tau 则切分。
    返回 [subA, subB] 或 None。"""
    poly = ensure_ccw(np.asarray(poly, dtype=np.float64))
    cls = classify_vertices(poly)
    if not concave_runs(cls):
        return None
    pockets = detect_pockets(poly)
    if not pockets:
        return None

    best = None
    best_conc = 0.0
    for pkt in pockets:
        if len(pkt['arc']) < 3:
            continue
        conc = pocket_concavity(poly, pkt)
        if conc > best_conc:
            best_conc = conc
            best = pkt
    if best is None or best_conc < tau:
        return None

    st, sl = classify_pocket(poly, best)
    if not sl:
        return None
    p0, p1 = sl
    i0 = int(np.argmin(np.linalg.norm(poly - np.array(p0), axis=1)))
    i1 = int(np.argmin(np.linalg.norm(poly - np.array(p1), axis=1)))
    return split_polygon_by_chord(poly, i0, i1, p0, p1)


def acd_decompose(poly, tau=0.3, max_depth=4):
    """近似凸分解（§4.1）：递归切分显著凹区域直到归一化凹度 < tau。
    返回子多边形列表（每个为 (n,2) 顶点坐标，隐式闭合）。
    tau 是尺度旋钮：高 tau 只切大凹，低 tau 切得更细。"""
    polys = [ensure_ccw(np.asarray(poly, dtype=np.float64))]
    for _ in range(max_depth):
        new_polys = []
        changed = False
        for p in polys:
            subs = _single_decompose(p, tau)
            if subs is None:
                new_polys.append(p)
            else:
                new_polys.extend(subs)
                changed = True
        polys = new_polys
        if not changed:
            break
    return polys


def adaptive_threshold(poly, base_tau=0.3, target_components=None):
    """自适应凹度阈值（§4.1.3 简化）：用包围盒对角线归一化 + 二分调整。

    若分解结果组件数 < target_components，减半 tau（更细）；反之加倍（更粗）。"""
    poly = ensure_ccw(np.asarray(poly, dtype=np.float64))
    bb = poly.max(axis=0) - poly.min(axis=0)
    diag = float(np.linalg.norm(bb))
    tau = base_tau / max(diag, 1e-9) if diag > 0 else base_tau

    if target_components is None:
        return tau
    n_comp = len(acd_decompose(poly, tau))
    for _ in range(6):
        if n_comp >= target_components:
            break
        tau /= 2.0
        n_comp = len(acd_decompose(poly, tau))
    return tau


# ═══════════════════════════════════════════════════════════════
# 主入口：完整多尺度凹性分析
# ═══════════════════════════════════════════════════════════════

def analyze_boundary(poly, tau=0.3, min_run_len=3, scales=(0, 1, 2, 4, 8), use_ph=True):
    """完整多尺度凹性分析（主入口，报告 §5.2 混合工作流）。

    参数:
      poly: 闭合边界多边形（numpy (n,2)，任意方向，内部会 CCW 化）
      tau: ACD 凹度阈值（归一化凹度 depth/width）
      min_run_len: 过滤孤立凹顶点（拉普拉斯平滑残留）的最小凹段长度
      scales: TAR 多尺度平滑的尺度集合（顶点数单位）
      use_ph: 是否附加 PH 持久同调（§4.3，需 ripser），作为交叉验证

    返回 dict:
      poly, cls, tar, runs, pockets, deficit,
      segments: 每个显著凹段的完整量化（按持久性降序）,
      split_lines: ACD 第一层生成的分割线 [(type, p0, p1), ...],
      sub_polygons: ACD 递归分解结果（2D 顶点列表）,
      ph: {'intervals': [(birth,death,persistence),...], 'note': str}（或 None）,
      tau: 实际使用的凹度阈值
    """
    poly_ccw = ensure_ccw(np.asarray(poly, dtype=np.float64))
    n = len(poly_ccw)
    cls = classify_vertices(poly_ccw)
    tar = turn_angles(poly_ccw)
    all_runs = concave_runs(cls)

    # 过滤微观凹（孤立凹顶点）
    runs = []
    for (s, e) in all_runs:
        ln = (e - s + 1) if e >= s else (n - s + e + 1)
        if ln >= min_run_len:
            runs.append((s, e))

    pa = polygon_area(poly_ccw)
    ha = convex_hull_area(poly_ccw)
    deficit = (1.0 - pa / ha) if ha > 1e-12 else 0.0

    pockets = detect_pockets(poly_ccw) if runs else []

    # §3.2 曲率（多尺度邻域半径取平均边长的 2 倍顶点数近似）
    curv = multiscale_curvature(poly_ccw, radius=2)
    si, curvedness = shape_index_curvedness(curv)

    # §4.2/§4.3 多尺度持久性（仅对过滤后的宏观凹段）
    persistence = concave_segment_persistence(poly_ccw, runs, scales)

    # 每个凹段（run）的完整量化
    segments = []
    for pr in persistence:
        (s, e) = pr['run']
        idx = run_indices(s, e, n)
        # 匹配该凹段所属口袋（凹段顶点属于口袋的凹边界弧）
        pocket = None
        for pkt in pockets:
            if any(vi in pkt['arc'] for vi in idx):
                pocket = pkt
                break
        depth_sl = sl_concavity(poly_ccw, pocket) if pocket else 0.0
        depth_sp = sp_concavity(poly_ccw, pocket) if pocket else 0.0
        width = pocket['width'] if pocket else 0.0
        conc = normalized_concavity(depth_sl, width) if width > 1e-9 else 0.0
        seg_si = float(np.mean([si[j] for j in idx]))
        seg_curv = float(np.mean([curvedness[j] for j in idx]))
        segments.append({
            'run': (s, e),
            'indices': idx,
            'depth_sl': depth_sl,
            'depth_sp': depth_sp,
            'width': width,
            'concavity': conc,
            'persistence': pr['persistence'],
            'shape_index': seg_si,
            'curvedness': seg_curv,
            'depths': pr['depths'],
        })

    # §4.3 PH 持久同调（可选，交叉验证）
    ph = None
    if use_ph:
        intervals, note = ph_persistence(poly_ccw)
        ph = {'intervals': intervals, 'note': note}

    # §4.1 ACD 递归分解 + 第一层分割线
    sub_polygons = acd_decompose(poly_ccw, tau) if runs else [poly_ccw]
    split_lines = []
    for pkt in pockets:
        st, sl = classify_pocket(poly_ccw, pkt, tau)
        if sl:
            split_lines.append((st, sl[0], sl[1]))

    return {
        'poly': poly_ccw,
        'cls': cls,
        'tar': tar,
        'runs': runs,
        'pockets': pockets,
        'deficit': deficit,
        'segments': segments,
        'split_lines': split_lines,
        'sub_polygons': sub_polygons,
        'ph': ph,
        'tau': tau,
    }
