"""
Minimal NURBS surface evaluator (Cox-de Boor algorithm).
Loads from nurbs_surface.txt exported by distillation.exe.
"""
import numpy as np

class NurbsSurface:
    def __init__(self, path):
        with open(path) as f:
            # Line 1: nU nV degU degV
            self.nU, self.nV, self.degU, self.degV = map(int, f.readline().split())
            # Line 2: nKnotsU + knot values (expanded with multiplicities)
            line2 = f.readline().split()
            nkU = int(line2[0])
            self.knotsU = np.array([float(x) for x in line2[1:1+nkU]])
            # Line 3: nKnotsV + knot values (expanded with multiplicities)
            line3 = f.readline().split()
            nkV = int(line3[0])
            self.knotsV = np.array([float(x) for x in line3[1:1+nkV]])
            # Lines 4+: control points (nU * nV lines)
            self.cp = np.zeros((self.nU, self.nV, 4))
            for i in range(self.nU):
                for j in range(self.nV):
                    vals = [float(x) for x in f.readline().split()]
                    self.cp[i, j] = vals

    def _basis(self, u, knots, degree, n_ctrl):
        """Evaluate B-spline basis at parameter u. Returns (n_ctrl,) array."""
        u = np.clip(u, knots[degree], knots[-degree-1])
        # Find span
        span = degree
        for k in range(degree, len(knots) - degree - 1):
            if knots[k] <= u < knots[k+1]:
                span = k
                break
        if u >= knots[-degree-1]:
            span = len(knots) - degree - 2

        N = np.zeros(degree + 1)
        N[0] = 1.0
        left = np.zeros(degree + 1, dtype=int)
        right = np.zeros(degree + 1, dtype=int)

        for j in range(1, degree + 1):
            left[j] = span + 1 - j
            right[j] = span + j
            saved = 0.0
            for r in range(j):
                temp = N[r] / (knots[right[r+1]] - knots[left[r+1]] + 1e-12)
                N[r] = saved + (knots[right[r+1]] - u) * temp
                saved = (u - knots[left[r+1]]) * temp
            N[j] = saved

        result = np.zeros(n_ctrl)
        result[span-degree:span+1] = N[:degree+1]
        return result

    def evaluate(self, u, v):
        """Evaluate NURBS surface at (u,v). Returns (3,) array."""
        Nu = self._basis(u, self.knotsU, self.degU, self.nU)
        Nv = self._basis(v, self.knotsV, self.degV, self.nV)

        point = np.zeros(4)
        for i in range(self.nU):
            for j in range(self.nV):
                w = self.cp[i, j, 3]
                point += Nu[i] * Nv[j] * self.cp[i, j] * w
        if point[3] > 1e-12:
            point[:3] /= point[3]
        return point[:3]
