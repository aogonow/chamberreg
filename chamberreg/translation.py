"""Stage 2: translation maximising the score of Eq. (12), sum CT over the model wall - sum CT over its cavity.

The model is a solid of revolution, so the cylindrical coordinates of the region of interest
are computed once and each candidate pose costs a few array passes.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from . import config as cfg


class Scorer:
    """Scores model placements on a CT region of interest."""
    # frame: the line (centre, direction) in array indices; a placement (lam, l1, l2, sign)
    # puts `anchor` at axial position lam [mm], shifts the axis by (l1, l2) [mm] along
    # (e1, e2), and sign = -1 flips the model

    def __init__(self, ct, roi, centre, direction, e1, e2, voxel_mm, anchor=0.0, stride=1):
        self.ct = ct
        self.anchor = np.float32(anchor)
        axes = [np.arange(a, b, stride, dtype=np.float32) - np.float32(c)
                for (a, b), c in zip(roi, centre)]

        def project(v):
            v = np.asarray(v, np.float32) * np.float32(voxel_mm)
            return (axes[0][:, None, None] * v[0] + axes[1][None, :, None] * v[1]
                    + axes[2][None, None, :] * v[2])

        self.z = project(np.asarray(direction) / np.linalg.norm(direction))
        self.p1 = project(e1)
        self.p2 = project(e2)
        rows = max(1, int(cfg.CHUNK_VOXELS / self.z[0].size))
        self.chunks = [slice(i, i + rows) for i in range(0, self.z.shape[0], rows)]

    def _regions(self, model, sl, lam, l1, l2, sign):
        r = np.hypot(self.p1[sl] - np.float32(l1), self.p2[sl] - np.float32(l2))
        z = (self.z[sl] - np.float32(lam)) * np.float32(sign) + self.anchor
        return model.in_wall(r, z), model.in_cavity(r, z)

    def score(self, model, lam=0.0, l1=0.0, l2=0.0, sign=1):
        total = 0.0
        for sl in self.chunks:
            wall, cavity = self._regions(model, sl, lam, l1, l2, sign)
            ct = self.ct[sl]
            total += float(np.sum(ct, where=wall, dtype=np.float64))
            total -= float(np.sum(ct, where=cavity, dtype=np.float64))
        return total

    def masks(self, model, lam=0.0, l1=0.0, l2=0.0, sign=1):
        wall = np.empty(self.z.shape, bool)
        cavity = np.empty(self.z.shape, bool)
        for sl in self.chunks:
            wall[sl], cavity[sl] = self._regions(model, sl, lam, l1, l2, sign)
        return wall, cavity

def _parallel(fn, items, workers):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, items))

def fit_parabola(samples):
    """Quadratic through the largest window centred on the sampled maximum; vertex if it is a
    maximum inside the window."""
    xs = np.array(sorted(samples))
    ys = np.array([samples[x] for x in xs])
    i = int(np.argmax(ys))
    n = min(i, len(xs) - 1 - i)
    if n < 2:
        return float(xs[i])
    bx, by = xs[i - n:i + n + 1], ys[i - n:i + n + 1]
    p = np.poly1d(np.polyfit(bx, by, 2))
    if p.coeffs[0] >= 0:                     # convex fit: its stationary point is a minimum
        return float(xs[i])
    roots = np.roots(p.deriv().coeffs)
    roots = roots[np.isreal(roots)].real
    roots = roots[(roots >= bx.min()) & (roots <= bx.max())]
    return float(roots[np.argmax(p(roots))]) if len(roots) else float(xs[i])

def fit_paraboloid(samples):
    """Bivariate quadratic on the largest sub-grid centred on the sampled maximum; Eq. (16)-(17)."""
    pts = np.array(list(samples))
    vals = np.array([samples[tuple(p)] for p in pts])
    k = int(np.argmax(vals))
    best = (float(pts[k, 0]), float(pts[k, 1]))
    keep = np.ones(len(pts), bool)
    windows = []
    for dim in range(2):
        u = np.unique(pts[:, dim])
        i = int(np.searchsorted(u, pts[k, dim]))
        n = min(i, len(u) - 1 - i)
        win = u[i - n:i + n + 1]
        keep &= np.isin(pts[:, dim], win)
        windows.append(win)
    if keep.sum() < 6:
        return best
    x, y = pts[keep, 0], pts[keep, 1]
    T = np.c_[np.ones_like(x), y, y * y, x, x * y, x * x]
    g00, g01, g02, g10, g11, g20 = np.linalg.lstsq(T, vals[keep], rcond=None)[0]
    if g20 >= 0 or 4 * g20 * g02 - g11 ** 2 <= 0:   # Hessian not negative definite: minimum or saddle
        return best
    try:
        sol = np.linalg.solve([[2 * g20, g11], [g11, 2 * g02]], [-g10, -g01])
    except np.linalg.LinAlgError:
        return best
    inside = all(w.min() <= s <= w.max() for s, w in zip(sol, windows))
    return (float(sol[0]), float(sol[1])) if inside else best

def _grid(centre, half, step):
    return np.round(np.arange(centre - half, centre + half + 1e-9, step), 4)

def axial_refine(scorer, model, lam0, sign, workers, half=cfg.FINE_HALF_MM,
                 step=cfg.FINE_STEP_MM, max_moves=cfg.MAX_MOVES):
    """Fine axial pass around lam0; re-centred if the maximum falls on the window border."""
    samples, moves = {}, 0
    for _ in range(max_moves + 1):
        window = _grid(lam0, half, step)
        todo = [x for x in window if x not in samples]
        samples.update(zip(todo, _parallel(lambda x: scorer.score(model, x, sign=sign), todo, workers)))
        best = max(window, key=samples.get)
        if best not in (window[0], window[-1]):
            break
        lam0 = best
        moves += 1
    return fit_parabola({x: samples[x] for x in window}), window, moves

def perpendicular_refine(scorer, model, lam, sign, workers, half=cfg.PERP_HALF_MM,
                         step=cfg.PERP_STEP_MM, max_moves=cfg.MAX_MOVES):
    """5x5 grid in the plane perpendicular to the axis; re-centred if the maximum is on its border."""
    samples, c1, c2, moves = {}, 0.0, 0.0, 0
    for _ in range(max_moves + 1):
        g1, g2 = _grid(c1, half, step), _grid(c2, half, step)
        grid = [(a, b) for a in g1 for b in g2]
        todo = [p for p in grid if p not in samples]
        samples.update(zip(todo, _parallel(
            lambda p: scorer.score(model, lam, p[0], p[1], sign), todo, workers)))
        b1, b2 = max(grid, key=samples.get)
        if b1 not in (g1[0], g1[-1]) and b2 not in (g2[0], g2[-1]):
            break
        c1, c2 = b1, b2
        moves += 1
    return fit_paraboloid({p: samples[p] for p in grid}), (c1, c2), moves

def estimate_translation(ct, roi, centre, direction, model, voxel_mm, workers=cfg.THREADS):
    """Model pose: oriented axis and outline centroid, in array indices, plus diagnostics."""
    from .cad import perpendicular_basis

    e1, e2 = perpendicular_basis(direction)
    anchor = model.outline_centroid          # lam = 0 puts the outline centroid at `centre`
    k = cfg.COARSE_SUBSAMPLE
    coarse = Scorer(ct[::k, ::k, ::k], roi, centre, direction, e1, e2, voxel_mm, anchor, stride=k)
    fine = Scorer(ct, roi, centre, direction, e1, e2, voxel_mm, anchor)

    # coarse pass: both orientations, global maximum fixes orientation and basin
    reach = max(cfg.COARSE_REACH_MM, cfg.COARSE_REACH_LENGTH_FRACTION * model.length)
    lams = _grid(0.0, reach, cfg.COARSE_STEP_MM)
    best = None
    for sign in (+1, -1):
        vals = _parallel(lambda x: coarse.score(model, x, sign=sign), lams, workers)
        i = int(np.argmax(vals))
        if best is None or vals[i] > best[0]:
            best = (vals[i], float(lams[i]), sign)
    _, lam0, sign = best

    lam, window, axial_moves = axial_refine(fine, model, lam0, sign, workers)
    (l1, l2), grid_centre, perp_moves = perpendicular_refine(fine, model, lam, sign, workers)

    d = np.asarray(direction, float)
    position = centre + (lam * d + l1 * e1 + l2 * e2) / voxel_mm
    info = {"sign": sign, "search_reach_mm": reach, "lam_coarse_mm": lam0, "lam_mm": lam,
            "perp_mm": [l1, l2], "axial_window_mm": [float(window[0]), float(window[-1])],
            "perp_grid_centre_mm": list(grid_centre),
            "axial_recentrings": axial_moves, "perp_recentrings": perp_moves}
    return sign * d, position, info
