"""Stage 1: symmetry axis from the trajectory of ellipse centres in axial slices."""
import cv2
import numpy as np
from scipy import ndimage
from . import config as cfg


# Scharr kernels normalised to unit gain (an ideal step gives a peak of half its height)
SCHARR_X = np.outer([3.0, 10.0, 3.0], [-1.0, 0.0, 1.0]) / 32.0
SCHARR_Y = SCHARR_X.T

def symmetry_axis(weights):
    """Symmetry axis and centroid of a mask, from weighted second moments."""
    # the axis is the eigenvector whose two *other* eigenvalues are closest to each other;
    # the largest principal component is wrong for squat bodies (H < sqrt(3) R)
    idx = np.argwhere(weights > 0)
    w = weights[weights > 0].astype(np.float64)
    c = (idx * w[:, None]).sum(0) / w.sum()
    q = idx - c
    ev, vec = np.linalg.eigh((q * w[:, None]).T @ q / w.sum())
    k = int(np.argmin([abs(ev[1] - ev[2]), abs(ev[0] - ev[2]), abs(ev[0] - ev[1])]))
    return vec[:, k] / np.linalg.norm(vec[:, k]), c

def slicing_axis(body):
    """Array axis closest to the rough chamber axis, and the permutation putting it first."""
    k = int(np.argmax(np.abs(symmetry_axis(body)[0])))
    return k, (k, (k + 1) % 3, (k + 2) % 3)      # cyclic, so a rotation and not a reflection

def largest_component(mask, min_fraction=cfg.MIN_COMPONENT_FRACTION):
    """Largest connected component; a border-free one wins if at least `min_fraction` of it."""
    # the cable leaves the field of view, so the chamber itself may touch the border
    labels, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest = int(sizes.argmax())
    border = np.unique(np.concatenate([
        labels[0].ravel(), labels[-1].ravel(), labels[:, 0].ravel(),
        labels[:, -1].ravel(), labels[:, :, 0].ravel(), labels[:, :, -1].ravel()]))
    inner = sizes.copy()
    inner[border] = 0
    if inner.max() >= min_fraction * sizes[largest]:
        return labels == int(inner.argmax())
    return labels == largest

def coarse_segmentation(vol, step, sigma=cfg.SEGMENTATION_SIGMA, k_mad=cfg.SEGMENTATION_K_MAD):
    """Segment the chamber on a volume subsampled by `step`."""
    # threshold = median + k * MAD of the volume itself, so nothing outside the scan is needed
    smooth = ndimage.gaussian_filter(vol, sigma)
    median = float(np.median(smooth))
    mad = float(np.median(np.abs(smooth - median)))
    if mad <= 0.0:      # over half the volume holds one value, so the threshold would be the median
        raise RuntimeError("coarse segmentation: the volume has no scatter to set a threshold from")
    threshold = median + k_mad * mad
    solid = largest_component(smooth > threshold)
    if solid is None:
        raise RuntimeError("chamber not found by coarse segmentation")
    body = ndimage.binary_fill_holes(solid)
    core = ndimage.binary_erosion(solid, iterations=1)
    air = float(np.median(vol))
    material = float(np.median(vol[core] if core.any() else vol[solid]))
    idx = np.argwhere(body)
    bbox = tuple((int(idx[:, i].min()) * step, (int(idx[:, i].max()) + 1) * step) for i in range(3))
    return {"body": body, "step": step, "bbox": bbox,
            "air": air, "material": material, "contrast": material - air}

def body_slab(seg, fraction=cfg.BODY_FRACTION):
    """Slices where the filled cross-section exceeds `fraction` of its maximum (drops neck and cable)."""
    area = seg["body"].sum(axis=(1, 2))
    idx = np.flatnonzero(area > fraction * area.max())
    return int(idx.min()) * seg["step"], int(idx.max() + 1) * seg["step"]

def gradient_magnitude(img):
    f = img.astype(np.float64)
    return np.hypot(ndimage.convolve(f, SCHARR_X, mode="nearest"),
                    ndimage.convolve(f, SCHARR_Y, mode="nearest"))

def fit_ellipse(points):
    """Fitzgibbon fit, Halir-Flusser form -> (cx, cy, a, b, rms_px) or None."""
    P = np.asarray(points, float)
    if len(P) < 6:
        return None
    m = P.mean(0)
    s = np.abs(P - m).max()
    if s <= 0:
        return None
    x, y = (P[:, 0] - m[0]) / s, (P[:, 1] - m[1]) / s    # conditioning
    D = np.c_[x * x, x * y, y * y, x, y, np.ones_like(x)]
    S = D.T @ D
    S11, S12, S22 = S[:3, :3], S[:3, 3:], S[3:, 3:]
    C1_inv = np.array([[0, 0, 0.5], [0, -1, 0], [0.5, 0, 0]])
    try:
        T = -np.linalg.solve(S22, S12.T)
        w, v = np.linalg.eig(C1_inv @ (S11 + S12 @ T))
    except np.linalg.LinAlgError:
        return None
    cond = (4 * v[0] * v[2] - v[1] ** 2).real
    k = int(np.argmax(np.where(np.isfinite(cond), cond, -np.inf)))
    if not np.isfinite(cond[k]) or cond[k] <= 0:
        return None
    a1 = np.real(v[:, k])
    A, B, C, Dx, Ey, F = np.concatenate([a1, T @ a1])

    den = B * B - 4 * A * C
    if abs(den) < 1e-14:
        return None
    cx = (2 * C * Dx - B * Ey) / den
    cy = (2 * A * Ey - B * Dx) / den
    num = 2 * (A * Ey * Ey + C * Dx * Dx + F * B * B - B * Dx * Ey - 4 * A * C * F)
    root = np.sqrt(max((A - C) ** 2 + B * B, 0.0))
    s1, s2 = num * ((A + C) + root), num * ((A + C) - root)
    if s1 <= 0 or s2 <= 0:
        return None
    ax1, ax2 = np.sqrt(s1) / abs(den), np.sqrt(s2) / abs(den)

    # algebraic residual divided by the gradient norm ~ geometric distance, in pixels
    val = A * x * x + B * x * y + C * y * y + Dx * x + Ey * y + F
    grad = np.hypot(2 * A * x + B * y + Dx, 2 * C * y + B * x + Ey)
    rms = float(np.sqrt(np.mean((val / np.maximum(grad, 1e-9)) ** 2)) * s)
    return cx * s + m[0], cy * s + m[1], max(ax1, ax2) * s, min(ax1, ax2) * s, rms

def slice_ellipses(img, tau, n_contours=cfg.N_CONTOURS, min_points=cfg.MIN_CONTOUR_POINTS,
                   ratio_range=cfg.RATIO_RANGE, ratio_tol=cfg.RATIO_TOL,
                   centre_tol=cfg.CENTRE_TOL_PX, rms_tol=cfg.RMS_TOL_PX):
    """Edge map -> N_c largest contours -> ellipses -> validation. Centres are (column, row)."""
    edges = (gradient_magnitude(img) > tau).astype(np.uint8)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    fits = []
    for c in contours:
        if len(fits) == n_contours:
            break
        pts = c.reshape(-1, 2).astype(float)
        if len(pts) >= min_points:
            e = fit_ellipse(pts)
            if e is not None:
                fits.append(e)
    fits.reverse()       # index radially outward: the cavity rims first, then the outer ones

    out = {"centres": [(e[0], e[1]) for e in fits], "rms": [e[4] for e in fits], "ok": False}
    if len(fits) < n_contours:
        return out
    ratio = np.array([e[2] / e[3] for e in fits])
    centres = np.array(out["centres"])
    out["ok"] = bool(
        ratio.min() >= ratio_range[0] - 1e-9 and ratio.max() <= ratio_range[1]
        and ratio.std() <= ratio_tol
        and np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=2).max() <= centre_tol
        and max(out["rms"]) <= rms_tol)
    return out

def calibrate_tau(slices, contrast, ladder=cfg.TAU_LADDER):
    """Pick tau = c * contrast: most probe slices passing validation, ties broken by lower residual."""
    rows = []
    for c in ladder:
        res = [slice_ellipses(img, c * contrast) for img in slices]
        rms = [np.mean(r["rms"]) for r in res if r["ok"]]
        rows.append({"c": c, "tau": c * contrast, "n_ok": sum(r["ok"] for r in res),
                     "rms": float(np.mean(rms)) if rms else np.inf})
    best_n = max(r["n_ok"] for r in rows)
    return min((r for r in rows if r["n_ok"] == best_n), key=lambda r: r["rms"])

def longest_valid_run(ok_slices, max_gap=cfg.MAX_GAP, margin=cfg.MARGIN,
                      min_slices=cfg.MIN_SLICES, min_span=None):
    """Longest run of valid slices (tolerating short gaps), trimmed by `margin` at both ends."""
    ks = sorted(ok_slices)
    if not ks:
        return None
    runs, start = [], ks[0]
    for prev, k in zip(ks, ks[1:]):
        if k - prev > max_gap + 1:
            runs.append((start, prev))
            start = k
    runs.append((start, ks[-1]))
    a, b = max(runs, key=lambda r: r[1] - r[0])
    a, b = a + margin, b - margin
    if b <= a or sum(a <= k <= b for k in ks) < min_slices:
        return None
    if min_span is not None and b - a < min_span:
        return None
    return a, b

def ransac_line(P, threshold, iterations=cfg.RANSAC_ITERATIONS, seed=cfg.RANSAC_SEED):
    rng = np.random.RandomState(seed)
    best, best_count = None, -1
    for _ in range(iterations):
        i, j = rng.choice(len(P), 2, replace=False)
        d = P[j] - P[i]
        n = np.linalg.norm(d)
        if n < 1e-9:
            continue
        d /= n
        v = P - P[i]
        inliers = np.linalg.norm(v - np.outer(v @ d, d), axis=1) < threshold
        if inliers.sum() > best_count:
            best, best_count = inliers, int(inliers.sum())
    if best is None or best_count < 3:
        raise RuntimeError("RANSAC found no line")
    return best

def pca_axis(P):
    mu = P.mean(0)
    Q = P - mu
    w, v = np.linalg.eigh(Q.T @ Q / len(P))
    d = v[:, int(np.argmax(w))]
    return (d if d[2] >= 0 else -d), mu

def estimate_axis(chamber, seg, crop, perm=(0, 1, 2)):
    """Returns the axis direction and a point on it, both in array-index coordinates (i0, i1, i2).

    Slices are taken along `perm[0]`, the array axis closest to the rough chamber axis, so `seg`
    and `crop` are given in that order and the result is mapped back to (i0, i1, i2).
    """
    vox = chamber.voxel_mm
    lo, hi = body_slab(seg)
    (y0, y1), (x0, x1) = crop
    box = [None] * 3
    for i, a in enumerate(((lo, hi), (y0, y1), (x0, x1))):
        box[perm[i]] = a
    slab = np.transpose(chamber.recon(roi=tuple(box)), perm)

    probes = [int(round((hi - 1 - lo) * q)) for q in np.linspace(0.1, 0.9, 9)]
    best = calibrate_tau([slab[k] for k in probes], seg["contrast"])

    results = {k + lo: slice_ellipses(slab[k], best["tau"]) for k in range(hi - lo)}
    run = longest_valid_run([k for k, r in results.items() if r["ok"]],
                            min_span=int(round(3.0 / vox)))
    if run is None:
        raise RuntimeError("no range of slices with complete, valid ellipses")

    # point cloud (x, y, z) = (column, row, slice) * voxel size, Eq. (6)
    P = np.array([(cx + x0, cy + y0, k) for k, r in sorted(results.items())
                  if r["ok"] and run[0] <= k <= run[1] for cx, cy in r["centres"]], float) * vox
    inliers = ransac_line(P, cfg.RANSAC_THRESHOLD_VOXELS * vox)
    d, mu = pca_axis(P[inliers])

    direction = d[[2, 1, 0]] / np.linalg.norm(d)      # back to slicing order
    centre = mu[[2, 1, 0]] / vox
    tilt = float(np.degrees(np.arccos(np.clip(d[2], -1, 1))))   # from the axis sliced along
    out = np.empty((2, 3))
    out[:, list(perm)] = [direction, centre]                    # back to array order (i0, i1, i2)
    direction, centre = out
    info = {"slicing_axis": int(perm[0]), "slab": [lo, hi],
            "tau_fraction": best["c"], "probe_slices_ok": best["n_ok"],
            "slice_range": list(run), "n_slices": sum(1 for k, r in results.items()
                                                      if r["ok"] and run[0] <= k <= run[1]),
            "n_centres": len(P), "inlier_fraction": float(inliers.mean()), "tilt_deg": tilt}
    return direction, centre, info
