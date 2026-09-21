"""Full registration of one chamber, batch processing and the summary."""
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .axis import coarse_segmentation, estimate_axis, slicing_axis
from .cad import ChamberModel
from .data import Chamber, list_chambers
from .evaluate import ground_truth_pose, iou, pose_errors, rasterize
from .translation import estimate_translation
from . import config as cfg

def register_chamber(dataset_dir, name, threads=cfg.THREADS, margin=cfg.ROI_MARGIN_VOXELS,
                     segmentation_voxel_mm=cfg.SEGMENTATION_VOXEL_MM):
    t0 = time.time()
    ch = Chamber(dataset_dir, name)
    vox = ch.voxel_mm
    model = ChamberModel(ch.params)
    times = {}

    # segment at ~0.4 mm (x2 at 0.2 mm, x4 at 0.1 mm), then crop to the chamber
    step = max(1, int(round(segmentation_voxel_mm / vox)))
    seg = coarse_segmentation(ch.recon(step=step), step)
    roi = tuple((max(0, a - margin), min(n, b + margin)) for (a, b), n in zip(seg["bbox"], ch.shape))
    times["segmentation_s"] = time.time() - t0

    t = time.time()
    # Stage 1 slices perpendicular to the array axis closest to the rough chamber axis (Sec. 3.3.1)
    k, perm = slicing_axis(seg["body"])
    seg_sliced = dict(seg, body=np.transpose(seg["body"], perm))
    direction, centre, axis_info = estimate_axis(ch, seg_sliced,
                                                 crop=tuple(roi[a] for a in perm)[1:], perm=perm)
    times["stage1_s"] = time.time() - t

    t = time.time()
    # air -> 0, brightest voxel -> 1
    ct = ch.recon(roi=roi)
    ct = (ct - np.float32(seg["air"])) / np.float32(float(ct.max()) - seg["air"])
    axis, position, trans_info = estimate_translation(ct, roi, centre, direction, model, vox, threads)
    times["stage2_s"] = time.time() - t

    # evaluation only; the method never reads the mask
    t = time.time()
    axis_gt, position_gt, cavity_sim = ground_truth_pose(ch, model, roi)
    wall_est, cav_est = rasterize(model, ct, roi, position, axis, vox)
    wall_gt, cav_gt = rasterize(model, ct, roi, position_gt, axis_gt, vox)
    result = {
        "chamber": name, "voxel_mm": vox, "shape": list(ch.shape), "roi": [list(r) for r in roi],
        "materials": ch.materials, "parameters": ch.params,
        "cavity_volume_mm3": model.cavity_volume(),
        "levels": {"air": seg["air"], "material": seg["material"], "contrast": seg["contrast"]},
        **pose_errors(axis, position, axis_gt, position_gt, vox, axis_info["slicing_axis"]),
        "iou_cavity": iou(cav_est, cav_gt),
        "iou_wall": iou(wall_est, wall_gt),
        "iou_cavity_vs_simulator": iou(cav_est, cavity_sim),
        "iou_cavity_ceiling": iou(cav_gt, cavity_sim),
        "axis": axis.tolist(), "axis_true": axis_gt.tolist(),
        "position_idx": np.asarray(position).tolist(), "position_true_idx": position_gt.tolist(),
        "stage1": axis_info, "stage2": trans_info,
    }
    result["success"] = bool(result["axis_error_deg"] < cfg.SUCCESS_AXIS_DEG
                             and result["translation_error_mm"] < cfg.SUCCESS_TRANSLATION_MM)
    times["evaluation_s"] = time.time() - t
    times["registration_s"] = times["segmentation_s"] + times["stage1_s"] + times["stage2_s"]
    times["total_s"] = time.time() - t0
    result["time"] = times
    return result

def _run_one(dataset_dir, name, out_dir, threads):
    try:
        res = register_chamber(dataset_dir, name, threads)
    except Exception as ex:                          # record the error instead of aborting the batch
        res = {"chamber": name, "error": f"{type(ex).__name__}: {ex}",
               "traceback": traceback.format_exc(), "success": False}
    with open(Path(out_dir, f"{name}.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res

def run_dataset(dataset_dir, out_dir, chambers=None, jobs=1, threads=cfg.THREADS, log=print):
    os.makedirs(out_dir, exist_ok=True)
    names = chambers or list_chambers(dataset_dir)
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futures = {ex.submit(_run_one, dataset_dir, n, out_dir, threads): n for n in names}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            log(format_line(r))
    summary = summarize(out_dir, names)
    summary["wall_time_s"] = time.time() - t0
    summary["jobs"], summary["threads"] = jobs, threads
    write_summary(summary, out_dir)
    return summary

def format_line(r):
    if "error" in r:
        return f"{r['chamber']:>11}  FAILED: {r['error']}"
    return (f"{r['chamber']:>11}  axis {r['axis_error_deg']:.4f} deg  "
            f"translation {r['translation_error_mm']:.4f} mm  "
            f"IoU cavity {r['iou_cavity']:.4f} wall {r['iou_wall']:.4f}  "
            f"{r['time']['total_s']:.0f} s")

def _stats(x):
    x = np.asarray(x, float)
    return {"median": float(np.median(x)), "mean": float(x.mean()),
            "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "min": float(x.min()), "max": float(x.max())}

def _order(row):
    tail = row["chamber"].rsplit("_", 1)[-1]
    return (0, int(tail), "") if tail.isdigit() else (1, 0, row["chamber"])

def summarize(out_dir, names=None):
    """Summary of `names`, or of every per-chamber file in `out_dir`."""
    if names is None:
        paths = [p for p in Path(out_dir).glob("*.json") if p.name != "summary.json"]
    else:
        paths = [p for p in (Path(out_dir, f"{n}.json") for n in names) if p.exists()]
    rows = [json.load(open(p)) for p in paths]
    rows.sort(key=_order)
    ok = [r for r in rows if "error" not in r]
    s = {"n_chambers": len(rows), "n_failed": len(rows) - len(ok),
         "n_success": sum(r["success"] for r in rows),
         "success_rule": f"axis < {cfg.SUCCESS_AXIS_DEG} deg and translation < {cfg.SUCCESS_TRANSLATION_MM} mm",
         }
    if ok:
        for key in ("axis_error_deg", "tilt_error_deg", "orientation_error_deg",
                    "translation_error_mm", "translation_axial_mm",
                    "translation_perpendicular_mm", "iou_cavity", "iou_wall"):
            vals = [abs(r[key]) if key == "translation_axial_mm" else r[key] for r in ok]
            s[key if key != "translation_axial_mm" else "translation_axial_abs_mm"] = _stats(vals)
        s["registration_time_s"] = _stats([r["time"]["registration_s"] for r in ok])
        s["total_time_s"] = _stats([r["time"]["total_s"] for r in ok])
    s["chambers"] = [{k: r.get(k) for k in ("chamber", "axis_error_deg", "translation_error_mm",
                                            "translation_axial_mm", "translation_perpendicular_mm",
                                            "iou_cavity", "iou_wall",
                                            "success", "error")} for r in rows]
    return s

def write_summary(s, out_dir):
    with open(Path(out_dir, "summary.json"), "w") as f:
        json.dump(s, f, indent=1)
    lines = ["| chamber | axis error [deg] | translation [mm] | axial [mm] | perpendicular [mm] "
             "| IoU cavity | IoU wall | success |", "|---|---|---|---|---|---|---|---|"]
    for r in s["chambers"]:
        if r.get("error"):
            lines.append(f"| {r['chamber']} | failed: {r['error']} | | | | | | no |")
            continue
        lines.append(f"| {r['chamber']} | {r['axis_error_deg']:.4f} | {r['translation_error_mm']:.4f} "
                     f"| {r['translation_axial_mm']:+.4f} | {r['translation_perpendicular_mm']:.4f} "
                     f"| {r['iou_cavity']:.4f} "
                     f"| {r['iou_wall']:.4f} | {'yes' if r['success'] else 'no'} |")
    if "axis_error_deg" in s:
        a, t = s["axis_error_deg"], s["translation_error_mm"]
        c, w = s["iou_cavity"], s["iou_wall"]
        lines.append(f"| median | {a['median']:.4f} | {t['median']:.4f} | | | {c['median']:.4f} "
                     f"| {w['median']:.4f} | |")
        lines.append(f"| mean +/- SD | {a['mean']:.4f} +/- {a['sd']:.4f} | {t['mean']:.4f} +/- {t['sd']:.4f} "
                     f"| | | {c['mean']:.4f} | {w['mean']:.4f} | |")
        lines.append(f"| max | {a['max']:.4f} | {t['max']:.4f} | | | | | |")
        lines.append(f"| all chambers | | | | | | | {s['n_success']} / {s['n_chambers']} |")
    Path(out_dir, "summary.md").write_text("\n".join(lines) + "\n")
