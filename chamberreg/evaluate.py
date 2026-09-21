"""Ground-truth pose from the simulator mask and the error metrics."""
import numpy as np

from .axis import symmetry_axis
from .cad import perpendicular_basis
from .data import CAVITY_ID
from .translation import Scorer

def ground_truth_pose(chamber, model, roi):
    """True axis (towards the neck) and outline centroid, in array indices."""
    # mask = voxelised CAD in the scan pose: cavity fraction gives axis and centroid,
    # the side carrying the wall mass gives the orientation
    offset = np.array([a for a, _ in roi], float)
    cavity = chamber.fraction(CAVITY_ID, roi)
    axis, centroid = symmetry_axis(cavity)
    centroid += offset
    wall_centroid = np.argwhere(chamber.fraction(chamber.wall_id, roi) >= 0.5).mean(0) + offset
    if np.sign((wall_centroid - centroid) @ axis) != np.sign(model.wall_centroid - model.cavity_centroid):
        axis = -axis
    position = centroid + (model.outline_centroid - model.cavity_centroid) * axis / chamber.voxel_mm
    return axis, position, cavity >= 0.5

def rasterize(model, ct, roi, position, axis, voxel_mm):
    """Wall and cavity masks of the model with its outline centroid at `position`, oriented along `axis`."""
    e1, e2 = perpendicular_basis(axis)
    return Scorer(ct, roi, position, axis, e1, e2, voxel_mm, model.outline_centroid).masks(model)

def iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0

def angle_deg(a, b):
    """Angle between two axes, ignoring their orientation."""
    c = abs(np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))

def tilt_orientation_deg(axis, slicing_axis=0):
    """Tilt and orientation of an axis given in array indices (Eq. 11), measured from the array
    axis the slices are normal to, as in the paper; d_z >= 0."""
    k = int(slicing_axis)
    a = np.asarray(axis, float)[[k, (k + 1) % 3, (k + 2) % 3]]
    a = a / np.linalg.norm(a)
    if a[0] < 0:
        a = -a
    tilt = np.degrees(np.arccos(np.clip(a[0], -1.0, 1.0)))
    orientation = np.degrees(np.arctan2(a[1], a[2])) % 360.0
    return float(tilt), float(orientation)

def pose_errors(axis, position, axis_gt, position_gt, voxel_mm, slicing_axis=0):
    delta = (np.asarray(position) - position_gt) * voxel_mm
    axial = float(delta @ axis_gt)
    tilt, orientation = tilt_orientation_deg(axis, slicing_axis)
    tilt_gt, orientation_gt = tilt_orientation_deg(axis_gt, slicing_axis)
    return {"axis_error_deg": angle_deg(axis, axis_gt),
            "tilt_error_deg": abs(tilt - tilt_gt),
            "orientation_error_deg": abs((orientation - orientation_gt + 180.0) % 360.0 - 180.0),
            "tilt_deg": tilt, "orientation_deg": orientation,
            "translation_error_mm": float(np.linalg.norm(delta)),
            "translation_axial_mm": axial,
            "translation_perpendicular_mm": float(np.linalg.norm(delta - axial * axis_gt))}
