
import numpy as np
from . import config as cfg

def perpendicular_basis(axis):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    ref = np.array([1.0, 0.0, 0.0]) if abs(a[0]) <= 0.9 else np.array([0.0, 0.0, 1.0])
    e1 = np.cross(a, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    return e1, e2 / np.linalg.norm(e2)

class ChamberModel:
    def __init__(self, params):
        p = params
        self.R = float(p["body_rad"])
        self.H = float(p["body_height"])
        self.t = float(p["thickness"])
        self.rn = float(p["neck_rad"])
        self.hn = float(p["neck_height"])
        self.electrode = bool(p.get("electrode", False))
        self.re = float(p.get("electrode_rad", 0.0))
        self.L = float(p.get("electrode_ratio", 0.0)) * self.H
        self.flat_tip = bool(p.get("electrode_flat", True))
        self.z_min = -(self.H / 2 + self.t)
        self.z_max = self.H / 2 + self.hn
        self.length = self.z_max - self.z_min

        self.cavity_centroid = self._centroid(self.in_cavity)
        self.wall_centroid = self._centroid(self.in_wall)
        # reference point of the axial search: centroid of the whole outline (wall + cavity)
        self.outline_centroid = self._centroid(lambda r, z: self.in_cavity(r, z) | self.in_wall(r, z))

    def in_cavity(self, r, z):
        h = self.H / 2
        inside = (r <= self.R) & (z >= -h) & (z <= h)
        if self.electrode:
            tip = h - self.L
            if self.flat_tip:
                in_electrode = (r <= self.re) & (z >= tip)
            else:
                cap = tip + self.re - np.sqrt(np.maximum(self.re ** 2 - r * r, 0.0))
                in_electrode = (r <= self.re) & (z >= cap)
            inside &= ~in_electrode
        return inside

    def in_wall(self, r, z):
        h, t, R, rn = self.H / 2, self.t, self.R, self.rn
        outer = R + t
        bottom = (r <= outer) & (z >= -h - t) & (z < -h)
        body = (r > R) & (r <= outer) & (z >= -h) & (z <= h)
        shoulder = (r > rn) & (r <= outer) & (z > h) & (z <= h + t)
        neck = (r > rn) & (r <= rn + t) & (z > h + t) & (z <= h + self.hn)
        return bottom | body | shoulder | neck

    def cavity_volume(self):
        """Cavity volume from the closed-form expression (exact for the generator geometry)."""
        v = np.pi * self.R ** 2 * self.H
        if self.electrode:
            v -= np.pi * self.re ** 2 * self.L
            if not self.flat_tip:
                v += np.pi * self.re ** 3 / 3.0
        return float(v)

    def _centroid(self, inside, res=cfg.CENTROID_RES_MM):
        r = np.arange(res / 2, self.R + self.t + 0.5, res)
        z = np.arange(self.z_min - 0.5, self.z_max + 0.5, res) + res / 2
        rr, zz = np.meshgrid(r, z, indexing="ij")
        w = inside(rr, zz) * rr           # volume element 2*pi*r*dr*dz
        return float((w * zz).sum() / w.sum())
