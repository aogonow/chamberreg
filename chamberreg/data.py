"""Reading the simulated chamber scans, one HDF5 file per chamber."""
import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  registers the compression filters; must precede any read
import numpy as np

CAVITY_ID = 1

def list_chambers(dataset_dir):
    files = sorted(Path(dataset_dir, "sinograms").glob("*_recon_sino.hdf5"))
    names = [f.name.replace("_recon_sino.hdf5", "") for f in files]
    return sorted(names, key=lambda s: (len(s), s))

def _slices(roi):
    return tuple(slice(a, b) for a, b in roi)

class Chamber:
    """One simulated scan: reconstruction, fractional material masks and generator metadata."""

    def __init__(self, dataset_dir, name):
        self.name = name
        self.path = Path(dataset_dir, "sinograms", f"{name}_recon_sino.hdf5")
        with h5py.File(self.path, "r") as f:
            info = json.loads(f["creation_data"][0].decode())
            self.shape = tuple(int(n) for n in f["recon"].shape)
            self.voxel_mm = float(f["input_voxel_size"][()])
            if f["mask"].ndim != 4:
                raise ValueError(f"{self.path}: expected a 4D fractional mask (one channel per material)")
            self.mask_channels = json.loads(f["voxelization_data"][0].decode())["materials"]
        self.kind = info.get("type")
        self.params = info["parameters"]
        self.materials = info["materials"]          # material ids are drawn per chamber
        self.wall_id = int(self.materials["wall"])
        self.cavity_volume_mm3 = float(info["empty interior volume"])

    def recon(self, roi=None, step=1):
        with h5py.File(self.path, "r") as f:
            d = f["recon"]
            if roi is not None:
                return d[_slices(roi)].astype(np.float32)
            return d[::step, ::step, ::step].astype(np.float32)

    def fraction(self, material_id, roi):
        """Volume fraction (0..1) of one material inside `roi`."""
        ch = self.mask_channels.index(material_id)
        with h5py.File(self.path, "r") as f:
            raw = f["mask"][(ch,) + _slices(roi)]
        return raw.astype(np.float32) / 255.0
