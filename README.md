# chamberreg: CAD-to-CT registration of ionization chambers

Official code for the paper *CAD-to-CT Registration of Cylindrical Objects via Ellipse-Based Axis
Estimation* (arXiv:2606.02935), to run the method on the simulated chambers published with it. It
registers the CAD model of a chamber to its CT reconstruction in two stages and scores the result
against the ground truth of the scan. The method is described in the paper.

## Contents

```
chamberreg/            the package
  config.py            all default parameters, with the section of the paper each comes from
  data.py              reading the HDF5 scans
  cad.py               analytic chamber model (solid of revolution) from generator parameters
  axis.py              Stage 1: segmentation, ellipses, RANSAC, PCA
  translation.py       Stage 2: overlap score, axial and perpendicular searches, polynomial peak fits
  evaluate.py          ground truth from the simulator mask, error metrics, IoU
  pipeline.py          one chamber / whole dataset / summary
  __main__.py          command-line interface
data/                  the scans, downloaded separately from Zenodo (see Data below)
results_reference/     the output of a run on the ten chambers, to compare a new one against
results/               where the commands below write (created on the first run)
requirements.txt, Dockerfile
```

## Data

The scans are not in this repository. The ten simulated chambers the package runs on are the Zenodo
record <https://doi.org/10.5281/zenodo.22878186> (13 files, 5.0 GB, MIT). That DOI always resolves
to the latest version of the record, at present <https://zenodo.org/records/22878187>.

The record holds its thirteen files flat, while the package expects the reconstructions in a
`sinograms` subdirectory. Download them into `data/baseline` and move the ten `.hdf5` files:

```bash
mkdir -p data/baseline/sinograms
# download the files of the record into data/baseline, then
mv data/baseline/chamber_*_recon_sino.hdf5 data/baseline/sinograms/
```

giving

```
data/baseline/config.json
data/baseline/stats.json
data/baseline/sinograms/chamber_0_recon_sino.hdf5 ... chamber_9_recon_sino.hdf5
```

Any other directory works; it is the argument of the commands below, which use `data/baseline`.

## Installation

Python 3.10 with numpy, scipy, h5py, hdf5plugin and opencv-python-headless, pinned in
`requirements.txt`:

```bash
pip install -r requirements.txt
```

`hdf5plugin` is required: the simulator writes compressed datasets. Without it, reading fails
with a misleading `OSError: can't open directory (.../plugin)`.

Docker (the data are mounted, not copied into the image):

```bash
docker build -t chamberreg .
docker run --rm -v "$PWD/data:/data:ro" -v "$PWD/results:/results" chamberreg \
    run /data/baseline --out /results --jobs 5 --threads 8
```

## Running

```bash
python -m chamberreg run data/baseline --out results --jobs 5 --threads 8   # all chambers
python -m chamberreg one data/baseline chamber_7                           # one chamber, JSON to stdout
python -m chamberreg summarize results                                     # rebuild the summary
```

## Output

`results/<chamber>.json`, one file per chamber:

| key | meaning |
|---|---|
| `axis_error_deg` | angle between the estimated and the true symmetry axis (orientation ignored) |
| `tilt_error_deg`, `orientation_error_deg` | errors of the two angles of Eq. (11), the quantities of Table 3 in the paper |
| `tilt_deg`, `orientation_deg` | the two angles of the estimated axis, measured from `stage1.slicing_axis` like the tilt reported by Stage 1 |
| `translation_error_mm` | displacement of the model reference point (centroid of the outline, wall + cavity) from its true position |
| `translation_axial_mm`, `translation_perpendicular_mm` | the same displacement along / across the true axis |
| `iou_cavity`, `iou_wall` | IoU of the model rasterised in the estimated pose against the same model rasterised in the true pose (same rasteriser on both sides) |
| `iou_cavity_vs_simulator`, `iou_cavity_ceiling` | cavity IoU against the simulator mask, and the IoU the true pose itself reaches against that mask |
| `success` | `axis_error_deg < 1` and `translation_error_mm < 1` |
| `stage1` | the array axis sliced along (`slicing_axis`), slab, edge threshold (`tau_fraction` = τ / contrast), slice range, number of slices and centres, RANSAC inlier fraction, tilt |
| `stage2` | chosen orientation `sign`, search reach, coarse and final axial offset λ, perpendicular offset, and how often each search window was re-centred (`axial_recentrings`, `perp_recentrings`) |
| `time` | seconds per stage; `registration_s` excludes the evaluation |

A failed chamber gets `error` and `traceback` instead. `results/summary.json` and `summary.md` hold
the per-chamber table plus the median, mean ± SD and maximum of each error, the number of successful
chambers; both are written by `run` and rebuilt by `summarize`.

The same output for the ten chambers of the record is kept in `results_reference/`, so
`diff -r results_reference results` after a run shows whether it reproduced them.

The ground truth comes from the simulator mask, which is the CAD model voxelised in the scan pose.
The true axis comes from the weighted second moments of the fractional cavity mask. The true
position comes from the cavity centroid, and the orientation from the side on which the wall mass
lies. The method itself never reads the mask.

## Expected results

The ten chambers of the published dataset:

| chamber | axis [°] | translation [mm] | IoU cavity | IoU wall |
|---|---|---|---|---|
| chamber_0 | 0.0167 | 0.0106 | 0.9990 | 0.9964 |
| chamber_1 | 0.0052 | 0.0127 | 0.9993 | 0.9974 |
| chamber_2 | 0.0042 | 0.0067 | 0.9996 | 0.9985 |
| chamber_3 | 0.0242 | 0.0051 | 0.9994 | 0.9975 |
| chamber_4 | 0.0016 | 0.0139 | 0.9994 | 0.9965 |
| chamber_5 | 0.0020 | 0.0040 | 0.9996 | 0.9982 |
| chamber_6 | 0.0058 | 0.0239 | 0.9976 | 0.9941 |
| chamber_7 | 0.0527 | 0.0062 | 0.9988 | 0.9948 |
| chamber_8 | 0.0059 | 0.0061 | 0.9996 | 0.9983 |
| chamber_9 | 0.0057 | 0.0102 | 0.9992 | 0.9974 |

| quantity | median | mean ± SD | max |
|---|---|---|---|
| axis error [°] | 0.0058 | 0.0124 ± 0.0158 | 0.0527 |
| tilt error [°] | 0.0029 | 0.0077 ± 0.0159 | 0.0527 |
| orientation error [°] | 0.0197 | 0.0413 ± 0.0561 | 0.1607 |
| translation error [mm] | 0.0085 | 0.0099 ± 0.0059 | 0.0239 |
| successful (axis < 1°, translation < 1 mm) | 10 / 10 | | |

For the paper's full series of 30 baseline chambers the reported values are a median axis error of
0.0063° and a median translation error of 0.0077 mm.

**chamber_7** has the largest axis error (0.053°): being short, wide and tilted by 15.2°, it has the
narrowest band of complete elliptical cross-sections in the set (6.2 mm, against 7.7 mm for the next
one), so only 33 slices pass the ellipse validation.

## Data format (HDF5, one file per chamber: `sinograms/<chamber>_recon_sino.hdf5`)

| dataset | content |
|---|---|
| `recon` | `(512, 512, 512)` float32 reconstruction (linear attenuation; air ≈ 0) |
| `mask` | `(n_materials, 512, 512, 512)` uint8, fractional occupancy 0–255 per material; channel order in `voxelization_data["materials"]` |
| `creation_data` | JSON: chamber `type`, generator `parameters`, `materials` (name → id), `empty interior volume` |
| `input_voxel_size` | voxel size in mm (0.2 here) |
| `volume_<id>` | voxelised volume of each material in mm³; `volume_1` is the cavity, its analytic value is `creation_data["empty interior volume"]` |

Material ids are drawn per chamber (for example, the wall is 13 in `chamber_0` and 7 in
`chamber_3`), so they are always read from `creation_data["materials"]`. Id 1 is the cavity (the
empty interior with the electrode subtracted). There are no STEP files: the CAD model is built from
`parameters` (`body_rad`, `body_height`, `thickness`, `neck_rad`, `neck_height` and the electrode
parameters) as a solid of revolution. Its closed-form cavity volume reproduces
`empty interior volume` exactly. Array axes map to the model as (i0, i1, i2); in Stage 1 a slice
is indexed (row, column) = (i1, i2), and the ellipse centres are (x, y) = (column, row).

## Method parameters

All of them are in `chamberreg/config.py`, one constant each, with the section or equation of
the paper they come from; nothing below is hard-coded elsewhere.

| step | setting |
|---|---|
| coarse segmentation | volume subsampled to ~0.4 mm (×2 at 0.2 mm, ×4 at 0.1 mm), Gaussian σ = 1.5, threshold at the median plus four median absolute deviations (refused if the MAD is zero), largest component; ROI = bounding box + 12 voxels |
| slicing direction | perpendicular to the array axis closest to the rough chamber axis, from the second moments of the coarse segmentation |
| slice band | slices whose filled cross-section exceeds 50 % of the maximum |
| edge threshold | τ = c · (material − air), c from {0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30}, chosen on 9 probe slices (most valid slices, then lowest residual) |
| ellipses | Scharr magnitude (kernels /32), `cv2.findContours` (list, no approximation), N_c = 4 largest contours with ≥ 30 points, indexed radially outward, Fitzgibbon fit; valid if axis ratios are within [1, 2] with SD ≤ 0.08, the centres are within 2 px of each other, residual < 1 px |
| slice range | longest run of valid slices (gaps ≤ 6), minus 3 at each end, ≥ 10 valid slices and ≥ 3 mm axial span |
| line | RANSAC, threshold 0.1 · s_v, 3000 iterations, seed 0; PCA of the inliers |
| score | CT normalised with air = 0 and the brightest voxel = 1; wall sum minus cavity sum; λ = 0 puts the outline centroid at the centroid μ of the inlier centres |
| coarse axial | ROI subsampled ×2, λ ∈ ±max(12 mm, 0.5 · model length) in 0.5 mm steps, both orientations, global maximum |
| coarse range | 0.5 · model length bounds the distance between the centroid of the model outline and the centroid of the ellipse centres, which lies inside the body slab; measured offsets are at most 4.6 mm |
| fine axial | ±1 mm in 0.1 mm steps at full resolution, parabola on the largest window centred on the sampled maximum |
| perpendicular | 5 × 5 grid, ±0.5 mm in 0.25 mm steps, bivariate quadratic |

Both searches re-centre their window if the sampled maximum lies on its border, and count how often
they do (`axial_recentrings`, `perp_recentrings`); on the ten chambers both counts are zero.

## Citation

If you use this software, please cite the paper it implements:

```bibtex
@misc{ogonowski2026cadtoctregistrationcylindricalobjects,
      title={CAD-to-CT Registration of Cylindrical Objects via Ellipse-Based Axis Estimation},
      author={Aleksander Ogonowski and Mikołaj Mrozowski and Daniel Więcek and Arkadiusz Ćwiek and Konrad Klimaszewski and Rafał Możdżonek and Adam Padee and Lech Raczyński and Piotr Wasiuk and Wojciech Wiślicki and Michał Matusiak and Sławomir Wronka},
      year={2026},
      eprint={2606.02935},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.02935},
}
```

## License

MIT, see `LICENSE`. It asks one thing of anyone who redistributes this code or a substantial part of
it: keep the copyright notice and the licence text with it. Citing the paper is not a condition of
the licence, it is the request above.

## Acknowledgement

This work was completed with resources provided by the Świerk Computing Centre at the National Centre
for Nuclear Research. The AI4GUM – INFOSTRATEG7/0001/2023 project is co-financed by the National Centre
for Research and Development under the 7th call of the Strategic Programme of Scientific Research and
Experimental Development ‘INFOSTRATEG – Advanced information, telecommunications and mechatronic
technologies’, agreement No. INFOSTRATEG7/0001/2023.
