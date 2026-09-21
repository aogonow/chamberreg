# chamberreg

Two-stage CAD-to-CT registration of cylindrical ionisation chambers: ellipse-based axis
estimation and region-aware translation search.

This is the reproduction package for

> A. Ogonowski, M. Mrozowski, D. Więcek, A. Ćwiek, K. Klimaszewski, R. Możdżonek, A. Padee,
> L. Raczyński, P. Wasiuk, W. Wiślicki, M. Matusiak, S. Wronka,
> *CAD-to-CT Registration of Cylindrical Objects via Ellipse-Based Axis Estimation*,
> [arXiv:2606.02935](https://arxiv.org/abs/2606.02935).

The method registers the CAD model of a cylindrical chamber to its CT reconstruction in two
stages. Stage 1 estimates the symmetry axis from ellipses fitted to the gradient contours of
axial slices, by RANSAC and PCA of their centres. Stage 2 places the model along that axis by
maximising a score that counts CT intensity inside the wall of the model and subtracts it inside
the cavity. Neither stage needs an intensity calibration, a starting pose or manual landmarks.

## Status

The code is being prepared for release together with the paper and will be added here shortly.

## Data

The ten simulated chambers the package runs on, and the real chamber scans used for the
segmentation experiment, are published separately:

| data | where |
|---|---|
| simulated chambers, baseline acquisition | Zenodo (DOI to be added) |
| real chamber CT scans, training and test | [10.5281/zenodo.20797364](https://doi.org/10.5281/zenodo.20797364) |

## Acknowledgement


Please cite as:

@misc{ogonowski2026cadtoctregistrationcylindricalobjects,
      title={CAD-to-CT Registration of Cylindrical Objects via Ellipse-Based Axis Estimation}, 
      author={Aleksander Ogonowski and Mikołaj Mrozowski and Daniel Więcek and Arkadiusz Ćwiek and Konrad Klimaszewski and Rafał Możdżonek and Adam Padee and Lech Raczyński and Piotr Wasiuk and Wojciech Wiślicki and Michał Matusiak and Sławomir Wronka},
      year={2026},
      eprint={2606.02935},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.02935}, 
}

This work was completed with resources provided by the Swierk Computing Centre at the National Centre for
Nuclear Research. The AI4GUM – INFOSTRATEG7/0001/2023 project is co-financed by the National Centre for
Research and Development under the 7th call of the Strategic Programme of Scientific Research and Experimental
Development ‘INFOSTRATEG – Advanced information, telecommunications and mechatronic technologies’, agree-
ment No. INFOSTRATEG7/0001/2023.
