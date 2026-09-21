"""Default parameters; sections refer to the paper."""

# Stage 1 (Sec. 3.3)
SEGMENTATION_VOXEL_MM = 0.4
SEGMENTATION_SIGMA = 1.5
SEGMENTATION_K_MAD = 4.0        # threshold = median + K * MAD over the whole volume
MIN_COMPONENT_FRACTION = 0.2    # border-free component preferred above this share of the largest
BODY_FRACTION = 0.5             # slab: cross-section above this share of the largest
ROI_MARGIN_VOXELS = 12
TAU_LADDER = (0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30)   # tau = c * (g_mat - g_air)
N_CONTOURS = 4
MIN_CONTOUR_POINTS = 30
RATIO_RANGE = (1.0, 2.0)
RATIO_TOL = 0.08
CENTRE_TOL_PX = 2.0
RMS_TOL_PX = 1.0
MAX_GAP = 6                     # invalid slices tolerated inside a run
MARGIN = 3                      # slices trimmed at each end
MIN_SLICES = 10                 # below it PCA can return a direction lying in one plane
RANSAC_THRESHOLD_VOXELS = 0.1
RANSAC_ITERATIONS = 3000
RANSAC_SEED = 0

# Stage 2 (Sec. 3.4)
COARSE_REACH_MM = 12.0          # reach = max(this, COARSE_REACH_LENGTH_FRACTION * model length)
COARSE_REACH_LENGTH_FRACTION = 0.5
COARSE_STEP_MM = 0.5
COARSE_SUBSAMPLE = 2
FINE_HALF_MM = 1.0
FINE_STEP_MM = 0.1
PERP_HALF_MM = 0.5              # with PERP_STEP_MM gives a 5 x 5 grid
PERP_STEP_MM = 0.25
MAX_MOVES = 10                  # re-centrings when the maximum falls on a window border
CHUNK_VOXELS = 4e6              # temporary arrays per thread, ~100 MB

CENTROID_RES_MM = 0.01
THREADS = 8

# a registration counts as successful below both (Tab. 5)
SUCCESS_AXIS_DEG = 1.0
SUCCESS_TRANSLATION_MM = 1.0
