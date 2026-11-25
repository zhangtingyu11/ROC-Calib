# Paper C++ solver

## Version and source integrity

The numerical source comes from the adopted 2026-10-26 frozen experiment. `native/paper-source-sha256.json` records hashes of the archived files. Geometry, region/contour objectives, support construction, optimizers and boundary correction remain byte-identical. Release changes are limited to `data.cpp` (portable input paths/admission), `main.cpp` (default root and group label), `core.hpp` (specific OpenCV public includes), `CMakeLists.txt` (component-only OpenCV installations), and README. Unused historical appearance source files are retained for provenance but are not linked or enabled.

## Inputs

Pass `--root DIRECTORY` containing `request.json`, ACP1 point clouds, and optionally `checksums.json`. The existing CARLA request is an example. Each pair specifies `annotation_id`, `cloud_url`, `point_indices`, `mask_width`, `mask_height`, alternating row-major `mask_rle` starting with background, row-major 3x3 `intrinsic`, `distortion`, and `distortion_model`. `/data/cache/` URLs resolve relative to this directory; relative paths also resolve there. The optional existing `temporal_projection` object is forwarded unchanged. Data must already have the temporal compensation appropriate to its camera model; the loader does not invent timing metadata or apply additional deskew.

ACP1: 16-byte header, magic ACP1, uint32 little-endian record count at byte4; each16-byte record stores float32 XYZ at offsets0,4,8 and a validity/type byte at12. Selected type2 or nonfinite records are rejected. All selected indices retain order; no downsampling is performed. Original masks and cloud files are hashed on entry and checked again before exit. Matching entries in optional checksums.json are verified; it is an integrity check, not reference geometry. Private research manifests are no longer required. `base_matrix` is not used to initialize the fit: initialization comes from the confirmed objects.

## Commands and output

Build and run as in the repository README. `--group` is an output label. `--prepare-only` reads inputs and computes PnP; `--state` and `--audit` are diagnostic modes. Normal calibration does not read expected results or reference extrinsics. `solve.matrix` is camera-from-LiDAR. `solve.completed` and `solve.finalAuditCompleted` report completion, not global optimality. Output and checkpoint files must be stored outside immutable input assets.

There is no internal time cutoff. An external `timeout` can be used by an operator, but interrupted runs must not be reported as complete. OpenMP objective evaluation explicitly uses4 threads. BLAS is limited to1 thread by the suggested command.

The example downloader uses Python's standard library only; no Python is required by the C++ solver. Building requires Linux/POSIX, C++17, OpenCV, Eigen3, OpenSSL, OpenMP, BLAS and LAPACK. Numerical results may vary slightly across native library versions. The archival experiment used OpenCV4.2; release validation uses the available OpenCV4.6 environment. See release-validation.json for actual results, not a speed claim.

## Legacy application

This release adds the paper C++ CLI. The existing web API and Python `roc-calib` command are unchanged legacy implementations. The native request uses the same saved pair fields, but users must supply accessible ACP1 assets and compatible camera/temporal fields. No claim is made that browser calibration has been upgraded.

## Release validation limits

The OpenCV4.6 smoke run completed with20 pairs/2258 selected points and internal audits passing. Compared with the archived OpenCV4.2 result, max PnP matrix difference is5.73e-9, final matrix difference0.00272131, and final loss difference0.00211402. This is NOT an exact numerical reproduction. The archived expected result remains unchanged; no tolerance or expected output was adjusted to hide the difference. See [validation record](../native/release-validation.json).

An optional Ubuntu20.04/OpenCV4.2 container recipe is provided to approach the original library environment:

```sh
docker build -f native/Dockerfile -t roc-calib-paper .
mkdir -p outputs
docker run --rm -v "$PWD/demo/carla:/input:ro" -v "$PWD/outputs:/output" roc-calib-paper --root /input --group 6 --output /output/paper-carla.json
```

This recipe was not built in the release validation environment (Docker unavailable). It is not a digest-pinned reconstruction of the original research container. The CPU source build above was compiled and run. Exact original-environment numerical reproduction remains unverified by this release.
